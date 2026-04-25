from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import scipy.sparse as sp
import torch
from py3gpp import nrLDPCDecode
from py3gpp.nrLDPCEncode import _encode, _gen_submat, _lift_basegraph, _load_basegraph

from nr_phy_simu.common.torch_utils import BIT_DTYPE, REAL_DTYPE, as_int_tensor, as_real_tensor, to_numpy


@dataclass(frozen=True)
class UlschLdpcInfo:
    crc: str
    tb_crc_bits: int
    base_graph: int
    num_code_blocks: int
    cb_crc_bits: int
    num_filler_bits: int
    zc: int
    cb_input_bits: int
    encoded_block_bits: int


@dataclass(frozen=True)
class LdpcDecoderStructure:
    parity_check: sp.csr_matrix
    edge_var_indices: torch.Tensor
    row_edge_groups: tuple[torch.Tensor, ...]
    col_edge_groups: tuple[torch.Tensor, ...]


def get_ulsch_ldpc_info(tbs: int, target_code_rate: float) -> UlschLdpcInfo:
    assert tbs >= 0
    assert isinstance(tbs, int)
    assert 0 < target_code_rate < 1

    bgn_info = _get_base_graph_info(tbs, target_code_rate)
    cbs_info = _get_code_block_info(bgn_info["B"], bgn_info["BGN"])
    encoded_block_bits = (66 if bgn_info["BGN"] == 1 else 50) * cbs_info["Zc"]
    return UlschLdpcInfo(
        crc=bgn_info["CRC"],
        tb_crc_bits=bgn_info["L"],
        base_graph=bgn_info["BGN"],
        num_code_blocks=cbs_info["C"],
        cb_crc_bits=cbs_info["Lcb"],
        num_filler_bits=cbs_info["F"],
        zc=cbs_info["Zc"],
        cb_input_bits=cbs_info["K"],
        encoded_block_bits=encoded_block_bits,
    )


def encode_ldpc_codeblocks(code_blocks: torch.Tensor | np.ndarray, base_graph: int) -> torch.Tensor:
    code_blocks_np = np.asarray(to_numpy(code_blocks), dtype=np.int8).copy()
    input_bits, num_code_blocks = code_blocks_np.shape
    if base_graph == 1:
        num_systematic_nodes = 22
        num_codeword_nodes = 66
    else:
        num_systematic_nodes = 10
        num_codeword_nodes = 50

    zc = int(input_bits / num_systematic_nodes)
    if zc not in _get_z_list():
        raise ValueError(f"Unsupported lifting size Zc={zc}")

    codeword_bits = int(zc * num_codeword_nodes)
    coded = np.zeros((codeword_bits + 2 * zc, num_code_blocks), dtype=np.int8)
    lifting_kb = _select_lifting_kb(input_bits, base_graph)
    lifting_set_index = _find_lifting_set_index(lifting_kb, input_bits)

    filler_indices = code_blocks_np[:, 0] == -1
    code_blocks_np[filler_indices, :] = 0

    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(base_matrix, num_systematic_nodes, zc, base_graph)
    for code_block_index in range(num_code_blocks):
        coded[:, code_block_index] = _encode(
            code_blocks_np[:, code_block_index],
            pcm_a,
            pcm_b_inv,
            pcm_c1,
            pcm_c2,
        )

    filler_indices_out = np.append(filler_indices, np.repeat(False, codeword_bits + 2 * zc - input_bits))
    coded[filler_indices_out, :] = -1
    return torch.as_tensor(coded[2 * zc :, :], dtype=BIT_DTYPE)


def rate_match_ulsch_ldpc(
    encoded_code_blocks: torch.Tensor | np.ndarray,
    out_length: int,
    rv: int,
    modulation: str,
    num_layers: int,
) -> torch.Tensor:
    assert num_layers == 1, "nLayers > 1 is not yet implemented"
    assert rv in [0, 1, 2, 3], "rv has to be in [0, 1, 2, 3]"

    encoded_code_blocks = as_int_tensor(encoded_code_blocks, dtype=BIT_DTYPE)
    qm = _modulation_order(modulation)
    n = encoded_code_blocks.shape[0]
    c = encoded_code_blocks.shape[1]
    z_list = torch.tensor(_get_z_list(), dtype=torch.int64, device=encoded_code_blocks.device)
    if bool(torch.any(n == z_list * 66)):
        base_graph = 1
        num_codeword_nodes = 66
    elif bool(torch.any(n == z_list * 50)):
        base_graph = 2
        num_codeword_nodes = 50
    else:
        raise ValueError(f"Unsupported encoded code-block length: N={n}")
    zc = int(n / num_codeword_nodes)
    ncb = n
    k0 = _rate_matching_start(base_graph, rv, ncb, n, zc)

    rematched: list[torch.Tensor] = []
    for code_block_index in range(c):
        if code_block_index <= c - (out_length / (num_layers * qm)) % c - 1:
            e = qm * num_layers * int(math.floor(out_length / (qm * num_layers * c)))
        else:
            e = qm * num_layers * int(math.ceil(out_length / (qm * num_layers * c)))
        rematched.append(_rate_match_single(encoded_code_blocks[:, code_block_index], e, k0, ncb, qm))
    return torch.cat(rematched).to(dtype=BIT_DTYPE)


def rate_recover_ulsch_ldpc(
    llrs: torch.Tensor | np.ndarray,
    trblklen: int,
    target_code_rate: float,
    rv: int,
    modulation: str,
    num_layers: int,
    num_code_blocks: int | None = None,
) -> torch.Tensor:
    assert 0 < target_code_rate < 1, "R has to satisfy 0 < R < 1"
    assert rv in [0, 1, 2, 3], "rv has to be in [0, 1, 2, 3]"

    llrs = as_real_tensor(llrs).reshape(-1)
    qm = _modulation_order(modulation)
    info = get_ulsch_ldpc_info(trblklen, target_code_rate)
    c = info.num_code_blocks if num_code_blocks is None else num_code_blocks
    n = info.encoded_block_bits
    ncb = n
    k0 = _rate_matching_start(info.base_graph, rv, ncb, n, info.zc)

    g = int(llrs.numel())
    out = torch.zeros((n, c), dtype=REAL_DTYPE, device=llrs.device)
    index = 0
    for code_block_index in range(c):
        if code_block_index <= c - (g / (num_layers * qm)) % c - 1:
            e = qm * num_layers * int(math.floor(g / (qm * num_layers * c)))
        else:
            e = qm * num_layers * int(math.ceil(g / (qm * num_layers * c)))

        deconcatenated = llrs[index : index + e]
        index += e
        out[:, code_block_index] = _rate_recover_single(
            deconcatenated,
            info=info,
            k0=k0,
            ncb=ncb,
            qm=qm,
        )
    return out


def decode_ulsch_ldpc(
    llrs: torch.Tensor | np.ndarray,
    info: UlschLdpcInfo,
    max_num_iter: int,
    min_sum_scaling: float = 0.75,
    enable_py3gpp_fallback: bool = True,
) -> torch.Tensor:
    llrs = as_real_tensor(llrs)
    decoded = torch.zeros((info.cb_input_bits, llrs.shape[1]), dtype=BIT_DTYPE, device=llrs.device)
    for code_block_index in range(llrs.shape[1]):
        decoded[:, code_block_index] = _decode_single_code_block(
            llrs[:, code_block_index],
            info,
            max_num_iter=max_num_iter,
            min_sum_scaling=min_sum_scaling,
            enable_py3gpp_fallback=enable_py3gpp_fallback,
        )
    return decoded


def _rate_match_single(codeword: torch.Tensor, out_length: int, k0: int, ncb: int, qm: int) -> torch.Tensor:
    codeword = as_int_tensor(codeword, dtype=BIT_DTYPE)
    codeword_buffer = codeword[:ncb]
    num_filler_bits = int(torch.count_nonzero(codeword_buffer == -1).item())
    repeat_count = int(math.ceil(out_length / (codeword_buffer.numel() - num_filler_bits)))
    tiled = codeword_buffer.repeat(repeat_count)
    tiled = torch.roll(tiled, shifts=-k0)
    selected = tiled[tiled != -1][:out_length]
    return selected.reshape(qm, int(out_length / qm)).T.reshape(-1)


def _rate_recover_single(
    deconcatenated: torch.Tensor,
    info: UlschLdpcInfo,
    k0: int,
    ncb: int,
    qm: int,
) -> torch.Tensor:
    deconcatenated = as_real_tensor(deconcatenated)
    e = int(deconcatenated.numel())
    deconcatenated = deconcatenated.reshape(int(e / qm), qm).T.reshape(-1)

    k = int(info.cb_input_bits - 2 * info.zc)
    kd = int(k - info.num_filler_bits)
    num_filler_bits = int(min(k, ncb) - kd)
    n_buffer = ncb - num_filler_bits

    indices = torch.arange(ncb, dtype=torch.int64, device=deconcatenated.device).repeat(int(math.ceil(e / n_buffer)))
    indices = torch.roll(indices, shifts=-k0)
    indices = indices[~((indices >= kd) & (indices < k))]
    indices = indices[:e]

    out = torch.zeros(info.encoded_block_bits, dtype=REAL_DTYPE, device=deconcatenated.device)
    out[kd:k] = torch.inf

    if e > n_buffer:
        repeats = int(math.floor(e / n_buffer))
        for repeat_idx in range(repeats):
            start = repeat_idx * n_buffer
            stop = (repeat_idx + 1) * n_buffer
            out[indices[:n_buffer]] += deconcatenated[start:stop]
        rem_bits = int(e % n_buffer)
        if rem_bits:
            out[indices[:rem_bits]] += deconcatenated[-rem_bits:]
    else:
        out[indices] = deconcatenated
    return out


def _rate_matching_start(base_graph: int, rv: int, ncb: int, n: int, zc: int) -> int:
    if base_graph == 1:
        starts = {0: 0, 1: math.floor(17 * ncb / n) * zc, 2: math.floor(33 * ncb / n) * zc, 3: math.floor(56 * ncb / n) * zc}
    else:
        starts = {0: 0, 1: math.floor(13 * ncb / n) * zc, 2: math.floor(25 * ncb / n) * zc, 3: math.floor(43 * ncb / n) * zc}
    return int(starts[rv])


def _modulation_order(modulation: str) -> int:
    normalized = modulation.upper()
    mapping = {
        "PI/2-BPSK": 1,
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
        "64QAM": 6,
        "256QAM": 8,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported modulation type: {modulation}")
    return mapping[normalized]


def _find_lifting_set_index(kb: int, k: int) -> int:
    z_array = _get_z_array()
    min_value = 100000
    lifting_set_index = None
    for idx, sizes in enumerate(z_array):
        for size in sizes:
            candidate = kb * size
            if candidate >= k and candidate < min_value:
                min_value = candidate
                lifting_set_index = idx
    if lifting_set_index is None:
        raise ValueError(f"Unable to find lifting-set index for K={k}")
    return lifting_set_index


def _select_lifting_kb(input_bits: int, base_graph: int) -> int:
    if base_graph == 1:
        return 22
    if input_bits > 640:
        return 10
    if input_bits > 560:
        return 9
    if input_bits > 192:
        return 8
    return 6


def _decode_single_code_block(
    llrs: torch.Tensor,
    info: UlschLdpcInfo,
    max_num_iter: int,
    min_sum_scaling: float,
    enable_py3gpp_fallback: bool,
) -> torch.Tensor:
    structure = _ldpc_decoder_structure(info.base_graph, info.cb_input_bits, info.zc)
    llrs = as_real_tensor(llrs)
    punctured_prefix = torch.zeros(2 * info.zc, dtype=REAL_DTYPE, device=llrs.device)
    channel_llr = torch.cat([punctured_prefix, llrs])
    channel_llr = torch.nan_to_num(channel_llr, nan=0.0, posinf=1e6, neginf=-1e6)

    posterior = _normalized_min_sum_decode(
        channel_llr,
        structure,
        max_num_iter=max_num_iter,
        scaling=min_sum_scaling,
    )
    if posterior is not None:
        return (posterior[: info.cb_input_bits] < 0).to(dtype=BIT_DTYPE)

    direct = _direct_decode_from_hard_decisions(llrs.reshape(-1, 1), info)
    if direct is not None:
        return direct[:, 0]

    if not enable_py3gpp_fallback:
        return (channel_llr[: info.cb_input_bits] < 0).to(dtype=BIT_DTYPE)

    decoded, _ = nrLDPCDecode(to_numpy(llrs.reshape(-1, 1)), info.base_graph, maxNumIter=max_num_iter)
    return torch.as_tensor(decoded[:, 0], dtype=BIT_DTYPE, device=llrs.device)


def _normalized_min_sum_decode(
    channel_llr: torch.Tensor,
    structure: LdpcDecoderStructure,
    max_num_iter: int,
    scaling: float,
) -> torch.Tensor | None:
    edge_var_indices = structure.edge_var_indices.to(device=channel_llr.device)
    row_edge_groups = structure.row_edge_groups
    col_edge_groups = structure.col_edge_groups

    v2c = channel_llr[edge_var_indices].clone()
    c2v = torch.zeros_like(v2c)
    posterior = channel_llr.clone()

    for _ in range(max_num_iter):
        for row_edges in row_edge_groups:
            row_edges_tensor = row_edges.to(device=channel_llr.device)
            incoming = v2c[row_edges_tensor]
            if incoming.numel() == 0:
                continue
            signs = torch.sign(incoming)
            signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
            abs_values = torch.abs(incoming)

            min_index = int(torch.argmin(abs_values).item())
            min1 = float(abs_values[min_index].item())
            if abs_values.numel() > 1:
                min2 = float(torch.min(torch.cat([abs_values[:min_index], abs_values[min_index + 1 :]])).item())
            else:
                min2 = min1
            total_sign = float(torch.prod(signs).item())

            outgoing = torch.full(abs_values.shape, scaling * min1, dtype=REAL_DTYPE, device=channel_llr.device)
            outgoing[min_index] = scaling * min2
            c2v[row_edges_tensor] = total_sign * signs * outgoing

        posterior = channel_llr.clone()
        for var_index, edge_ids in enumerate(col_edge_groups):
            if edge_ids.numel():
                edge_ids_tensor = edge_ids.to(device=channel_llr.device)
                posterior[var_index] += torch.sum(c2v[edge_ids_tensor])

        hard = (posterior < 0).to(dtype=BIT_DTYPE)
        if _parity_check_satisfied(hard, structure.parity_check):
            return posterior

        v2c = posterior[edge_var_indices] - c2v

    if _parity_check_satisfied((posterior < 0).to(dtype=BIT_DTYPE), structure.parity_check):
        return posterior
    return None


def _parity_check_satisfied(bits: torch.Tensor | np.ndarray, parity_check: sp.csr_matrix) -> bool:
    syndrome = parity_check.dot(np.asarray(to_numpy(bits), dtype=np.uint8)) % 2
    return not np.any(syndrome)


@lru_cache(maxsize=None)
def _ldpc_decoder_structure(base_graph: int, cb_input_bits: int, zc: int) -> LdpcDecoderStructure:
    lifting_kb = _select_lifting_kb(cb_input_bits, base_graph)
    lifting_set_index = _find_lifting_set_index(lifting_kb, cb_input_bits)
    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    parity_check = _lift_basegraph(base_matrix, zc).tocsr()

    edge_var_indices: list[int] = []
    row_edge_groups: list[torch.Tensor] = []
    col_edges: list[list[int]] = [[] for _ in range(parity_check.shape[1])]

    edge_id = 0
    for row_index in range(parity_check.shape[0]):
        cols = parity_check.indices[parity_check.indptr[row_index] : parity_check.indptr[row_index + 1]]
        row_edge_ids = torch.arange(edge_id, edge_id + len(cols), dtype=torch.int64)
        row_edge_groups.append(row_edge_ids)
        for col in cols:
            edge_var_indices.append(int(col))
            col_edges[int(col)].append(edge_id)
            edge_id += 1

    col_edge_groups = tuple(torch.tensor(edges, dtype=torch.int64) for edges in col_edges)
    return LdpcDecoderStructure(
        parity_check=parity_check,
        edge_var_indices=torch.tensor(edge_var_indices, dtype=torch.int64),
        row_edge_groups=tuple(row_edge_groups),
        col_edge_groups=col_edge_groups,
    )


def _direct_decode_from_hard_decisions(llrs: torch.Tensor, info: UlschLdpcInfo) -> torch.Tensor | None:
    hard = (as_real_tensor(llrs) < 0).to(dtype=BIT_DTYPE)
    if hard.ndim != 2:
        raise TypeError("LLR matrix must be 2-dimensional")

    decoded = torch.zeros((info.cb_input_bits, hard.shape[1]), dtype=BIT_DTYPE, device=hard.device)
    for code_block_index in range(hard.shape[1]):
        code_block = _recover_code_block_from_hard_bits(hard[:, code_block_index], info)
        if code_block is None:
            return None
        decoded[:, code_block_index] = code_block
    return decoded


def _recover_code_block_from_hard_bits(hard_bits: torch.Tensor, info: UlschLdpcInfo) -> torch.Tensor | None:
    zc = info.zc
    hard_bits = as_int_tensor(hard_bits, dtype=BIT_DTYPE)
    full_codeword = torch.cat([torch.zeros(2 * zc, dtype=BIT_DTYPE, device=hard_bits.device), hard_bits])
    kd = info.cb_input_bits - info.num_filler_bits
    full_codeword[kd : info.cb_input_bits] = 0

    parity_check, punctured_submatrix = _punctured_solver_matrices(info.base_graph, info.cb_input_bits, zc)
    parity_check = parity_check.to(device=hard_bits.device)
    punctured_submatrix = punctured_submatrix.to(device=hard_bits.device)
    rhs = torch.remainder(parity_check[:, 2 * zc :] @ full_codeword[2 * zc :].to(dtype=torch.int64), 2).to(dtype=BIT_DTYPE)
    solution = _solve_gf2(punctured_submatrix, rhs)
    if solution is None:
        return None

    full_codeword[: 2 * zc] = solution
    syndrome = torch.remainder(parity_check @ full_codeword.to(dtype=torch.int64), 2)
    if torch.any(syndrome):
        return None

    return full_codeword[: info.cb_input_bits]


@lru_cache(maxsize=None)
def _punctured_solver_matrices(base_graph: int, cb_input_bits: int, zc: int) -> tuple[torch.Tensor, torch.Tensor]:
    lifting_kb = _select_lifting_kb(cb_input_bits, base_graph)
    lifting_set_index = _find_lifting_set_index(lifting_kb, cb_input_bits)
    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    parity_check = _lift_basegraph(base_matrix, zc).astype(np.uint8).toarray() % 2
    parity_check_tensor = torch.as_tensor(parity_check, dtype=torch.int64)
    punctured_submatrix = parity_check_tensor[:, : 2 * zc].clone()
    return parity_check_tensor, punctured_submatrix


def _solve_gf2(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor | None:
    augmented = torch.cat(
        [torch.remainder(matrix.clone(), 2), torch.remainder(rhs.reshape(-1, 1), 2).to(dtype=torch.int64)],
        dim=1,
    )
    num_rows, num_cols_aug = augmented.shape
    num_cols = num_cols_aug - 1
    pivot_columns: list[int] = []
    pivot_row = 0

    for col in range(num_cols):
        candidate_rows = torch.nonzero(augmented[pivot_row:, col], as_tuple=False).reshape(-1) + pivot_row
        if candidate_rows.numel() == 0:
            continue
        row = int(candidate_rows[0].item())
        if row != pivot_row:
            augmented[[pivot_row, row], :] = augmented[[row, pivot_row], :].clone()
        for other_row in range(num_rows):
            if other_row != pivot_row and augmented[other_row, col]:
                augmented[other_row, :] ^= augmented[pivot_row, :]
        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row == num_rows:
            break

    for row in range(num_rows):
        if not torch.any(augmented[row, :num_cols]) and bool(augmented[row, num_cols]):
            return None

    solution = torch.zeros(num_cols, dtype=BIT_DTYPE, device=matrix.device)
    for row, col in enumerate(pivot_columns):
        solution[col] = augmented[row, num_cols].to(dtype=BIT_DTYPE)
    return solution


def _get_base_graph_info(a: int, r: float) -> dict[str, int | str]:
    if a <= 292 or (a <= 3824 and r <= 0.67) or r <= 0.25:
        bgn = 2
    else:
        bgn = 1

    if a > 3824:
        l = 24
        crc = "24A"
    else:
        l = 16
        crc = "16"
    return {"BGN": bgn, "B": a + l, "L": l, "CRC": crc}


def _get_code_block_info(b: int, base_graph: int) -> dict[str, int]:
    kcb = 8448 if base_graph == 1 else 3840

    if b <= kcb:
        l = 0
        c = 1
        bd = b
    else:
        l = 24
        c = int(math.ceil(b / (kcb - l)))
        bd = b + c * l

    kd = int(math.ceil(bd / c))
    if base_graph == 1:
        kb = 22
    else:
        kb = 10 if b > 640 else 9 if b > 560 else 8 if b > 192 else 6

    zc = min([x for x in _get_z_list() if kb * x >= kd])
    k = (22 if base_graph == 1 else 10) * zc
    return {
        "C": c,
        "Lcb": l,
        "F": int(k - kd),
        "K": int(k),
        "Zc": int(zc),
    }


def _get_z_array() -> list[list[int]]:
    return [
        [2, 4, 8, 16, 32, 64, 128, 256],
        [3, 6, 12, 24, 48, 96, 192, 384],
        [5, 10, 20, 40, 80, 160, 320],
        [7, 14, 28, 56, 112, 224],
        [9, 18, 36, 72, 144, 288],
        [11, 22, 44, 88, 176, 352],
        [13, 26, 52, 104, 208],
        [15, 30, 60, 120, 240],
    ]


def _get_z_list() -> list[int]:
    z_list: list[int] = []
    z_list += list(range(2, 17))
    z_list += list(range(18, 33, 2))
    z_list += list(range(36, 65, 4))
    z_list += list(range(72, 129, 8))
    z_list += list(range(144, 257, 16))
    z_list += list(range(288, 385, 32))
    return z_list
