from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from py3gpp import nrLDPCDecode
from py3gpp.nrLDPCEncode import _encode, _gen_submat, _lift_basegraph, _load_basegraph


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
    edge_var_indices: np.ndarray
    row_edge_groups: tuple[np.ndarray, ...]
    col_edge_groups: tuple[np.ndarray, ...]


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


def encode_ldpc_codeblocks(code_blocks: np.ndarray, base_graph: int) -> np.ndarray:
    code_blocks = code_blocks.copy()
    input_bits, num_code_blocks = code_blocks.shape
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

    filler_indices = code_blocks[:, 0] == -1
    code_blocks[filler_indices, :] = 0

    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(base_matrix, num_systematic_nodes, zc, base_graph)
    for code_block_index in range(num_code_blocks):
        coded[:, code_block_index] = _encode(
            code_blocks[:, code_block_index],
            pcm_a,
            pcm_b_inv,
            pcm_c1,
            pcm_c2,
        )

    filler_indices_out = np.append(filler_indices, np.repeat(False, codeword_bits + 2 * zc - input_bits))
    coded[filler_indices_out, :] = -1
    return coded[2 * zc :, :]


def rate_match_ulsch_ldpc(
    encoded_code_blocks: np.ndarray,
    out_length: int,
    rv: int,
    modulation: str,
    num_layers: int,
) -> np.ndarray:
    assert num_layers == 1, "nLayers > 1 is not yet implemented"
    assert rv in [0, 1, 2, 3], "rv has to be in [0, 1, 2, 3]"

    qm = _modulation_order(modulation)
    n = encoded_code_blocks.shape[0]
    c = encoded_code_blocks.shape[1]
    z_list = np.array(_get_z_list())
    if n in z_list * 66:
        base_graph = 1
        num_codeword_nodes = 66
    elif n in z_list * 50:
        base_graph = 2
        num_codeword_nodes = 50
    else:
        raise ValueError(f"Unsupported encoded code-block length: N={n}")
    zc = int(n / num_codeword_nodes)
    ncb = n
    k0 = _rate_matching_start(base_graph, rv, ncb, n, zc)

    rematched = np.empty(0, dtype=np.int8)
    for code_block_index in np.arange(c):
        if code_block_index <= c - np.mod(out_length / (num_layers * qm), c) - 1:
            e = qm * num_layers * int(np.floor(out_length / (qm * num_layers * c)))
        else:
            e = qm * num_layers * int(np.ceil(out_length / (qm * num_layers * c)))
        rematched = np.append(
            rematched,
            _rate_match_single(encoded_code_blocks[:, code_block_index], e, k0, ncb, qm),
        )
    return rematched.astype(np.int8)


def rate_recover_ulsch_ldpc(
    llrs: np.ndarray,
    trblklen: int,
    target_code_rate: float,
    rv: int,
    modulation: str,
    num_layers: int,
    num_code_blocks: int | None = None,
) -> np.ndarray:
    assert 0 < target_code_rate < 1, "R has to satisfy 0 < R < 1"
    assert rv in [0, 1, 2, 3], "rv has to be in [0, 1, 2, 3]"

    qm = _modulation_order(modulation)
    info = get_ulsch_ldpc_info(trblklen, target_code_rate)
    c = info.num_code_blocks if num_code_blocks is None else num_code_blocks
    n = info.encoded_block_bits
    ncb = n
    k0 = _rate_matching_start(info.base_graph, rv, ncb, n, info.zc)

    g = len(llrs)
    out = np.zeros((n, c), dtype=np.float64)
    index = 0
    for code_block_index in np.arange(c):
        if code_block_index <= c - np.mod(g / (num_layers * qm), c) - 1:
            e = qm * num_layers * int(np.floor(g / (qm * num_layers * c)))
        else:
            e = qm * num_layers * int(np.ceil(g / (qm * num_layers * c)))

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
    llrs: np.ndarray,
    info: UlschLdpcInfo,
    max_num_iter: int,
    min_sum_scaling: float = 0.75,
    enable_py3gpp_fallback: bool = True,
) -> np.ndarray:
    decoded = np.zeros((info.cb_input_bits, llrs.shape[1]), dtype=np.uint8)
    for code_block_index in range(llrs.shape[1]):
        decoded[:, code_block_index] = _decode_single_code_block(
            llrs[:, code_block_index],
            info,
            max_num_iter=max_num_iter,
            min_sum_scaling=min_sum_scaling,
            enable_py3gpp_fallback=enable_py3gpp_fallback,
        )
    return decoded


def _rate_match_single(codeword: np.ndarray, out_length: int, k0: int, ncb: int, qm: int) -> np.ndarray:
    num_filler_bits = np.count_nonzero(codeword[:ncb] == -1)
    tiled = np.tile(codeword, int(np.ceil(out_length / (len(codeword[:ncb]) - num_filler_bits))))
    tiled = np.roll(tiled, -k0)
    selected = tiled[tiled != -1][:out_length]
    return np.reshape(selected, (qm, int(out_length / qm))).ravel(order="F")


def _rate_recover_single(
    deconcatenated: np.ndarray,
    info: UlschLdpcInfo,
    k0: int,
    ncb: int,
    qm: int,
) -> np.ndarray:
    e = len(deconcatenated)
    deconcatenated = np.reshape(deconcatenated, (int(e / qm), qm)).ravel("F")

    k = int(info.cb_input_bits - 2 * info.zc)
    kd = int(k - info.num_filler_bits)
    num_filler_bits = int(min(k, ncb) - kd)
    n_buffer = ncb - num_filler_bits

    indices = np.tile(np.arange(ncb), int(np.ceil(e / n_buffer)))
    indices = np.roll(indices, -k0)
    indices = np.delete(indices, (indices >= kd) & (indices < k))
    indices = indices[:e]

    out = np.zeros(info.encoded_block_bits, dtype=np.float64)
    out[kd:k] = np.inf

    if e > n_buffer:
        repeats = int(np.floor(e / n_buffer))
        for repeat_idx in range(repeats):
            start = repeat_idx * n_buffer
            stop = (repeat_idx + 1) * n_buffer
            out[indices[:n_buffer]] += deconcatenated[start:stop]
        rem_bits = int(np.mod(e, n_buffer))
        if rem_bits:
            out[indices[:rem_bits]] += deconcatenated[-rem_bits:]
    else:
        out[indices] = deconcatenated
    return out


def _rate_matching_start(base_graph: int, rv: int, ncb: int, n: int, zc: int) -> int:
    if base_graph == 1:
        starts = {0: 0, 1: np.floor(17 * ncb / n) * zc, 2: np.floor(33 * ncb / n) * zc, 3: np.floor(56 * ncb / n) * zc}
    else:
        starts = {0: 0, 1: np.floor(13 * ncb / n) * zc, 2: np.floor(25 * ncb / n) * zc, 3: np.floor(43 * ncb / n) * zc}
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
    llrs: np.ndarray,
    info: UlschLdpcInfo,
    max_num_iter: int,
    min_sum_scaling: float,
    enable_py3gpp_fallback: bool,
) -> np.ndarray:
    structure = _ldpc_decoder_structure(info.base_graph, info.cb_input_bits, info.zc)
    punctured_prefix = np.zeros(2 * info.zc, dtype=np.float64)
    channel_llr = np.concatenate([punctured_prefix, np.asarray(llrs, dtype=np.float64)])
    channel_llr = np.nan_to_num(channel_llr, nan=0.0, posinf=1e6, neginf=-1e6)

    posterior = _normalized_min_sum_decode(
        channel_llr,
        structure,
        max_num_iter=max_num_iter,
        scaling=min_sum_scaling,
    )
    if posterior is not None:
        return (posterior[: info.cb_input_bits] < 0).astype(np.uint8)

    direct = _direct_decode_from_hard_decisions(llrs.reshape(-1, 1), info)
    if direct is not None:
        return direct[:, 0]

    if not enable_py3gpp_fallback:
        return (channel_llr[: info.cb_input_bits] < 0).astype(np.uint8)

    decoded, _ = nrLDPCDecode(llrs.reshape(-1, 1), info.base_graph, maxNumIter=max_num_iter)
    return decoded[:, 0].astype(np.uint8)


def _normalized_min_sum_decode(
    channel_llr: np.ndarray,
    structure: LdpcDecoderStructure,
    max_num_iter: int,
    scaling: float,
) -> np.ndarray | None:
    edge_var_indices = structure.edge_var_indices
    row_edge_groups = structure.row_edge_groups
    col_edge_groups = structure.col_edge_groups

    v2c = channel_llr[edge_var_indices].astype(np.float64, copy=True)
    c2v = np.zeros_like(v2c)
    posterior = channel_llr.copy()

    for _ in range(max_num_iter):
        for row_edges in row_edge_groups:
            incoming = v2c[row_edges]
            if incoming.size == 0:
                continue
            signs = np.sign(incoming)
            signs[signs == 0.0] = 1.0
            abs_values = np.abs(incoming)

            min_index = int(np.argmin(abs_values))
            min1 = float(abs_values[min_index])
            min2 = float(np.min(np.delete(abs_values, min_index))) if abs_values.size > 1 else min1
            total_sign = float(np.prod(signs))

            outgoing = np.full(abs_values.shape, scaling * min1, dtype=np.float64)
            outgoing[min_index] = scaling * min2
            c2v[row_edges] = total_sign * signs * outgoing

        posterior = channel_llr.copy()
        for var_index, edge_ids in enumerate(col_edge_groups):
            if edge_ids.size:
                posterior[var_index] += float(np.sum(c2v[edge_ids]))

        hard = (posterior < 0).astype(np.uint8)
        if _parity_check_satisfied(hard, structure.parity_check):
            return posterior

        v2c = posterior[edge_var_indices] - c2v

    if _parity_check_satisfied((posterior < 0).astype(np.uint8), structure.parity_check):
        return posterior
    return None


def _parity_check_satisfied(bits: np.ndarray, parity_check: sp.csr_matrix) -> bool:
    syndrome = parity_check.dot(bits.astype(np.uint8)) % 2
    return not np.any(syndrome)


@lru_cache(maxsize=None)
def _ldpc_decoder_structure(base_graph: int, cb_input_bits: int, zc: int) -> LdpcDecoderStructure:
    lifting_kb = _select_lifting_kb(cb_input_bits, base_graph)
    lifting_set_index = _find_lifting_set_index(lifting_kb, cb_input_bits)
    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    parity_check = _lift_basegraph(base_matrix, zc).tocsr()

    edge_var_indices: list[int] = []
    row_edge_groups: list[np.ndarray] = []
    col_edges: list[list[int]] = [[] for _ in range(parity_check.shape[1])]

    edge_id = 0
    for row_index in range(parity_check.shape[0]):
        cols = parity_check.indices[parity_check.indptr[row_index] : parity_check.indptr[row_index + 1]]
        row_edge_ids = np.arange(edge_id, edge_id + len(cols), dtype=np.int32)
        row_edge_groups.append(row_edge_ids)
        for col in cols:
            edge_var_indices.append(int(col))
            col_edges[int(col)].append(edge_id)
            edge_id += 1

    col_edge_groups = tuple(np.asarray(edges, dtype=np.int32) for edges in col_edges)
    return LdpcDecoderStructure(
        parity_check=parity_check,
        edge_var_indices=np.asarray(edge_var_indices, dtype=np.int32),
        row_edge_groups=tuple(row_edge_groups),
        col_edge_groups=col_edge_groups,
    )


def _direct_decode_from_hard_decisions(llrs: np.ndarray, info: UlschLdpcInfo) -> np.ndarray | None:
    hard = (llrs < 0).astype(np.uint8)
    if hard.ndim != 2:
        raise TypeError("LLR matrix must be 2-dimensional")

    decoded = np.zeros((info.cb_input_bits, hard.shape[1]), dtype=np.uint8)
    for code_block_index in range(hard.shape[1]):
        code_block = _recover_code_block_from_hard_bits(hard[:, code_block_index], info)
        if code_block is None:
            return None
        decoded[:, code_block_index] = code_block
    return decoded


def _recover_code_block_from_hard_bits(hard_bits: np.ndarray, info: UlschLdpcInfo) -> np.ndarray | None:
    zc = info.zc
    full_codeword = np.concatenate([np.zeros(2 * zc, dtype=np.uint8), hard_bits.astype(np.uint8)])
    kd = info.cb_input_bits - info.num_filler_bits
    full_codeword[kd : info.cb_input_bits] = 0

    parity_check, punctured_submatrix = _punctured_solver_matrices(info.base_graph, info.cb_input_bits, zc)
    rhs = (parity_check[:, 2 * zc :] @ full_codeword[2 * zc :]) % 2
    solution = _solve_gf2(punctured_submatrix, rhs.astype(np.uint8))
    if solution is None:
        return None

    full_codeword[: 2 * zc] = solution
    syndrome = (parity_check @ full_codeword) % 2
    if np.any(syndrome):
        return None

    return full_codeword[: info.cb_input_bits]


@lru_cache(maxsize=None)
def _punctured_solver_matrices(base_graph: int, cb_input_bits: int, zc: int) -> tuple[np.ndarray, np.ndarray]:
    lifting_kb = _select_lifting_kb(cb_input_bits, base_graph)
    lifting_set_index = _find_lifting_set_index(lifting_kb, cb_input_bits)
    base_matrix = _load_basegraph(lifting_set_index, base_graph)
    parity_check = _lift_basegraph(base_matrix, zc).astype(np.uint8).toarray() % 2
    punctured_submatrix = parity_check[:, : 2 * zc].copy()
    return parity_check, punctured_submatrix


def _solve_gf2(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    augmented = np.concatenate([matrix.copy() % 2, rhs.reshape(-1, 1) % 2], axis=1).astype(np.uint8)
    num_rows, num_cols_aug = augmented.shape
    num_cols = num_cols_aug - 1
    pivot_columns: list[int] = []
    pivot_row = 0

    for col in range(num_cols):
        candidate_rows = np.flatnonzero(augmented[pivot_row:, col]) + pivot_row
        if candidate_rows.size == 0:
            continue
        row = int(candidate_rows[0])
        if row != pivot_row:
            augmented[[pivot_row, row], :] = augmented[[row, pivot_row], :]
        for other_row in range(num_rows):
            if other_row != pivot_row and augmented[other_row, col]:
                augmented[other_row, :] ^= augmented[pivot_row, :]
        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row == num_rows:
            break

    for row in range(num_rows):
        if not np.any(augmented[row, :num_cols]) and augmented[row, num_cols]:
            return None

    solution = np.zeros(num_cols, dtype=np.uint8)
    for row, col in enumerate(pivot_columns):
        solution[col] = augmented[row, num_cols]
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
        c = int(np.ceil(b / (kcb - l)))
        bd = b + c * l

    kd = int(np.ceil(bd / c))
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
