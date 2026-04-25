from __future__ import annotations

import contextlib
import io

import torch
from py3gpp import nrCRCDecode, nrCodeBlockDesegmentLDPC

from nr_phy_simu.common.interfaces import ChannelDecoder
from nr_phy_simu.common.torch_utils import BIT_DTYPE, as_real_tensor, to_numpy
from nr_phy_simu.common.ulsch_ldpc import (
    decode_ulsch_ldpc,
    get_ulsch_ldpc_info,
    rate_recover_ulsch_ldpc,
)
from nr_phy_simu.config import SimulationConfig


class NrLdpcDecoder(ChannelDecoder):
    def __init__(self) -> None:
        self.last_crc_ok: bool | None = None

    def decode(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Decode one transport block from descrambled soft bits."""
        tbs = int(config.link.transport_block_size or 0)
        if tbs <= 0:
            raise ValueError("transport_block_size must be resolved before LDPC decoding.")

        llrs = as_real_tensor(llrs)
        info = get_ulsch_ldpc_info(tbs, config.link.code_rate)
        recovered = rate_recover_ulsch_ldpc(
            to_numpy(llrs),
            trblklen=tbs,
            target_code_rate=config.link.code_rate,
            rv=int(config.link.mcs.rv),
            modulation=config.link.modulation,
            num_layers=config.link.num_layers,
        )
        decoded_cbs = decode_ulsch_ldpc(
            recovered,
            info,
            max_num_iter=int(config.decoder.ldpc_max_iterations),
            min_sum_scaling=float(config.decoder.ldpc_min_sum_scaling),
            enable_py3gpp_fallback=bool(config.decoder.ldpc_enable_py3gpp_fallback),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tb_with_crc, _ = nrCodeBlockDesegmentLDPC(to_numpy(decoded_cbs), info.base_graph, tbs + info.tb_crc_bits)
        decoded, crc_error = nrCRCDecode(tb_with_crc.astype("int8"), info.crc)
        self.last_crc_ok = bool(crc_error == 0)
        return torch.as_tensor(decoded, dtype=BIT_DTYPE).reshape(-1)[:tbs]


class HardDecisionBypassDecoder(ChannelDecoder):
    def __init__(self) -> None:
        self.last_crc_ok: bool | None = None

    def decode(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Convert descrambled LLRs to hard bits when channel decoding is bypassed."""
        del config
        self.last_crc_ok = None
        llrs = as_real_tensor(llrs)
        return (llrs.reshape(-1) < 0.0).to(dtype=BIT_DTYPE)
