from __future__ import annotations

import contextlib
import io

import numpy as np
from py3gpp import (
    nrCRCDecode,
    nrCodeBlockDesegmentLDPC,
)

from nr_phy_simu.common.interfaces import ChannelDecoder
from nr_phy_simu.common.ulsch_ldpc import (
    decode_ulsch_ldpc,
    get_ulsch_ldpc_info,
    rate_recover_ulsch_ldpc,
)
from nr_phy_simu.config import SimulationConfig


class NrLdpcDecoder(ChannelDecoder):
    def __init__(self) -> None:
        self.last_crc_ok: bool | None = None

    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Decode one transport block from descrambled soft bits.

        Args:
            llrs: Descrambled LLR sequence for the scheduled codeword.
            config: Full simulation configuration that defines UL-SCH parameters.

        Returns:
            Decoded transport-block bit sequence after CRC removal.
        """
        tbs = int(config.link.transport_block_size or 0)
        if tbs <= 0:
            raise ValueError("transport_block_size must be resolved before LDPC decoding.")

        info = get_ulsch_ldpc_info(tbs, config.link.code_rate)
        recovered = rate_recover_ulsch_ldpc(
            llrs,
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
            tb_with_crc, _ = nrCodeBlockDesegmentLDPC(decoded_cbs, info.base_graph, tbs + info.tb_crc_bits)
        decoded, crc_error = nrCRCDecode(tb_with_crc.astype(np.int8), info.crc)
        self.last_crc_ok = bool(crc_error == 0)
        return np.asarray(decoded).reshape(-1)[:tbs].astype(np.int8)


class HardDecisionBypassDecoder(ChannelDecoder):
    def __init__(self) -> None:
        self.last_crc_ok: bool | None = None

    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Convert descrambled LLRs to hard bits when channel decoding is bypassed.

        Args:
            llrs: Descrambled LLR sequence for the scheduled codeword.
            config: Full simulation configuration, unused by this bypass decoder.

        Returns:
            Hard-decision bit sequence derived directly from the LLR signs.
        """
        del config
        self.last_crc_ok = None
        return (np.asarray(llrs).reshape(-1) < 0.0).astype(np.int8)
