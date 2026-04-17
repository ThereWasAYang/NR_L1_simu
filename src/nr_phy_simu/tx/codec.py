from __future__ import annotations

import numpy as np
from py3gpp import (
    nrCRCEncode,
    nrCodeBlockSegmentLDPC,
)

from nr_phy_simu.common.interfaces import ChannelCoder
from nr_phy_simu.common.ulsch_ldpc import (
    encode_ldpc_codeblocks,
    get_ulsch_ldpc_info,
    rate_match_ulsch_ldpc,
)
from nr_phy_simu.config import SimulationConfig


class NrLdpcCoder(ChannelCoder):
    """
    NR LDPC coding chain:
    TB CRC -> code block segmentation -> LDPC encode -> rate matching.
    """

    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Encode one transport block with the NR UL-SCH LDPC chain.

        Args:
            bits: Input transport-block bits before CRC and LDPC processing.
            config: Full simulation configuration that defines rate-matching targets.

        Returns:
            Rate-matched coded bit sequence ready for scrambling.
        """
        if config.link.coded_bit_capacity is None:
            raise ValueError("coded_bit_capacity must be resolved before LDPC encoding.")

        tbs = int(bits.size)
        info = get_ulsch_ldpc_info(tbs, config.link.code_rate)
        tb_crc = nrCRCEncode(bits.astype(np.int8), info.crc)[:, 0].astype(np.int8)
        cbs = nrCodeBlockSegmentLDPC(tb_crc, info.base_graph)
        coded_cbs = encode_ldpc_codeblocks(cbs, info.base_graph)
        return rate_match_ulsch_ldpc(
            coded_cbs,
            out_length=int(config.link.coded_bit_capacity),
            rv=int(config.link.mcs.rv),
            modulation=config.link.modulation,
            num_layers=config.link.num_layers,
        ).astype(np.int8)


class RandomBitCoder(ChannelCoder):
    """
    Bypass mode for link-level EVM or waveform studies.

    When channel coding is disabled, generate a deterministic pseudo-random
    bitstream with the same length as the coded-bit capacity and feed it
    directly into scrambling/modulation.
    """

    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Generate deterministic pseudo-random coded bits in bypass mode.

        Args:
            bits: Unused transport-block input kept for interface compatibility.
            config: Full simulation configuration that defines coded-bit capacity.

        Returns:
            Pseudo-random bit sequence with the same length as coded-bit capacity.
        """
        del bits
        if config.link.coded_bit_capacity is None:
            raise ValueError("coded_bit_capacity must be resolved before bypass coding.")

        rng = np.random.default_rng(config.random_seed + 1000)
        return rng.integers(0, 2, size=int(config.link.coded_bit_capacity), dtype=np.int8)
