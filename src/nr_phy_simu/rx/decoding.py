from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import ChannelDecoder
from nr_phy_simu.config import SimulationConfig


class HardDecisionRepetitionDecoder(ChannelDecoder):
    """
    Pairs with the placeholder repetition coder.
    """

    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        target_len = config.link.transport_block_size
        hard_bits = (llrs < 0).astype(np.int8)
        if hard_bits.size < target_len:
            hard_bits = np.pad(hard_bits, (0, target_len - hard_bits.size))

        repeats = int(np.ceil(hard_bits.size / target_len))
        padded = np.pad(hard_bits, (0, repeats * target_len - hard_bits.size))
        blocks = padded.reshape(repeats, target_len)
        return (np.mean(blocks, axis=0) >= 0.5).astype(np.int8)


class NrLdpcDecoder(ChannelDecoder):
    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError("NR LDPC decoding is not implemented yet.")
