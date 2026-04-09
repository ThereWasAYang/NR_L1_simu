from __future__ import annotations

import math

import numpy as np

from nr_phy_simu.common.interfaces import ChannelCoder
from nr_phy_simu.config import SimulationConfig


class RepetitionCoder(ChannelCoder):
    """
    Runnable placeholder for the coding stage.

    This keeps the chain executable before a full NR LDPC implementation is
    integrated. Replace with an R18-compliant LDPC coder in production use.
    """

    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        target_len = math.ceil(bits.size / config.link.code_rate)
        repeat_factor = max(1, math.ceil(target_len / bits.size))
        coded = np.tile(bits, repeat_factor)
        return coded[:target_len].astype(np.int8)


class NrLdpcCoder(ChannelCoder):
    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError("NR LDPC coding is not implemented yet.")

