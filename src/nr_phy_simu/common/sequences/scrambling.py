from __future__ import annotations

import numpy as np

from nr_phy_simu.common.sequences.dmrs import gold_sequence
from nr_phy_simu.config import SimulationConfig


class NrDataScrambler:
    """3GPP NR data scrambling/descrambling helper for shared channels."""

    def scramble(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        sequence = self._scrambling_sequence(bits.size, config)
        return np.bitwise_xor(bits.astype(np.int8), sequence)

    def descramble_llrs(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        sequence = self._scrambling_sequence(llrs.size, config)
        signs = 1.0 - 2.0 * sequence.astype(np.float64)
        return llrs * signs

    def _scrambling_sequence(self, length: int, config: SimulationConfig) -> np.ndarray:
        return gold_sequence(self._c_init(config), length).astype(np.int8)

    @staticmethod
    def _c_init(config: SimulationConfig) -> int:
        return (
            (int(config.scrambling.rnti) << 15)
            + (int(config.scrambling.codeword_index) << 14)
            + int(config.scrambling.effective_data_scrambling_id)
        ) % (1 << 31)
