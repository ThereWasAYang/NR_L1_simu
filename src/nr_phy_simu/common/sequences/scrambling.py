from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import BitScrambler
from nr_phy_simu.common.sequences.dmrs import gold_sequence
from nr_phy_simu.config import SimulationConfig


class NrDataScrambler(BitScrambler):
    """3GPP NR data scrambling/descrambling helper for shared channels."""

    def scramble(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Scramble coded bits with the configured NR data scrambling sequence.

        Args:
            bits: One-dimensional encoded bit array with shape ``(coded_bits,)``;
                axis 0 is coded-bit index before modulation.
            config: Full simulation configuration that provides scrambling seeds.

        Returns:
            Scrambled bit sequence.
        """
        sequence = self._scrambling_sequence(bits.size, config)
        return np.bitwise_xor(bits.astype(np.int8), sequence)

    def descramble_llrs(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply the inverse scrambling sequence to demodulated LLRs.

        Args:
            llrs: One-dimensional demodulated LLR array with shape ``(coded_bits,)``;
                axis 0 is coded-bit index before descrambling.
            config: Full simulation configuration that provides scrambling seeds.

        Returns:
            Descrambled LLR sequence ready for channel decoding.
        """
        sequence = self._scrambling_sequence(llrs.size, config)
        signs = 1.0 - 2.0 * sequence.astype(np.float64)
        return llrs * signs

    def _scrambling_sequence(self, length: int, config: SimulationConfig) -> np.ndarray:
        """Generate the scrambling binary sequence for one codeword.

        Args:
            length: Number of bits or LLRs that need scrambling signs.
            config: Full simulation configuration that provides initialization state.

        Returns:
            One-dimensional binary scrambling sequence with shape ``(length,)``.
        """
        return gold_sequence(self._c_init(config), length).astype(np.int8)

    @staticmethod
    def _c_init(config: SimulationConfig) -> int:
        """Build the 31-bit scrambling initialization value.

        Args:
            config: Full simulation configuration that carries RNTI and scrambling IDs.

        Returns:
            Integer initialization value used by the Gold-sequence generator.
        """
        return (
            (int(config.scrambling.rnti) << 15)
            + (int(config.scrambling.codeword_index) << 14)
            + int(config.scrambling.effective_data_scrambling_id)
        ) % (1 << 31)
