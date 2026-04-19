from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import BitScrambler
from nr_phy_simu.common.sequences.dmrs import gold_sequence
from nr_phy_simu.common.torch_utils import BIT_DTYPE, REAL_DTYPE, as_int_tensor
from nr_phy_simu.config import SimulationConfig


class NrDataScrambler(BitScrambler):
    """3GPP NR data scrambling/descrambling helper for shared channels."""

    def scramble(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Scramble coded bits with the configured NR data scrambling sequence."""
        bits = as_int_tensor(bits, dtype=BIT_DTYPE)
        sequence = self._scrambling_sequence(bits.numel(), config).to(device=bits.device)
        return torch.bitwise_xor(bits, sequence)

    def descramble_llrs(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Apply the inverse scrambling sequence to demodulated LLRs."""
        sequence = self._scrambling_sequence(llrs.numel(), config).to(device=llrs.device)
        signs = 1.0 - 2.0 * sequence.to(dtype=REAL_DTYPE, device=llrs.device)
        return llrs * signs

    def _scrambling_sequence(self, length: int, config: SimulationConfig) -> torch.Tensor:
        """Generate the scrambling binary sequence for one codeword."""
        return gold_sequence(self._c_init(config), length).to(dtype=BIT_DTYPE)

    @staticmethod
    def _c_init(config: SimulationConfig) -> int:
        """Build the 31-bit scrambling initialization value."""
        return (
            (int(config.scrambling.rnti) << 15)
            + (int(config.scrambling.codeword_index) << 14)
            + int(config.scrambling.effective_data_scrambling_id)
        ) % (1 << 31)
