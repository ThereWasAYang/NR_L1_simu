from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import BitScrambler
from nr_phy_simu.common.sequences.dmrs import gold_sequence
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import BIT_DTYPE, REAL_DTYPE, as_int_tensor


class NrDataScrambler(BitScrambler):
    """3GPP NR data scrambling/descrambling helper for shared channels."""

    def scramble(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        bits = as_int_tensor(bits, dtype=BIT_DTYPE)
        sequence = self._scrambling_sequence(bits.numel(), config)
        return torch.bitwise_xor(bits, sequence)

    def descramble_llrs(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        sequence = self._scrambling_sequence(llrs.numel(), config)
        signs = 1.0 - 2.0 * sequence.to(dtype=REAL_DTYPE, device=llrs.device)
        return llrs * signs

    def _scrambling_sequence(self, length: int, config: SimulationConfig) -> torch.Tensor:
        return gold_sequence(self._c_init(config), length).to(dtype=BIT_DTYPE)

    @staticmethod
    def _c_init(config: SimulationConfig) -> int:
        return (
            (int(config.scrambling.rnti) << 15)
            + (int(config.scrambling.codeword_index) << 14)
            + int(config.scrambling.effective_data_scrambling_id)
        ) % (1 << 31)
