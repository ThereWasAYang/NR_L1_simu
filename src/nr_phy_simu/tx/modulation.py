from __future__ import annotations

import torch
from py3gpp import nrSymbolModulate

from nr_phy_simu.common.interfaces import Modulator
from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import BIT_DTYPE, COMPLEX_DTYPE, as_int_tensor, to_numpy


class QamModulator(Modulator):
    def map_bits(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        return self.map_bits_for_modulation(bits, config.link.modulation)

    @staticmethod
    def map_bits_for_modulation(bits: torch.Tensor, modulation: str) -> torch.Tensor:
        bps = bits_per_symbol(modulation)
        padded = QamModulator._pad_bits(bits, bps)
        symbols = nrSymbolModulate(to_numpy(padded.to(dtype=BIT_DTYPE)), modulation)
        return torch.as_tensor(symbols, dtype=COMPLEX_DTYPE)

    @staticmethod
    def _pad_bits(bits: torch.Tensor, bits_per_mod_symbol: int) -> torch.Tensor:
        bits = as_int_tensor(bits, dtype=BIT_DTYPE)
        remainder = bits.numel() % bits_per_mod_symbol
        if remainder == 0:
            return bits
        return torch.nn.functional.pad(bits, (0, bits_per_mod_symbol - remainder))
