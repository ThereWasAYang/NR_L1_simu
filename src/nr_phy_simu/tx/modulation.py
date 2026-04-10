from __future__ import annotations

import numpy as np
from py3gpp import nrSymbolModulate

from nr_phy_simu.common.interfaces import Modulator
from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.config import SimulationConfig


class QamModulator(Modulator):
    def map_bits(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        return self.map_bits_for_modulation(bits, config.link.modulation)

    @staticmethod
    def map_bits_for_modulation(bits: np.ndarray, modulation: str) -> np.ndarray:
        bps = bits_per_symbol(modulation)
        padded = QamModulator._pad_bits(bits, bps)
        return nrSymbolModulate(padded.astype(np.int8), modulation)

    @staticmethod
    def _pad_bits(bits: np.ndarray, bits_per_mod_symbol: int) -> np.ndarray:
        remainder = bits.size % bits_per_mod_symbol
        if remainder == 0:
            return bits
        return np.pad(bits, (0, bits_per_mod_symbol - remainder))
