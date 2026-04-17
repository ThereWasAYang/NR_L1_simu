from __future__ import annotations

import numpy as np
from py3gpp import nrSymbolModulate

from nr_phy_simu.common.interfaces import Modulator
from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.config import SimulationConfig


class QamModulator(Modulator):
    def map_bits(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Map coded bits to modulation symbols selected by the link config.

        Args:
            bits: Scrambled coded-bit sequence to be modulated.
            config: Full simulation configuration that defines modulation order.

        Returns:
            Serialized complex modulation symbols.
        """
        return self.map_bits_for_modulation(bits, config.link.modulation)

    @staticmethod
    def map_bits_for_modulation(bits: np.ndarray, modulation: str) -> np.ndarray:
        """Map bits to symbols for an explicitly selected modulation format.

        Args:
            bits: Bit sequence to modulate.
            modulation: Modulation name understood by ``py3gpp.nrSymbolModulate``.

        Returns:
            Complex modulation symbols in serial order.
        """
        bps = bits_per_symbol(modulation)
        padded = QamModulator._pad_bits(bits, bps)
        return nrSymbolModulate(padded.astype(np.int8), modulation)

    @staticmethod
    def _pad_bits(bits: np.ndarray, bits_per_mod_symbol: int) -> np.ndarray:
        """Pad the bit sequence so it aligns with the modulation order.

        Args:
            bits: Bit sequence that may have a partial tail symbol.
            bits_per_mod_symbol: Number of bits represented by one modulated symbol.

        Returns:
            Bit sequence padded with zeros to a multiple of ``bits_per_mod_symbol``.
        """
        remainder = bits.size % bits_per_mod_symbol
        if remainder == 0:
            return bits
        return np.pad(bits, (0, bits_per_mod_symbol - remainder))
