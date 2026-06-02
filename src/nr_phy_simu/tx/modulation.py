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
            bits: One-dimensional scrambled coded-bit array with shape
                ``(coded_bits,)``; axis 0 is the coded-bit index.
            config: Full simulation configuration that defines modulation order.

        Returns:
            Serialized complex modulation symbols with shape ``(num_mod_symbols,)``.
        """
        return self.map_bits_for_modulation(bits, config.link.modulation)

    @staticmethod
    def map_bits_for_modulation(bits: np.ndarray, modulation: str) -> np.ndarray:
        """Map bits to symbols for an explicitly selected modulation format.

        Args:
            bits: One-dimensional bit array with shape ``(num_bits,)``; axis 0 is
                grouped by ``bits_per_symbol(modulation)`` into output symbols.
            modulation: Modulation name understood by ``py3gpp.nrSymbolModulate``.

        Returns:
            Complex modulation symbols in serial order with shape
            ``(ceil(num_bits / bits_per_symbol),)``.
        """
        bps = bits_per_symbol(modulation)
        padded = QamModulator._pad_bits(bits, bps)
        if modulation.upper() == "1024QAM":
            return QamModulator._map_1024qam(padded)
        return nrSymbolModulate(padded.astype(np.int8), modulation)

    @staticmethod
    def _map_1024qam(bits: np.ndarray) -> np.ndarray:
        """Map bits to 1024QAM symbols using the 38.211 square-QAM labeling."""
        grouped = np.asarray(bits, dtype=np.int8).reshape(-1, 10)
        i_bits = grouped[:, [0, 2, 4, 6, 8]]
        q_bits = grouped[:, [1, 3, 5, 7, 9]]
        i_values = QamModulator._map_32pam_axis(i_bits)
        q_values = QamModulator._map_32pam_axis(q_bits)
        return (i_values + 1j * q_values) / np.sqrt(682.0)

    @staticmethod
    def _map_32pam_axis(axis_bits: np.ndarray) -> np.ndarray:
        b = 1 - 2 * axis_bits.astype(np.float64)
        return b[:, 0] * (16.0 - b[:, 1] * (8.0 - b[:, 2] * (4.0 - b[:, 3] * (2.0 - b[:, 4]))))

    @staticmethod
    def _pad_bits(bits: np.ndarray, bits_per_mod_symbol: int) -> np.ndarray:
        """Pad the bit sequence so it aligns with the modulation order.

        Args:
            bits: One-dimensional bit array with shape ``(num_bits,)``; axis 0 is
                the coded-bit index before grouping into modulation symbols.
            bits_per_mod_symbol: Number of bits represented by one modulated symbol.

        Returns:
            Bit sequence padded with zeros to a multiple of ``bits_per_mod_symbol``.
        """
        remainder = bits.size % bits_per_mod_symbol
        if remainder == 0:
            return bits
        return np.pad(bits, (0, bits_per_mod_symbol - remainder))
