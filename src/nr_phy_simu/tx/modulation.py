from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import Modulator
from nr_phy_simu.config import SimulationConfig


class QamModulator(Modulator):
    def map_bits(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        order = config.link.modulation.upper()
        if order == "QPSK":
            return self._map_qpsk(bits)
        if order == "16QAM":
            return self._map_16qam(bits)
        raise ValueError(f"Unsupported modulation: {config.link.modulation}")

    def _map_qpsk(self, bits: np.ndarray) -> np.ndarray:
        padded = self._pad_bits(bits, 2)
        pairs = padded.reshape(-1, 2)
        real = 1 - 2 * pairs[:, 0]
        imag = 1 - 2 * pairs[:, 1]
        return (real + 1j * imag) / np.sqrt(2.0)

    def _map_16qam(self, bits: np.ndarray) -> np.ndarray:
        padded = self._pad_bits(bits, 4)
        groups = padded.reshape(-1, 4)
        real = self._gray_2bit_to_level(groups[:, 0], groups[:, 1])
        imag = self._gray_2bit_to_level(groups[:, 2], groups[:, 3])
        return (real + 1j * imag) / np.sqrt(10.0)

    @staticmethod
    def _gray_2bit_to_level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        idx = (msb << 1) | lsb
        levels = np.array([3, 1, -3, -1], dtype=np.float64)
        return levels[idx]

    @staticmethod
    def _pad_bits(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
        remainder = bits.size % bits_per_symbol
        if remainder == 0:
            return bits.astype(np.int8)
        pad_len = bits_per_symbol - remainder
        return np.pad(bits.astype(np.int8), (0, pad_len))

