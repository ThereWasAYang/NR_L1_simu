from __future__ import annotations

import itertools

import numpy as np

from nr_phy_simu.common.interfaces import Demodulator
from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.tx.modulation import QamModulator


class QamDemodulator(Demodulator):
    def demap_symbols(
        self,
        symbols: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        constellation, bit_labels = self._constellation(config.link.modulation)
        metric = np.empty((symbols.size, constellation.size), dtype=np.float64)
        for idx, point in enumerate(constellation):
            metric[:, idx] = np.abs(symbols - point) ** 2

        llrs = []
        variance = max(noise_variance, 1e-12)
        for bit_idx in range(bit_labels.shape[1]):
            zero_metric = np.min(metric[:, bit_labels[:, bit_idx] == 0], axis=1)
            one_metric = np.min(metric[:, bit_labels[:, bit_idx] == 1], axis=1)
            llrs.append((one_metric - zero_metric) / variance)
        return np.column_stack(llrs).reshape(-1)

    def _constellation(self, modulation: str) -> tuple[np.ndarray, np.ndarray]:
        bps = bits_per_symbol(modulation)
        bit_patterns = np.array(list(itertools.product([0, 1], repeat=bps)), dtype=np.int8)
        symbols = QamModulator.map_bits_for_modulation(bit_patterns.reshape(-1), modulation)
        return symbols, bit_patterns
