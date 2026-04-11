from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import MimoEqualizer
from nr_phy_simu.config import SimulationConfig


class OneTapMmseEqualizer(MimoEqualizer):
    def equalize(
        self,
        rx_symbols: np.ndarray,
        channel_estimate: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        del config
        if rx_symbols.ndim == 2:
            numerator = np.sum(np.conj(channel_estimate) * rx_symbols, axis=0)
            denominator = np.sum(np.abs(channel_estimate) ** 2, axis=0) + noise_variance
            return numerator / np.maximum(denominator, 1e-12)

        denom = (np.abs(channel_estimate) ** 2) + noise_variance
        weights = np.conj(channel_estimate) / np.maximum(denom, 1e-12)
        return weights * rx_symbols
