from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import Demodulator
from nr_phy_simu.config import SimulationConfig


class QamDemodulator(Demodulator):
    def demap_symbols(
        self,
        symbols: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        modulation = config.link.modulation.upper()
        if modulation == "QPSK":
            scale = np.sqrt(2.0)
            llr_i = 2 * scale * symbols.real / max(noise_variance, 1e-12)
            llr_q = 2 * scale * symbols.imag / max(noise_variance, 1e-12)
            return np.column_stack([llr_i, llr_q]).reshape(-1)
        if modulation == "16QAM":
            scale = np.sqrt(10.0)
            re = scale * symbols.real
            im = scale * symbols.imag
            llr0 = 2 * re / max(noise_variance, 1e-12)
            llr1 = (2 - np.abs(re)) * 2 / max(noise_variance, 1e-12)
            llr2 = 2 * im / max(noise_variance, 1e-12)
            llr3 = (2 - np.abs(im)) * 2 / max(noise_variance, 1e-12)
            return np.column_stack([llr0, llr1, llr2, llr3]).reshape(-1)
        raise ValueError(f"Unsupported modulation: {config.link.modulation}")

