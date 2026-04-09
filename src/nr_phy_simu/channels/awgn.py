from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig


class AwgnChannel(ChannelModel):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        snr_linear = 10 ** (config.snr_db / 10.0)
        signal_power = np.mean(np.abs(waveform) ** 2)
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise = (
            self.rng.normal(0.0, np.sqrt(noise_variance / 2), waveform.shape)
            + 1j * self.rng.normal(0.0, np.sqrt(noise_variance / 2), waveform.shape)
        )
        return waveform + noise, {"noise_variance": noise_variance}

