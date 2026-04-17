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
        """Apply receive-branch expansion and AWGN impairment.

        Args:
            waveform: Transmit waveform for one or more transmit branches.
            config: Full simulation configuration that defines receive antennas and SNR.

        Returns:
            Tuple of ``(rx_waveform, channel_info)`` with added noise statistics.
        """
        tx_waveform = self._expand_receive_branches(waveform, config)
        if not bool(config.channel.params.get("add_noise", True)):
            return tx_waveform, {"noise_variance": 0.0, "snr_db": float("inf")}

        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(tx_waveform) ** 2)
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise = (
            self.rng.normal(0.0, np.sqrt(noise_variance / 2), tx_waveform.shape)
            + 1j * self.rng.normal(0.0, np.sqrt(noise_variance / 2), tx_waveform.shape)
        )
        return tx_waveform + noise, {"noise_variance": noise_variance, "snr_db": snr_db}

    @staticmethod
    def _expand_receive_branches(waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Replicate a single received stream across configured receive branches.

        Args:
            waveform: Input waveform before receive-antenna expansion.
            config: Full simulation configuration that defines ``num_rx_ant``.

        Returns:
            Waveform stacked by receive antenna when expansion is required.
        """
        if waveform.ndim == 2:
            return waveform

        num_rx_ant = int(config.link.num_rx_ant)
        if num_rx_ant <= 1:
            return waveform
        return np.repeat(waveform[np.newaxis, :], num_rx_ant, axis=0)
