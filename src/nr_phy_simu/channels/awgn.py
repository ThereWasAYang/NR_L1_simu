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
            waveform: Transmit waveform with shape ``(slot_samples,)`` for one TX
                branch or ``(num_tx_ant, slot_samples)`` for multiple TX branches;
                last axis is time-sample index.
            config: Full simulation configuration that defines receive antennas and SNR.

        Returns:
            Tuple of ``(rx_waveform, channel_info)``. ``rx_waveform`` has shape
            ``(num_rx_ant, slot_samples)`` for both SISO and MIMO. The receive
            antenna axis is never omitted.
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
            waveform: Input waveform with shape ``(slot_samples,)`` or
                ``(num_rx_ant, slot_samples)``; last axis is time-sample index.
            config: Full simulation configuration that defines ``num_rx_ant``.

        Returns:
            Waveform with shape ``(num_rx_ant, slot_samples)``.
        """
        num_rx_ant = int(config.link.num_rx_ant)
        samples = np.asarray(waveform, dtype=np.complex128)
        if samples.ndim == 1:
            return np.repeat(samples[np.newaxis, :], num_rx_ant, axis=0)
        if samples.ndim != 2:
            raise ValueError("AWGN channel expects waveform shape (slot_samples,) or (num_ant, slot_samples).")
        if samples.shape[0] == num_rx_ant:
            return samples
        if samples.shape[0] == 1:
            return np.repeat(samples, num_rx_ant, axis=0)
        reference_stream = np.sum(samples, axis=0) / np.sqrt(samples.shape[0])
        return np.repeat(reference_stream[np.newaxis, :], num_rx_ant, axis=0)
