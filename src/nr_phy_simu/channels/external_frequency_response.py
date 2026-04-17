from __future__ import annotations

from abc import ABC

import numpy as np
from scipy.signal import fftconvolve

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.io.frequency_response_loader import load_frequency_response


class ExternalFrequencyResponseBase(ChannelModel, ABC):
    """Shared helper for channels driven by externally supplied frequency response."""

    @staticmethod
    def load_frequency_response(config: SimulationConfig) -> np.ndarray:
        """Load and validate the configured per-subcarrier frequency response."""
        params = config.channel.params
        frequency_response = load_frequency_response(
            values=params.get("frequency_response"),
            path=params.get("frequency_response_path"),
        )
        expected = config.carrier.n_subcarriers
        if frequency_response.size != expected:
            raise ValueError(
                f"Frequency-response length ({frequency_response.size}) must equal cell-bandwidth subcarriers ({expected})."
            )
        return frequency_response

    @staticmethod
    def require_siso(config: SimulationConfig) -> None:
        """Reject unsupported multi-antenna external-frequency-response cases."""
        if int(config.link.num_tx_ant) != 1 or int(config.link.num_rx_ant) != 1:
            raise ValueError(
                "External frequency-response channels currently support only SISO operation "
                "(num_tx_ant = 1 and num_rx_ant = 1)."
            )

    @staticmethod
    def full_fft_response(frequency_response: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Embed active-subcarrier response into the full FFT bin grid."""
        fft_size = config.carrier.fft_size_effective
        n_sc = config.carrier.n_subcarriers
        fft_bins = np.ones(fft_size, dtype=np.complex128)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc
        fft_bins[start:stop] = frequency_response
        return fft_bins

    @staticmethod
    def add_awgn(samples: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, float, float]:
        """Add AWGN using the configured SNR."""
        if not bool(config.channel.params.get("add_noise", True)):
            return samples, 0.0, float("inf")

        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(samples) ** 2)
        noise_variance = signal_power / max(snr_linear, 1e-12)
        rng = np.random.default_rng(config.random_seed)
        noise = (
            rng.normal(0.0, np.sqrt(noise_variance / 2.0), samples.shape)
            + 1j * rng.normal(0.0, np.sqrt(noise_variance / 2.0), samples.shape)
        )
        return samples + noise, float(noise_variance), snr_db


class ExternalFrequencyResponseTimeDomainChannel(ExternalFrequencyResponseBase):
    """Time-domain channel built from an externally supplied frequency response."""

    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Convert external frequency response to FIR taps and filter the waveform."""
        self.require_siso(config)
        if waveform.ndim != 1:
            raise ValueError("External frequency-response time-domain channel expects a single TX waveform branch.")

        frequency_response = self.load_frequency_response(config)
        fft_response = self.full_fft_response(frequency_response, config)
        impulse_response = np.fft.ifft(np.fft.ifftshift(fft_response))
        tap_length = int(config.channel.params.get("time_domain_tap_length", impulse_response.size))
        taps = impulse_response[:tap_length]
        filtered = fftconvolve(waveform, taps, mode="full")[: waveform.size]
        rx_waveform, noise_variance, snr_db = self.add_awgn(filtered, config)
        return rx_waveform, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": frequency_response,
            "time_domain_taps": taps,
        }


class ExternalFrequencyResponseFrequencyDomainChannel(ExternalFrequencyResponseBase):
    """Frequency-domain direct channel that bypasses OFDM mod/demod."""

    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Reject time-domain propagation for the direct frequency-domain channel."""
        del waveform, config
        raise NotImplementedError(
            "Frequency-domain direct channel must be used through propagate_grid(...), not propagate(...)."
        )

    def propagate_grid(
        self,
        grid: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Apply the configured frequency response directly on the slot grid."""
        self.require_siso(config)
        if grid.ndim != 2:
            raise ValueError("Frequency-domain direct channel expects a single-layer 2D resource grid.")

        frequency_response = self.load_frequency_response(config)
        rx_grid = grid * frequency_response[:, np.newaxis]
        rx_grid, noise_variance, snr_db = self.add_awgn(rx_grid, config)
        return rx_grid, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": frequency_response,
        }
