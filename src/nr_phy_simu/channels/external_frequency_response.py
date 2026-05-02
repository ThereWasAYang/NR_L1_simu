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
        """Load and validate the configured per-subcarrier frequency response.

        Args:
            config: Full simulation configuration containing channel params.

        Returns:
            Complex response with shape ``(num_subcarriers,)`` for SISO or
            ``(num_subcarriers, num_rx_ant, num_tx_ant)`` for MIMO.
        """
        params = config.channel.params
        frequency_response = load_frequency_response(
            values=params.get("frequency_response"),
            path=params.get("frequency_response_path"),
        )
        expected = config.carrier.n_subcarriers
        if frequency_response.shape[0] != expected:
            raise ValueError(
                f"Frequency-response first dimension ({frequency_response.shape[0]}) must equal cell-bandwidth subcarriers ({expected})."
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
        """Embed active-subcarrier response into the full FFT bin grid.

        Args:
            frequency_response: One-dimensional active-band response with shape
                ``(num_subcarriers,)``; axis 0 is cell subcarrier index.
            config: Full simulation configuration that defines FFT size.

        Returns:
            One-dimensional FFT-bin response with shape ``(fft_size,)``; axis 0 is
            shifted FFT-bin index before IFFT conversion to taps.
        """
        fft_size = config.carrier.fft_size_effective
        n_sc = config.carrier.n_subcarriers
        fft_bins = np.ones(fft_size, dtype=np.complex128)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc
        fft_bins[start:stop] = frequency_response
        return fft_bins

    @staticmethod
    def add_awgn(samples: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, float, float]:
        """Add AWGN using the configured SNR.

        Args:
            samples: Complex samples or grid with arbitrary shape; all axes are
                preserved and noise is generated element-wise.
            config: Full simulation configuration that provides SNR.

        Returns:
            Tuple of noisy samples with the same shape as ``samples``, scalar noise
            variance, and scalar SNR in dB.
        """
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
        """Convert external frequency response to FIR taps and filter the waveform.

        Args:
            waveform: One-dimensional SISO TX waveform with shape ``(slot_samples,)``.
            config: Full simulation configuration with external frequency response.

        Returns:
            Tuple of RX waveform with shape ``(slot_samples,)`` and channel metadata.
        """
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
        """Reject time-domain propagation for the direct frequency-domain channel.

        Args:
            waveform: Time-domain waveform input, unused by this channel mode.
            config: Full simulation configuration, unused by this channel mode.

        Returns:
            This method always raises; use :meth:`propagate_grid` instead.
        """
        del waveform, config
        raise NotImplementedError(
            "Frequency-domain direct channel must be used through propagate_grid(...), not propagate(...)."
        )

    def propagate_grid(
        self,
        grid: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Apply the configured frequency response directly on the slot grid.

        Args:
            grid: Frequency-domain slot grid with shape
                ``(num_subcarriers, num_symbols)`` for SISO/single-stream input or
                ``(num_tx_ant, num_subcarriers, num_symbols)`` for explicit MIMO TX.
            config: Full simulation configuration with external frequency response.

        Returns:
            Tuple of RX grid with shape ``(num_subcarriers, num_symbols)`` for one
            RX antenna or ``(num_rx_ant, num_subcarriers, num_symbols)`` for MIMO.
        """
        frequency_response = self.load_frequency_response(config)
        channel_matrix = self._frequency_response_matrix(frequency_response, config)
        tx_grid = self._tx_grid_matrix(grid, config)
        rx_grid = np.einsum("krt,tks->rks", channel_matrix, tx_grid)
        if rx_grid.shape[0] == 1:
            rx_grid = rx_grid[0]
        rx_grid, noise_variance, snr_db = self.add_awgn(rx_grid, config)
        return rx_grid, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": frequency_response,
        }

    @staticmethod
    def _frequency_response_matrix(frequency_response: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Normalize external frequency response to ``(K, Nrx, Ntx)``."""
        num_sc = int(config.carrier.n_subcarriers)
        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        response = np.asarray(frequency_response, dtype=np.complex128)
        if response.ndim == 1:
            if num_rx_ant != 1 or num_tx_ant != 1:
                raise ValueError(
                    "MIMO EXTERNAL_FREQRESP_FD requires frequency_response shape "
                    "(num_subcarriers, num_rx_ant, num_tx_ant)."
                )
            if response.shape[0] != num_sc:
                raise ValueError(f"Frequency-response length must be {num_sc}.")
            return response[:, np.newaxis, np.newaxis]
        if response.ndim == 2 and response.shape == (num_sc, num_rx_ant * num_tx_ant):
            return response.reshape(num_sc, num_rx_ant, num_tx_ant)
        expected_shape = (num_sc, num_rx_ant, num_tx_ant)
        if response.shape != expected_shape:
            raise ValueError(
                f"MIMO frequency_response shape must be {expected_shape} or "
                f"({num_sc}, {num_rx_ant * num_tx_ant}), got {response.shape}."
            )
        return response

    @staticmethod
    def _tx_grid_matrix(grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Normalize TX grid to ``(Ntx, K, L)`` for frequency-domain MIMO."""
        num_tx_ant = int(config.link.num_tx_ant)
        tx_grid = np.asarray(grid, dtype=np.complex128)
        if tx_grid.ndim == 2:
            if num_tx_ant == 1:
                return tx_grid[np.newaxis, ...]
            return np.repeat(tx_grid[np.newaxis, ...], num_tx_ant, axis=0) / np.sqrt(num_tx_ant)
        if tx_grid.ndim == 3:
            if tx_grid.shape[0] != num_tx_ant:
                raise ValueError(f"TX grid antenna dimension must be {num_tx_ant}, got {tx_grid.shape[0]}.")
            return tx_grid
        raise ValueError(
            "Frequency-domain direct channel expects grid shape "
            "(num_subcarriers, num_symbols) or (num_tx_ant, num_subcarriers, num_symbols)."
        )
