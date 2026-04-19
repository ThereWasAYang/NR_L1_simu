from __future__ import annotations

from abc import ABC

import torch
from scipy.signal import fftconvolve

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, REAL_DTYPE, as_complex_tensor, complex_randn
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.io.frequency_response_loader import load_frequency_response


class ExternalFrequencyResponseBase(ChannelModel, ABC):
    """Shared helper for channels driven by externally supplied frequency response."""

    @staticmethod
    def load_frequency_response(config: SimulationConfig) -> torch.Tensor:
        """Load and validate the configured per-subcarrier frequency response."""
        params = config.channel.params
        frequency_response = as_complex_tensor(
            load_frequency_response(
                values=params.get("frequency_response"),
                path=params.get("frequency_response_path"),
            )
        )
        expected = config.carrier.n_subcarriers
        if frequency_response.numel() != expected:
            raise ValueError(
                f"Frequency-response length ({frequency_response.numel()}) must equal cell-bandwidth subcarriers ({expected})."
            )
        return frequency_response.reshape(-1)

    @staticmethod
    def require_siso(config: SimulationConfig) -> None:
        """Reject unsupported multi-antenna external-frequency-response cases."""
        if int(config.link.num_tx_ant) != 1 or int(config.link.num_rx_ant) != 1:
            raise ValueError(
                "External frequency-response channels currently support only SISO operation "
                "(num_tx_ant = 1 and num_rx_ant = 1)."
            )

    @staticmethod
    def full_fft_response(frequency_response: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Embed active-subcarrier response into the full FFT bin grid."""
        fft_size = config.carrier.fft_size_effective
        n_sc = config.carrier.n_subcarriers
        fft_bins = torch.ones(fft_size, dtype=COMPLEX_DTYPE, device=frequency_response.device)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc
        fft_bins[start:stop] = frequency_response
        return fft_bins

    @staticmethod
    def add_awgn(samples: torch.Tensor, config: SimulationConfig) -> tuple[torch.Tensor, float, float]:
        """Add AWGN using the configured SNR."""
        if not bool(config.channel.params.get("add_noise", True)):
            return samples, 0.0, float("inf")

        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = float(torch.mean(torch.abs(samples) ** 2).item())
        noise_variance = signal_power / max(snr_linear, 1e-12)
        generator = torch.Generator(device=samples.device.type if samples.device.type != "mps" else "cpu").manual_seed(
            config.random_seed
        )
        noise = complex_randn(
            tuple(samples.shape),
            generator=generator,
            std=(noise_variance / 2.0) ** 0.5,
            device=samples.device,
        )
        return samples + noise, float(noise_variance), snr_db


class ExternalFrequencyResponseTimeDomainChannel(ExternalFrequencyResponseBase):
    """Time-domain channel built from an externally supplied frequency response."""

    def propagate(
        self,
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        """Convert external frequency response to FIR taps and filter the waveform."""
        self.require_siso(config)
        waveform = as_complex_tensor(waveform)
        if waveform.ndim != 1:
            raise ValueError("External frequency-response time-domain channel expects a single TX waveform branch.")

        frequency_response = self.load_frequency_response(config).to(device=waveform.device)
        fft_response = self.full_fft_response(frequency_response, config)
        impulse_response = torch.fft.ifft(torch.fft.ifftshift(fft_response))
        tap_length = int(config.channel.params.get("time_domain_tap_length", impulse_response.numel()))
        taps = impulse_response[:tap_length]
        filtered = as_complex_tensor(fftconvolve(waveform.detach().cpu().numpy(), taps.detach().cpu().numpy(), mode="full"))[
            : waveform.numel()
        ].to(device=waveform.device)
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
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        """Reject time-domain propagation for the direct frequency-domain channel."""
        del waveform, config
        raise NotImplementedError(
            "Frequency-domain direct channel must be used through propagate_grid(...), not propagate(...)."
        )

    def propagate_grid(
        self,
        grid: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        """Apply the configured frequency response directly on the slot grid."""
        self.require_siso(config)
        grid = as_complex_tensor(grid)
        if grid.ndim != 2:
            raise ValueError("Frequency-domain direct channel expects a single-layer 2D resource grid.")

        frequency_response = self.load_frequency_response(config).to(device=grid.device)
        rx_grid = grid * frequency_response[:, None]
        rx_grid, noise_variance, snr_db = self.add_awgn(rx_grid, config)
        return rx_grid, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": frequency_response,
        }
