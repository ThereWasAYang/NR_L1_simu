from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F

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

    @staticmethod
    def _frequency_response_matrix(frequency_response: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Normalize external frequency response to ``(K, Nrx, Ntx)``."""
        num_sc = int(config.carrier.n_subcarriers)
        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        response = as_complex_tensor(frequency_response)
        if response.ndim == 1:
            if num_rx_ant != 1 or num_tx_ant != 1:
                raise ValueError(
                    "MIMO EXTERNAL_FREQRESP_FD requires frequency_response shape "
                    "(num_subcarriers, num_rx_ant, num_tx_ant)."
                )
            if response.shape[0] != num_sc:
                raise ValueError(f"Frequency-response length must be {num_sc}.")
            return response[:, None, None]
        if response.ndim == 2 and tuple(response.shape) == (num_sc, num_rx_ant * num_tx_ant):
            return response.reshape(num_sc, num_rx_ant, num_tx_ant)
        expected_shape = (num_sc, num_rx_ant, num_tx_ant)
        if tuple(response.shape) != expected_shape:
            raise ValueError(
                f"MIMO frequency_response shape must be {expected_shape} or "
                f"({num_sc}, {num_rx_ant * num_tx_ant}), got {tuple(response.shape)}."
            )
        return response


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
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim != 1:
            raise ValueError("External frequency-response time-domain channel expects exactly one TX waveform branch.")

        frequency_response = self.load_frequency_response(config).to(device=waveform.device)
        channel_matrix = self._frequency_response_matrix(frequency_response, config)
        siso_response = channel_matrix[:, 0, 0]
        fft_response = self.full_fft_response(siso_response, config)
        impulse_response = torch.fft.ifft(torch.fft.ifftshift(fft_response))
        tap_length = int(config.channel.params.get("time_domain_tap_length", impulse_response.numel()))
        taps = impulse_response[:tap_length]
        filtered = F.conv1d(
            waveform.reshape(1, 1, -1),
            torch.flip(taps, dims=(0,)).reshape(1, 1, -1),
            padding=taps.numel() - 1,
        ).reshape(-1)[: waveform.numel()]
        rx_waveform, noise_variance, snr_db = self.add_awgn(filtered, config)
        rx_waveform = rx_waveform.unsqueeze(0)
        return rx_waveform, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": channel_matrix,
            "time_domain_taps": taps.reshape(1, 1, -1),
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
        grid = as_complex_tensor(grid)
        frequency_response = self.load_frequency_response(config).to(device=grid.device)
        channel_matrix = self._frequency_response_matrix(frequency_response, config)
        tx_grid = self._tx_grid_matrix(grid, config)
        rx_grid = torch.einsum("krt,tks->rks", channel_matrix, tx_grid)
        rx_grid, noise_variance, snr_db = self.add_awgn(rx_grid, config)
        return rx_grid, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "frequency_response": channel_matrix,
        }

    @staticmethod
    def _tx_grid_matrix(grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Normalize TX grid to ``(Ntx, K, L)`` for frequency-domain MIMO."""
        num_tx_ant = int(config.link.num_tx_ant)
        tx_grid = as_complex_tensor(grid)
        if tx_grid.ndim == 2:
            if num_tx_ant == 1:
                return tx_grid.unsqueeze(0)
            scale = torch.sqrt(torch.tensor(float(num_tx_ant), dtype=REAL_DTYPE, device=tx_grid.device))
            return tx_grid.unsqueeze(0).repeat(num_tx_ant, 1, 1) / scale
        if tx_grid.ndim == 3:
            if tx_grid.shape[0] != num_tx_ant:
                raise ValueError(f"TX grid antenna dimension must be {num_tx_ant}, got {tx_grid.shape[0]}.")
            return tx_grid
        raise ValueError(
            "Frequency-domain direct channel expects grid shape "
            "(num_subcarriers, num_symbols) or (num_tx_ant, num_subcarriers, num_symbols)."
        )
