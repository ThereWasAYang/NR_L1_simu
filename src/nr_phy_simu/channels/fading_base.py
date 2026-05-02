from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, REAL_DTYPE, as_complex_tensor, complex_randn, to_numpy
from nr_phy_simu.config import SimulationConfig


class FadingChannelBase(ChannelModel, ABC):
    def __init__(self, rng: torch.Generator | None = None) -> None:
        self.rng = rng or torch.Generator(device="cpu").manual_seed(0)

    def propagate(
        self,
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        tx_waveform = self._expand_tx_branches(waveform, config)
        sample_rate = config.carrier.sample_rate_effective_hz
        delays_s, coeff = self._generate_path_coefficients(tx_waveform.shape[-1], sample_rate, config)
        delays_s = delays_s.to(device=tx_waveform.device)
        coeff = coeff.to(device=tx_waveform.device)
        rx_waveform = self._apply_time_varying_channel(tx_waveform, delays_s, coeff, sample_rate)
        if not bool(config.channel.params.get("add_noise", True)):
            return rx_waveform, {
                "noise_variance": 0.0,
                "snr_db": float("inf"),
                "path_delays_s": to_numpy(delays_s),
                "path_coefficients": coeff,
            }
        noisy_waveform, noise_variance, snr_db = self._add_awgn(rx_waveform, config)
        return noisy_waveform, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "path_delays_s": to_numpy(delays_s),
            "path_coefficients": coeff,
        }

    @abstractmethod
    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _apply_time_varying_channel(
        self,
        tx_waveform: torch.Tensor,
        delays_s: torch.Tensor,
        coefficients: torch.Tensor,
        sample_rate_hz: float,
    ) -> torch.Tensor:
        num_rx_ant, num_tx_ant, _, _ = coefficients.shape
        rx_waveform = torch.zeros(
            (num_rx_ant, tx_waveform.shape[-1]),
            dtype=COMPLEX_DTYPE,
            device=tx_waveform.device,
        )

        for rx_idx in range(num_rx_ant):
            for tx_idx in range(num_tx_ant):
                tx_branch = tx_waveform[tx_idx]
                for path_idx, delay_s in enumerate(delays_s):
                    delayed = self._fractional_delay(tx_branch, delay_s * sample_rate_hz)
                    rx_waveform[rx_idx] += delayed * coefficients[rx_idx, tx_idx, path_idx, : tx_branch.numel()]

        return rx_waveform

    @staticmethod
    def _fractional_delay(waveform: torch.Tensor, delay_samples: float, filter_half_len: int = 8) -> torch.Tensor:
        waveform = as_complex_tensor(waveform)
        integer_delay = int(math.floor(delay_samples))
        fractional = delay_samples - integer_delay
        taps_index = torch.arange(-filter_half_len, filter_half_len + 1, dtype=REAL_DTYPE, device=waveform.device)
        taps = torch.sinc(taps_index - fractional) * torch.hamming_window(
            2 * filter_half_len + 1,
            periodic=False,
            dtype=REAL_DTYPE,
            device=waveform.device,
        )
        taps = taps / torch.sum(taps)
        delayed = F.conv1d(
            waveform.reshape(1, 1, -1),
            torch.flip(taps, dims=(0,)).reshape(1, 1, -1).to(dtype=COMPLEX_DTYPE),
            padding=taps.numel() - 1,
        ).reshape(-1)
        start = filter_half_len + integer_delay
        stop = start + waveform.numel()
        if stop > delayed.numel():
            delayed = F.pad(delayed, (0, stop - delayed.numel()))
        return delayed[start:stop]

    def _rayleigh_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        num_sinusoids: int,
    ) -> torch.Tensor:
        if abs(max_doppler_hz) < 1e-12:
            sample = complex_randn((1,), generator=self.rng, std=1.0 / math.sqrt(2.0))[0]
            return sample.repeat(num_samples)

        device = torch.device("cpu")
        time = torch.arange(num_samples, dtype=REAL_DTYPE, device=device) / sample_rate_hz
        alpha = 2 * math.pi * (torch.arange(1, num_sinusoids + 1, dtype=REAL_DTYPE, device=device) - 0.5) / num_sinusoids
        phase_i = 2 * math.pi * torch.rand(num_sinusoids, generator=self.rng, dtype=REAL_DTYPE, device=device)
        phase_q = 2 * math.pi * torch.rand(num_sinusoids, generator=self.rng, dtype=REAL_DTYPE, device=device)
        real = torch.zeros(num_samples, dtype=REAL_DTYPE, device=device)
        imag = torch.zeros(num_samples, dtype=REAL_DTYPE, device=device)
        for idx in range(num_sinusoids):
            freq = max_doppler_hz * torch.cos(alpha[idx])
            real += torch.cos(2 * math.pi * freq * time + phase_i[idx])
            imag += torch.cos(2 * math.pi * freq * time + phase_q[idx])
        return torch.complex(real, imag) / math.sqrt(2.0 * num_sinusoids)

    def _rician_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        k_factor_linear: float,
        num_sinusoids: int,
        specular_doppler_hz: float | None = None,
        initial_phase: float | None = None,
    ) -> torch.Tensor:
        diffuse = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)
        phase0 = (
            float(2 * math.pi * torch.rand(1, generator=self.rng, dtype=REAL_DTYPE).item())
            if initial_phase is None
            else initial_phase
        )
        spec_freq = specular_doppler_hz if specular_doppler_hz is not None else max_doppler_hz
        time = torch.arange(num_samples, dtype=REAL_DTYPE) / sample_rate_hz
        specular = torch.exp(1j * (2 * math.pi * spec_freq * time + phase0))
        return (
            math.sqrt(k_factor_linear / (k_factor_linear + 1.0)) * specular
            + math.sqrt(1.0 / (k_factor_linear + 1.0)) * diffuse
        )

    @staticmethod
    def _normalize_powers_db(power_db: torch.Tensor) -> torch.Tensor:
        linear = torch.pow(10.0, power_db / 10.0)
        linear = linear / torch.sum(linear)
        return linear

    def _add_awgn(self, waveform: torch.Tensor, config: SimulationConfig) -> tuple[torch.Tensor, float, float]:
        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = float(torch.mean(torch.abs(waveform) ** 2).item())
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise = complex_randn(
            tuple(waveform.shape),
            generator=self.rng,
            std=math.sqrt(noise_variance / 2.0),
            device=waveform.device,
        )
        return waveform + noise, float(noise_variance), snr_db

    @staticmethod
    def _carrier_frequency_hz(config: SimulationConfig) -> float:
        return float(config.channel.params.get("carrier_frequency_hz", 3.5e9))

    @classmethod
    def _wavelength_m(cls, config: SimulationConfig) -> float:
        return 299792458.0 / cls._carrier_frequency_hz(config)

    @classmethod
    def _max_doppler_hz(cls, config: SimulationConfig) -> float:
        params = config.channel.params
        if "max_doppler_hz" in params:
            return float(params["max_doppler_hz"])
        ue_speed_mps = float(params.get("ue_speed_mps", 0.0))
        return ue_speed_mps / cls._wavelength_m(config)

    @staticmethod
    def _resolve_path_parameters(
        config: SimulationConfig,
        normalized_delays: torch.Tensor,
        power_db: torch.Tensor,
        *,
        delay_key: str = "path_delays_ns",
        power_key: str = "path_powers_db",
        delay_spread_key: str = "delay_spread_ns",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = config.channel.params
        custom_delays = params.get(delay_key)
        custom_powers = params.get(power_key)
        if custom_delays is not None or custom_powers is not None:
            custom_delays_ns = torch.as_tensor(custom_delays if custom_delays is not None else [], dtype=REAL_DTYPE)
            custom_powers_db = torch.as_tensor(custom_powers if custom_powers is not None else [], dtype=REAL_DTYPE)
            if custom_delays_ns.numel() == 0 or custom_powers_db.numel() == 0:
                raise ValueError(
                    f"Channel params '{delay_key}' and '{power_key}' must both be provided when overriding path taps."
                )
            if custom_delays_ns.numel() != custom_powers_db.numel():
                raise ValueError(
                    f"Channel params '{delay_key}' and '{power_key}' must have the same number of elements."
                )
            return custom_delays_ns * 1e-9, custom_powers_db

        desired_ds_s = float(params.get(delay_spread_key, 300.0)) * 1e-9
        return normalized_delays * desired_ds_s, power_db

    @staticmethod
    def _expand_tx_branches(waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        waveform = as_complex_tensor(waveform)
        if waveform.ndim == 2:
            return waveform

        num_tx_ant = int(config.link.num_tx_ant)
        if num_tx_ant <= 1:
            return waveform.unsqueeze(0)
        return waveform.unsqueeze(0).repeat(num_tx_ant, 1) / math.sqrt(num_tx_ant)

    @staticmethod
    def _array_response(
        num_ant: int,
        spatial_frequency: float,
        spacing_lambda: float,
    ) -> torch.Tensor:
        antenna_index = torch.arange(num_ant, dtype=REAL_DTYPE)
        return torch.exp(1j * 2.0 * math.pi * spacing_lambda * spatial_frequency * antenna_index)
