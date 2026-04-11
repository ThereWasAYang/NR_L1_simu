from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np
from scipy.signal import fftconvolve

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig


class FadingChannelBase(ChannelModel, ABC):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        tx_waveform = self._expand_tx_branches(waveform, config)
        sample_rate = config.carrier.sample_rate_effective_hz
        delays_s, coeff = self._generate_path_coefficients(tx_waveform.shape[-1], sample_rate, config)
        rx_waveform = self._apply_time_varying_channel(tx_waveform, delays_s, coeff, sample_rate)
        if not bool(config.channel.params.get("add_noise", True)):
            return rx_waveform, {
                "noise_variance": 0.0,
                "snr_db": float("inf"),
                "path_delays_s": delays_s,
                "path_coefficients": coeff,
            }
        noisy_waveform, noise_variance, snr_db = self._add_awgn(rx_waveform, config)
        return noisy_waveform, {
            "noise_variance": noise_variance,
            "snr_db": snr_db,
            "path_delays_s": delays_s,
            "path_coefficients": coeff,
        }

    @abstractmethod
    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _apply_time_varying_channel(
        self,
        tx_waveform: np.ndarray,
        delays_s: np.ndarray,
        coefficients: np.ndarray,
        sample_rate_hz: float,
    ) -> np.ndarray:
        num_rx_ant, num_tx_ant, _, _ = coefficients.shape
        rx_waveform = np.zeros((num_rx_ant, tx_waveform.shape[-1]), dtype=np.complex128)

        for rx_idx in range(num_rx_ant):
            for tx_idx in range(num_tx_ant):
                tx_branch = tx_waveform[tx_idx]
                for path_idx, delay_s in enumerate(delays_s):
                    delayed = self._fractional_delay(tx_branch, delay_s * sample_rate_hz)
                    rx_waveform[rx_idx] += delayed * coefficients[rx_idx, tx_idx, path_idx, : tx_branch.size]

        if num_rx_ant == 1:
            return rx_waveform[0]
        return rx_waveform

    @staticmethod
    def _fractional_delay(waveform: np.ndarray, delay_samples: float, filter_half_len: int = 8) -> np.ndarray:
        integer_delay = int(math.floor(delay_samples))
        fractional = delay_samples - integer_delay
        taps_index = np.arange(-filter_half_len, filter_half_len + 1, dtype=np.float64)
        taps = np.sinc(taps_index - fractional) * np.hamming(2 * filter_half_len + 1)
        taps /= np.sum(taps)
        delayed = fftconvolve(waveform, taps, mode="full")
        start = filter_half_len + integer_delay
        stop = start + waveform.size
        if stop > delayed.size:
            delayed = np.pad(delayed, (0, stop - delayed.size))
        return delayed[start:stop]

    def _rayleigh_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        num_sinusoids: int,
    ) -> np.ndarray:
        if abs(max_doppler_hz) < 1e-12:
            sample = (self.rng.normal() + 1j * self.rng.normal()) / np.sqrt(2.0)
            return np.full(num_samples, sample, dtype=np.complex128)

        time = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
        alpha = 2 * np.pi * (np.arange(1, num_sinusoids + 1) - 0.5) / num_sinusoids
        phase_i = self.rng.uniform(0.0, 2 * np.pi, size=num_sinusoids)
        phase_q = self.rng.uniform(0.0, 2 * np.pi, size=num_sinusoids)
        real = np.zeros(num_samples, dtype=np.float64)
        imag = np.zeros(num_samples, dtype=np.float64)
        for idx in range(num_sinusoids):
            freq = max_doppler_hz * np.cos(alpha[idx])
            real += np.cos(2 * np.pi * freq * time + phase_i[idx])
            imag += np.cos(2 * np.pi * freq * time + phase_q[idx])
        return (real + 1j * imag) / np.sqrt(2.0 * num_sinusoids)

    def _rician_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        k_factor_linear: float,
        num_sinusoids: int,
        specular_doppler_hz: float | None = None,
        initial_phase: float | None = None,
    ) -> np.ndarray:
        diffuse = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)
        phase0 = self.rng.uniform(0.0, 2 * np.pi) if initial_phase is None else initial_phase
        spec_freq = specular_doppler_hz if specular_doppler_hz is not None else max_doppler_hz
        time = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
        specular = np.exp(1j * (2 * np.pi * spec_freq * time + phase0))
        return (
            np.sqrt(k_factor_linear / (k_factor_linear + 1.0)) * specular
            + np.sqrt(1.0 / (k_factor_linear + 1.0)) * diffuse
        )

    @staticmethod
    def _normalize_powers_db(power_db: np.ndarray) -> np.ndarray:
        linear = 10 ** (power_db / 10.0)
        linear /= np.sum(linear)
        return linear

    def _add_awgn(self, waveform: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, float, float]:
        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(waveform) ** 2)
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise = (
            self.rng.normal(0.0, np.sqrt(noise_variance / 2.0), waveform.shape)
            + 1j * self.rng.normal(0.0, np.sqrt(noise_variance / 2.0), waveform.shape)
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
        normalized_delays: np.ndarray,
        power_db: np.ndarray,
        *,
        delay_key: str = "path_delays_ns",
        power_key: str = "path_powers_db",
        delay_spread_key: str = "delay_spread_ns",
    ) -> tuple[np.ndarray, np.ndarray]:
        params = config.channel.params
        custom_delays = params.get(delay_key)
        custom_powers = params.get(power_key)
        if custom_delays is not None or custom_powers is not None:
            custom_delays_ns = np.asarray(custom_delays if custom_delays is not None else [], dtype=np.float64)
            custom_powers_db = np.asarray(custom_powers if custom_powers is not None else [], dtype=np.float64)
            if custom_delays_ns.size == 0 or custom_powers_db.size == 0:
                raise ValueError(
                    f"Channel params '{delay_key}' and '{power_key}' must both be provided when overriding path taps."
                )
            if custom_delays_ns.size != custom_powers_db.size:
                raise ValueError(
                    f"Channel params '{delay_key}' and '{power_key}' must have the same number of elements."
                )
            return custom_delays_ns * 1e-9, custom_powers_db

        desired_ds_s = float(params.get(delay_spread_key, 300.0)) * 1e-9
        return normalized_delays * desired_ds_s, power_db

    @staticmethod
    def _expand_tx_branches(waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        if waveform.ndim == 2:
            return waveform

        num_tx_ant = int(config.link.num_tx_ant)
        if num_tx_ant <= 1:
            return waveform[np.newaxis, :]
        return np.repeat(waveform[np.newaxis, :], num_tx_ant, axis=0) / np.sqrt(num_tx_ant)

    @staticmethod
    def _array_response(
        num_ant: int,
        spatial_frequency: float,
        spacing_lambda: float,
    ) -> np.ndarray:
        antenna_index = np.arange(num_ant, dtype=np.float64)
        return np.exp(1j * 2.0 * np.pi * spacing_lambda * spatial_frequency * antenna_index)
