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
        if config.link.num_tx_ant != 1 or config.link.num_rx_ant != 1:
            raise NotImplementedError("Current 38.901 fading channel implementation supports SISO only.")

        sample_rate = config.carrier.sample_rate_effective_hz
        delays_s, coeff = self._generate_path_coefficients(waveform.size, sample_rate, config)
        rx_waveform = self._apply_time_varying_channel(waveform, delays_s, coeff, sample_rate)
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
        waveform: np.ndarray,
        delays_s: np.ndarray,
        coefficients: np.ndarray,
        sample_rate_hz: float,
    ) -> np.ndarray:
        output = np.zeros(waveform.size, dtype=np.complex128)
        for path_idx, delay_s in enumerate(delays_s):
            delayed = self._fractional_delay(waveform, delay_s * sample_rate_hz)
            output += delayed * coefficients[path_idx, : waveform.size]
        return output

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
            sample = (
                self.rng.normal() + 1j * self.rng.normal()
            ) / np.sqrt(2.0)
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
