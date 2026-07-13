from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig


@dataclass(frozen=True)
class AntennaArrayDescription:
    """Resolved antenna-array geometry used by link-level TDL/CDL models.

    Shape conventions:
        ``positions_lambda``: ``(num_ports, 3)`` element positions measured in
        wavelengths along x/y/z.
        ``polarization_slants_deg``: ``(num_ports,)`` polarization slant angles
        used by the isotropic dual-polarized field model.
    """

    positions_lambda: np.ndarray
    polarization_slants_deg: np.ndarray
    polarization: str


class FadingChannelBase(ChannelModel, ABC):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self._time_offset_s = 0.0
        self._continuous_initial_rng_state: dict | None = None
        self._continuous_noise_rng: np.random.Generator | None = None
        self._continuous_mode = False

    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Propagate a time-domain waveform through a time-varying tapped channel.

        Args:
            waveform: Transmit waveform with shape ``(num_tx_ant, slot_samples)``;
                axis 0 is TX antenna and axis 1 is time-sample index.
            config: Full simulation configuration that defines antennas, channel
                profile parameters, sample rate, and SNR.

        Returns:
            Tuple of ``(rx_waveform, channel_info)``. ``rx_waveform`` has shape
            ``(num_rx_ant, slot_samples)`` for both SISO and MIMO.
            ``channel_info["path_coefficients"]`` has
            shape ``(num_rx_ant, num_tx_ant, num_paths, slot_samples)``.
        """
        self._validate_link_level_limits(config)
        evolution = str(config.channel.params.get("tti_evolution", "independent")).lower()
        if evolution not in {"independent", "continuous"}:
            raise ValueError("channel.params.tti_evolution must be 'independent' or 'continuous'.")
        self._continuous_mode = evolution == "continuous"
        if self._continuous_mode:
            if self._continuous_initial_rng_state is None:
                self._continuous_initial_rng_state = deepcopy(self.rng.bit_generator.state)
            else:
                self.rng.bit_generator.state = deepcopy(self._continuous_initial_rng_state)
            self._time_offset_s = (
                int(config.carrier.slot_start_sample(config.slot_index))
                / float(config.carrier.sample_rate_effective_hz)
            )
        else:
            self._time_offset_s = 0.0
        geometry_info = self._channel_geometry_info(config)
        tx_waveform = self._expand_tx_branches(waveform, config)
        sample_rate = config.carrier.sample_rate_effective_hz
        delays_s, coeff = self._generate_path_coefficients(tx_waveform.shape[-1], sample_rate, config)
        if self._continuous_mode and self._continuous_noise_rng is None:
            self._continuous_noise_rng = np.random.default_rng()
            self._continuous_noise_rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        rx_waveform = self._apply_time_varying_channel(tx_waveform, delays_s, coeff, sample_rate)
        base_info = {
            "path_delays_s": delays_s,
            "path_coefficients": coeff,
            "carrier_frequency_hz": self._carrier_frequency_hz(config),
            "max_doppler_hz": self._max_doppler_hz(config),
            **geometry_info,
        }
        if not bool(config.channel.params.get("add_noise", True)):
            return rx_waveform, {"noise_variance": 0.0, "snr_db": float("inf"), **base_info}
        noisy_waveform, noise_variance, snr_db = self._add_awgn(rx_waveform, config)
        return noisy_waveform, {"noise_variance": noise_variance, "snr_db": snr_db, **base_info}

    @abstractmethod
    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate path delays and time-varying MIMO path coefficients.

        Args:
            num_samples: Number of time samples in the slot waveform.
            sample_rate_hz: Baseband sample rate in Hz.
            config: Full simulation configuration with channel and antenna settings.

        Returns:
            Tuple ``(delays_s, coefficients)`` where ``delays_s`` has shape
            ``(num_paths,)`` and ``coefficients`` has shape
            ``(num_rx_ant, num_tx_ant, num_paths, num_samples)``. Coefficient axes
            are RX antenna, TX antenna, path index, and time-sample index.
        """
        raise NotImplementedError

    def _apply_time_varying_channel(
        self,
        tx_waveform: np.ndarray,
        delays_s: np.ndarray,
        coefficients: np.ndarray,
        sample_rate_hz: float,
    ) -> np.ndarray:
        """Apply delayed time-varying MIMO taps to all TX branches.

        Args:
            tx_waveform: TX waveform matrix with shape ``(num_tx_ant, slot_samples)``.
            delays_s: Path delays with shape ``(num_paths,)`` in seconds.
            coefficients: Channel taps with shape
                ``(num_rx_ant, num_tx_ant, num_paths, slot_samples)``.
            sample_rate_hz: Baseband sample rate used to convert delay to samples.

        Returns:
            RX waveform with shape ``(num_rx_ant, slot_samples)``. The receive
            antenna axis is never omitted, including SISO.
        """
        num_rx_ant, num_tx_ant, _, _ = coefficients.shape
        rx_waveform = np.zeros((num_rx_ant, tx_waveform.shape[-1]), dtype=np.complex128)

        delayed_branches = {
            (tx_idx, path_idx): self._fractional_delay(
                tx_waveform[tx_idx],
                delay_s * sample_rate_hz,
            )
            for tx_idx in range(num_tx_ant)
            for path_idx, delay_s in enumerate(delays_s)
        }
        for rx_idx in range(num_rx_ant):
            for tx_idx in range(num_tx_ant):
                tx_branch = tx_waveform[tx_idx]
                for path_idx, _delay_s in enumerate(delays_s):
                    delayed = delayed_branches[(tx_idx, path_idx)]
                    rx_waveform[rx_idx] += delayed * coefficients[rx_idx, tx_idx, path_idx, : tx_branch.size]

        return rx_waveform

    @staticmethod
    def _fractional_delay(waveform: np.ndarray, delay_samples: float, filter_half_len: int = 8) -> np.ndarray:
        """Apply a fractional sample delay to one time-domain branch.

        Args:
            waveform: One-dimensional waveform with shape ``(slot_samples,)``.
            delay_samples: Delay measured in samples, including fractional part.
            filter_half_len: Half length of the sinc interpolation filter.

        Returns:
            Delayed waveform with shape ``(slot_samples,)``.
        """
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
        """Generate one complex Rayleigh fading process.

        Args:
            num_samples: Number of time samples to generate.
            sample_rate_hz: Baseband sample rate in Hz.
            max_doppler_hz: Maximum Doppler frequency in Hz.
            num_sinusoids: Number of sinusoids used by the fading approximation.

        Returns:
            One-dimensional complex fading process with shape ``(num_samples,)``;
            axis 0 is time-sample index.
        """
        if abs(max_doppler_hz) < 1e-12:
            sample = (self.rng.normal() + 1j * self.rng.normal()) / np.sqrt(2.0)
            return np.full(num_samples, sample, dtype=np.complex128)

        time = self._time_axis(num_samples, sample_rate_hz)
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
        """Generate one complex Rician fading process.

        Args:
            num_samples: Number of time samples to generate.
            sample_rate_hz: Baseband sample rate in Hz.
            max_doppler_hz: Diffuse component maximum Doppler frequency in Hz.
            k_factor_linear: Rician K-factor in linear scale.
            num_sinusoids: Number of sinusoids used for the diffuse component.
            specular_doppler_hz: Optional Doppler frequency for the LOS component.
            initial_phase: Optional initial LOS phase in radians.

        Returns:
            One-dimensional complex fading process with shape ``(num_samples,)``.
        """
        diffuse = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)
        phase0 = self.rng.uniform(0.0, 2 * np.pi) if initial_phase is None else initial_phase
        spec_freq = specular_doppler_hz if specular_doppler_hz is not None else max_doppler_hz
        time = self._time_axis(num_samples, sample_rate_hz)
        specular = np.exp(1j * (2 * np.pi * spec_freq * time + phase0))
        return (
            np.sqrt(k_factor_linear / (k_factor_linear + 1.0)) * specular
            + np.sqrt(1.0 / (k_factor_linear + 1.0)) * diffuse
        )

    @staticmethod
    def _normalize_powers_db(power_db: np.ndarray) -> np.ndarray:
        """Normalize per-path powers from dB to linear fractions.

        Args:
            power_db: One-dimensional path power array with shape ``(num_paths,)``.

        Returns:
            One-dimensional linear power array with shape ``(num_paths,)`` whose
            values sum to 1.
        """
        linear = 10 ** (power_db / 10.0)
        linear /= np.sum(linear)
        return linear

    def _add_awgn(self, waveform: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, float, float]:
        """Add AWGN to a time-domain RX waveform.

        Args:
            waveform: RX waveform with shape ``(num_rx_ant, slot_samples)``; axis
                0 is RX antenna and axis 1 is time-sample index.
            config: Full simulation configuration that provides SNR.

        Returns:
            Tuple of noisy waveform with the same shape as ``waveform``, scalar
            noise variance, and scalar SNR in dB.
        """
        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(waveform) ** 2)
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise_rng = self._continuous_noise_rng if self._continuous_mode else self.rng
        noise = (
            noise_rng.normal(0.0, np.sqrt(noise_variance / 2.0), waveform.shape)
            + 1j * noise_rng.normal(0.0, np.sqrt(noise_variance / 2.0), waveform.shape)
        )
        return waveform + noise, float(noise_variance), snr_db

    @staticmethod
    def _carrier_frequency_hz(config: SimulationConfig) -> float:
        carrier_frequency_hz = float(config.carrier.center_frequency_hz)
        if not 0.5e9 <= carrier_frequency_hz <= 100.0e9:
            raise ValueError(
                "TDL/CDL link-level models are specified for carrier.center_frequency_hz "
                "between 0.5 GHz and 100 GHz."
            )
        return carrier_frequency_hz

    @classmethod
    def _validate_link_level_limits(cls, config: SimulationConfig) -> None:
        """Validate link-level 38.901 operating limits common to TDL/CDL.

        Args:
            config: Full simulation configuration containing carrier/channel params.

        Returns:
            None. A ``ValueError`` is raised when the configuration is outside the
            link-level TDL/CDL applicability range.
        """
        cls._carrier_frequency_hz(config)
        active_bandwidth_hz = (
            int(config.carrier.n_subcarriers)
            * float(config.carrier.subcarrier_spacing_khz)
            * 1e3
        )
        if active_bandwidth_hz > 2.0e9:
            raise ValueError("TDL/CDL link-level models support bandwidths up to 2 GHz.")

    @classmethod
    def _wavelength_m(cls, config: SimulationConfig) -> float:
        return 299792458.0 / cls._carrier_frequency_hz(config)

    @classmethod
    def _max_doppler_hz(cls, config: SimulationConfig) -> float:
        params = config.channel.params
        if "max_doppler_hz" in params:
            return float(params["max_doppler_hz"])
        ue_speed_mps = cls._ue_speed_mps(config)
        return ue_speed_mps / cls._wavelength_m(config)

    @classmethod
    def _ue_speed_mps(cls, config: SimulationConfig) -> float:
        velocity = cls._ue_velocity_vector_mps(config)
        if velocity is not None:
            return float(np.linalg.norm(velocity))
        return float(config.channel.params.get("ue_speed_mps", 0.0))

    @classmethod
    def _ue_motion_angles_deg(cls, config: SimulationConfig) -> tuple[float, float]:
        velocity = cls._ue_velocity_vector_mps(config)
        if velocity is not None and np.linalg.norm(velocity) > 1e-12:
            return cls._vector_to_azimuth_zenith_deg(velocity)
        return (
            float(config.channel.params.get("ue_azimuth_deg", 0.0)),
            float(config.channel.params.get("ue_zenith_deg", 90.0)),
        )

    @classmethod
    def _ue_motion_unit_vector(cls, config: SimulationConfig) -> np.ndarray:
        velocity = cls._ue_velocity_vector_mps(config)
        if velocity is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > 1e-12:
                return velocity / speed
        azimuth_deg, zenith_deg = cls._ue_motion_angles_deg(config)
        return cls._unit_vector(azimuth_deg, zenith_deg)

    @staticmethod
    def _ue_velocity_vector_mps(config: SimulationConfig) -> np.ndarray | None:
        geometry = config.channel.geometry
        value = geometry.get("ue_velocity_vector_mps")
        if value is None:
            value = config.channel.params.get("ue_velocity_vector_mps")
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError("channel.geometry.ue_velocity_vector_mps must be a 3-element vector.")
        return vector

    @classmethod
    def _channel_geometry_info(cls, config: SimulationConfig) -> dict[str, Any]:
        geometry = config.channel.geometry
        tx_position = cls._optional_position_m(geometry.get("tx_position_m"), "tx_position_m")
        rx_position = cls._optional_position_m(geometry.get("rx_position_m"), "rx_position_m")
        if (tx_position is None) != (rx_position is None):
            raise ValueError("channel.geometry.tx_position_m and rx_position_m must be configured together.")

        info: dict[str, Any] = {
            "ue_speed_mps": cls._ue_speed_mps(config),
        }
        azimuth_deg, zenith_deg = cls._ue_motion_angles_deg(config)
        info["ue_azimuth_deg"] = azimuth_deg
        info["ue_zenith_deg"] = zenith_deg
        velocity = cls._ue_velocity_vector_mps(config)
        if velocity is not None:
            info["ue_velocity_vector_mps"] = velocity

        if tx_position is None or rx_position is None:
            return info

        vector = rx_position - tx_position
        distance = float(np.linalg.norm(vector))
        if distance <= 0.0:
            raise ValueError("channel.geometry.tx_position_m and rx_position_m must not be identical.")
        info.update(
            {
                "tx_position_m": tx_position,
                "rx_position_m": rx_position,
                "tx_rx_distance_m": distance,
                "los_unit_vector_tx_to_rx": vector / distance,
            }
        )
        return info

    @staticmethod
    def _optional_position_m(value: Any, label: str) -> np.ndarray | None:
        if value is None:
            return None
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"channel.geometry.{label} must be a 3-element vector.")
        return position

    @staticmethod
    def _vector_to_azimuth_zenith_deg(vector: np.ndarray) -> tuple[float, float]:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return 0.0, 90.0
        unit = np.asarray(vector, dtype=np.float64) / norm
        azimuth_deg = float(np.rad2deg(np.arctan2(unit[1], unit[0])))
        zenith_deg = float(np.rad2deg(np.arccos(np.clip(unit[2], -1.0, 1.0))))
        return azimuth_deg, zenith_deg

    def _time_axis(self, num_samples: int, sample_rate_hz: float) -> np.ndarray:
        """Build a sample-time vector.

        Args:
            num_samples: Number of time samples.
            sample_rate_hz: Sampling rate in Hz.

        Returns:
            One-dimensional time array with shape ``(num_samples,)`` in seconds.
        """
        return self._time_offset_s + np.arange(num_samples, dtype=np.float64) / sample_rate_hz

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
        """Resolve default or user-provided path delays and powers.

        Args:
            config: Full simulation configuration containing channel params.
            normalized_delays: One-dimensional default delay array with shape
                ``(num_paths,)`` in profile-normalized units.
            power_db: One-dimensional default path power array with shape
                ``(num_paths,)`` in dB.
            delay_key: Config key for explicit path/cluster delays in ns.
            power_key: Config key for explicit path/cluster powers in dB.
            delay_spread_key: Config key for RMS delay spread in ns.

        Returns:
            Tuple ``(delays_s, power_db)`` where both arrays have shape
            ``(num_paths,)``.
        """
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

        desired_ds_s = FadingChannelBase._resolve_delay_spread_ns(params, delay_spread_key) * 1e-9
        return normalized_delays * desired_ds_s, power_db

    @staticmethod
    def _resolve_delay_spread_ns(params: dict[str, Any], delay_spread_key: str) -> float:
        """Resolve delay spread from explicit value or named lookup profile.

        Args:
            params: Channel parameter dictionary.
            delay_spread_key: Primary key, normally ``"delay_spread_ns"``.

        Returns:
            Delay spread in ns.
        """
        if params.get(delay_spread_key) is not None:
            return float(params[delay_spread_key])
        profile_name = str(params.get("delay_spread_profile", "nominal"))
        lookup = params.get("scenario_delay_spread_lookup")
        if isinstance(lookup, dict) and profile_name in lookup:
            return float(lookup[profile_name])
        defaults = {
            "very_short": 10.0,
            "short": 30.0,
            "nominal": 300.0,
            "long": 1000.0,
            "very_long": 3000.0,
        }
        return defaults.get(profile_name, defaults["nominal"])

    @staticmethod
    def _expand_tx_branches(waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Ensure the TX waveform has an explicit transmit-antenna axis.

        Args:
            waveform: Input waveform with shape ``(num_tx_ant, slot_samples)``.
            config: Full simulation configuration that defines ``num_tx_ant``.

        Returns:
            TX waveform matrix with shape ``(num_tx_ant, slot_samples)``; axis 0 is
            TX antenna and axis 1 is time-sample index.
        """
        if waveform.ndim != 2:
            raise ValueError("Fading channel expects waveform shape (num_tx_ant, slot_samples).")
        expected_tx_ant = int(config.link.num_tx_ant)
        if waveform.shape[0] != expected_tx_ant:
            raise ValueError(f"TX waveform antenna dimension must be {expected_tx_ant}, got {waveform.shape[0]}.")
        return waveform

    @staticmethod
    def _array_response(
        num_ant: int,
        spatial_frequency: float,
        spacing_lambda: float,
    ) -> np.ndarray:
        """Generate a uniform-linear-array response vector.

        Args:
            num_ant: Number of antenna elements.
            spatial_frequency: Direction cosine projected onto the array axis.
            spacing_lambda: Element spacing measured in wavelengths.

        Returns:
            One-dimensional complex response vector with shape ``(num_ant,)``;
            axis 0 is antenna-element index.
        """
        antenna_index = np.arange(num_ant, dtype=np.float64)
        return np.exp(1j * 2.0 * np.pi * spacing_lambda * spatial_frequency * antenna_index)

    @staticmethod
    def _antenna_array(
        config: SimulationConfig,
        side: str,
        num_ports: int,
    ) -> AntennaArrayDescription:
        """Resolve antenna-array geometry from channel params.

        Args:
            config: Full simulation configuration containing channel params.
            side: ``"tx"`` or ``"rx"``.
            num_ports: Number of TX/RX antenna ports.

        Returns:
            Resolved antenna array with one row per port.
        """
        params = config.channel.params
        array_cfg = params.get(f"{side}_array")
        spacing_key = f"{side}_antenna_spacing_lambda"
        if array_cfg is None:
            array_cfg = {}
        if not isinstance(array_cfg, dict):
            raise ValueError(f"{side}_array must be a mapping when provided.")

        polarization = str(array_cfg.get("polarization", "single")).lower()
        spacing = float(array_cfg.get("element_spacing_lambda", params.get(spacing_key, 0.5)))
        positions_value = array_cfg.get("positions_lambda")
        if positions_value is not None:
            positions = np.asarray(positions_value, dtype=np.float64)
            if positions.shape != (num_ports, 3):
                raise ValueError(f"{side}_array.positions_lambda must have shape ({num_ports}, 3).")
        elif polarization in {"dual", "dual_slant", "cross"}:
            if num_ports % 2 != 0:
                raise ValueError(f"{side}_array dual polarization requires an even number of ports.")
            num_elements = num_ports // 2
            base_positions = np.zeros((num_elements, 3), dtype=np.float64)
            base_positions[:, 0] = np.arange(num_elements, dtype=np.float64) * spacing
            positions = np.repeat(base_positions, 2, axis=0)
        else:
            positions = np.zeros((num_ports, 3), dtype=np.float64)
            positions[:, 0] = np.arange(num_ports, dtype=np.float64) * spacing

        slants_value = array_cfg.get("polarization_slants_deg")
        if slants_value is not None:
            slants = np.asarray(slants_value, dtype=np.float64)
            if slants.shape != (num_ports,):
                raise ValueError(f"{side}_array.polarization_slants_deg must have shape ({num_ports},).")
        elif polarization in {"dual", "dual_slant", "cross"}:
            slants = np.tile(np.array([45.0, -45.0], dtype=np.float64), num_ports // 2)
        else:
            slants = np.zeros(num_ports, dtype=np.float64)

        return AntennaArrayDescription(
            positions_lambda=positions,
            polarization_slants_deg=slants,
            polarization=polarization,
        )

    @staticmethod
    def _array_phase(array: AntennaArrayDescription, direction: np.ndarray) -> np.ndarray:
        """Compute array phase for one propagation direction.

        Args:
            array: Resolved antenna array.
            direction: Cartesian unit vector with shape ``(3,)``.

        Returns:
            Complex phase vector with shape ``(num_ports,)``.
        """
        return np.exp(1j * 2.0 * np.pi * (array.positions_lambda @ direction))

    @staticmethod
    def _field_pattern(array: AntennaArrayDescription, _direction: np.ndarray) -> np.ndarray:
        """Return isotropic theta/phi field components for all ports.

        Args:
            array: Resolved antenna array.
            _direction: Cartesian unit vector with shape ``(3,)``. The current
                isotropic element model is direction independent.

        Returns:
            Field array with shape ``(num_ports, 2)`` where axis 1 is
            ``[theta, phi]`` polarization component.
        """
        slant_rad = np.deg2rad(array.polarization_slants_deg)
        return np.stack([np.cos(slant_rad), np.sin(slant_rad)], axis=1).astype(np.complex128)

    @staticmethod
    def _matrix_sqrt_hermitian(matrix: np.ndarray, expected_size: int, label: str) -> np.ndarray:
        """Compute a Hermitian matrix square root with validation.

        Args:
            matrix: Correlation matrix with shape ``(expected_size, expected_size)``.
            expected_size: Required matrix dimension.
            label: User-facing label for validation errors.

        Returns:
            Hermitian square root matrix with the same shape.
        """
        value = np.asarray(matrix, dtype=np.complex128)
        if value.shape != (expected_size, expected_size):
            raise ValueError(f"{label} must have shape ({expected_size}, {expected_size}).")
        if not np.allclose(value, value.conj().T, atol=1e-10):
            raise ValueError(f"{label} must be Hermitian.")
        eigvals, eigvecs = np.linalg.eigh(value)
        if np.min(eigvals) < -1e-10:
            raise ValueError(f"{label} must be positive semidefinite.")
        eigvals = np.maximum(eigvals, 0.0)
        return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T

    @classmethod
    def _apply_mimo_correlation(
        cls,
        matrix: np.ndarray,
        rx_correlation: np.ndarray | None,
        tx_correlation: np.ndarray | None,
    ) -> np.ndarray:
        """Apply separable RX/TX spatial correlation to MIMO coefficients.

        Args:
            matrix: IID MIMO process with shape ``(num_rx_ant, num_tx_ant, num_samples)``.
            rx_correlation: Optional RX correlation matrix.
            tx_correlation: Optional TX correlation matrix.

        Returns:
            Correlated process with the same shape as ``matrix``.
        """
        num_rx_ant, num_tx_ant, _ = matrix.shape
        rx_sqrt = (
            np.eye(num_rx_ant, dtype=np.complex128)
            if rx_correlation is None
            else cls._matrix_sqrt_hermitian(rx_correlation, num_rx_ant, "tdl_rx_correlation")
        )
        tx_sqrt = (
            np.eye(num_tx_ant, dtype=np.complex128)
            if tx_correlation is None
            else cls._matrix_sqrt_hermitian(tx_correlation, num_tx_ant, "tdl_tx_correlation")
        )
        return np.einsum("ab,bct,dc->adt", rx_sqrt, matrix, tx_sqrt.conj())

    @staticmethod
    def _angle_scale_values(
        values_deg: np.ndarray,
        power_weights: np.ndarray,
        desired_spread_deg: float | None,
        desired_mean_deg: float | None,
        *,
        circular: bool,
    ) -> np.ndarray:
        """Scale and shift cluster angles to requested mean/spread.

        Args:
            values_deg: Angle values with shape ``(num_clusters,)``.
            power_weights: Linear cluster power weights with the same shape.
            desired_spread_deg: Optional target RMS angular spread.
            desired_mean_deg: Optional target mean angle.
            circular: Whether angles wrap around at +/-180 degrees.

        Returns:
            Scaled angle values with shape ``(num_clusters,)``.
        """
        values = np.asarray(values_deg, dtype=np.float64)
        weights = np.asarray(power_weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        if circular:
            radians = np.deg2rad(values)
            mean_rad = np.angle(np.sum(weights * np.exp(1j * radians)))
            centered = np.rad2deg(np.angle(np.exp(1j * (radians - mean_rad))))
            current_mean = np.rad2deg(mean_rad)
        else:
            current_mean = float(np.sum(weights * values))
            centered = values - current_mean

        current_spread = float(np.sqrt(np.sum(weights * centered**2)))
        target_center = current_mean if desired_mean_deg is None else float(desired_mean_deg)
        if desired_spread_deg is not None and current_spread > 1e-12:
            centered = centered * (float(desired_spread_deg) / current_spread)
        result = target_center + centered
        if circular:
            for _ in range(4):
                result_rad = np.deg2rad(result)
                actual_mean = np.rad2deg(np.angle(np.sum(weights * np.exp(1j * result_rad))))
                result += float(target_center) - actual_mean
                centered_result = np.rad2deg(
                    np.angle(np.exp(1j * np.deg2rad(result - float(target_center))))
                )
                if desired_spread_deg is not None:
                    spread = float(np.sqrt(np.sum(weights * centered_result**2)))
                    if spread > 1e-12:
                        result = float(target_center) + centered_result * (float(desired_spread_deg) / spread)
            result = ((result + 180.0) % 360.0) - 180.0
        return result

    @staticmethod
    def _unit_vector(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
        """Convert spherical angles to a Cartesian unit vector.

        Args:
            azimuth_deg: Azimuth angle in degrees.
            zenith_deg: Zenith angle in degrees.

        Returns:
            One-dimensional real vector with shape ``(3,)``; axes are x, y, z.
        """
        az = np.deg2rad(azimuth_deg)
        ze = np.deg2rad(zenith_deg)
        sin_ze = np.sin(ze)
        return np.array(
            [
                sin_ze * np.cos(az),
                sin_ze * np.sin(az),
                np.cos(ze),
            ],
            dtype=np.float64,
        )
