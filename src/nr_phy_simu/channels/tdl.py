from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import TDL_LOS_K_DB, TDL_PROFILES
from nr_phy_simu.config import SimulationConfig


class TdlChannel(FadingChannelBase):
    """3GPP TR 38.901 link-level TDL channel model."""

    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate TDL path delays and MIMO coefficients.

        Args:
            num_samples: Number of time samples in the slot waveform.
            sample_rate_hz: Baseband sample rate in Hz.
            config: Full simulation configuration with TDL profile and antennas.

        Returns:
            Tuple ``(delays_s, coeff)`` where ``delays_s`` has shape ``(num_paths,)``
            and ``coeff`` has shape
            ``(num_rx_ant, num_tx_ant, num_paths, num_samples)``.
        """
        profile_name = str(config.channel.params.get("profile", "TDL-A")).upper()
        if profile_name not in TDL_PROFILES:
            raise ValueError(f"Unsupported TDL profile '{profile_name}'.")

        taps = TDL_PROFILES[profile_name]
        normalized_delays = np.array([tap.normalized_delay for tap in taps], dtype=np.float64)
        default_power_db = np.array([tap.power_db for tap in taps], dtype=np.float64)
        delays_s, power_db = self._resolve_path_parameters(config, normalized_delays, default_power_db)
        path_powers = self._normalize_powers_db(power_db)

        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        max_doppler_hz = self._max_doppler_hz(config)
        num_sinusoids = int(config.channel.params.get("num_sinusoids", 32))
        k_factor_db = float(config.channel.params.get("k_factor_db", TDL_LOS_K_DB.get(profile_name, 0.0)))
        method = str(config.channel.params.get("tdl_mimo_method", "iid")).lower()
        rx_correlation = config.channel.params.get("tdl_rx_correlation")
        tx_correlation = config.channel.params.get("tdl_tx_correlation")
        spatial_filter = self._resolve_spatial_filter(config, delays_s.size)

        coeff = np.zeros((num_rx_ant, num_tx_ant, delays_s.size, num_samples), dtype=np.complex128)
        tap_fading = list(config.channel.params.get("path_fading", [tap.fading for tap in taps]))
        if len(tap_fading) != delays_s.size:
            if len(tap_fading) == len(taps):
                tap_fading = tap_fading[: delays_s.size]
            else:
                raise ValueError("TDL 'path_fading' length must match the number of channel paths.")

        for path_idx in range(delays_s.size):
            fading = str(tap_fading[path_idx]).upper()
            if spatial_filter is not None:
                process = self._scalar_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    fading=fading,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                )
                process_matrix = spatial_filter[path_idx, :, :, np.newaxis] * process[np.newaxis, np.newaxis, :]
            elif method in {"spatial_filter", "spatial_filter_from_cdl"}:
                process_matrix = self._spatial_filter_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    fading=fading,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                    config=config,
                )
            else:
                process_matrix = self._iid_mimo_process(
                    num_rx_ant=num_rx_ant,
                    num_tx_ant=num_tx_ant,
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    fading=fading,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                )
                if method in {"correlated", "correlation", "kronecker"} or rx_correlation is not None or tx_correlation is not None:
                    process_matrix = self._apply_mimo_correlation(
                        process_matrix,
                        None if rx_correlation is None else np.asarray(rx_correlation, dtype=np.complex128),
                        None if tx_correlation is None else np.asarray(tx_correlation, dtype=np.complex128),
                    )
                elif method not in {"iid", "zero_correlation", "zero-correlation"}:
                    raise ValueError(
                        "Unsupported tdl_mimo_method. Use 'iid', 'correlated', or 'spatial_filter'."
                    )

            coeff[:, :, path_idx, :] = np.sqrt(path_powers[path_idx]) * process_matrix

        return delays_s, coeff

    def _scalar_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        fading: str,
        k_factor_linear: float,
        num_sinusoids: int,
    ) -> np.ndarray:
        """Generate one scalar TDL tap process.

        Args:
            num_samples: Number of generated samples.
            sample_rate_hz: Baseband sample rate in Hz.
            max_doppler_hz: Maximum Doppler frequency in Hz.
            fading: Tap fading type.
            k_factor_linear: Linear K-factor for LOS taps.
            num_sinusoids: Number of sinusoids for diffuse fading.

        Returns:
            Complex process with shape ``(num_samples,)``.
        """
        if fading == "LOS":
            return self._rician_process(
                num_samples=num_samples,
                sample_rate_hz=sample_rate_hz,
                max_doppler_hz=max_doppler_hz,
                k_factor_linear=k_factor_linear,
                num_sinusoids=num_sinusoids,
                specular_doppler_hz=0.7 * max_doppler_hz,
            )
        return self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)

    def _iid_mimo_process(
        self,
        num_rx_ant: int,
        num_tx_ant: int,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        fading: str,
        k_factor_linear: float,
        num_sinusoids: int,
    ) -> np.ndarray:
        """Generate zero-correlation IID TDL MIMO branch processes.

        Args:
            num_rx_ant: Number of receive antennas.
            num_tx_ant: Number of transmit antennas.
            num_samples: Number of generated samples.
            sample_rate_hz: Baseband sample rate in Hz.
            max_doppler_hz: Maximum Doppler frequency in Hz.
            fading: Tap fading type, ``"LOS"`` or Rayleigh-like.
            k_factor_linear: Linear K-factor for LOS taps.
            num_sinusoids: Number of sinusoids for diffuse fading.

        Returns:
            MIMO process with shape ``(num_rx_ant, num_tx_ant, num_samples)``.
        """
        process = np.zeros((num_rx_ant, num_tx_ant, num_samples), dtype=np.complex128)
        for rx_idx in range(num_rx_ant):
            for tx_idx in range(num_tx_ant):
                if fading == "LOS":
                    process[rx_idx, tx_idx] = self._rician_process(
                        num_samples=num_samples,
                        sample_rate_hz=sample_rate_hz,
                        max_doppler_hz=max_doppler_hz,
                        k_factor_linear=k_factor_linear,
                        num_sinusoids=num_sinusoids,
                        specular_doppler_hz=0.7 * max_doppler_hz,
                    )
                else:
                    process[rx_idx, tx_idx] = self._rayleigh_process(
                        num_samples,
                        sample_rate_hz,
                        max_doppler_hz,
                        num_sinusoids,
                    )
        return process

    def _spatial_filter_process(
        self,
        num_samples: int,
        sample_rate_hz: float,
        max_doppler_hz: float,
        fading: str,
        k_factor_linear: float,
        num_sinusoids: int,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Generate a spatially filtered TDL branch process from array responses.

        Args:
            num_samples: Number of generated samples.
            sample_rate_hz: Baseband sample rate in Hz.
            max_doppler_hz: Maximum Doppler frequency in Hz.
            fading: Tap fading type.
            k_factor_linear: Linear K-factor for LOS taps.
            num_sinusoids: Number of sinusoids for diffuse fading.
            config: Full simulation configuration.

        Returns:
            MIMO process with shape ``(num_rx_ant, num_tx_ant, num_samples)``.
        """
        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        rx_array = self._antenna_array(config, "rx", num_rx_ant)
        tx_array = self._antenna_array(config, "tx", num_tx_ant)
        arrival = self._unit_vector(self.rng.uniform(-180.0, 180.0), self.rng.uniform(45.0, 135.0))
        departure = self._unit_vector(self.rng.uniform(-180.0, 180.0), self.rng.uniform(45.0, 135.0))
        rx_response = self._array_phase(rx_array, arrival)
        tx_response = self._array_phase(tx_array, departure)
        spatial = rx_response[:, np.newaxis] * tx_response[np.newaxis, :]
        if fading == "LOS":
            process = self._rician_process(
                num_samples=num_samples,
                sample_rate_hz=sample_rate_hz,
                max_doppler_hz=max_doppler_hz,
                k_factor_linear=k_factor_linear,
                num_sinusoids=num_sinusoids,
                specular_doppler_hz=0.7 * max_doppler_hz,
            )
        else:
            process = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)
        return spatial[:, :, np.newaxis] * process[np.newaxis, np.newaxis, :]

    def _resolve_spatial_filter(self, config: SimulationConfig, num_paths: int) -> np.ndarray | None:
        """Resolve an optional explicit TDL spatial filter.

        Args:
            config: Full simulation configuration.
            num_paths: Number of TDL paths.

        Returns:
            Spatial filter with shape ``(num_paths, num_rx_ant, num_tx_ant)``, or
            ``None`` when no explicit filter is configured.
        """
        value = config.channel.params.get("spatial_filter")
        if value is None:
            return None
        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        spatial_filter = np.asarray(value, dtype=np.complex128)
        if spatial_filter.shape == (num_rx_ant, num_tx_ant):
            return np.repeat(spatial_filter[np.newaxis, :, :], num_paths, axis=0)
        expected = (num_paths, num_rx_ant, num_tx_ant)
        if spatial_filter.shape != expected:
            raise ValueError(f"spatial_filter must have shape ({num_rx_ant}, {num_tx_ant}) or {expected}.")
        return spatial_filter
