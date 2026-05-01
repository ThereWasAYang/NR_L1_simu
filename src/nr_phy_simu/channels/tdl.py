from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import TDL_LOS_K_DB, TDL_PROFILES
from nr_phy_simu.config import SimulationConfig


class TdlChannel(FadingChannelBase):
    """3GPP TR 38.901 TDL channel model with multi-TX multi-RX support."""

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
        tx_spacing = float(config.channel.params.get("tx_antenna_spacing_lambda", 0.5))
        rx_spacing = float(config.channel.params.get("rx_antenna_spacing_lambda", 0.5))

        coeff = np.zeros((num_rx_ant, num_tx_ant, delays_s.size, num_samples), dtype=np.complex128)
        tap_fading = list(config.channel.params.get("path_fading", [tap.fading for tap in taps]))
        if len(tap_fading) != delays_s.size:
            if len(tap_fading) == len(taps):
                tap_fading = tap_fading[: delays_s.size]
            else:
                raise ValueError("TDL 'path_fading' length must match the number of channel paths.")

        for path_idx in range(delays_s.size):
            fading = str(tap_fading[path_idx]).upper()
            if fading == "LOS":
                process = self._rician_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                    specular_doppler_hz=0.7 * max_doppler_hz,
                )
            else:
                process = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)

            tx_spatial_freq = self.rng.uniform(-1.0, 1.0)
            rx_spatial_freq = self.rng.uniform(-1.0, 1.0)
            tx_response = self._array_response(num_tx_ant, tx_spatial_freq, tx_spacing)
            rx_response = self._array_response(num_rx_ant, rx_spatial_freq, rx_spacing)
            spatial = np.outer(rx_response, np.conj(tx_response))
            coeff[:, :, path_idx, :] = (
                np.sqrt(path_powers[path_idx]) * spatial[:, :, np.newaxis] * process[np.newaxis, np.newaxis, :]
            )

        return delays_s, coeff
