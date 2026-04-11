from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import TDL_LOS_K_DB, TDL_PROFILES
from nr_phy_simu.config import SimulationConfig


class TdlChannel(FadingChannelBase):
    """3GPP TR 38.901 TDL channel model for SISO link-level simulation."""

    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        profile_name = str(config.channel.params.get("profile", "TDL-A")).upper()
        if profile_name not in TDL_PROFILES:
            raise ValueError(f"Unsupported TDL profile '{profile_name}'.")

        taps = TDL_PROFILES[profile_name]
        normalized_delays = np.array([tap.normalized_delay for tap in taps], dtype=np.float64)
        power_db = np.array([tap.power_db for tap in taps], dtype=np.float64)
        k_factor_db = float(config.channel.params.get("k_factor_db", TDL_LOS_K_DB.get(profile_name, 0.0)))
        desired_ds_s = float(config.channel.params.get("delay_spread_ns", 300.0)) * 1e-9
        max_doppler_hz = self._max_doppler_hz(config)
        num_sinusoids = int(config.channel.params.get("num_sinusoids", 32))

        path_powers = self._normalize_powers_db(power_db)
        delays_s = normalized_delays * desired_ds_s

        coeff = np.zeros((len(taps), num_samples), dtype=np.complex128)
        for path_idx, tap in enumerate(taps):
            process = self._rayleigh_process(num_samples, sample_rate_hz, max_doppler_hz, num_sinusoids)
            if tap.fading == "LOS":
                process = self._rician_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                    specular_doppler_hz=0.7 * max_doppler_hz,
                )
            coeff[path_idx, :] = np.sqrt(path_powers[path_idx]) * process

        return delays_s, coeff
