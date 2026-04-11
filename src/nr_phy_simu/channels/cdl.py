from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import CDL_LOS_K_DB, CDL_PROFILES, RAY_OFFSETS
from nr_phy_simu.config import SimulationConfig


class CdlChannel(FadingChannelBase):
    """3GPP TR 38.901 CDL channel model for SISO link-level simulation."""

    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        profile_name = str(config.channel.params.get("profile", "CDL-A")).upper()
        if profile_name not in CDL_PROFILES:
            raise ValueError(f"Unsupported CDL profile '{profile_name}'.")

        profile = CDL_PROFILES[profile_name]
        desired_ds_s = float(config.channel.params.get("delay_spread_ns", 300.0)) * 1e-9
        max_doppler_hz = self._max_doppler_hz(config)
        power_db = np.array([cluster.power_db for cluster in profile.clusters], dtype=np.float64)
        path_powers = self._normalize_powers_db(power_db)
        delays_s = np.array([cluster.normalized_delay for cluster in profile.clusters], dtype=np.float64) * desired_ds_s
        coeff = np.zeros((len(profile.clusters), num_samples), dtype=np.complex128)

        velocity_az_deg = float(config.channel.params.get("ue_azimuth_deg", 0.0))
        velocity_ze_deg = float(config.channel.params.get("ue_zenith_deg", 90.0))
        velocity_vec = self._unit_vector(velocity_az_deg, velocity_ze_deg)
        wavelength = self._wavelength_m(config)

        for cluster_idx, cluster in enumerate(profile.clusters):
            cluster_signal = np.zeros(num_samples, dtype=np.complex128)
            for ray_offset in RAY_OFFSETS:
                ray_aoa = cluster.aoa_deg + ray_offset * profile.c_asa_deg
                ray_zoa = cluster.zoa_deg + ray_offset * profile.c_zsa_deg
                arrival_vec = self._unit_vector(ray_aoa, ray_zoa)
                ray_doppler = np.dot(arrival_vec, velocity_vec) / wavelength
                phase = self.rng.uniform(0.0, 2 * np.pi)
                time = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
                ray = np.exp(1j * (2 * np.pi * ray_doppler * time + phase))
                cluster_signal += ray

            cluster_signal /= np.sqrt(len(RAY_OFFSETS))
            if cluster.fading == "LOS":
                k_factor_db = float(config.channel.params.get("k_factor_db", CDL_LOS_K_DB.get(profile_name, 0.0)))
                cluster_signal = self._rician_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=int(config.channel.params.get("num_sinusoids", 32)),
                    specular_doppler_hz=np.dot(self._unit_vector(cluster.aoa_deg, cluster.zoa_deg), velocity_vec)
                    / wavelength,
                )
            coeff[cluster_idx, :] = np.sqrt(path_powers[cluster_idx]) * cluster_signal

        return delays_s, coeff

    @staticmethod
    def _unit_vector(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
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
