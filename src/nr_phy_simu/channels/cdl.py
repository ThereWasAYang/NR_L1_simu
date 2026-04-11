from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import CDL_LOS_K_DB, CDL_PROFILES, RAY_OFFSETS
from nr_phy_simu.config import SimulationConfig


class CdlChannel(FadingChannelBase):
    """3GPP TR 38.901 CDL channel model with multi-TX multi-RX support."""

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
        normalized_delays = np.array([cluster.normalized_delay for cluster in profile.clusters], dtype=np.float64)
        default_power_db = np.array([cluster.power_db for cluster in profile.clusters], dtype=np.float64)
        delays_s, power_db = self._resolve_path_parameters(
            config,
            normalized_delays,
            default_power_db,
            delay_key="cluster_delays_ns" if "cluster_delays_ns" in config.channel.params else "path_delays_ns",
            power_key="cluster_powers_db" if "cluster_powers_db" in config.channel.params else "path_powers_db",
        )
        if delays_s.size != len(profile.clusters):
            raise ValueError("CDL override delays/powers must match the number of clusters in the selected profile.")

        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        path_powers = self._normalize_powers_db(power_db)
        coeff = np.zeros((num_rx_ant, num_tx_ant, delays_s.size, num_samples), dtype=np.complex128)

        velocity_az_deg = float(config.channel.params.get("ue_azimuth_deg", 0.0))
        velocity_ze_deg = float(config.channel.params.get("ue_zenith_deg", 90.0))
        velocity_vec = self._unit_vector(velocity_az_deg, velocity_ze_deg)
        wavelength = self._wavelength_m(config)
        max_doppler_hz = self._max_doppler_hz(config)
        num_sinusoids = int(config.channel.params.get("num_sinusoids", 32))
        tx_spacing = float(config.channel.params.get("tx_antenna_spacing_lambda", 0.5))
        rx_spacing = float(config.channel.params.get("rx_antenna_spacing_lambda", 0.5))

        for cluster_idx, cluster in enumerate(profile.clusters):
            cluster_matrix = np.zeros((num_rx_ant, num_tx_ant, num_samples), dtype=np.complex128)
            for ray_offset in RAY_OFFSETS:
                ray_aod = cluster.aod_deg + ray_offset * profile.c_asd_deg
                ray_aoa = cluster.aoa_deg + ray_offset * profile.c_asa_deg
                ray_zod = cluster.zod_deg + ray_offset * profile.c_zsd_deg
                ray_zoa = cluster.zoa_deg + ray_offset * profile.c_zsa_deg

                arrival_vec = self._unit_vector(ray_aoa, ray_zoa)
                departure_vec = self._unit_vector(ray_aod, ray_zod)
                ray_doppler = np.dot(arrival_vec, velocity_vec) / wavelength

                time = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
                phase = self.rng.uniform(0.0, 2 * np.pi)
                ray_process = np.exp(1j * (2 * np.pi * ray_doppler * time + phase))
                rx_response = self._array_response(num_rx_ant, arrival_vec[0], rx_spacing)
                tx_response = self._array_response(num_tx_ant, departure_vec[0], tx_spacing)
                cluster_matrix += (
                    rx_response[:, np.newaxis, np.newaxis]
                    * np.conj(tx_response)[np.newaxis, :, np.newaxis]
                    * ray_process[np.newaxis, np.newaxis, :]
                )

            cluster_matrix /= np.sqrt(len(RAY_OFFSETS))
            if cluster.fading.upper() == "LOS":
                k_factor_db = float(config.channel.params.get("k_factor_db", CDL_LOS_K_DB.get(profile_name, 0.0)))
                los_process = self._rician_process(
                    num_samples=num_samples,
                    sample_rate_hz=sample_rate_hz,
                    max_doppler_hz=max_doppler_hz,
                    k_factor_linear=10 ** (k_factor_db / 10.0),
                    num_sinusoids=num_sinusoids,
                    specular_doppler_hz=np.dot(self._unit_vector(cluster.aoa_deg, cluster.zoa_deg), velocity_vec)
                    / wavelength,
                )
                rx_response = self._array_response(num_rx_ant, self._unit_vector(cluster.aoa_deg, cluster.zoa_deg)[0], rx_spacing)
                tx_response = self._array_response(num_tx_ant, self._unit_vector(cluster.aod_deg, cluster.zod_deg)[0], tx_spacing)
                cluster_matrix = (
                    rx_response[:, np.newaxis, np.newaxis]
                    * np.conj(tx_response)[np.newaxis, :, np.newaxis]
                    * los_process[np.newaxis, np.newaxis, :]
                )

            coeff[:, :, cluster_idx, :] = np.sqrt(path_powers[cluster_idx]) * cluster_matrix

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
