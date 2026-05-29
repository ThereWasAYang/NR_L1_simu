from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import CDL_LOS_K_DB, CDL_PROFILES, RAY_OFFSETS
from nr_phy_simu.config import SimulationConfig


class CdlChannel(FadingChannelBase):
    """3GPP TR 38.901 link-level CDL channel model."""

    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate CDL cluster delays and MIMO coefficients.

        Args:
            num_samples: Number of time samples in the slot waveform.
            sample_rate_hz: Baseband sample rate in Hz.
            config: Full simulation configuration with CDL profile and antennas.

        Returns:
            Tuple ``(delays_s, coeff)`` where ``delays_s`` has shape
            ``(num_clusters,)`` and ``coeff`` has shape
            ``(num_rx_ant, num_tx_ant, num_clusters, num_samples)``.
        """
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

        velocity_vec = self._ue_motion_unit_vector(config)
        max_doppler_hz = self._max_doppler_hz(config)
        time = self._time_axis(num_samples, sample_rate_hz)
        rx_array = self._antenna_array(config, "rx", num_rx_ant)
        tx_array = self._antenna_array(config, "tx", num_tx_ant)
        xpr_param = config.channel.params.get("xpr_db")
        xpr_db = profile.xpr_db if xpr_param is None else float(xpr_param)
        xpr_sigma_db = float(config.channel.params.get("xpr_sigma_db", 0.0))
        scaled_angles = self._resolve_cluster_angles(config, profile, path_powers)

        for cluster_idx, cluster in enumerate(profile.clusters):
            cluster_matrix = np.zeros((num_rx_ant, num_tx_ant, num_samples), dtype=np.complex128)
            ray_angles = self._ray_angles_for_cluster(
                cluster_idx=cluster_idx,
                scaled_angles=scaled_angles,
                profile=profile,
            )

            for ray_aod, ray_aoa, ray_zod, ray_zoa in ray_angles:
                arrival_vec = self._unit_vector(ray_aoa, ray_zoa)
                departure_vec = self._unit_vector(ray_aod, ray_zod)
                ray_doppler = max_doppler_hz * float(np.dot(arrival_vec, velocity_vec))
                ray_process = np.exp(1j * 2.0 * np.pi * ray_doppler * time)
                ray_xpr_db = xpr_db
                if xpr_sigma_db > 0.0:
                    ray_xpr_db += float(self.rng.normal(0.0, xpr_sigma_db))
                spatial = self._polarized_spatial_matrix(
                    rx_array=rx_array,
                    tx_array=tx_array,
                    arrival_vec=arrival_vec,
                    departure_vec=departure_vec,
                    xpr_linear=10 ** (ray_xpr_db / 10.0),
                )
                cluster_matrix += spatial[:, :, np.newaxis] * ray_process[np.newaxis, np.newaxis, :]

            cluster_matrix /= np.sqrt(len(RAY_OFFSETS))
            if cluster.fading.upper() == "LOS":
                k_factor_db = float(config.channel.params.get("k_factor_db", CDL_LOS_K_DB.get(profile_name, 0.0)))
                k_factor_linear = 10 ** (k_factor_db / 10.0)
                los_arrival = self._unit_vector(scaled_angles["aoa"][cluster_idx], scaled_angles["zoa"][cluster_idx])
                los_departure = self._unit_vector(scaled_angles["aod"][cluster_idx], scaled_angles["zod"][cluster_idx])
                los_doppler = max_doppler_hz * float(np.dot(los_arrival, velocity_vec))
                los_process = np.exp(
                    1j
                    * (
                        2.0 * np.pi * los_doppler * time
                        + self.rng.uniform(0.0, 2.0 * np.pi)
                    )
                )
                los_spatial = self._los_spatial_matrix(
                    rx_array=rx_array,
                    tx_array=tx_array,
                    arrival_vec=los_arrival,
                    departure_vec=los_departure,
                )
                los_matrix = los_spatial[:, :, np.newaxis] * los_process[np.newaxis, np.newaxis, :]
                cluster_matrix = (
                    np.sqrt(1.0 / (k_factor_linear + 1.0)) * cluster_matrix
                    + np.sqrt(k_factor_linear / (k_factor_linear + 1.0)) * los_matrix
                )

            coeff[:, :, cluster_idx, :] = np.sqrt(path_powers[cluster_idx]) * cluster_matrix

        return delays_s, coeff

    def _resolve_cluster_angles(
        self,
        config: SimulationConfig,
        profile,
        path_powers: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Resolve optional CDL angle scaling targets.

        Args:
            config: Full simulation configuration.
            profile: Selected CDL profile table.
            path_powers: Linear cluster powers with shape ``(num_clusters,)``.

        Returns:
            Mapping with ``aod/aoa/zod/zoa`` arrays, each shape
            ``(num_clusters,)``.
        """
        params = config.channel.params
        angles = {
            "aod": np.array([cluster.aod_deg for cluster in profile.clusters], dtype=np.float64),
            "aoa": np.array([cluster.aoa_deg for cluster in profile.clusters], dtype=np.float64),
            "zod": np.array([cluster.zod_deg for cluster in profile.clusters], dtype=np.float64),
            "zoa": np.array([cluster.zoa_deg for cluster in profile.clusters], dtype=np.float64),
        }
        if not bool(params.get("angle_scaling_enabled", False)):
            return angles
        return {
            "aod": self._angle_scale_values(
                angles["aod"],
                path_powers,
                params.get("desired_asd_deg"),
                params.get("desired_mean_aod_deg"),
                circular=True,
            ),
            "aoa": self._angle_scale_values(
                angles["aoa"],
                path_powers,
                params.get("desired_asa_deg"),
                params.get("desired_mean_aoa_deg"),
                circular=True,
            ),
            "zod": self._angle_scale_values(
                angles["zod"],
                path_powers,
                params.get("desired_zsd_deg"),
                params.get("desired_mean_zod_deg"),
                circular=False,
            ),
            "zoa": self._angle_scale_values(
                angles["zoa"],
                path_powers,
                params.get("desired_zsa_deg"),
                params.get("desired_mean_zoa_deg"),
                circular=False,
            ),
        }

    def _ray_angles_for_cluster(
        self,
        cluster_idx: int,
        scaled_angles: dict[str, np.ndarray],
        profile,
    ) -> list[tuple[float, float, float, float]]:
        """Generate randomly coupled ray angles for one CDL cluster.

        Args:
            cluster_idx: Cluster index.
            scaled_angles: Mapping returned by ``_resolve_cluster_angles``.
            profile: Selected CDL profile table.

        Returns:
            List of ``(AOD, AOA, ZOD, ZOA)`` tuples, one per ray.
        """
        offsets = np.asarray(RAY_OFFSETS, dtype=np.float64)
        aod_offsets = offsets
        aoa_offsets = self.rng.permutation(offsets)
        zod_offsets = self.rng.permutation(offsets)
        zoa_offsets = self.rng.permutation(offsets)
        return [
            (
                float(scaled_angles["aod"][cluster_idx] + aod_offsets[ray_idx] * profile.c_asd_deg),
                float(scaled_angles["aoa"][cluster_idx] + aoa_offsets[ray_idx] * profile.c_asa_deg),
                float(scaled_angles["zod"][cluster_idx] + zod_offsets[ray_idx] * profile.c_zsd_deg),
                float(scaled_angles["zoa"][cluster_idx] + zoa_offsets[ray_idx] * profile.c_zsa_deg),
            )
            for ray_idx in range(offsets.size)
        ]

    def _polarized_spatial_matrix(
        self,
        rx_array,
        tx_array,
        arrival_vec: np.ndarray,
        departure_vec: np.ndarray,
        xpr_linear: float,
    ) -> np.ndarray:
        """Build one CDL ray MIMO matrix including polarization coupling.

        Args:
            rx_array: Resolved RX antenna array.
            tx_array: Resolved TX antenna array.
            arrival_vec: Arrival unit vector with shape ``(3,)``.
            departure_vec: Departure unit vector with shape ``(3,)``.
            xpr_linear: Cross-polarization ratio in linear scale.

        Returns:
            Complex MIMO matrix with shape ``(num_rx_ant, num_tx_ant)``.
        """
        phase = self.rng.uniform(0.0, 2.0 * np.pi, size=4)
        cross = np.sqrt(1.0 / max(xpr_linear, 1e-12))
        polarization = np.array(
            [
                [np.exp(1j * phase[0]), cross * np.exp(1j * phase[1])],
                [cross * np.exp(1j * phase[2]), np.exp(1j * phase[3])],
            ],
            dtype=np.complex128,
        )
        return self._field_spatial_matrix(
            rx_array=rx_array,
            tx_array=tx_array,
            arrival_vec=arrival_vec,
            departure_vec=departure_vec,
            polarization=polarization,
        )

    def _los_spatial_matrix(
        self,
        rx_array,
        tx_array,
        arrival_vec: np.ndarray,
        departure_vec: np.ndarray,
    ) -> np.ndarray:
        """Build deterministic LOS spatial matrix for a CDL LOS cluster.

        Args:
            rx_array: Resolved RX antenna array.
            tx_array: Resolved TX antenna array.
            arrival_vec: Arrival unit vector with shape ``(3,)``.
            departure_vec: Departure unit vector with shape ``(3,)``.

        Returns:
            Complex MIMO matrix with shape ``(num_rx_ant, num_tx_ant)``.
        """
        return self._field_spatial_matrix(
            rx_array=rx_array,
            tx_array=tx_array,
            arrival_vec=arrival_vec,
            departure_vec=departure_vec,
            polarization=np.eye(2, dtype=np.complex128),
        )

    def _field_spatial_matrix(
        self,
        rx_array,
        tx_array,
        arrival_vec: np.ndarray,
        departure_vec: np.ndarray,
        polarization: np.ndarray,
    ) -> np.ndarray:
        """Combine element fields, polarization matrix, and array phases.

        Args:
            rx_array: Resolved RX antenna array.
            tx_array: Resolved TX antenna array.
            arrival_vec: Arrival unit vector with shape ``(3,)``.
            departure_vec: Departure unit vector with shape ``(3,)``.
            polarization: Polarization coupling matrix with shape ``(2, 2)``.

        Returns:
            Complex MIMO spatial matrix with shape ``(num_rx_ant, num_tx_ant)``.
        """
        rx_field = self._field_pattern(rx_array, arrival_vec)
        tx_field = self._field_pattern(tx_array, departure_vec)
        polarization_gain = np.einsum("ri,ij,tj->rt", rx_field, polarization, tx_field)
        rx_phase = self._array_phase(rx_array, arrival_vec)
        tx_phase = self._array_phase(tx_array, departure_vec)
        return polarization_gain * rx_phase[:, np.newaxis] * tx_phase[np.newaxis, :]
