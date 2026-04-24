from __future__ import annotations

import math

import torch

from nr_phy_simu.channels.fading_base import FadingChannelBase
from nr_phy_simu.channels.profile_tables import TDL_LOS_K_DB, TDL_PROFILES
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, REAL_DTYPE
from nr_phy_simu.config import SimulationConfig


class TdlChannel(FadingChannelBase):
    """3GPP TR 38.901 TDL channel model with multi-TX multi-RX support."""

    def _generate_path_coefficients(
        self,
        num_samples: int,
        sample_rate_hz: float,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile_name = str(config.channel.params.get("profile", "TDL-A")).upper()
        if profile_name not in TDL_PROFILES:
            raise ValueError(f"Unsupported TDL profile '{profile_name}'.")

        taps = TDL_PROFILES[profile_name]
        normalized_delays = torch.as_tensor([tap.normalized_delay for tap in taps], dtype=REAL_DTYPE)
        default_power_db = torch.as_tensor([tap.power_db for tap in taps], dtype=REAL_DTYPE)
        delays_s, power_db = self._resolve_path_parameters(config, normalized_delays, default_power_db)
        path_powers = self._normalize_powers_db(power_db)

        num_rx_ant = int(config.link.num_rx_ant)
        num_tx_ant = int(config.link.num_tx_ant)
        max_doppler_hz = self._max_doppler_hz(config)
        num_sinusoids = int(config.channel.params.get("num_sinusoids", 32))
        k_factor_db = float(config.channel.params.get("k_factor_db", TDL_LOS_K_DB.get(profile_name, 0.0)))
        tx_spacing = float(config.channel.params.get("tx_antenna_spacing_lambda", 0.5))
        rx_spacing = float(config.channel.params.get("rx_antenna_spacing_lambda", 0.5))

        coeff = torch.zeros((num_rx_ant, num_tx_ant, delays_s.numel(), num_samples), dtype=COMPLEX_DTYPE)
        tap_fading = list(config.channel.params.get("path_fading", [tap.fading for tap in taps]))
        if len(tap_fading) != delays_s.numel():
            if len(tap_fading) == len(taps):
                tap_fading = tap_fading[: delays_s.numel()]
            else:
                raise ValueError("TDL 'path_fading' length must match the number of channel paths.")

        for path_idx in range(delays_s.numel()):
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

            tx_spatial_freq = float((2 * torch.rand(1, generator=self.rng, dtype=REAL_DTYPE) - 1.0).item())
            rx_spatial_freq = float((2 * torch.rand(1, generator=self.rng, dtype=REAL_DTYPE) - 1.0).item())
            tx_response = self._array_response(num_tx_ant, tx_spatial_freq, tx_spacing)
            rx_response = self._array_response(num_rx_ant, rx_spatial_freq, rx_spacing)
            spatial = torch.outer(rx_response, torch.conj(tx_response))
            coeff[:, :, path_idx, :] = (
                math.sqrt(float(path_powers[path_idx].item()))
                * spatial[:, :, None]
                * process[None, None, :]
            )

        return delays_s, coeff
