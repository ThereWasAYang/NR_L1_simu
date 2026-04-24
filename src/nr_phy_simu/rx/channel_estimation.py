from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import ChannelEstimator
from nr_phy_simu.common.torch_utils import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    as_complex_tensor,
    ensure_antenna_axis,
    interp1d_complex,
)
from nr_phy_simu.common.types import ChannelEstimateResult
from nr_phy_simu.config import SimulationConfig


class LeastSquaresEstimator(ChannelEstimator):
    def estimate(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        """Estimate the full slot channel response from DMRS observations."""
        rx_grid = ensure_antenna_axis(as_complex_tensor(rx_grid))
        dmrs_symbols = as_complex_tensor(dmrs_symbols, device=rx_grid.device)
        dmrs_mask = torch.as_tensor(dmrs_mask, dtype=torch.bool, device=rx_grid.device)
        channel_estimate = torch.stack(
            [
                self._estimate_single(rx_grid[antenna_idx], dmrs_symbols, dmrs_mask, config)
                for antenna_idx in range(rx_grid.shape[0])
            ],
            dim=0,
        )
        pilot_estimates, pilot_symbol_indices = self._extract_pilot_estimates(channel_estimate, dmrs_mask)
        return ChannelEstimateResult(
            channel_estimate=channel_estimate,
            pilot_estimates=pilot_estimates,
            pilot_symbol_indices=pilot_symbol_indices,
        )

    def _estimate_single(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> torch.Tensor:
        """Estimate the channel for one receive antenna across the full slot."""
        del config
        if dmrs_symbols.numel() == 0:
            return torch.ones_like(rx_grid, dtype=COMPLEX_DTYPE)

        dmrs_symbol_indices, dmrs_estimates = self._estimate_dmrs_symbols(rx_grid, dmrs_symbols, dmrs_mask)
        return self._interpolate_time(dmrs_symbol_indices, dmrs_estimates, rx_grid.shape[1])

    def _estimate_dmrs_symbols(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate channel values on all DMRS symbols after frequency interpolation."""
        dmrs_symbol_indices = torch.where(torch.any(dmrs_mask, dim=0))[0]
        dmrs_estimates = torch.zeros(
            (dmrs_symbol_indices.numel(), rx_grid.shape[0]),
            dtype=COMPLEX_DTYPE,
            device=rx_grid.device,
        )
        dmrs_cursor = 0
        for dmrs_idx, symbol_idx_tensor in enumerate(dmrs_symbol_indices):
            symbol_idx = int(symbol_idx_tensor.item())
            pilot_subcarriers = torch.where(dmrs_mask[:, symbol_idx])[0]
            symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_subcarriers.numel()]
            dmrs_cursor += pilot_subcarriers.numel()
            pilot_values = self._ls_estimate(rx_grid[pilot_subcarriers, symbol_idx], symbol_dmrs)
            dmrs_estimates[dmrs_idx] = self._interpolate_frequency(
                pilot_subcarriers,
                pilot_values,
                rx_grid.shape[0],
                device=rx_grid.device,
            )
        return dmrs_symbol_indices, dmrs_estimates

    def _ls_estimate(self, rx_pilots: torch.Tensor, reference_pilots: torch.Tensor) -> torch.Tensor:
        """Compute least-squares channel estimates on pilot RE locations."""
        return rx_pilots / reference_pilots

    def _interpolate_frequency(
        self,
        pilot_subcarriers: torch.Tensor,
        pilot_values: torch.Tensor,
        num_subcarriers: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Interpolate pilot-only estimates across the frequency axis."""
        full_subcarriers = torch.arange(num_subcarriers, dtype=REAL_DTYPE, device=device)
        return interp1d_complex(
            full_subcarriers,
            pilot_subcarriers.to(dtype=REAL_DTYPE, device=device),
            pilot_values.to(device=device, dtype=COMPLEX_DTYPE),
        )

    def _interpolate_time(
        self,
        dmrs_symbol_indices: torch.Tensor,
        dmrs_estimates: torch.Tensor,
        num_symbols: int,
    ) -> torch.Tensor:
        """Interpolate full-band DMRS estimates across OFDM symbols."""
        if dmrs_symbol_indices.numel() == 1:
            return dmrs_estimates[:1].T.repeat(1, num_symbols)

        channel = torch.zeros((dmrs_estimates.shape[1], num_symbols), dtype=COMPLEX_DTYPE, device=dmrs_estimates.device)
        known_symbols = dmrs_symbol_indices.to(dtype=REAL_DTYPE, device=dmrs_estimates.device)
        target_symbols = torch.arange(num_symbols, dtype=REAL_DTYPE, device=dmrs_estimates.device)
        for sc_idx in range(dmrs_estimates.shape[1]):
            known_values = dmrs_estimates[:, sc_idx]
            channel[sc_idx, :] = interp1d_complex(target_symbols, known_symbols, known_values)
        return channel

    def _extract_pilot_estimates(
        self,
        channel_estimate: torch.Tensor,
        dmrs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract pilot-only channel estimates for plotting and debug views."""
        channel_estimate = ensure_antenna_axis(as_complex_tensor(channel_estimate))
        pilot_symbol_indices: list[torch.Tensor] = []
        estimates_per_ant = []
        for antenna_idx in range(channel_estimate.shape[0]):
            estimates = []
            symbol_indices = []
            for symbol_idx in range(channel_estimate.shape[2]):
                symbol_mask = dmrs_mask[:, symbol_idx]
                if not torch.any(symbol_mask):
                    continue
                pilot_sc = torch.where(symbol_mask)[0]
                estimates.append(channel_estimate[antenna_idx, pilot_sc, symbol_idx])
                symbol_indices.append(
                    torch.full((pilot_sc.numel(),), symbol_idx, dtype=torch.int64, device=channel_estimate.device)
                )
            estimates_per_ant.append(
                torch.cat(estimates) if estimates else torch.zeros(0, dtype=COMPLEX_DTYPE, device=channel_estimate.device)
            )
            pilot_symbol_indices.append(
                torch.cat(symbol_indices) if symbol_indices else torch.zeros(0, dtype=torch.int64, device=channel_estimate.device)
            )

        pilot_estimates = torch.stack(estimates_per_ant, dim=0)
        symbol_index_ref = (
            pilot_symbol_indices[0]
            if pilot_symbol_indices
            else torch.zeros(0, dtype=torch.int64, device=channel_estimate.device)
        )
        return pilot_estimates, symbol_index_ref
