from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import ChannelEstimator
from nr_phy_simu.common.types import ChannelEstimateResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    as_complex_tensor,
    ensure_antenna_axis,
    interp1d_complex,
)


class LeastSquaresEstimator(ChannelEstimator):
    def estimate(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        rx_grid = ensure_antenna_axis(as_complex_tensor(rx_grid))
        dmrs_symbols = as_complex_tensor(dmrs_symbols, device=rx_grid.device)
        channel_estimate = torch.stack(
            [self._estimate_single(rx_grid[antenna_idx], dmrs_symbols, dmrs_mask, config) for antenna_idx in range(rx_grid.shape[0])],
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
        del config
        channel = torch.ones_like(rx_grid, dtype=COMPLEX_DTYPE)
        if dmrs_symbols.numel() == 0:
            return channel

        dmrs_symbol_indices = torch.where(torch.any(dmrs_mask, dim=0))[0]
        dmrs_estimates: dict[int, torch.Tensor] = {}
        dmrs_cursor = 0
        full_sc = torch.arange(rx_grid.shape[0], dtype=REAL_DTYPE, device=rx_grid.device)
        for symbol_idx_tensor in dmrs_symbol_indices:
            symbol_idx = int(symbol_idx_tensor.item())
            symbol_mask = dmrs_mask[:, symbol_idx]
            pilot_sc = torch.where(symbol_mask)[0]
            symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_sc.numel()]
            dmrs_cursor += pilot_sc.numel()
            pilot_values = rx_grid[pilot_sc, symbol_idx] / symbol_dmrs
            channel[pilot_sc, symbol_idx] = pilot_values
            dmrs_estimates[symbol_idx] = interp1d_complex(full_sc, pilot_sc.to(dtype=REAL_DTYPE), pilot_values)
            channel[:, symbol_idx] = dmrs_estimates[symbol_idx]

        if len(dmrs_symbol_indices) == 1:
            channel[:, :] = channel[:, int(dmrs_symbol_indices[0].item())].unsqueeze(1)
            return channel

        for sc_idx in range(rx_grid.shape[0]):
            known_symbols = dmrs_symbol_indices.to(dtype=REAL_DTYPE)
            known_values = torch.stack([dmrs_estimates[int(symbol_idx.item())][sc_idx] for symbol_idx in dmrs_symbol_indices], dim=0)
            target_symbols = torch.arange(rx_grid.shape[1], dtype=REAL_DTYPE, device=rx_grid.device)
            channel[sc_idx, :] = interp1d_complex(target_symbols, known_symbols, known_values)

        return channel

    @staticmethod
    def _extract_pilot_estimates(
        channel_estimate: torch.Tensor,
        dmrs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                symbol_indices.append(torch.full((pilot_sc.numel(),), symbol_idx, dtype=torch.int64, device=channel_estimate.device))
            estimates_per_ant.append(
                torch.cat(estimates) if estimates else torch.zeros(0, dtype=COMPLEX_DTYPE, device=channel_estimate.device)
            )
            pilot_symbol_indices.append(
                torch.cat(symbol_indices) if symbol_indices else torch.zeros(0, dtype=torch.int64, device=channel_estimate.device)
            )

        pilot_estimates = torch.stack(estimates_per_ant, dim=0)
        symbol_index_ref = pilot_symbol_indices[0] if pilot_symbol_indices else torch.zeros(0, dtype=torch.int64, device=channel_estimate.device)
        return pilot_estimates, symbol_index_ref
