from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import ChannelEstimator
from nr_phy_simu.common.types import ChannelEstimateResult
from nr_phy_simu.config import SimulationConfig


class LeastSquaresEstimator(ChannelEstimator):
    def estimate(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        if rx_grid.ndim == 2:
            rx_grid = rx_grid[np.newaxis, ...]
        channel_estimate = np.stack(
            [self._estimate_single(rx_grid[antenna_idx], dmrs_symbols, dmrs_mask, config) for antenna_idx in range(rx_grid.shape[0])],
            axis=0,
        )
        pilot_estimates, pilot_symbol_indices = self._extract_pilot_estimates(channel_estimate, dmrs_mask)
        return ChannelEstimateResult(
            channel_estimate=channel_estimate,
            pilot_estimates=pilot_estimates,
            pilot_symbol_indices=pilot_symbol_indices,
        )

    def _estimate_single(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:
        del config
        channel = np.ones_like(rx_grid, dtype=np.complex128)
        if dmrs_symbols.size == 0:
            return channel

        dmrs_symbol_indices = np.where(np.any(dmrs_mask, axis=0))[0]
        dmrs_estimates: dict[int, np.ndarray] = {}
        dmrs_cursor = 0
        full_sc = np.arange(rx_grid.shape[0])
        for symbol_idx in dmrs_symbol_indices:
            symbol_mask = dmrs_mask[:, symbol_idx]
            pilot_sc = np.flatnonzero(symbol_mask)
            symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_sc.size]
            dmrs_cursor += pilot_sc.size
            pilot_values = rx_grid[pilot_sc, symbol_idx] / symbol_dmrs
            channel[pilot_sc, symbol_idx] = pilot_values
            real = np.interp(full_sc, pilot_sc, pilot_values.real)
            imag = np.interp(full_sc, pilot_sc, pilot_values.imag)
            dmrs_estimates[symbol_idx] = real + 1j * imag
            channel[:, symbol_idx] = dmrs_estimates[symbol_idx]

        if len(dmrs_symbol_indices) == 1:
            channel[:, :] = channel[:, dmrs_symbol_indices[0]][:, np.newaxis]
            return channel

        for sc_idx in range(rx_grid.shape[0]):
            known_symbols = dmrs_symbol_indices.astype(np.float64)
            known_values = np.array([dmrs_estimates[int(symbol_idx)][sc_idx] for symbol_idx in dmrs_symbol_indices])
            real_interp = np.interp(np.arange(rx_grid.shape[1], dtype=np.float64), known_symbols, known_values.real)
            imag_interp = np.interp(np.arange(rx_grid.shape[1], dtype=np.float64), known_symbols, known_values.imag)
            channel[sc_idx, :] = real_interp + 1j * imag_interp

        return channel

    @staticmethod
    def _extract_pilot_estimates(
        channel_estimate: np.ndarray,
        dmrs_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if channel_estimate.ndim == 2:
            channel_estimate = channel_estimate[np.newaxis, ...]

        pilot_symbol_indices: list[np.ndarray] = []
        estimates_per_ant = []
        for antenna_idx in range(channel_estimate.shape[0]):
            estimates = []
            symbol_indices = []
            for symbol_idx in range(channel_estimate.shape[2]):
                symbol_mask = dmrs_mask[:, symbol_idx]
                if not np.any(symbol_mask):
                    continue
                pilot_sc = np.flatnonzero(symbol_mask)
                estimates.append(channel_estimate[antenna_idx, pilot_sc, symbol_idx])
                symbol_indices.append(np.full(pilot_sc.size, symbol_idx, dtype=int))
            estimates_per_ant.append(
                np.concatenate(estimates) if estimates else np.array([], dtype=np.complex128)
            )
            pilot_symbol_indices.append(
                np.concatenate(symbol_indices) if symbol_indices else np.array([], dtype=int)
            )

        pilot_estimates = np.stack(estimates_per_ant, axis=0)
        symbol_index_ref = pilot_symbol_indices[0] if pilot_symbol_indices else np.array([], dtype=int)
        return pilot_estimates, symbol_index_ref
