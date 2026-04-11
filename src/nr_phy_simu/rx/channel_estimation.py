from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import ChannelEstimator
from nr_phy_simu.config import SimulationConfig


class LeastSquaresEstimator(ChannelEstimator):
    def estimate(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:
        if rx_grid.ndim == 2:
            rx_grid = rx_grid[np.newaxis, ...]
        return np.stack(
            [self._estimate_single(rx_grid[antenna_idx], dmrs_symbols, dmrs_mask, config) for antenna_idx in range(rx_grid.shape[0])],
            axis=0,
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
        dmrs_cursor = 0
        for symbol_idx in range(rx_grid.shape[1]):
            symbol_mask = dmrs_mask[:, symbol_idx]
            if np.any(symbol_mask):
                pilot_sc = np.flatnonzero(symbol_mask)
                symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_sc.size]
                dmrs_cursor += pilot_sc.size
                pilot_values = rx_grid[pilot_sc, symbol_idx] / symbol_dmrs
                channel[pilot_sc, symbol_idx] = pilot_values
                full_sc = np.arange(rx_grid.shape[0])
                real = np.interp(full_sc, pilot_sc, pilot_values.real)
                imag = np.interp(full_sc, pilot_sc, pilot_values.imag)
                channel[:, symbol_idx] = real + 1j * imag
                continue

            nearest = dmrs_symbol_indices[np.argmin(np.abs(dmrs_symbol_indices - symbol_idx))]
            channel[:, symbol_idx] = channel[:, nearest]

        return channel

    @staticmethod
    def pilot_estimates(
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
    ) -> np.ndarray:
        if rx_grid.ndim == 2:
            rx_grid = rx_grid[np.newaxis, ...]
        if dmrs_symbols.size == 0:
            return np.zeros((rx_grid.shape[0], 0), dtype=np.complex128)

        antenna_estimates = []
        for antenna_idx in range(rx_grid.shape[0]):
            estimates = []
            dmrs_cursor = 0
            for symbol_idx in range(rx_grid.shape[2]):
                symbol_mask = dmrs_mask[:, symbol_idx]
                if not np.any(symbol_mask):
                    continue
                pilot_sc = np.flatnonzero(symbol_mask)
                symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_sc.size]
                dmrs_cursor += pilot_sc.size
                estimates.append(rx_grid[antenna_idx, pilot_sc, symbol_idx] / symbol_dmrs)
            antenna_estimates.append(
                np.concatenate(estimates) if estimates else np.array([], dtype=np.complex128)
            )
        return np.stack(antenna_estimates, axis=0)
