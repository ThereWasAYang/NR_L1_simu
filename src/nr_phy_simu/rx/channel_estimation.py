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
        del config
        channel = np.ones_like(rx_grid, dtype=np.complex128)
        if dmrs_symbols.size == 0:
            return channel

        pilot_est = rx_grid[dmrs_mask] / dmrs_symbols[: np.count_nonzero(dmrs_mask)]
        channel[dmrs_mask] = pilot_est

        dmrs_symbol_indices = np.where(np.any(dmrs_mask, axis=0))[0]
        for symbol_idx in range(rx_grid.shape[1]):
            symbol_mask = dmrs_mask[:, symbol_idx]
            if np.any(symbol_mask):
                pilot_sc = np.flatnonzero(symbol_mask)
                pilot_values = channel[pilot_sc, symbol_idx]
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
        if dmrs_symbols.size == 0:
            return np.array([], dtype=np.complex128)
        return rx_grid[dmrs_mask] / dmrs_symbols[: np.count_nonzero(dmrs_mask)]
