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
        """Estimate the full slot channel response from DMRS observations.

        Args:
            rx_grid: Received slot grid with shape ``(num_subcarriers, num_symbols)``
                or ``(num_rx_ant, num_subcarriers, num_symbols)``; axes are RX
                antenna when present, cell subcarrier index, and OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            config: Full simulation configuration for estimation context.

        Returns:
            Structured channel-estimation result with full-grid and pilot-only views.
        """
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
        """Estimate the channel for one receive antenna across the full slot.

        Args:
            rx_grid: Single-antenna received slot grid with shape
                ``(num_subcarriers, num_symbols)``; axis 0 is cell subcarrier index,
                axis 1 is OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.
            config: Full simulation configuration, unused by this LS implementation.

        Returns:
            Full-grid complex channel estimate for one receive antenna.
        """
        del config
        if dmrs_symbols.size == 0:
            return np.ones_like(rx_grid, dtype=np.complex128)

        dmrs_symbol_indices, dmrs_estimates = self._estimate_dmrs_symbols(rx_grid, dmrs_symbols, dmrs_mask)
        return self._interpolate_time(dmrs_symbol_indices, dmrs_estimates, rx_grid.shape[1])

    def _estimate_dmrs_symbols(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate channel values on all DMRS symbols after frequency interpolation.

        Args:
            rx_grid: Single-antenna received slot grid with shape
                ``(num_subcarriers, num_symbols)``.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.

        Returns:
            Tuple of ``(dmrs_symbol_indices, dmrs_estimates)`` where:
            - ``dmrs_symbol_indices`` lists OFDM symbols carrying DMRS.
            - ``dmrs_estimates`` stores one full-band estimate per DMRS symbol.
        """
        dmrs_symbol_indices = np.where(np.any(dmrs_mask, axis=0))[0]
        dmrs_estimates = np.zeros((dmrs_symbol_indices.size, rx_grid.shape[0]), dtype=np.complex128)
        dmrs_cursor = 0
        for dmrs_idx, symbol_idx in enumerate(dmrs_symbol_indices):
            pilot_subcarriers = np.flatnonzero(dmrs_mask[:, symbol_idx])
            symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_subcarriers.size]
            dmrs_cursor += pilot_subcarriers.size
            pilot_values = self._ls_estimate(
                rx_grid[pilot_subcarriers, symbol_idx],
                symbol_dmrs,
            )
            dmrs_estimates[dmrs_idx] = self._interpolate_frequency(
                pilot_subcarriers,
                pilot_values,
                rx_grid.shape[0],
            )
        return dmrs_symbol_indices, dmrs_estimates

    def _ls_estimate(self, rx_pilots: np.ndarray, reference_pilots: np.ndarray) -> np.ndarray:
        """Compute least-squares channel estimates on pilot RE locations.

        Args:
            rx_pilots: One-dimensional received pilot values with shape
                ``(num_pilot_re_in_symbol,)``.
            reference_pilots: One-dimensional reference DMRS values with the same
                shape as ``rx_pilots`` and the same RE ordering.

        Returns:
            LS channel estimates on the pilot RE locations.
        """
        return rx_pilots / reference_pilots

    def _interpolate_frequency(
        self,
        pilot_subcarriers: np.ndarray,
        pilot_values: np.ndarray,
        num_subcarriers: int,
    ) -> np.ndarray:
        """Interpolate pilot-only estimates across the frequency axis.

        Args:
            pilot_subcarriers: One-dimensional integer array with shape
                ``(num_pilot_re_in_symbol,)``; values are cell subcarrier indices.
            pilot_values: One-dimensional complex LS estimates with the same shape
                and ordering as ``pilot_subcarriers``.
            num_subcarriers: Full subcarrier count of the slot grid.

        Returns:
            Full-band estimate for one OFDM symbol after frequency interpolation.
        """
        full_subcarriers = np.arange(num_subcarriers, dtype=np.float64)
        real = np.interp(full_subcarriers, pilot_subcarriers, pilot_values.real)
        imag = np.interp(full_subcarriers, pilot_subcarriers, pilot_values.imag)
        return real + 1j * imag

    def _interpolate_time(
        self,
        dmrs_symbol_indices: np.ndarray,
        dmrs_estimates: np.ndarray,
        num_symbols: int,
    ) -> np.ndarray:
        """Interpolate full-band DMRS estimates across OFDM symbols.

        Args:
            dmrs_symbol_indices: One-dimensional integer array with shape
                ``(num_dmrs_symbols,)``; values are OFDM symbol indices.
            dmrs_estimates: Complex array with shape
                ``(num_dmrs_symbols, num_subcarriers)``; axis 0 is DMRS symbol,
                axis 1 is cell subcarrier index.
            num_symbols: Total OFDM symbol count in the slot.

        Returns:
            Full-grid channel estimate after time interpolation.
        """
        if dmrs_symbol_indices.size == 1:
            return np.repeat(dmrs_estimates[:1].T, num_symbols, axis=1)

        channel = np.zeros((dmrs_estimates.shape[1], num_symbols), dtype=np.complex128)
        known_symbols = dmrs_symbol_indices.astype(np.float64)
        target_symbols = np.arange(num_symbols, dtype=np.float64)
        for sc_idx in range(dmrs_estimates.shape[1]):
            known_values = dmrs_estimates[:, sc_idx]
            real_interp = np.interp(target_symbols, known_symbols, known_values.real)
            imag_interp = np.interp(target_symbols, known_symbols, known_values.imag)
            channel[sc_idx, :] = real_interp + 1j * imag_interp
        return channel

    def _extract_pilot_estimates(
        self,
        channel_estimate: np.ndarray,
        dmrs_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract pilot-only channel estimates for plotting and debug views.

        Args:
            channel_estimate: Complex estimate with shape
                ``(num_subcarriers, num_symbols)`` or
                ``(num_rx_ant, num_subcarriers, num_symbols)``.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.

        Returns:
            Tuple of ``(pilot_estimates, pilot_symbol_indices)`` where:
            - ``pilot_estimates`` is grouped by receive antenna.
            - ``pilot_symbol_indices`` identifies which OFDM symbol each pilot came from.
        """
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
