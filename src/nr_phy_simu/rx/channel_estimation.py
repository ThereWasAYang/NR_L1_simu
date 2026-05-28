from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nr_phy_simu.common.interfaces import ChannelEstimator
from nr_phy_simu.common.types import ChannelEstimateResult
from nr_phy_simu.config import SimulationConfig


@dataclass(frozen=True)
class PilotLeastSquaresResult:
    """Pilot-RE LS estimates for all receive antennas and DMRS symbols."""

    dmrs_symbol_indices: np.ndarray
    pilot_subcarriers_by_symbol: tuple[np.ndarray, ...]
    pilot_estimates_by_symbol: tuple[np.ndarray, ...]
    num_rx_ant: int


@dataclass(frozen=True)
class FrequencyInterpolatedChannelResult:
    """Frequency-interpolated channel estimates on DMRS symbols."""

    dmrs_symbol_indices: np.ndarray
    channel_estimates: np.ndarray


class LeastSquaresEstimator(ChannelEstimator):
    def estimate(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        """Estimate the user-allocation channel response from DMRS observations.

        Args:
            rx_grid: Received user-allocation grid with shape
                ``(num_rx_ant, num_user_subcarriers, num_symbols)``; axes are RX
                antenna, user-allocation subcarrier index, and OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape
                ``(num_user_subcarriers, num_symbols)``; axis 0 is user-allocation
                subcarrier index and axis 1 is OFDM symbol index.
            config: Full simulation configuration for estimation context.

        Returns:
            Structured channel-estimation result with user-grid and pilot-only views.
        """
        if rx_grid.ndim != 3:
            raise ValueError(
                "LeastSquaresEstimator.estimate expects rx_grid shape "
                "(num_rx_ant, num_user_subcarriers, num_symbols)."
            )
        if dmrs_mask.shape != rx_grid.shape[1:]:
            raise ValueError(
                "dmrs_mask shape must match rx_grid user-grid axes "
                "(num_user_subcarriers, num_symbols)."
            )
        del config

        if dmrs_symbols.size == 0:
            if np.any(dmrs_mask):
                raise ValueError("dmrs_symbols is empty but dmrs_mask contains DMRS REs.")
            channel_estimate = np.ones_like(rx_grid, dtype=np.complex128)
        else:
            pilot_ls_result = self.estimate_pilot_re_ls(rx_grid, dmrs_symbols, dmrs_mask)
            frequency_interpolated_result = self.interpolate_frequency(
                pilot_ls_result,
                rx_grid.shape[1],
            )
            channel_estimate = self.interpolate_time(
                frequency_interpolated_result,
                rx_grid.shape[2],
            )

        pilot_estimates, pilot_symbol_indices = self._extract_pilot_estimates(
            channel_estimate,
            dmrs_mask,
        )
        return ChannelEstimateResult(
            channel_estimate=channel_estimate,
            pilot_estimates=pilot_estimates,
            pilot_symbol_indices=pilot_symbol_indices,
        )

    def estimate_pilot_re_ls(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
    ) -> PilotLeastSquaresResult:
        """Compute LS channel estimates on DMRS REs for all receive antennas.

        Args:
            rx_grid: Received user-allocation grid with shape
                ``(num_rx_ant, num_user_subcarriers, num_symbols)``; axes are RX
                antenna, user-allocation subcarrier index and OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape
                ``(num_user_subcarriers, num_symbols)``; ``True`` marks a DMRS RE.

        Returns:
            ``PilotLeastSquaresResult`` containing complete LS results for all
            receive antennas and all DMRS symbols. Each entry in
            ``pilot_estimates_by_symbol`` has shape
            ``(num_rx_ant, num_pilot_re_in_symbol)``.
        """
        if rx_grid.ndim != 3:
            raise ValueError(
                "LeastSquaresEstimator.estimate_pilot_re_ls expects rx_grid shape "
                "(num_rx_ant, num_user_subcarriers, num_symbols)."
            )
        if dmrs_mask.shape != rx_grid.shape[1:]:
            raise ValueError(
                "dmrs_mask shape must match rx_grid user-grid axes "
                "(num_user_subcarriers, num_symbols)."
            )

        expected_dmrs_re = int(np.count_nonzero(dmrs_mask))
        if dmrs_symbols.size != expected_dmrs_re:
            raise ValueError(
                "dmrs_symbols length must match the number of True entries in dmrs_mask "
                f"({dmrs_symbols.size} != {expected_dmrs_re})."
            )

        dmrs_symbol_indices = np.where(np.any(dmrs_mask, axis=0))[0]
        pilot_subcarriers_by_symbol: list[np.ndarray] = []
        pilot_estimates_by_symbol: list[np.ndarray] = []
        dmrs_cursor = 0
        for symbol_idx in dmrs_symbol_indices:
            pilot_subcarriers = np.flatnonzero(dmrs_mask[:, symbol_idx])
            symbol_dmrs = dmrs_symbols[dmrs_cursor : dmrs_cursor + pilot_subcarriers.size]
            dmrs_cursor += pilot_subcarriers.size
            pilot_subcarriers_by_symbol.append(pilot_subcarriers)
            pilot_estimates_by_symbol.append(
                self._ls_estimate(
                    rx_grid[:, pilot_subcarriers, symbol_idx],
                    symbol_dmrs,
                )
            )
        return PilotLeastSquaresResult(
            dmrs_symbol_indices=dmrs_symbol_indices,
            pilot_subcarriers_by_symbol=tuple(pilot_subcarriers_by_symbol),
            pilot_estimates_by_symbol=tuple(pilot_estimates_by_symbol),
            num_rx_ant=rx_grid.shape[0],
        )

    def interpolate_frequency(
        self,
        pilot_ls_result: PilotLeastSquaresResult,
        num_subcarriers: int,
    ) -> FrequencyInterpolatedChannelResult:
        """Interpolate pilot-RE LS estimates across frequency for all antennas.

        Args:
            pilot_ls_result: Complete LS result returned by ``estimate_pilot_re_ls``.
            num_subcarriers: User-allocation subcarrier count.

        Returns:
            ``FrequencyInterpolatedChannelResult`` whose ``channel_estimates`` has
            shape ``(num_rx_ant, num_dmrs_symbols, num_user_subcarriers)``.
        """
        dmrs_symbol_indices = pilot_ls_result.dmrs_symbol_indices
        pilot_subcarriers_by_symbol = pilot_ls_result.pilot_subcarriers_by_symbol
        pilot_estimates_by_symbol = pilot_ls_result.pilot_estimates_by_symbol
        if len(pilot_subcarriers_by_symbol) != dmrs_symbol_indices.size:
            raise ValueError("pilot_subcarriers_by_symbol length must match dmrs_symbol_indices.")
        if len(pilot_estimates_by_symbol) != dmrs_symbol_indices.size:
            raise ValueError("pilot_estimates_by_symbol length must match dmrs_symbol_indices.")

        channel_estimates = np.zeros(
            (pilot_ls_result.num_rx_ant, dmrs_symbol_indices.size, num_subcarriers),
            dtype=np.complex128,
        )
        for dmrs_idx, (pilot_subcarriers, pilot_values_by_ant) in enumerate(
            zip(pilot_subcarriers_by_symbol, pilot_estimates_by_symbol, strict=True)
        ):
            if pilot_values_by_ant.shape[0] != pilot_ls_result.num_rx_ant:
                raise ValueError("pilot estimate antenna axis must match num_rx_ant.")
            for antenna_idx in range(pilot_ls_result.num_rx_ant):
                channel_estimates[antenna_idx, dmrs_idx] = self._interpolate_frequency(
                    pilot_subcarriers,
                    pilot_values_by_ant[antenna_idx],
                    num_subcarriers,
                )
        return FrequencyInterpolatedChannelResult(
            dmrs_symbol_indices=dmrs_symbol_indices,
            channel_estimates=channel_estimates,
        )

    def interpolate_time(
        self,
        frequency_result: FrequencyInterpolatedChannelResult,
        num_symbols: int,
    ) -> np.ndarray:
        """Interpolate frequency-complete DMRS estimates across time for all antennas.

        Args:
            frequency_result: Complete frequency-interpolated result returned by
                ``interpolate_frequency``. ``channel_estimates`` has shape
                ``(num_rx_ant, num_dmrs_symbols, num_user_subcarriers)``.
            num_symbols: Total OFDM symbol count in the slot.

        Returns:
            Full user-grid channel estimate with shape
            ``(num_rx_ant, num_user_subcarriers, num_symbols)``.
        """
        channel_estimates = frequency_result.channel_estimates
        if channel_estimates.ndim != 3:
            raise ValueError(
                "frequency_result.channel_estimates must have shape "
                "(num_rx_ant, num_dmrs_symbols, num_user_subcarriers)."
            )
        if channel_estimates.shape[1] != frequency_result.dmrs_symbol_indices.size:
            raise ValueError("DMRS symbol axis must match dmrs_symbol_indices.")

        channel = np.zeros(
            (channel_estimates.shape[0], channel_estimates.shape[2], num_symbols),
            dtype=np.complex128,
        )
        for antenna_idx in range(channel_estimates.shape[0]):
            channel[antenna_idx] = self._interpolate_time(
                frequency_result.dmrs_symbol_indices,
                channel_estimates[antenna_idx],
                num_symbols,
            )
        return channel

    def _ls_estimate(self, rx_pilots: np.ndarray, reference_pilots: np.ndarray) -> np.ndarray:
        """Compute least-squares channel estimates on pilot RE locations.

        Args:
            rx_pilots: Received pilot values with shape
                ``(..., num_pilot_re_in_symbol)``.
            reference_pilots: One-dimensional reference DMRS values with shape
                ``(num_pilot_re_in_symbol,)`` and the same RE ordering.

        Returns:
            LS channel estimates on the pilot RE locations with the same shape as
            ``rx_pilots``.
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
                ``(num_pilot_re_in_symbol,)``; values are user-allocation
                subcarrier indices.
            pilot_values: One-dimensional complex LS estimates with the same shape
                and ordering as ``pilot_subcarriers``.
            num_subcarriers: User-allocation subcarrier count of the slot grid.

        Returns:
            User-band estimate for one OFDM symbol after frequency interpolation.
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
        """Interpolate user-band DMRS estimates across OFDM symbols.

        Args:
            dmrs_symbol_indices: One-dimensional integer array with shape
                ``(num_dmrs_symbols,)``; values are OFDM symbol indices.
            dmrs_estimates: Complex array with shape
                ``(num_dmrs_symbols, num_subcarriers)``; axis 0 is DMRS symbol,
                axis 1 is user-allocation subcarrier index.
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
                ``(num_rx_ant, num_user_subcarriers, num_symbols)``.
            dmrs_mask: Boolean mask with shape
                ``(num_user_subcarriers, num_symbols)``.

        Returns:
            Tuple of ``(pilot_estimates, pilot_symbol_indices)`` where:
            - ``pilot_estimates`` is grouped by receive antenna.
            - ``pilot_symbol_indices`` identifies which OFDM symbol each pilot came from.
        """
        if channel_estimate.ndim != 3:
            raise ValueError(
                "LeastSquaresEstimator._extract_pilot_estimates expects channel_estimate "
                "shape (num_rx_ant, num_user_subcarriers, num_symbols)."
            )

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
