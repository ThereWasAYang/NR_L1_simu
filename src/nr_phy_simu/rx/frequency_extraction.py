from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import FrequencyExtractor
from nr_phy_simu.config import SimulationConfig


class FrequencyDomainExtractor(FrequencyExtractor):
    """RX-side frequency-domain data extraction and optional de-spreading."""

    def extract_user_grid(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Crop the received full-cell grid to the scheduled user allocation.

        Args:
            grid: Full-cell frequency-domain grid with shape
                ``(num_rx_ant, num_subcarriers, num_symbols)``; axis 0 is RX antenna,
                axis 1 is cell subcarrier index, and axis 2 is OFDM symbol index.
            config: Full simulation configuration that defines PRB start and width.

        Returns:
            User-allocation grid with shape
            ``(num_rx_ant, num_user_subcarriers, num_symbols)``.
        """
        if grid.ndim != 3:
            raise ValueError(
                "FrequencyDomainExtractor.extract_user_grid expects grid shape "
                "(num_rx_ant, num_subcarriers, num_symbols)."
            )
        return grid[:, self.allocated_subcarriers(config), :]

    def extract_user_mask(self, mask: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Crop a full-cell RE mask to the scheduled user allocation.

        Args:
            mask: Full-cell boolean RE mask with shape
                ``(num_subcarriers, num_symbols)``.
            config: Full simulation configuration that defines PRB start and width.

        Returns:
            User-allocation mask with shape ``(num_user_subcarriers, num_symbols)``.
        """
        if mask.ndim != 2:
            raise ValueError(
                "FrequencyDomainExtractor.extract_user_mask expects mask shape "
                "(num_subcarriers, num_symbols)."
            )
        return mask[self.allocated_subcarriers(config), :]

    def extract(
        self,
        grid: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
        despread: bool = True,
    ) -> np.ndarray:
        """Extract scheduled data REs from a slot grid or estimate grid.

        Args:
            grid: Input grid with shape
                ``(num_rx_ant, num_user_subcarriers, num_symbols)``; axes are RX
                antenna, user-allocation subcarrier index, and OFDM symbol index.
            data_mask: Boolean mask with shape
                ``(num_user_subcarriers, num_symbols)``; axis 0 is user-allocation
                subcarrier index and axis 1 is OFDM symbol index.
            config: Full simulation configuration that defines waveform behavior.
            despread: Whether to undo DFT spreading for DFT-s-OFDM PUSCH.

        Returns:
            Serialized extracted values with shape ``(num_rx_ant, num_data_re)``.
        """
        if grid.ndim != 3:
            raise ValueError(
                "FrequencyDomainExtractor.extract expects grid shape "
                "(num_rx_ant, num_user_subcarriers, num_symbols)."
            )
        per_antenna = [
            self._extract_single(grid[antenna_idx], data_mask, config, despread)
            for antenna_idx in range(grid.shape[0])
        ]
        return np.stack(per_antenna, axis=0)

    @staticmethod
    def allocated_subcarriers(config: SimulationConfig) -> np.ndarray:
        """Build absolute subcarrier indices for the scheduled PRB allocation.

        Args:
            config: Full simulation configuration that provides PRB start and width.

        Returns:
            One-dimensional integer array with shape ``(num_prbs * 12,)``; axis 0
            is the user-allocation subcarrier index and values are absolute cell
            subcarrier indices.
        """
        start = int(config.link.prb_start) * 12
        stop = start + int(config.link.num_prbs) * 12
        return np.arange(start, stop, dtype=int)

    @staticmethod
    def _extract_single(
        grid: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
        despread: bool,
    ) -> np.ndarray:
        """Extract scheduled data REs from a single-antenna grid.

        Args:
            grid: Single-antenna frequency-domain grid with shape
                ``(num_subcarriers, num_symbols)``.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.
            config: Full simulation configuration that defines waveform behavior.
            despread: Whether to undo DFT spreading for DFT-s-OFDM PUSCH.

        Returns:
            Serialized extracted values for the scheduled data REs.
        """
        symbols = []
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            symbol_values = grid[:, symbol_idx][data_mask[:, symbol_idx]]
            if symbol_values.size == 0:
                continue
            if (
                despread
                and config.link.channel_type.upper() == "PUSCH"
                and config.link.waveform.upper() == "DFT-S-OFDM"
            ):
                symbol_values = np.fft.ifft(symbol_values, n=symbol_values.size) * np.sqrt(symbol_values.size)
            symbols.append(symbol_values)

        if not symbols:
            return np.array([], dtype=np.complex128)
        return np.concatenate(symbols)
