from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import FrequencyExtractor
from nr_phy_simu.config import SimulationConfig


class FrequencyDomainExtractor(FrequencyExtractor):
    """RX-side frequency-domain data extraction and optional de-spreading."""

    def extract(
        self,
        grid: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
        despread: bool = True,
    ) -> np.ndarray:
        if grid.ndim == 3:
            per_antenna = [
                self._extract_single(grid[antenna_idx], data_mask, config, despread)
                for antenna_idx in range(grid.shape[0])
            ]
            return np.stack(per_antenna, axis=0)
        return self._extract_single(grid, data_mask, config, despread)

    @staticmethod
    def _extract_single(
        grid: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
        despread: bool,
    ) -> np.ndarray:
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
