from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import FrequencyExtractor
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor


class FrequencyDomainExtractor(FrequencyExtractor):
    """RX-side frequency-domain data extraction and optional de-spreading."""

    def extract(
        self,
        grid: torch.Tensor,
        data_mask: torch.Tensor,
        config: SimulationConfig,
        despread: bool = True,
    ) -> torch.Tensor:
        grid = as_complex_tensor(grid)
        if grid.ndim == 3:
            per_antenna = [
                self._extract_single(grid[antenna_idx], data_mask, config, despread)
                for antenna_idx in range(grid.shape[0])
            ]
            return torch.stack(per_antenna, dim=0)
        return self._extract_single(grid, data_mask, config, despread)

    @staticmethod
    def _extract_single(
        grid: torch.Tensor,
        data_mask: torch.Tensor,
        config: SimulationConfig,
        despread: bool,
    ) -> torch.Tensor:
        symbols = []
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            symbol_values = grid[:, symbol_idx][data_mask[:, symbol_idx]]
            if symbol_values.numel() == 0:
                continue
            if (
                despread
                and config.link.channel_type.upper() == "PUSCH"
                and config.link.waveform.upper() == "DFT-S-OFDM"
            ):
                symbol_values = torch.fft.ifft(symbol_values, n=symbol_values.numel()) * torch.sqrt(
                    torch.tensor(float(symbol_values.numel()), dtype=torch.float64, device=symbol_values.device)
                )
            symbols.append(symbol_values)

        if not symbols:
            return torch.zeros(0, dtype=COMPLEX_DTYPE, device=grid.device)
        return torch.cat(symbols)
