from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import TimeDomainProcessor
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor
from nr_phy_simu.config import SimulationConfig


class OfdmProcessor(TimeDomainProcessor):
    """CP-OFDM processor shared by CP-OFDM and DFT-s-OFDM chains."""

    def modulate(self, grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Apply OFDM modulation and cyclic-prefix insertion."""
        grid = as_complex_tensor(grid)
        if grid.ndim == 3:
            return torch.stack([self._modulate_single(antenna_grid, config) for antenna_grid in grid], dim=0)
        if grid.ndim != 2:
            raise ValueError(
                "OFDM modulation expects grid shape (num_subcarriers, num_symbols) "
                "or (num_tx_ant, num_subcarriers, num_symbols)."
            )

        return self._modulate_single(grid, config).unsqueeze(0)

    def _modulate_single(self, grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Modulate one transmit branch."""
        fft_size = config.carrier.fft_size_effective
        cp_lengths = config.carrier.cyclic_prefix_lengths
        n_sc = config.carrier.n_subcarriers

        waveform_symbols = []
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        for symbol_idx in range(grid.shape[1]):
            cp_length = cp_lengths[symbol_idx % len(cp_lengths)]
            fft_bins = torch.zeros(fft_size, dtype=COMPLEX_DTYPE, device=grid.device)
            fft_bins[start:stop] = grid[:, symbol_idx]
            time_domain = torch.fft.ifft(torch.fft.ifftshift(fft_bins))
            cp = time_domain[-cp_length:]
            waveform_symbols.append(torch.cat([cp, time_domain]))

        return torch.cat(waveform_symbols)

    def demodulate(self, waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Apply cyclic-prefix removal and FFT demodulation."""
        waveform = as_complex_tensor(waveform)
        if waveform.ndim == 2:
            return torch.stack([self._demodulate_single(antenna_waveform, config) for antenna_waveform in waveform], dim=0)
        return self._demodulate_single(waveform, config).unsqueeze(0)

    def _demodulate_single(self, waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Demodulate a single-antenna waveform into one slot grid."""
        fft_size = config.carrier.fft_size_effective
        cp_lengths = config.carrier.cyclic_prefix_lengths
        n_sc = config.carrier.n_subcarriers
        symbols_per_slot = config.carrier.symbols_per_slot

        grid = torch.zeros((n_sc, symbols_per_slot), dtype=COMPLEX_DTYPE, device=waveform.device)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        offset = 0
        for symbol_idx in range(symbols_per_slot):
            cp_length = cp_lengths[symbol_idx % len(cp_lengths)]
            symbol_length = fft_size + cp_length
            symbol = waveform[offset + cp_length : offset + symbol_length]
            fft_bins = torch.fft.fftshift(torch.fft.fft(symbol, n=fft_size))
            grid[:, symbol_idx] = fft_bins[start:stop]
            offset += symbol_length

        return grid
