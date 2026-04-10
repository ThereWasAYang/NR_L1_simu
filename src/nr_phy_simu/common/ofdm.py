from __future__ import annotations

import numpy as np

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.interfaces import TimeDomainProcessor


class OfdmProcessor(TimeDomainProcessor):
    """CP-OFDM processor shared by CP-OFDM and DFT-s-OFDM chains."""

    def modulate(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        fft_size = config.carrier.fft_size_effective
        cp_length = config.carrier.cp_length
        n_sc = config.carrier.n_subcarriers

        waveform_symbols = []
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        for symbol_idx in range(grid.shape[1]):
            fft_bins = np.zeros(fft_size, dtype=np.complex128)
            fft_bins[start:stop] = grid[:, symbol_idx]
            time_domain = np.fft.ifft(np.fft.ifftshift(fft_bins))
            cp = time_domain[-cp_length:]
            waveform_symbols.append(np.concatenate([cp, time_domain]))

        return np.concatenate(waveform_symbols)

    def demodulate(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        fft_size = config.carrier.fft_size_effective
        cp_length = config.carrier.cp_length
        n_sc = config.carrier.n_subcarriers
        symbols_per_slot = config.carrier.symbols_per_slot
        symbol_length = fft_size + cp_length

        grid = np.zeros((n_sc, symbols_per_slot), dtype=np.complex128)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        for symbol_idx in range(symbols_per_slot):
            offset = symbol_idx * symbol_length
            symbol = waveform[offset + cp_length : offset + symbol_length]
            fft_bins = np.fft.fftshift(np.fft.fft(symbol, n=fft_size))
            grid[:, symbol_idx] = fft_bins[start:stop]

        return grid
