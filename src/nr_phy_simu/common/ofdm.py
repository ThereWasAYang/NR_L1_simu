from __future__ import annotations

import numpy as np

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.interfaces import TimeDomainProcessor


class OfdmProcessor(TimeDomainProcessor):
    """CP-OFDM processor shared by CP-OFDM and DFT-s-OFDM chains."""

    def modulate(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply OFDM modulation and cyclic-prefix insertion.

        Args:
            grid: Frequency-domain slot grid with shape
                ``(num_subcarriers, num_symbols)``; axis 0 is cell subcarrier index,
                axis 1 is OFDM symbol index.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Serialized time-domain waveform with shape ``(slot_samples,)``; axis 0
            is time-sample index after CP insertion.
        """
        fft_size = config.carrier.fft_size_effective
        cp_lengths = config.carrier.cyclic_prefix_lengths
        n_sc = config.carrier.n_subcarriers

        waveform_symbols = []
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        for symbol_idx in range(grid.shape[1]):
            cp_length = cp_lengths[symbol_idx % len(cp_lengths)]
            fft_bins = np.zeros(fft_size, dtype=np.complex128)
            fft_bins[start:stop] = grid[:, symbol_idx]
            time_domain = np.fft.ifft(np.fft.ifftshift(fft_bins))
            cp = time_domain[-cp_length:]
            waveform_symbols.append(np.concatenate([cp, time_domain]))

        return np.concatenate(waveform_symbols)

    def demodulate(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply cyclic-prefix removal and FFT demodulation.

        Args:
            waveform: Time-domain waveform with shape ``(slot_samples,)`` for SISO
                or ``(num_rx_ant, slot_samples)`` for multiple RX branches; axis 0
                is RX antenna when present, last axis is time-sample index.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Frequency-domain grid with shape
            ``(num_rx_ant, num_subcarriers, num_symbols)``; axes are RX antenna,
            cell subcarrier index, and OFDM symbol index.
        """
        if waveform.ndim == 2:
            return np.stack([self._demodulate_single(antenna_waveform, config) for antenna_waveform in waveform], axis=0)
        return self._demodulate_single(waveform, config)[np.newaxis, ...]

    def _demodulate_single(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Demodulate a single-antenna waveform into one slot grid.

        Args:
            waveform: One-dimensional time-domain waveform with shape
                ``(slot_samples,)``; axis 0 is time-sample index for one RX antenna.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Frequency-domain resource grid with shape ``(num_subcarriers, num_symbols)``;
            axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
        """
        fft_size = config.carrier.fft_size_effective
        cp_lengths = config.carrier.cyclic_prefix_lengths
        n_sc = config.carrier.n_subcarriers
        symbols_per_slot = config.carrier.symbols_per_slot

        grid = np.zeros((n_sc, symbols_per_slot), dtype=np.complex128)
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        offset = 0
        for symbol_idx in range(symbols_per_slot):
            cp_length = cp_lengths[symbol_idx % len(cp_lengths)]
            symbol_length = fft_size + cp_length
            symbol = waveform[offset + cp_length : offset + symbol_length]
            fft_bins = np.fft.fftshift(np.fft.fft(symbol, n=fft_size))
            grid[:, symbol_idx] = fft_bins[start:stop]
            offset += symbol_length

        return grid
