from __future__ import annotations

import numpy as np

from nr_phy_simu.common.bwp import ofdm_phase_compensation_vector
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.interfaces import TimeDomainProcessor


def time_to_frequency_noise_variance(noise_variance: float, config: SimulationConfig) -> float:
    """Convert time-sample AWGN variance to FFT-bin noise variance.

    Args:
        noise_variance: Complex time-domain sample noise variance.
        config: Full simulation configuration defining the FFT size.

    Returns:
        Frequency-domain RE noise variance after the unnormalized FFT used by
        :class:`OfdmProcessor`.
    """
    if not np.isfinite(noise_variance):
        return float(noise_variance)
    return float(noise_variance) * float(config.carrier.fft_size_effective)


class OfdmProcessor(TimeDomainProcessor):
    """CP-OFDM processor shared by CP-OFDM and DFT-s-OFDM chains."""

    def modulate(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply OFDM modulation and cyclic-prefix insertion.

        Args:
            grid: Frequency-domain slot grid with shape
                ``(num_tx_ant, num_subcarriers, num_symbols)``; axes are TX antenna,
                cell subcarrier index, and OFDM symbol index.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Serialized time-domain waveform with shape ``(num_tx_ant, slot_samples)``;
            axis 0 is TX antenna and axis 1 is time-sample index after CP insertion.
        """
        grid = np.asarray(grid, dtype=np.complex128)
        if grid.ndim != 3:
            raise ValueError(
                "OFDM modulation expects grid shape "
                "(num_tx_ant, num_subcarriers, num_symbols)."
            )
        return np.stack([self._modulate_single(antenna_grid, config) for antenna_grid in grid], axis=0)

    def _modulate_single(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Modulate one transmit branch.

        Args:
            grid: Single-antenna frequency-domain grid with shape
                ``(num_subcarriers, num_symbols)``.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Serialized time-domain waveform with shape ``(slot_samples,)``.
        """
        fft_size = config.carrier.fft_size_effective
        cp_lengths = config.carrier.cyclic_prefix_lengths
        n_sc = config.carrier.n_subcarriers

        waveform_symbols = []
        start = (fft_size - n_sc) // 2
        stop = start + n_sc

        offset = 0
        for symbol_idx in range(grid.shape[1]):
            cp_length = cp_lengths[symbol_idx % len(cp_lengths)]
            fft_bins = np.zeros(fft_size, dtype=np.complex128)
            fft_bins[start:stop] = grid[:, symbol_idx]
            time_domain = np.fft.ifft(np.fft.ifftshift(fft_bins))
            cp = time_domain[-cp_length:]
            symbol_with_cp = np.concatenate([cp, time_domain])
            symbol_with_cp = symbol_with_cp * ofdm_phase_compensation_vector(
                config,
                symbol_start_sample=offset,
                cp_length=cp_length,
                symbol_length=symbol_with_cp.size,
                inverse=False,
            )
            waveform_symbols.append(symbol_with_cp)
            offset += symbol_with_cp.size

        return np.concatenate(waveform_symbols)

    def demodulate(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply cyclic-prefix removal and FFT demodulation.

        Args:
            waveform: Time-domain waveform with shape ``(num_rx_ant, slot_samples)``;
                axis 0 is RX antenna and axis 1 is time-sample index.
            config: Full simulation configuration that defines FFT size and CP lengths.

        Returns:
            Frequency-domain grid with shape
            ``(num_rx_ant, num_subcarriers, num_symbols)``; axes are RX antenna,
            cell subcarrier index, and OFDM symbol index.
        """
        if waveform.ndim != 2:
            raise ValueError("OFDM demodulation expects waveform shape (num_rx_ant, slot_samples).")
        return np.stack([self._demodulate_single(antenna_waveform, config) for antenna_waveform in waveform], axis=0)

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
            symbol_with_cp = waveform[offset : offset + symbol_length]
            symbol_with_cp = symbol_with_cp * ofdm_phase_compensation_vector(
                config,
                symbol_start_sample=offset,
                cp_length=cp_length,
                symbol_length=symbol_length,
                inverse=True,
            )
            symbol = symbol_with_cp[cp_length:]
            fft_bins = np.fft.fftshift(np.fft.fft(symbol, n=fft_size))
            grid[:, symbol_idx] = fft_bins[start:stop]
            offset += symbol_length

        return grid
