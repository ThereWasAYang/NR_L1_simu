from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.types import ChannelEstimateResult


class ChannelCoder(ABC):
    @abstractmethod
    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Encode a transport block into channel-coded bits.

        Args:
            bits: Input transport-block bit sequence before CRC/coding.
            config: Full simulation configuration that defines coding parameters.

        Returns:
            Channel-coded bit sequence ready for scrambling and modulation.
        """
        raise NotImplementedError


class ChannelDecoder(ABC):
    @abstractmethod
    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Decode descrambled LLRs back to transport-block bits.

        Args:
            llrs: Soft-bit log-likelihood ratios after equalization and descrambling.
            config: Full simulation configuration that defines decoding parameters.

        Returns:
            Decoded bit sequence after rate recovery and channel decoding.
        """
        raise NotImplementedError


class BitScrambler(ABC):
    @abstractmethod
    def scramble(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply data scrambling to encoded bits.

        Args:
            bits: Encoded bit sequence before modulation.
            config: Full simulation configuration that provides scrambling seeds.

        Returns:
            Scrambled bit sequence mapped to the modulation stage.
        """
        raise NotImplementedError

    @abstractmethod
    def descramble_llrs(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply inverse scrambling in the LLR domain.

        Args:
            llrs: Demodulated LLR sequence before descrambling.
            config: Full simulation configuration that provides scrambling seeds.

        Returns:
            Descrambled LLR sequence ready for channel decoding.
        """
        raise NotImplementedError


class Modulator(ABC):
    @abstractmethod
    def map_bits(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Map bits to complex-valued modulation symbols.

        Args:
            bits: Scrambled bit sequence to modulate.
            config: Full simulation configuration that defines modulation order.

        Returns:
            Complex modulation symbols in serial order.
        """
        raise NotImplementedError


class Demodulator(ABC):
    @abstractmethod
    def demap_symbols(
        self,
        symbols: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Demap equalized symbols into soft-bit LLRs.

        Args:
            symbols: Equalized complex symbols to demodulate.
            noise_variance: Receiver noise variance used for LLR scaling.
            config: Full simulation configuration that defines modulation order.

        Returns:
            Flat LLR sequence for all coded bits carried by ``symbols``.
        """
        raise NotImplementedError


class ResourceMapper(ABC):
    @abstractmethod
    def map_to_grid(
        self,
        data_symbols: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map data and DMRS symbols into a slot resource grid.

        Args:
            data_symbols: Serial data symbols produced by the modulator.
            config: Full simulation configuration that defines grid allocation.

        Returns:
            Tuple of ``(grid, dmrs_mask, data_mask, dmrs_symbols)`` where:
            - ``grid`` is the frequency-domain resource grid.
            - ``dmrs_mask`` marks DMRS RE locations in the grid.
            - ``data_mask`` marks data RE locations in the grid.
            - ``dmrs_symbols`` is the serialized DMRS sequence mapped to the grid.
        """
        raise NotImplementedError

    @abstractmethod
    def count_data_re(self, config: SimulationConfig) -> int:
        """Count the number of REs available for data mapping.

        Args:
            config: Full simulation configuration that defines allocation rules.

        Returns:
            Total number of data-carrying REs inside the scheduled slot region.
        """
        raise NotImplementedError


class TimeDomainProcessor(ABC):
    @abstractmethod
    def modulate(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Convert a frequency-domain grid into time-domain waveform samples.

        Args:
            grid: Frequency-domain resource grid for one slot.
            config: Full simulation configuration that defines OFDM numerology.

        Returns:
            Time-domain waveform for transmission.
        """
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Convert time-domain waveform samples back into a frequency-domain grid.

        Args:
            waveform: Received time-domain waveform, optionally stacked by antenna.
            config: Full simulation configuration that defines OFDM numerology.

        Returns:
            Frequency-domain grid, keeping the receive-antenna dimension when present.
        """
        raise NotImplementedError


class ChannelModel(ABC):
    @abstractmethod
    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Propagate a transmit waveform through the configured channel model.

        Args:
            waveform: Time-domain waveform generated by the transmitter.
            config: Full simulation configuration that defines channel settings.

        Returns:
            Tuple of ``(rx_waveform, channel_state)`` where ``channel_state`` stores
            implementation-specific metadata such as tap coefficients or noise power.
        """
        raise NotImplementedError


class FrequencyExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        grid: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
        despread: bool = True,
    ) -> np.ndarray:
        """Extract scheduled data REs from a frequency-domain grid.

        Args:
            grid: Frequency-domain grid or channel estimate to sample from.
            data_mask: Boolean mask that marks data RE locations.
            config: Full simulation configuration that defines waveform behavior.
            despread: Whether DFT-s-OFDM data should be de-spread during extraction.

        Returns:
            Serialized extracted symbols, optionally stacked by receive antenna.
        """
        raise NotImplementedError


class ChannelEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        """Estimate the channel response on the full slot grid.

        Args:
            rx_grid: Received frequency-domain grid, with optional antenna dimension.
            dmrs_symbols: Serialized transmitted DMRS sequence used as reference.
            dmrs_mask: Boolean mask that marks DMRS RE locations.
            config: Full simulation configuration that defines estimation context.

        Returns:
            Structured channel-estimation result containing the full estimate and
            pilot-only views used for plotting/debug.
        """
        raise NotImplementedError


class MimoEqualizer(ABC):
    @abstractmethod
    def equalize(
        self,
        rx_symbols: np.ndarray,
        channel_estimate: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Equalize extracted data symbols with the estimated channel.

        Args:
            rx_symbols: Extracted received data symbols, optionally stacked by antenna.
            channel_estimate: Channel estimate sampled at the same REs as ``rx_symbols``.
            noise_variance: Receiver noise variance used by the equalizer.
            config: Full simulation configuration for equalizer-specific options.

        Returns:
            Equalized symbol stream ready for demodulation.
        """
        raise NotImplementedError


class DmrsSequenceGenerator(ABC):
    @abstractmethod
    def get_dmrs_info(self, config: SimulationConfig):
        """Build static DMRS placement metadata for the configured slot.

        Args:
            config: Full simulation configuration that defines DMRS layout.

        Returns:
            DMRS metadata object describing symbol locations and per-PRB offsets.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_for_symbol(self, symbol: int, config: SimulationConfig) -> np.ndarray:
        """Generate the DMRS sequence for one OFDM symbol.

        Args:
            symbol: OFDM symbol index inside the slot.
            config: Full simulation configuration that defines DMRS generation.

        Returns:
            Serialized complex DMRS sequence mapped within the scheduled bandwidth.
        """
        raise NotImplementedError
