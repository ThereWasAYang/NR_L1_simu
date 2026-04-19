from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from nr_phy_simu.common.types import ChannelEstimateResult
from nr_phy_simu.config import SimulationConfig


class ChannelCoder(ABC):
    @abstractmethod
    def encode(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Encode a transport block into channel-coded bits."""
        raise NotImplementedError


class ChannelDecoder(ABC):
    @abstractmethod
    def decode(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Decode descrambled LLRs back to transport-block bits."""
        raise NotImplementedError


class BitScrambler(ABC):
    @abstractmethod
    def scramble(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Apply data scrambling to encoded bits."""
        raise NotImplementedError

    @abstractmethod
    def descramble_llrs(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Apply inverse scrambling in the LLR domain."""
        raise NotImplementedError


class Modulator(ABC):
    @abstractmethod
    def map_bits(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Map bits to complex-valued modulation symbols."""
        raise NotImplementedError


class Demodulator(ABC):
    @abstractmethod
    def demap_symbols(
        self,
        symbols: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> torch.Tensor:
        """Demap equalized symbols into soft-bit LLRs."""
        raise NotImplementedError


class ResourceMapper(ABC):
    @abstractmethod
    def map_to_grid(
        self,
        data_symbols: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map data and DMRS symbols into a slot resource grid."""
        raise NotImplementedError

    @abstractmethod
    def count_data_re(self, config: SimulationConfig) -> int:
        """Count the number of REs available for data mapping."""
        raise NotImplementedError


class TimeDomainProcessor(ABC):
    @abstractmethod
    def modulate(self, grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Convert a frequency-domain grid into time-domain waveform samples."""
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Convert time-domain waveform samples back into a frequency-domain grid."""
        raise NotImplementedError


class ChannelModel(ABC):
    @abstractmethod
    def propagate(
        self,
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        """Propagate a transmit waveform through the configured channel model."""
        raise NotImplementedError


class FrequencyExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        grid: torch.Tensor,
        data_mask: torch.Tensor,
        config: SimulationConfig,
        despread: bool = True,
    ) -> torch.Tensor:
        """Extract scheduled data REs from a frequency-domain grid."""
        raise NotImplementedError


class ChannelEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> ChannelEstimateResult:
        """Estimate the channel response on the full slot grid."""
        raise NotImplementedError


class MimoEqualizer(ABC):
    @abstractmethod
    def equalize(
        self,
        rx_symbols: torch.Tensor,
        channel_estimate: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> torch.Tensor:
        """Equalize extracted data symbols with the estimated channel."""
        raise NotImplementedError


class DmrsSequenceGenerator(ABC):
    @abstractmethod
    def get_dmrs_info(self, config: SimulationConfig):
        """Build static DMRS placement metadata for the configured slot."""
        raise NotImplementedError

    @abstractmethod
    def generate_for_symbol(self, symbol: int, config: SimulationConfig) -> torch.Tensor:
        """Generate the DMRS sequence for one OFDM symbol."""
        raise NotImplementedError
