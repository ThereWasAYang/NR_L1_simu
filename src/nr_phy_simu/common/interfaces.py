from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.types import ChannelEstimateResult


class ChannelCoder(ABC):
    @abstractmethod
    def encode(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError


class ChannelDecoder(ABC):
    @abstractmethod
    def decode(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError


class BitScrambler(ABC):
    @abstractmethod
    def scramble(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def descramble_llrs(self, llrs: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError


class Modulator(ABC):
    @abstractmethod
    def map_bits(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError


class Demodulator(ABC):
    @abstractmethod
    def demap_symbols(
        self,
        symbols: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> torch.Tensor:
        raise NotImplementedError


class ResourceMapper(ABC):
    @abstractmethod
    def map_to_grid(
        self,
        data_symbols: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def count_data_re(self, config: SimulationConfig) -> int:
        raise NotImplementedError


class TimeDomainProcessor(ABC):
    @abstractmethod
    def modulate(self, grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError


class ChannelModel(ABC):
    @abstractmethod
    def propagate(
        self,
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
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
        raise NotImplementedError


class DmrsSequenceGenerator(ABC):
    @abstractmethod
    def get_dmrs_info(self, config: SimulationConfig):
        raise NotImplementedError

    @abstractmethod
    def generate_for_symbol(self, symbol: int, config: SimulationConfig) -> torch.Tensor:
        raise NotImplementedError
