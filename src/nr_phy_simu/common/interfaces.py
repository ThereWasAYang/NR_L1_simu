from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nr_phy_simu.config import SimulationConfig


class ChannelCoder(ABC):
    @abstractmethod
    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError


class ChannelDecoder(ABC):
    @abstractmethod
    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError


class Modulator(ABC):
    @abstractmethod
    def map_bits(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError


class Demodulator(ABC):
    @abstractmethod
    def demap_symbols(
        self,
        symbols: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> np.ndarray:
        raise NotImplementedError


class ResourceMapper(ABC):
    @abstractmethod
    def map_to_grid(
        self,
        data_symbols: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class TimeDomainProcessor(ABC):
    @abstractmethod
    def modulate(self, grid: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, waveform: np.ndarray, config: SimulationConfig) -> np.ndarray:
        raise NotImplementedError


class ChannelModel(ABC):
    @abstractmethod
    def propagate(
        self,
        waveform: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, dict]:
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
        raise NotImplementedError


class ChannelEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:
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
        raise NotImplementedError
