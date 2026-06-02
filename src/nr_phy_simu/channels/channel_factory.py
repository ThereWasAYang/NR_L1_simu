from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nr_phy_simu.channels.awgn import AwgnChannel
from nr_phy_simu.channels.cdl import CdlChannel
from nr_phy_simu.channels.external_frequency_response import (
    ExternalFrequencyResponseFrequencyDomainChannel,
    ExternalFrequencyResponseTimeDomainChannel,
)
from nr_phy_simu.channels.tdl import TdlChannel
from nr_phy_simu.config import SimulationConfig


class ChannelFactory(ABC):
    @abstractmethod
    def create(self, config: SimulationConfig):
        raise NotImplementedError


class DefaultChannelFactory(ChannelFactory):
    """Creates channel-model instances from simulation config."""

    def create(self, config: SimulationConfig):
        model = config.channel.model.upper()
        if model == "AWGN":
            return AwgnChannel(rng=_channel_rng(config))
        if model == "TDL":
            return TdlChannel(rng=_channel_rng(config))
        if model == "CDL":
            return CdlChannel(rng=_channel_rng(config))
        if model == "EXTERNAL_FREQRESP_TD":
            return ExternalFrequencyResponseTimeDomainChannel(rng=_channel_rng(config))
        if model == "EXTERNAL_FREQRESP_FD":
            return ExternalFrequencyResponseFrequencyDomainChannel(rng=_channel_rng(config))
        raise NotImplementedError(f"Channel model '{config.channel.model}' is not implemented yet.")


def _channel_rng(config: SimulationConfig) -> np.random.Generator:
    seed = config.channel.seed
    if seed is None:
        seed = config.random_seed
    if isinstance(seed, str):
        if seed.lower() == "auto":
            return np.random.default_rng()
        seed = int(seed)
    return np.random.default_rng(int(seed))
