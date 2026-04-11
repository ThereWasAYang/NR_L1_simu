from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nr_phy_simu.channels.awgn import AwgnChannel
from nr_phy_simu.channels.cdl import CdlChannel
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
            return AwgnChannel(rng=np.random.default_rng(config.random_seed))
        if model == "TDL":
            return TdlChannel(rng=np.random.default_rng(config.random_seed))
        if model == "CDL":
            return CdlChannel(rng=np.random.default_rng(config.random_seed))
        raise NotImplementedError(f"Channel model '{config.channel.model}' is not implemented yet.")
