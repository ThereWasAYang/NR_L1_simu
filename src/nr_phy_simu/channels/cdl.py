from __future__ import annotations

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig


class CdlChannel(ChannelModel):
    def propagate(self, waveform, config: SimulationConfig):
        raise NotImplementedError(
            "CDL channel model is reserved for a 3GPP 38.901-compliant implementation."
        )

