from __future__ import annotations

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.scenarios.base import SharedChannelSimulation
from nr_phy_simu.scenarios.component_factory import SimulationComponentFactory


class PdschSimulation(SharedChannelSimulation):
    def __init__(
        self,
        config: SimulationConfig | None = None,
        component_factory: SimulationComponentFactory | None = None,
    ) -> None:
        cfg = config or SimulationConfig()
        cfg.link.channel_type = "PDSCH"
        cfg.link.waveform = "CP-OFDM"
        super().__init__(cfg, component_factory=component_factory)
