"""NR PHY simulation package."""

from importlib.metadata import PackageNotFoundError, version
import logging

from .config import (
    CarrierConfig,
    ChannelConfig,
    ConfigNode,
    DmrsConfig,
    LinkConfig,
    McsConfig,
    SimulationConfig,
    WaveformInputConfig,
)
from .scenarios.component_factory import DefaultSimulationComponentFactory, SimulationComponentFactory
from .scenarios.multi_tti import MultiTtiSimulationRunner
from .scenarios.pdsch import PdschSimulation
from .scenarios.pusch import PuschSimulation
from .io.config_loader import load_simulation_config
from .visualization import save_simulation_plots

try:
    __version__ = version("nr-phy-simu")
except PackageNotFoundError:
    __version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CarrierConfig",
    "ChannelConfig",
    "ConfigNode",
    "DefaultSimulationComponentFactory",
    "DmrsConfig",
    "LinkConfig",
    "McsConfig",
    "MultiTtiSimulationRunner",
    "PdschSimulation",
    "PuschSimulation",
    "SimulationComponentFactory",
    "SimulationConfig",
    "WaveformInputConfig",
    "__version__",
    "load_simulation_config",
    "save_simulation_plots",
]
