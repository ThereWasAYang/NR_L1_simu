from __future__ import annotations

from copy import deepcopy

from nr_phy_simu.common.types import MultiTtiSimulationResult, SimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.scenarios.component_factory import SimulationComponentFactory
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation


class MultiTtiSimulationRunner:
    def __init__(
        self,
        config: SimulationConfig,
        component_factory: SimulationComponentFactory | None = None,
    ) -> None:
        self.config = config
        self.component_factory = component_factory

    def run(self) -> MultiTtiSimulationResult:
        num_ttis = int(self.config.simulation.num_ttis)
        if num_ttis <= 0:
            raise ValueError("simulation.num_ttis must be a positive integer.")

        tti_results: list[SimulationResult] = []
        packet_errors = 0
        crc_checked_ttis = 0
        final_config = None
        evm_values: list[float] = []
        evm_snr_values: list[float] = []
        for tti_idx in range(num_ttis):
            tti_config = deepcopy(self.config)
            tti_config.random_seed = self.config.random_seed + tti_idx
            tti_config.slot_index = self.config.slot_index + tti_idx
            tti_result = self._build_simulation(tti_config).run()
            tti_results.append(tti_result)
            final_config = tti_config
            if tti_result.crc_ok is not None:
                crc_checked_ttis += 1
            if tti_result.crc_ok is False:
                packet_errors += 1
            if tti_result.evm_percent is not None:
                evm_values.append(tti_result.evm_percent)
            if tti_result.evm_snr_linear is not None:
                evm_snr_values.append(tti_result.evm_snr_linear)

        return MultiTtiSimulationResult(
            num_ttis=num_ttis,
            packet_errors=packet_errors,
            block_error_rate=(packet_errors / crc_checked_ttis) if crc_checked_ttis else float("nan"),
            average_evm_percent=(sum(evm_values) / len(evm_values)) if evm_values else None,
            average_evm_snr_linear=(sum(evm_snr_values) / len(evm_snr_values)) if evm_snr_values else None,
            tti_results=tuple(tti_results),
            final_config=final_config,
        )

    def _build_simulation(self, config: SimulationConfig):
        if config.waveform_input.enabled:
            return WaveformReplaySimulation(config, component_factory=self.component_factory)
        if config.link.channel_type.upper() == "PUSCH":
            return PuschSimulation(config, component_factory=self.component_factory)
        return PdschSimulation(config, component_factory=self.component_factory)
