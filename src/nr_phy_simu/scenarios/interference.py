from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.config import InterferenceSourceConfig, SimulationConfig
from nr_phy_simu.scenarios.component_factory import SimulationComponentFactory, build_transmitter
from nr_phy_simu.common.torch_utils import BIT_DTYPE, COMPLEX_DTYPE, as_complex_tensor, to_numpy


@dataclass(frozen=True)
class InterferenceReport:
    label: str
    inr_db: float
    channel_model: str
    prb_start: int
    num_prbs: int
    rx_power: float
    scale: float


class InterferenceMixer:
    def __init__(self, component_factory: SimulationComponentFactory) -> None:
        self.component_factory = component_factory

    def apply(
        self,
        waveform: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, tuple[InterferenceReport, ...]]:
        if not config.interference.enabled:
            return waveform, ()

        composite = as_complex_tensor(waveform).clone()
        reports: list[InterferenceReport] = []
        for index, source in enumerate(config.interference.sources):
            if not source.enabled:
                continue
            interferer_cfg = self._build_interferer_config(config, source, index)
            interferer_waveform = self._generate_interferer_waveform(interferer_cfg)
            scaled_waveform, report = self._scale_to_inr(
                interferer_waveform,
                noise_variance=noise_variance,
                source=source,
                index=index,
            )
            composite = composite + scaled_waveform
            reports.append(report)
        return composite, tuple(reports)

    def _generate_interferer_waveform(self, config: SimulationConfig) -> torch.Tensor:
        components = self.component_factory.create_components(config)
        transmitter = build_transmitter(components)
        channel = self.component_factory.create_channel_factory().create(config)
        apply_mcs_to_link(config)
        data_re = components.transmitter.mapper.count_data_re(config)
        bits_per_symbol = _bits_per_symbol(config)
        config.link.coded_bit_capacity = data_re * bits_per_symbol
        if not config.link.transport_block_size:
            config.link.transport_block_size = resolve_transport_block_size(config, data_re)
        generator = torch.Generator().manual_seed(config.random_seed)
        transport_block = torch.randint(0, 2, (int(config.link.transport_block_size),), dtype=BIT_DTYPE, generator=generator)
        tx_payload = transmitter.transmit(transport_block, config)
        rx_waveform, _ = channel.propagate(to_numpy(tx_payload.waveform), config)
        return as_complex_tensor(rx_waveform)

    def _build_interferer_config(
        self,
        base_config: SimulationConfig,
        source: InterferenceSourceConfig,
        index: int,
    ) -> SimulationConfig:
        config = deepcopy(base_config)
        config.waveform_input.waveform_path = None
        config.channel.model = source.channel_model
        config.channel.params = dict(source.channel_params)
        config.channel.params["add_noise"] = False
        config.link.channel_type = source.channel_type or base_config.link.channel_type
        config.link.waveform = source.waveform or base_config.link.waveform
        config.link.num_tx_ant = int(source.num_tx_ant or base_config.link.num_tx_ant)
        config.link.num_rx_ant = base_config.link.num_rx_ant
        config.link.prb_start = int(source.prb_start if source.prb_start is not None else base_config.link.prb_start)
        config.link.num_prbs = int(source.num_prbs if source.num_prbs is not None else base_config.link.num_prbs)
        config.link.start_symbol = int(
            source.start_symbol if source.start_symbol is not None else base_config.link.start_symbol
        )
        config.link.num_symbols = int(
            source.num_symbols if source.num_symbols is not None else base_config.link.num_symbols
        )
        config.link.transport_block_size = None
        config.link.coded_bit_capacity = None
        if source.mcs.table is not None:
            config.link.mcs.table = source.mcs.table
        if source.mcs.index is not None:
            config.link.mcs.index = source.mcs.index
        if source.mcs.modulation is not None:
            config.link.mcs.modulation = source.mcs.modulation
        if source.mcs.target_code_rate is not None:
            config.link.mcs.target_code_rate = source.mcs.target_code_rate
        config.random_seed = base_config.random_seed + 1000 * (index + 1)
        config.scrambling.rnti = int((base_config.scrambling.rnti + index + 1) % 65536)
        config.scrambling.n_id = int((base_config.scrambling.n_id + index + 1) % 1024)
        if config.dmrs.scrambling_id0 is not None:
            config.dmrs.scrambling_id0 = int((config.dmrs.scrambling_id0 + index + 1) % (1 << 16))
        if config.dmrs.scrambling_id1 is not None:
            config.dmrs.scrambling_id1 = int((config.dmrs.scrambling_id1 + index + 1) % (1 << 16))
        if config.dmrs.n_pusch_identity is not None:
            config.dmrs.n_pusch_identity = int((config.dmrs.n_pusch_identity + index + 1) % 1008)
        config.interference.sources = ()
        return config

    @staticmethod
    def _scale_to_inr(
        waveform: torch.Tensor,
        noise_variance: float,
        source: InterferenceSourceConfig,
        index: int,
    ) -> tuple[torch.Tensor, InterferenceReport]:
        waveform = as_complex_tensor(waveform)
        rx_power = float(torch.mean(torch.abs(waveform) ** 2).item())
        target_power = float(noise_variance * (10 ** (source.inr_db / 10.0)))
        scale = 0.0 if rx_power <= 0.0 else float(target_power / rx_power) ** 0.5
        label = source.label or f"interferer_{index}"
        return waveform * scale, InterferenceReport(
            label=label,
            inr_db=float(source.inr_db),
            channel_model=source.channel_model,
            prb_start=int(source.prb_start) if source.prb_start is not None else -1,
            num_prbs=int(source.num_prbs) if source.num_prbs is not None else -1,
            rx_power=rx_power,
            scale=float(scale),
        )


def _bits_per_symbol(config: SimulationConfig) -> int:
    modulation = config.link.modulation.upper()
    mapping = {
        "PI/2-BPSK": 1,
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
        "64QAM": 6,
        "256QAM": 8,
        "1024QAM": 10,
    }
    return mapping[modulation]
