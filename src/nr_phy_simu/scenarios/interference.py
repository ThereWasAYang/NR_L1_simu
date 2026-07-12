from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.config import InterferenceSourceConfig, SimulationConfig
from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.component_factory import SimulationComponentFactory, build_transmitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterferenceReport:
    label: str
    inr_db: float
    channel_model: str
    config_path: str | None
    prb_start: int
    num_prbs: int
    rx_power: float
    scale: float


class InterferenceMixer:
    def __init__(self, component_factory: SimulationComponentFactory) -> None:
        self.component_factory = component_factory
        self._pipelines: dict[int, tuple[Any, Any, Any]] = {}

    def apply(
        self,
        waveform: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, tuple[InterferenceReport, ...]]:
        """Add all configured interferer waveforms to the desired waveform.

        Args:
            waveform: Desired RX waveform with shape ``(slot_samples,)`` or
                ``(num_rx_ant, slot_samples)``; last axis is time-sample index.
            noise_variance: Scalar receiver noise variance used to define INR.
            config: Full simulation configuration containing interference sources.

        Returns:
            Tuple of composite waveform with the same shape as ``waveform`` and
            per-interferer scalar reports.
        """
        if not config.interference.enabled:
            return waveform, ()

        composite = np.array(waveform, copy=True)
        reports: list[InterferenceReport] = []
        for index, source in enumerate(config.interference.sources):
            if not source.enabled:
                continue
            interferer_cfg = self._build_interferer_config(config, source, index)
            interferer_waveform = self._generate_interferer_waveform(interferer_cfg, index)
            scaled_waveform, report = self._scale_to_inr(
                interferer_waveform,
                noise_variance=noise_variance,
                source=source,
                config=interferer_cfg,
                index=index,
            )
            composite = composite + scaled_waveform
            reports.append(report)
        return composite, tuple(reports)

    def _generate_interferer_waveform(self, config: SimulationConfig, index: int) -> np.ndarray:
        """Generate one interferer after its own TX and channel path.

        Args:
            config: Interferer-specific simulation configuration.

        Returns:
            Complex interferer waveform with shape ``(slot_samples,)`` or
            ``(num_rx_ant, slot_samples)``; last axis is time-sample index.
        """
        pipeline = self._pipelines.get(index)
        if pipeline is None:
            components = self.component_factory.create_components(config)
            transmitter = build_transmitter(components)
            channel = self.component_factory.create_channel_factory().create(config)
            self._pipelines[index] = (components, transmitter, channel)
        else:
            components, transmitter, channel = pipeline
        apply_mcs_to_link(config)
        data_re = components.transmitter.mapper.count_data_re(config)
        bits_per_symbol = _bits_per_symbol(config)
        config.link.coded_bit_capacity = data_re * bits_per_symbol
        if not config.link.transport_block_size:
            config.link.transport_block_size = resolve_transport_block_size(config, data_re)
        rng = np.random.default_rng(config.random_seed)
        transport_block = rng.integers(
            0,
            2,
            size=int(config.link.transport_block_size),
            dtype=np.int8,
        )
        tx_payload = transmitter.transmit(transport_block, config)
        rx_waveform, _ = channel.propagate(tx_payload.waveform, config)
        return np.asarray(rx_waveform, dtype=np.complex128)

    def _build_interferer_config(
        self,
        base_config: SimulationConfig,
        source: InterferenceSourceConfig,
        index: int,
    ) -> SimulationConfig:
        if source.config_path:
            config = load_simulation_config(Path(source.config_path))
            self._sanitize_file_based_interferer_config(config, base_config)
            self._apply_file_based_inline_overrides(config, source)
        else:
            config = self._build_legacy_inline_interferer_config(base_config, source, index)

        self._finalize_interferer_config(config)
        self._validate_interferer_allocation(config)
        return config

    @staticmethod
    def _sanitize_file_based_interferer_config(
        config: SimulationConfig,
        base_config: SimulationConfig,
    ) -> None:
        """Keep only the referenced config parts needed for interferer generation.

        Args:
            config: Interferer config loaded from ``interference.sources[].config_path``.
            base_config: Main-link config providing global timing and receiver dimensions.
        """
        config.carrier = deepcopy(base_config.carrier)
        config.bwp = deepcopy(base_config.bwp)
        config.slot_index = int(base_config.slot_index)
        config.link.num_rx_ant = int(base_config.link.num_rx_ant)
        config.waveform_input.waveform_path = None
        config.interference.sources = ()
        config.plotting.enabled = False
        config.simulation.result_output_path = None
        config.simulation.num_ttis = 1
        config.simulation.bypass_channel_coding = True

    def _build_legacy_inline_interferer_config(
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
        config.plotting.enabled = False
        config.simulation.result_output_path = None
        config.simulation.num_ttis = 1
        config.simulation.bypass_channel_coding = True
        return config

    @staticmethod
    def _apply_file_based_inline_overrides(
        config: SimulationConfig,
        source: InterferenceSourceConfig,
    ) -> None:
        """Apply only user-written source fields over a file-based interferer config.

        Args:
            config: Interferer config loaded from ``config_path`` and already sanitized.
            source: Interference source row from the main config.
        """
        explicit = source.explicit_fields
        if "channel_model" in explicit:
            config.channel.model = source.channel_model
        if "channel_params" in explicit:
            params = dict(config.channel.params)
            params.update(dict(source.channel_params))
            config.channel.params = params
        if "channel_type" in explicit and source.channel_type is not None:
            config.link.channel_type = source.channel_type
        if "waveform" in explicit and source.waveform is not None:
            config.link.waveform = source.waveform
        if "num_tx_ant" in explicit and source.num_tx_ant is not None:
            config.link.num_tx_ant = int(source.num_tx_ant)
        if "prb_start" in explicit and source.prb_start is not None:
            config.link.prb_start = int(source.prb_start)
        if "num_prbs" in explicit and source.num_prbs is not None:
            config.link.num_prbs = int(source.num_prbs)
        if "start_symbol" in explicit and source.start_symbol is not None:
            config.link.start_symbol = int(source.start_symbol)
        if "num_symbols" in explicit and source.num_symbols is not None:
            config.link.num_symbols = int(source.num_symbols)

        if "mcs" in explicit:
            _apply_mcs_overrides(config, source)
        if "dmrs" in explicit and hasattr(source, "dmrs"):
            _apply_section_overrides(config.dmrs, source.dmrs)
        if "scrambling" in explicit and hasattr(source, "scrambling"):
            _apply_section_overrides(config.scrambling, source.scrambling)

    @staticmethod
    def _finalize_interferer_config(config: SimulationConfig) -> None:
        config.channel.params["add_noise"] = False
        config.link.transport_block_size = None
        config.link.coded_bit_capacity = None
        config.interference.sources = ()
        config.simulation.bypass_channel_coding = True
        config._validate_protocol_constraints()

    @staticmethod
    def _validate_interferer_allocation(config: SimulationConfig) -> None:
        prb_start = int(config.link.prb_start)
        num_prbs = int(config.link.num_prbs)
        start_symbol = int(config.link.start_symbol)
        num_symbols = int(config.link.num_symbols)
        if prb_start < 0 or num_prbs <= 0:
            raise ValueError("Interferer PRB allocation must use non-negative prb_start and positive num_prbs.")
        bwp_num_rbs = int(config.active_bwp_num_rbs)
        if prb_start + num_prbs > bwp_num_rbs:
            raise ValueError(
                "Interferer PRB allocation exceeds the active BWP bandwidth: "
                f"prb_start={prb_start}, num_prbs={num_prbs}, bwp.num_rbs={bwp_num_rbs}."
            )
        if start_symbol < 0 or num_symbols <= 0:
            raise ValueError("Interferer symbol allocation must use non-negative start_symbol and positive num_symbols.")
        if start_symbol + num_symbols > int(config.carrier.symbols_per_slot):
            raise ValueError(
                "Interferer symbol allocation exceeds the slot: "
                f"start_symbol={start_symbol}, num_symbols={num_symbols}, symbols_per_slot={config.carrier.symbols_per_slot}."
            )

    @staticmethod
    def _scale_to_inr(
        waveform: np.ndarray,
        noise_variance: float,
        source: InterferenceSourceConfig,
        config: SimulationConfig,
        index: int,
    ) -> tuple[np.ndarray, InterferenceReport]:
        """Scale one interferer waveform to its target INR.

        Args:
            waveform: Interferer RX waveform with shape ``(slot_samples,)`` or
                ``(num_rx_ant, slot_samples)``.
            noise_variance: Scalar receiver noise variance.
            source: Interference source configuration containing target INR.
            config: Effective interferer config after file loading and inline overrides.
            index: Scalar source index used for default labels.

        Returns:
            Tuple of scaled waveform with the same shape as ``waveform`` and a
            scalar report describing the applied scale.
        """
        rx_power = float(np.mean(np.abs(waveform) ** 2))
        if rx_power <= 0.0:
            logger.warning(
                "Interference source '%s' has zero measured receive power; applied scale is zero.",
                source.label or index,
            )
        target_power = float(noise_variance * (10 ** (source.inr_db / 10.0)))
        scale = 0.0 if rx_power <= 0.0 else np.sqrt(target_power / rx_power)
        label = source.label or f"interferer_{index}"
        return waveform * scale, InterferenceReport(
            label=label,
            inr_db=float(source.inr_db),
            channel_model=config.channel.model,
            config_path=source.config_path,
            prb_start=int(config.link.prb_start),
            num_prbs=int(config.link.num_prbs),
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


def _apply_mcs_overrides(config: SimulationConfig, source: InterferenceSourceConfig) -> None:
    for field_name in source.explicit_mcs_fields:
        if hasattr(config.link.mcs, field_name):
            setattr(config.link.mcs, field_name, getattr(source.mcs, field_name))


def _apply_section_overrides(target: Any, values: Any) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current_value = getattr(target, key)
        if isinstance(current_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(target, key, value)
