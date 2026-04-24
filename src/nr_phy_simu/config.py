from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass
class CarrierConfig:
    cell_bandwidth_rbs: int = 52
    subcarrier_spacing_khz: int = 30
    cyclic_prefix: str = "NormalCP"
    fft_size: int | None = None
    sample_rate_hz: float | None = None

    @property
    def n_size_grid(self) -> int:
        return self.cell_bandwidth_rbs

    @property
    def n_subcarriers(self) -> int:
        return self.cell_bandwidth_rbs * 12

    @property
    def slots_per_frame(self) -> int:
        mu = int(round(math.log2(self.subcarrier_spacing_khz / 15)))
        return 10 * (2**mu)

    @property
    def numerology(self) -> int:
        return int(round(math.log2(self.subcarrier_spacing_khz / 15)))

    @property
    def cyclic_prefix_mode(self) -> str:
        normalized = self.cyclic_prefix.strip().upper()
        mapping = {
            "NORMAL": "NORMAL",
            "NORMALCP": "NORMAL",
            "NORMAL_CP": "NORMAL",
            "ECP": "EXTENDED",
            "EXTENDED": "EXTENDED",
            "EXTENDEDCP": "EXTENDED",
            "EXTENDED_CP": "EXTENDED",
        }
        if normalized not in mapping:
            raise ValueError(
                f"Unsupported cyclic prefix '{self.cyclic_prefix}'. Use NormalCP or ECP."
            )
        mode = mapping[normalized]
        if mode == "EXTENDED" and self.subcarrier_spacing_khz != 60:
            raise ValueError("Extended cyclic prefix is only supported for 60 kHz SCS in NR.")
        return mode

    @property
    def symbols_per_slot(self) -> int:
        return 12 if self.cyclic_prefix_mode == "EXTENDED" else 14

    @property
    def fft_size_effective(self) -> int:
        if self.fft_size is not None:
            return int(self.fft_size)
        required = max(128, self.n_subcarriers)
        return 1 << math.ceil(math.log2(required))

    @property
    def sample_rate_effective_hz(self) -> float:
        if self.sample_rate_hz is not None:
            return float(self.sample_rate_hz)
        return float(self.fft_size_effective * self.subcarrier_spacing_khz * 1e3)

    @property
    def cyclic_prefix_lengths(self) -> tuple[int, ...]:
        scale = self.sample_rate_effective_hz / 30.72e6
        mu = self.numerology
        if self.cyclic_prefix_mode == "NORMAL":
            first = int(((144 * (2 ** (-mu))) + 16) * scale)
            other = int((144 * (2 ** (-mu))) * scale)
            lengths = [
                first if idx == 0 or idx == 7 * (2**mu) else other
                for idx in range(self.symbols_per_slot)
            ]
            return tuple(lengths)

        extended = int((512 * (2 ** (-mu))) * scale)
        return tuple([extended] * self.symbols_per_slot)

    @property
    def slot_length_samples(self) -> int:
        return sum(self.cyclic_prefix_lengths) + self.symbols_per_slot * self.fft_size_effective


@dataclass
class DmrsConfig:
    additional_positions: int = 0
    config_type: int = 1
    symbol_positions: tuple[int, ...] = ()
    data_mux_enabled: bool = True
    mapping_type: str = "A"
    type_a_position: int = 2
    max_length: int = 1
    scrambling_id0: int | None = None
    scrambling_id1: int | None = None
    nid_nscid: int | None = None
    n_scid: int = 0
    port_set: tuple[int, ...] = (0,)
    num_cdm_groups_without_data: int | None = None
    n_pusch_identity: int | None = None
    sequence_hopping: bool = False
    group_hopping: bool = False
    uplink_transform_precoding: bool = False
    pi2bpsk_scrambling_id0: int | None = None
    pi2bpsk_scrambling_id1: int | None = None


@dataclass
class ScramblingConfig:
    rnti: int = 1
    n_id: int = 0
    data_scrambling_id: int | None = None
    codeword_index: int = 0

    @property
    def effective_data_scrambling_id(self) -> int:
        if self.data_scrambling_id is not None:
            return int(self.data_scrambling_id)
        return int(self.n_id)


@dataclass
class McsConfig:
    table: str | None = None
    index: int | None = None
    modulation: str | None = None
    target_code_rate: float | None = None
    tp_pi2bpsk: bool = False
    x_overhead: int = 0
    rv: int = 0


@dataclass
class DecoderConfig:
    ldpc_max_iterations: int = 25
    ldpc_min_sum_scaling: float = 0.75
    ldpc_enable_py3gpp_fallback: bool = True


@dataclass
class HarqConfig:
    enabled: bool = False
    num_processes: int = 4
    max_retransmissions: int = 3
    rv_sequence: tuple[int, ...] = (0, 2, 3, 1)


@dataclass
class ChannelConfig:
    model: str = "AWGN"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterferenceSourceConfig:
    label: str | None = None
    enabled: bool = True
    inr_db: float = 0.0
    channel_model: str = "AWGN"
    channel_params: dict[str, Any] = field(default_factory=dict)
    prb_start: int | None = None
    num_prbs: int | None = None
    start_symbol: int | None = None
    num_symbols: int | None = None
    waveform: str | None = None
    channel_type: str | None = None
    num_tx_ant: int | None = None
    mcs: McsConfig = field(default_factory=McsConfig)


@dataclass
class InterferenceConfig:
    sources: tuple[InterferenceSourceConfig, ...] = ()

    @property
    def enabled(self) -> bool:
        return any(source.enabled for source in self.sources)


@dataclass
class PlottingConfig:
    enabled: bool = True


@dataclass
class SimulationControlConfig:
    num_ttis: int = 1
    result_output_path: str | None = None
    bypass_channel_coding: bool = False


@dataclass
class WaveformInputConfig:
    waveform_path: str | None = None
    format: str = "text_iq"
    num_samples_per_tti: int | None = None
    noise_variance: float | None = None

    @property
    def enabled(self) -> bool:
        return self.waveform_path is not None


@dataclass
class LinkConfig:
    channel_type: str = "PUSCH"
    waveform: str = "CP-OFDM"
    modulation: str = "QPSK"
    num_layers: int = 1
    num_codewords: int = 1
    num_tx_ant: int = 1
    num_rx_ant: int = 1
    code_rate: float = 0.5
    transport_block_size: int | None = None
    prb_start: int = 0
    num_prbs: int = 24
    start_symbol: int = 0
    num_symbols: int = 14
    mcs: McsConfig = field(default_factory=McsConfig)
    coded_bit_capacity: int | None = None

    @property
    def user_bandwidth_rbs(self) -> int:
        return self.num_prbs


@dataclass
class SimulationConfig:
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    dmrs: DmrsConfig = field(default_factory=DmrsConfig)
    scrambling: ScramblingConfig = field(default_factory=ScramblingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    harq: HarqConfig = field(default_factory=HarqConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    simulation: SimulationControlConfig = field(default_factory=SimulationControlConfig)
    waveform_input: WaveformInputConfig = field(default_factory=WaveformInputConfig)
    snr_db: float = 10.0
    random_seed: int = 7
    slot_index: int = 0

    def __post_init__(self) -> None:
        self._validate_protocol_constraints()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SimulationConfig":
        carrier_data = data.get("carrier", {})
        dmrs_data = _normalize_tuple_fields(data.get("dmrs", {}), {"symbol_positions", "port_set"})
        scrambling_data = data.get("scrambling", {})
        decoder_data = data.get("decoder", {})
        harq_data = _normalize_tuple_fields(data.get("harq", {}), {"rv_sequence"})
        link_data = dict(data.get("link", {}))
        mcs_data = link_data.pop("mcs", {})
        channel_data = data.get("channel", {})
        interference_data = data.get("interference", {})
        waveform_input_data = data.get("waveform_input", {})

        carrier = CarrierConfig(**carrier_data)
        dmrs = DmrsConfig(**dmrs_data)
        scrambling = ScramblingConfig(**scrambling_data)
        decoder = DecoderConfig(**decoder_data)
        harq = HarqConfig(**harq_data)
        mcs = McsConfig(**mcs_data)
        link = LinkConfig(**link_data, mcs=mcs)
        channel = ChannelConfig(**channel_data)
        interference = _parse_interference_config(interference_data)
        plotting = PlottingConfig(**data.get("plotting", {}))
        simulation = SimulationControlConfig(**data.get("simulation", {}))
        waveform_input = WaveformInputConfig(**waveform_input_data)

        snr_db = float(channel.params.get("snr_db", data.get("snr_db", 10.0)))
        return cls(
            carrier=carrier,
            dmrs=dmrs,
            scrambling=scrambling,
            decoder=decoder,
            harq=harq,
            link=link,
            channel=channel,
            interference=interference,
            plotting=plotting,
            simulation=simulation,
            waveform_input=waveform_input,
            snr_db=snr_db,
            random_seed=int(data.get("random_seed", 7)),
            slot_index=int(data.get("slot_index", 0)),
        )

    def _validate_protocol_constraints(self) -> None:
        channel_type = self.link.channel_type.upper()
        waveform = self.link.waveform.upper()
        transform_precoded_pusch = channel_type == "PUSCH" and waveform == "DFT-S-OFDM"
        if transform_precoded_pusch and int(self.dmrs.config_type) != 1:
            raise ValueError(
                "Transform-precoded PUSCH (DFT-s-OFDM) only supports DMRS configuration type 1."
            )
        if transform_precoded_pusch and bool(self.dmrs.data_mux_enabled):
            raise ValueError(
                "Transform-precoded PUSCH (DFT-s-OFDM) does not support data/DMRS symbol multiplexing."
            )
        if (
            transform_precoded_pusch
            and self.dmrs.num_cdm_groups_without_data is not None
            and int(self.dmrs.num_cdm_groups_without_data) != 2
        ):
            raise ValueError(
                "Transform-precoded PUSCH (DFT-s-OFDM) requires num_cdm_groups_without_data = 2."
            )
        if int(self.link.num_layers) <= 0:
            raise ValueError("link.num_layers must be a positive integer.")
        if int(self.link.num_codewords) <= 0:
            raise ValueError("link.num_codewords must be a positive integer.")
        if self.link.channel_type.upper() == "PUSCH" and int(self.link.num_codewords) != 1:
            raise ValueError("Current PUSCH implementation supports exactly one codeword.")
        if self.link.channel_type.upper() == "PDSCH" and int(self.link.num_codewords) not in (1, 2):
            raise ValueError("Current PDSCH configuration must use one or two codewords.")
        if self.harq.enabled:
            if int(self.harq.num_processes) <= 0:
                raise ValueError("harq.num_processes must be a positive integer.")
            if int(self.harq.max_retransmissions) < 0:
                raise ValueError("harq.max_retransmissions must be non-negative.")
            if not self.harq.rv_sequence:
                raise ValueError("harq.rv_sequence must contain at least one redundancy version.")
            invalid_rv = [rv for rv in self.harq.rv_sequence if int(rv) not in (0, 1, 2, 3)]
            if invalid_rv:
                raise ValueError(f"harq.rv_sequence contains unsupported RV values: {invalid_rv}")


def _normalize_tuple_fields(data: dict[str, Any], tuple_fields: set[str]) -> dict[str, Any]:
    normalized = dict(data)
    for field_name in tuple_fields:
        if field_name in normalized and isinstance(normalized[field_name], list):
            normalized[field_name] = tuple(normalized[field_name])
    return normalized


def _parse_interference_config(data: dict[str, Any]) -> InterferenceConfig:
    if not data:
        return InterferenceConfig()

    sources = []
    for source_data in data.get("sources", []):
        normalized = dict(source_data)
        mcs_data = normalized.pop("mcs", {})
        sources.append(InterferenceSourceConfig(mcs=McsConfig(**mcs_data), **normalized))
    return InterferenceConfig(sources=tuple(sources))


def config_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
