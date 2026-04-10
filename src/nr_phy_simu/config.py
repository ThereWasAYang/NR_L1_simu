from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass
class CarrierConfig:
    cell_bandwidth_rbs: int = 52
    subcarrier_spacing_khz: int = 30
    fft_size: int | None = None
    sample_rate_hz: float | None = None
    cp_length: int = 72
    symbols_per_slot: int = 14
    n_cell_id: int = 0

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


@dataclass
class DmrsConfig:
    additional_positions: int = 0
    config_type: int = 1
    symbol_positions: tuple[int, ...] = ()
    mapping_type: str = "A"
    type_a_position: int = 2
    max_length: int = 1
    scrambling_id0: int | None = None
    scrambling_id1: int | None = None
    nid_nscid: int | None = None
    n_scid: int = 0
    port_set: tuple[int, ...] = (0,)
    num_cdm_groups_without_data: int = 1
    n_pusch_identity: int | None = None
    sequence_hopping: bool = False
    group_hopping: bool = False


@dataclass
class McsConfig:
    table: str | None = None
    index: int | None = None
    modulation: str | None = None
    target_code_rate: float | None = None
    x_overhead: int = 0
    rv: int = 0


@dataclass
class ChannelConfig:
    model: str = "AWGN"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkConfig:
    channel_type: str = "PUSCH"
    waveform: str = "CP-OFDM"
    modulation: str = "QPSK"
    num_layers: int = 1
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
    link: LinkConfig = field(default_factory=LinkConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    snr_db: float = 10.0
    random_seed: int = 7
    slot_index: int = 0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SimulationConfig":
        carrier_data = data.get("carrier", {})
        dmrs_data = _normalize_tuple_fields(data.get("dmrs", {}), {"symbol_positions", "port_set"})
        link_data = dict(data.get("link", {}))
        mcs_data = link_data.pop("mcs", {})
        channel_data = data.get("channel", {})

        carrier = CarrierConfig(**carrier_data)
        dmrs = DmrsConfig(**dmrs_data)
        mcs = McsConfig(**mcs_data)
        link = LinkConfig(**link_data, mcs=mcs)
        channel = ChannelConfig(**channel_data)

        snr_db = float(channel.params.get("snr_db", data.get("snr_db", 10.0)))
        return cls(
            carrier=carrier,
            dmrs=dmrs,
            link=link,
            channel=channel,
            snr_db=snr_db,
            random_seed=int(data.get("random_seed", 7)),
            slot_index=int(data.get("slot_index", 0)),
        )


def _normalize_tuple_fields(data: dict[str, Any], tuple_fields: set[str]) -> dict[str, Any]:
    normalized = dict(data)
    for field_name in tuple_fields:
        if field_name in normalized and isinstance(normalized[field_name], list):
            normalized[field_name] = tuple(normalized[field_name])
    return normalized


def config_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
