from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class CarrierConfig:
    n_size_grid: int = 52
    subcarrier_spacing_khz: int = 30
    fft_size: int = 1024
    cp_length: int = 72
    symbols_per_slot: int = 14
    n_cell_id: int = 0

    @property
    def n_subcarriers(self) -> int:
        return self.n_size_grid * 12

    @property
    def slots_per_frame(self) -> int:
        mu = int(round(math.log2(self.subcarrier_spacing_khz / 15)))
        return 10 * (2**mu)


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
class LinkConfig:
    channel_type: str = "PUSCH"
    waveform: str = "CP-OFDM"
    modulation: str = "QPSK"
    num_layers: int = 1
    num_tx_ant: int = 1
    num_rx_ant: int = 1
    code_rate: float = 0.5
    transport_block_size: int = 1024
    prb_start: int = 0
    num_prbs: int = 24
    start_symbol: int = 0
    num_symbols: int = 14


@dataclass
class SimulationConfig:
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    dmrs: DmrsConfig = field(default_factory=DmrsConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    snr_db: float = 10.0
    random_seed: int = 7
    slot_index: int = 0
