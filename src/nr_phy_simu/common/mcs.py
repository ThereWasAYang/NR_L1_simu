from __future__ import annotations

from dataclasses import dataclass

from py3gpp import nrPDSCHMCSTables, nrTBS

from nr_phy_simu.config import SimulationConfig


@dataclass(frozen=True)
class McsEntry:
    index: int
    modulation: str
    bits_per_symbol: int
    target_code_rate: float


def resolve_mcs(config: SimulationConfig) -> McsEntry:
    if config.link.mcs.table is None or config.link.mcs.index is None:
        modulation = config.link.modulation
        return McsEntry(
            index=-1,
            modulation=modulation,
            bits_per_symbol=bits_per_symbol(modulation),
            target_code_rate=float(config.link.code_rate),
        )

    if config.link.mcs.modulation and config.link.mcs.target_code_rate is not None:
        modulation = config.link.mcs.modulation
        return McsEntry(
            index=config.link.mcs.index,
            modulation=modulation,
            bits_per_symbol=bits_per_symbol(modulation),
            target_code_rate=float(config.link.mcs.target_code_rate),
        )

    table_name = config.link.mcs.table.lower()
    if table_name != "qam64":
        raise ValueError(
            f"Unsupported MCS table '{config.link.mcs.table}'. Currently supported: qam64."
        )

    table = nrPDSCHMCSTables().QAM64Table
    index = int(config.link.mcs.index)
    modulation = table.Modulation(index)
    return McsEntry(
        index=index,
        modulation=modulation,
        bits_per_symbol=int(table.Qm(index)),
        target_code_rate=float(table.Rate(index)),
    )


def bits_per_symbol(modulation: str) -> int:
    mapping = {
        "PI/2-BPSK": 1,
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
        "64QAM": 6,
        "256QAM": 8,
        "1024QAM": 10,
    }
    key = modulation.upper()
    if key not in mapping:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return mapping[key]


def resolve_transport_block_size(config: SimulationConfig, data_re_count: int) -> int:
    mcs = resolve_mcs(config)
    nre_per_prb = data_re_count // max(config.link.num_prbs, 1)
    return int(
        nrTBS(
            mcs.modulation,
            config.link.num_layers,
            config.link.num_prbs,
            nre_per_prb,
            mcs.target_code_rate,
            xOh=config.link.mcs.x_overhead,
        )
    )


def apply_mcs_to_link(config: SimulationConfig) -> McsEntry:
    entry = resolve_mcs(config)
    config.link.modulation = entry.modulation
    config.link.code_rate = entry.target_code_rate
    return entry
