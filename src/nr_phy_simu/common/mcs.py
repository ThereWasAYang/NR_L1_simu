from __future__ import annotations

from dataclasses import dataclass

from py3gpp import nrTBS

from nr_phy_simu.config import SimulationConfig


@dataclass(frozen=True)
class McsEntry:
    index: int
    modulation: str
    bits_per_symbol: int
    target_code_rate: float


@dataclass(frozen=True)
class McsTableRow:
    modulation: str | None
    target_code_rate_x1024: float | None

    @property
    def is_reserved(self) -> bool:
        return self.modulation is None or self.target_code_rate_x1024 is None


def _rows(values: list[tuple[str | None, float | None]]) -> tuple[McsTableRow, ...]:
    return tuple(McsTableRow(modulation=modulation, target_code_rate_x1024=rate) for modulation, rate in values)


MCS_TABLES: dict[str, tuple[McsTableRow, ...]] = {
    "qam64": _rows(
        [
            ("QPSK", 120), ("QPSK", 157), ("QPSK", 193), ("QPSK", 251),
            ("QPSK", 308), ("QPSK", 379), ("QPSK", 449), ("QPSK", 526),
            ("QPSK", 602), ("QPSK", 679), ("16QAM", 340), ("16QAM", 378),
            ("16QAM", 434), ("16QAM", 490), ("16QAM", 553), ("16QAM", 616),
            ("16QAM", 658), ("64QAM", 438), ("64QAM", 466), ("64QAM", 517),
            ("64QAM", 567), ("64QAM", 616), ("64QAM", 666), ("64QAM", 719),
            ("64QAM", 772), ("64QAM", 822), ("64QAM", 873), ("64QAM", 910),
            ("64QAM", 948), (None, None), (None, None), (None, None),
        ]
    ),
    "qam256": _rows(
        [
            ("QPSK", 120), ("QPSK", 193), ("QPSK", 308), ("QPSK", 449),
            ("QPSK", 602), ("16QAM", 378), ("16QAM", 434), ("16QAM", 490),
            ("16QAM", 553), ("16QAM", 616), ("16QAM", 658), ("64QAM", 466),
            ("64QAM", 517), ("64QAM", 567), ("64QAM", 616), ("64QAM", 666),
            ("64QAM", 719), ("64QAM", 772), ("64QAM", 822), ("64QAM", 873),
            ("256QAM", 682.5), ("256QAM", 711), ("256QAM", 754), ("256QAM", 797),
            ("256QAM", 841), ("256QAM", 885), ("256QAM", 916.5), ("256QAM", 948),
            (None, None), (None, None), (None, None), (None, None),
        ]
    ),
    "qam64lowse": _rows(
        [
            ("QPSK", 30), ("QPSK", 40), ("QPSK", 50), ("QPSK", 64),
            ("QPSK", 78), ("QPSK", 99), ("QPSK", 120), ("QPSK", 157),
            ("QPSK", 193), ("QPSK", 251), ("QPSK", 308), ("QPSK", 379),
            ("QPSK", 449), ("QPSK", 526), ("QPSK", 602), ("16QAM", 340),
            ("16QAM", 378), ("16QAM", 434), ("16QAM", 490), ("16QAM", 553),
            ("16QAM", 616), ("64QAM", 438), ("64QAM", 466), ("64QAM", 517),
            ("64QAM", 567), ("64QAM", 616), ("64QAM", 666), ("64QAM", 719),
            ("64QAM", 772), (None, None), (None, None), (None, None),
        ]
    ),
    "qam1024": _rows(
        [
            ("QPSK", 120), ("QPSK", 193), ("QPSK", 449), ("16QAM", 378),
            ("16QAM", 490), ("16QAM", 616), ("64QAM", 466), ("64QAM", 517),
            ("64QAM", 567), ("64QAM", 616), ("64QAM", 666), ("64QAM", 719),
            ("64QAM", 772), ("64QAM", 822), ("64QAM", 873), ("256QAM", 682.5),
            ("256QAM", 711), ("256QAM", 754), ("256QAM", 797), ("256QAM", 841),
            ("256QAM", 885), ("256QAM", 916.5), ("256QAM", 948), ("1024QAM", 805.5),
            ("1024QAM", 853), ("1024QAM", 900.5), ("1024QAM", 948), (None, None),
            (None, None), (None, None), (None, None), (None, None),
        ]
    ),
    "tp64qam": _rows(
        [
            ("PI/2-BPSK", 240), ("PI/2-BPSK", 314), ("QPSK", 193), ("QPSK", 251),
            ("QPSK", 308), ("QPSK", 379), ("QPSK", 449), ("QPSK", 526),
            ("QPSK", 602), ("QPSK", 679), ("16QAM", 340), ("16QAM", 378),
            ("16QAM", 434), ("16QAM", 490), ("16QAM", 553), ("16QAM", 616),
            ("16QAM", 658), ("64QAM", 466), ("64QAM", 517), ("64QAM", 567),
            ("64QAM", 616), ("64QAM", 666), ("64QAM", 719), ("64QAM", 772),
            ("64QAM", 822), ("64QAM", 873), ("64QAM", 910), ("64QAM", 948),
            (None, None), (None, None), (None, None), (None, None),
        ]
    ),
    "tp64lowse": _rows(
        [
            ("PI/2-BPSK", 60), ("PI/2-BPSK", 80), ("PI/2-BPSK", 100), ("PI/2-BPSK", 128),
            ("PI/2-BPSK", 156), ("PI/2-BPSK", 198), ("QPSK", 120), ("QPSK", 157),
            ("QPSK", 193), ("QPSK", 251), ("QPSK", 308), ("QPSK", 379),
            ("QPSK", 449), ("QPSK", 526), ("QPSK", 602), ("QPSK", 679),
            ("16QAM", 378), ("16QAM", 434), ("16QAM", 490), ("16QAM", 553),
            ("16QAM", 616), ("16QAM", 658), ("16QAM", 699), ("16QAM", 772),
            ("64QAM", 567), ("64QAM", 616), ("64QAM", 666), ("64QAM", 772),
            (None, None), (None, None), (None, None), (None, None),
        ]
    ),
}

MCS_TABLE_ALIASES = {
    "table1": "qam64",
    "table2": "qam256",
    "table3": "qam64lowse",
    "table4": "qam1024",
    "qam64lowse": "qam64lowse",
    "qam64-lowse": "qam64lowse",
    "qam64_lowse": "qam64lowse",
    "qam64": "qam64",
    "qam256": "qam256",
    "qam1024": "qam1024",
    "tp64qam": "tp64qam",
    "transformprecoder64qam": "tp64qam",
    "tp64lowse": "tp64lowse",
    "tp64-lowse": "tp64lowse",
    "tp64_lowse": "tp64lowse",
    "transformprecoder64lowse": "tp64lowse",
}


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
            index=int(config.link.mcs.index),
            modulation=modulation,
            bits_per_symbol=bits_per_symbol(modulation),
            target_code_rate=float(config.link.mcs.target_code_rate),
        )

    table_name = canonical_mcs_table_name(config.link.mcs.table)
    _validate_mcs_table_applicability(config, table_name)

    index = int(config.link.mcs.index)
    table = MCS_TABLES[table_name]
    if not 0 <= index < len(table):
        raise ValueError(f"MCS index {index} is out of range for table '{table_name}'.")

    row = table[index]
    if row.is_reserved:
        raise ValueError(f"MCS index {index} is reserved in table '{table_name}'.")

    modulation = _resolve_modulation_for_row(row, config)
    return McsEntry(
        index=index,
        modulation=modulation,
        bits_per_symbol=bits_per_symbol(modulation),
        target_code_rate=float(row.target_code_rate_x1024 / 1024.0),
    )


def canonical_mcs_table_name(table_name: str) -> str:
    key = table_name.strip().lower()
    if key not in MCS_TABLE_ALIASES:
        supported = ", ".join(sorted(set(MCS_TABLE_ALIASES.values())))
        raise ValueError(f"Unsupported MCS table '{table_name}'. Supported tables: {supported}.")
    return MCS_TABLE_ALIASES[key]


def _validate_mcs_table_applicability(config: SimulationConfig, table_name: str) -> None:
    channel_type = config.link.channel_type.upper()
    waveform = config.link.waveform.upper()
    transform_precoded = channel_type == "PUSCH" and waveform == "DFT-S-OFDM"

    if table_name == "qam1024" and channel_type != "PDSCH":
        raise ValueError("MCS table 'qam1024' is defined for PDSCH, not for PUSCH.")
    if table_name.startswith("tp") and not transform_precoded:
        raise ValueError(
            f"MCS table '{table_name}' requires transform-precoded PUSCH (DFT-s-OFDM waveform)."
        )
    if transform_precoded and table_name == "qam64":
        raise ValueError("Transform-precoded PUSCH should use 'tp64qam', 'tp64lowse', 'qam256', or 'qam64lowse'.")


def _resolve_modulation_for_row(row: McsTableRow, config: SimulationConfig) -> str:
    if row.modulation != "PI/2-BPSK":
        return str(row.modulation)
    return "PI/2-BPSK" if config.link.mcs.tp_pi2bpsk else "QPSK"


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
