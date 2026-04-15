from __future__ import annotations

from pathlib import Path
import math

from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.common.types import MultiTtiSimulationResult
from nr_phy_simu.config import SimulationConfig

_HEADER = ["信噪比", "BLER", "EVM", "EVM_SNR", "RB位置", "MCS阶数", "总TTI数", "误包数", "码率", "调制阶数", "TBsize"]


def append_multi_tti_report(
    path: str | Path,
    result: MultiTtiSimulationResult,
    config: SimulationConfig,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    needs_header = (not output_path.exists()) or output_path.stat().st_size == 0
    row = _build_row(result, config)

    with output_path.open("a", encoding="utf-8") as handle:
        if needs_header:
            handle.write(",".join(_HEADER) + "\n")
        handle.write(",".join(row) + "\n")

    return output_path


def _build_row(result: MultiTtiSimulationResult, config: SimulationConfig) -> list[str]:
    effective_config = result.final_config or config
    rb_end = effective_config.link.prb_start + effective_config.link.num_prbs - 1
    modulation_order = bits_per_symbol(effective_config.link.modulation)
    return [
        f"{result.last_result.snr_db:.2f}" if result.last_result is not None else f"{effective_config.snr_db:.2f}",
        _format_optional(result.block_error_rate if math.isfinite(result.block_error_rate) else None, precision=6),
        _format_optional(result.average_evm_percent, precision=6),
        _format_optional_db(result.average_evm_snr_linear, precision=6),
        f"{effective_config.link.prb_start}-{rb_end}",
        str(effective_config.link.mcs.index),
        str(result.num_ttis),
        str(result.packet_errors),
        f"{effective_config.link.code_rate:.6f}",
        str(modulation_order),
        str(effective_config.link.transport_block_size),
    ]


def _format_optional(value: float | None, precision: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _format_optional_db(value_linear: float | None, precision: int) -> str:
    if value_linear is None:
        return "N/A"
    value_db = 10.0 * math.log10(max(value_linear, 1e-24))
    return f"{value_db:.{precision}f}"
