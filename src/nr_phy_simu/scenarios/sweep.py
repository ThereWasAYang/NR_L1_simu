from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csv

from nr_phy_simu.common.types import MultiTtiSimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.scenarios.component_factory import SimulationComponentFactory
from nr_phy_simu.scenarios.multi_tti import MultiTtiSimulationRunner


@dataclass(frozen=True)
class SnrSweepPoint:
    snr_db: float
    bler: float
    ber: float
    average_evm_percent: float | None
    average_evm_snr_linear: float | None


def run_snr_sweep(
    config: SimulationConfig,
    snr_points_db: list[float],
    component_factory: SimulationComponentFactory | None = None,
) -> list[SnrSweepPoint]:
    """Run multi-TTI BLER/BER sweep across a list of SNR points."""
    results: list[SnrSweepPoint] = []
    for snr_db in snr_points_db:
        config.channel.params["snr_db"] = float(snr_db)
        config.snr_db = float(snr_db)
        batch = MultiTtiSimulationRunner(config, component_factory=component_factory).run()
        ber = float("nan")
        total_bits = 0
        total_bit_errors = 0
        for tti_result in batch.tti_results:
            if tti_result.crc_ok is not None:
                ref_size = int(tti_result.tx.transport_block.numel())
            else:
                ref_size = int(tti_result.tx.coded_bits.numel())
            total_bits += ref_size
            total_bit_errors += int(tti_result.bit_errors)
        if total_bits > 0:
            ber = total_bit_errors / total_bits
        results.append(
            SnrSweepPoint(
                snr_db=float(snr_db),
                bler=float(batch.block_error_rate),
                ber=ber,
                average_evm_percent=batch.average_evm_percent,
                average_evm_snr_linear=batch.average_evm_snr_linear,
            )
        )
    return results


def write_snr_sweep_csv(path: str | Path, points: list[SnrSweepPoint]) -> Path:
    """Write SNR sweep results to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr_db", "bler", "ber", "average_evm_percent", "average_evm_snr_linear"])
        for point in points:
            writer.writerow(
                [
                    f"{point.snr_db:.6f}",
                    f"{point.bler:.12f}",
                    f"{point.ber:.12f}",
                    "" if point.average_evm_percent is None else f"{point.average_evm_percent:.12f}",
                    "" if point.average_evm_snr_linear is None else f"{point.average_evm_snr_linear:.12f}",
                ]
            )
    return output_path
