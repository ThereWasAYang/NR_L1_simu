from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig")))

import matplotlib.pyplot as plt
import numpy as np

from nr_phy_simu.common.types import SimulationResult


def save_simulation_plots(
    result: SimulationResult,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    constellation_path = output_path / f"{prefix}_constellation.png"
    pilots_path = output_path / f"{prefix}_pilot_channel_estimates.png"

    _save_constellation(result, constellation_path)
    _save_pilot_estimates(result, pilots_path)
    return {"constellation": constellation_path, "pilot_estimates": pilots_path}


def _save_constellation(result: SimulationResult, path: Path) -> None:
    symbols = result.rx.equalized_symbols
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, s=10, alpha=0.7)
    ax.set_title(f"Equalized Constellation (SNR={result.snr_db:.2f} dB)")
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pilot_estimates(result: SimulationResult, path: Path) -> None:
    pilots = result.rx.pilot_estimates
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = np.arange(pilots.size)

    axes[0].plot(x, np.abs(pilots), marker="o", markersize=3, linewidth=1)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Pilot-Based Channel Estimate")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(x, np.unwrap(np.angle(pilots)), marker="o", markersize=3, linewidth=1)
    axes[1].set_xlabel("Pilot Index")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
