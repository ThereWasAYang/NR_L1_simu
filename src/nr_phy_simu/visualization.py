from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig")))

import matplotlib

_PREFERRED_BACKEND_ENV = "NR_PHY_SIMU_PLOT_BACKEND"


def _configure_matplotlib_backend() -> None:
    requested_backend = os.environ.get(_PREFERRED_BACKEND_ENV)
    if requested_backend:
        matplotlib.use(requested_backend)
        return

    if os.environ.get("MPLBACKEND"):
        return

    if platform.system() == "Darwin":
        matplotlib.use("macosx")
        return

    if _is_foreground_session() and _has_tkinter():
        matplotlib.use("TkAgg")


def _is_foreground_session() -> bool:
    return bool(os.environ.get("PYCHARM_HOSTED")) or sys.stdout.isatty()


def _has_tkinter() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


_configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np

from nr_phy_simu.common.types import SimulationResult


def save_simulation_plots(
    result: SimulationResult,
    output_dir: str | Path,
    prefix: str,
    show: bool = False,
    block: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    constellation_path = output_path / f"{prefix}_constellation.png"
    pilots_path = output_path / f"{prefix}_pilot_channel_estimates.png"

    constellation_fig = _build_constellation_figure(result)
    pilot_fig = _build_pilot_estimate_figure(result)

    constellation_fig.savefig(constellation_path, dpi=160)
    pilot_fig.savefig(pilots_path, dpi=160)
    plt.close(constellation_fig)
    plt.close(pilot_fig)

    if show:
        _show_plots([constellation_path, pilots_path], block=block)

    return {"constellation": constellation_path, "pilot_estimates": pilots_path}


def _build_constellation_figure(result: SimulationResult):
    symbols = result.rx.equalized_symbols
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, s=10, alpha=0.7)
    ax.set_title(f"Equalized Constellation (SNR={result.snr_db:.2f} dB)")
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _build_pilot_estimate_figure(result: SimulationResult):
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
    return fig


def _show_plots(paths: list[Path], block: bool) -> None:
    if _use_system_viewer():
        _open_with_system_viewer(paths)
        return

    figures = [plt.figure() for _ in paths]
    for figure, path in zip(figures, paths, strict=True):
        image = plt.imread(path)
        axis = figure.subplots()
        axis.imshow(image)
        axis.axis("off")

    manager_backend = plt.get_backend().lower()
    if "agg" in manager_backend:
        for figure in figures:
            plt.close(figure)
        return

    plt.ioff()
    plt.show(block=block)
    plt.pause(0.001)
    if block:
        for figure in figures:
            plt.close(figure)


def _use_system_viewer() -> bool:
    return platform.system() == "Darwin" and _is_foreground_session()


def _open_with_system_viewer(paths: list[Path]) -> None:
    for path in paths:
        subprocess.Popen(
            ["open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
