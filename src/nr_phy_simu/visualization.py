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
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _build_pilot_estimate_figure(result: SimulationResult):
    channel_estimate = result.rx.channel_estimate
    dmrs_mask = result.tx.dmrs_mask
    if channel_estimate.ndim == 2:
        channel_estimate = channel_estimate[np.newaxis, ...]
    num_ant = channel_estimate.shape[0]
    dmrs_symbols = np.where(np.any(dmrs_mask, axis=0))[0]
    fig, axes = plt.subplots(2, num_ant, figsize=(max(10, 5 * num_ant), 6), sharex=True)
    if num_ant == 1:
        axes = np.asarray(axes).reshape(2, 1)
    all_dmrs_sc = np.flatnonzero(np.any(dmrs_mask, axis=1))
    x_min = int(all_dmrs_sc[0]) if all_dmrs_sc.size else 0
    x_max = int(all_dmrs_sc[-1]) if all_dmrs_sc.size else 1

    for ant_idx in range(num_ant):
        mag_ax = axes[0, ant_idx]
        phase_ax = axes[1, ant_idx]
        mag_ax.set_ylabel("Magnitude")
        mag_ax.set_title(f"Pilot-Based Channel Estimate RX{ant_idx}")
        mag_ax.grid(True, linestyle="--", alpha=0.4)
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, linestyle="--", alpha=0.4)

        for symbol_idx in dmrs_symbols:
            pilot_sc = np.flatnonzero(dmrs_mask[:, symbol_idx])
            pilot_values = channel_estimate[ant_idx, pilot_sc, symbol_idx]
            mag_ax.plot(
                pilot_sc,
                np.abs(pilot_values),
                marker="o",
                markersize=3,
                linewidth=1,
                label=f"sym {symbol_idx}",
            )
            phase_ax.plot(
                pilot_sc,
                np.angle(pilot_values),
                marker="o",
                markersize=3,
                linewidth=1,
                label=f"sym {symbol_idx}",
            )

        if dmrs_symbols.size > 1:
            mag_ax.legend(fontsize=8)
            phase_ax.legend(fontsize=8)
        mag_ax.set_xlim(x_min, x_max)
        phase_ax.set_xlim(x_min, x_max)
        phase_ax.set_ylim(-np.pi, np.pi)
        phase_ax.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        phase_ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])

    fig.supxlabel("DMRS Subcarrier Index")
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
    subprocess.Popen(
        ["open", *[str(path) for path in paths]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
