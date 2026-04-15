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
from nr_phy_simu.common.torch_utils import to_numpy
from nr_phy_simu.config import SimulationConfig


def save_simulation_plots(
    result: SimulationResult,
    config: SimulationConfig,
    output_dir: str | Path,
    prefix: str,
    show: bool = False,
    block: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots: dict[str, Path] = {}
    figure_builders = (
        _build_constellation_figures,
        _build_pilot_estimate_figures,
        _build_rx_time_domain_figures,
        _build_rx_frequency_domain_figures,
    )
    figures: dict[str, object] = {}
    for builder in figure_builders:
        figures.update(builder(result, config))

    for name, figure in figures.items():
        path = output_path / f"{prefix}_{name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plots[name] = path

    if show:
        _show_plots(list(plots.values()), block=block)

    return plots


def _build_constellation_figures(
    result: SimulationResult,
    config: SimulationConfig,
) -> dict[str, object]:
    del config
    symbols = to_numpy(result.rx.equalized_symbols)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, s=10, alpha=0.7)
    ax.set_title(f"Equalized Constellation (SNR={result.snr_db:.2f} dB)")
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    fig.tight_layout()
    return {"constellation": fig}


def _build_pilot_estimate_figures(
    result: SimulationResult,
    config: SimulationConfig,
) -> dict[str, object]:
    del config
    channel_estimation = result.rx.channel_estimation
    channel_estimate = to_numpy(channel_estimation.channel_estimate)
    if channel_estimate.ndim == 2:
        channel_estimate = channel_estimate[np.newaxis, ...]
    num_ant = channel_estimate.shape[0]
    dmrs_mask = to_numpy(result.tx.dmrs_mask)
    dmrs_symbols = np.where(np.any(dmrs_mask, axis=0))[0]
    pilot_estimates = to_numpy(channel_estimation.pilot_estimates)
    pilot_symbol_indices = to_numpy(channel_estimation.pilot_symbol_indices)
    max_cols = 4
    ant_cols = min(num_ant, max_cols)
    ant_rows = int(np.ceil(num_ant / ant_cols))
    fig, axes = plt.subplots(
        ant_rows * 2,
        ant_cols,
        figsize=(max(ant_cols * 4.5, 10), max(ant_rows * 5.5, 6)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(ant_rows * 2, ant_cols)
    all_dmrs_sc = np.flatnonzero(np.any(dmrs_mask, axis=1))
    x_min = int(all_dmrs_sc[0]) if all_dmrs_sc.size else 0
    x_max = int(all_dmrs_sc[-1]) if all_dmrs_sc.size else 1

    for ant_idx in range(num_ant):
        row_idx = ant_idx // ant_cols
        col_idx = ant_idx % ant_cols
        mag_ax = axes[row_idx * 2, col_idx]
        phase_ax = axes[row_idx * 2 + 1, col_idx]
        mag_ax.set_ylabel("Magnitude")
        mag_ax.set_title(f"Pilot-Based Channel Estimate RX{ant_idx}")
        mag_ax.grid(True, linestyle="--", alpha=0.4)
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, linestyle="--", alpha=0.4)

        for symbol_idx in dmrs_symbols:
            pilot_sc = np.flatnonzero(dmrs_mask[:, symbol_idx])
            pilot_values = pilot_estimates[ant_idx, pilot_symbol_indices == symbol_idx]
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

    for ant_idx in range(num_ant, ant_rows * ant_cols):
        row_idx = ant_idx // ant_cols
        col_idx = ant_idx % ant_cols
        axes[row_idx * 2, col_idx].set_visible(False)
        axes[row_idx * 2 + 1, col_idx].set_visible(False)

    fig.supxlabel("DMRS Subcarrier Index")
    fig.tight_layout()
    return {"pilot_estimates": fig}


def _build_rx_time_domain_figures(result: SimulationResult, config: SimulationConfig) -> dict[str, object]:
    waveform = to_numpy(result.rx.rx_waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    cp_lengths = config.carrier.cyclic_prefix_lengths
    fft_size = config.carrier.fft_size_effective
    boundaries = [0]
    labels: list[tuple[float, int]] = []
    offset = 0
    for symbol_idx, cp_len in enumerate(cp_lengths):
        symbol_len = fft_size + cp_len
        center = offset + symbol_len / 2
        labels.append((center, symbol_idx))
        offset += symbol_len
        boundaries.append(offset)

    num_ant = waveform.shape[0]
    fig, axes = plt.subplots(num_ant, 1, figsize=(12, max(3.5 * num_ant, 4)), sharex=True)
    axes = np.atleast_1d(axes)
    for ant_idx, ax in enumerate(axes):
        x = np.arange(waveform.shape[1])
        ax.plot(x, np.abs(waveform[ant_idx]), linewidth=0.9)
        for boundary in boundaries:
            ax.axvline(boundary, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
        ymax = max(float(np.max(np.abs(waveform[ant_idx]))), 1e-6)
        for center, symbol_idx in labels:
            ax.text(center, 0.98 * ymax, f"sym {symbol_idx}", ha="center", va="top", fontsize=8)
        ax.set_title(f"Received Time-Domain Magnitude RX{ant_idx}")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Sample Index")
    fig.tight_layout()
    return {"rx_time": fig}


def _build_rx_frequency_domain_figures(result: SimulationResult, config: SimulationConfig) -> dict[str, object]:
    rx_grid = to_numpy(result.rx.rx_grid)
    if rx_grid.ndim == 2:
        rx_grid = rx_grid[np.newaxis, ...]
    n_sc = config.carrier.n_subcarriers
    num_ant = rx_grid.shape[0]
    fig, axes = plt.subplots(num_ant, 1, figsize=(12, max(3.5 * num_ant, 4)), sharex=True)
    axes = np.atleast_1d(axes)
    for ant_idx, ax in enumerate(axes):
        concatenated = np.abs(rx_grid[ant_idx, :n_sc, :]).reshape(-1, order="F")
        x = np.arange(concatenated.size)
        ax.plot(x, concatenated, linewidth=0.9)
        for symbol_idx in range(config.carrier.symbols_per_slot + 1):
            ax.axvline(symbol_idx * n_sc, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
        ymax = max(float(np.max(concatenated)), 1e-6)
        for symbol_idx in range(config.carrier.symbols_per_slot):
            center = symbol_idx * n_sc + n_sc / 2
            ax.text(center, 0.98 * ymax, f"sym {symbol_idx}", ha="center", va="top", fontsize=8)
        ax.set_title(f"Received Frequency-Domain Magnitude RX{ant_idx} (Full Cell BW)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Flattened Subcarrier Index Across Symbols")
    fig.tight_layout()
    return {"rx_freq": fig}


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
