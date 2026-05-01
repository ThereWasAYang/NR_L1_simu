from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nr_phy_simu_mplconfig"))

import matplotlib

_PREFERRED_BACKEND_ENV = "NR_PHY_SIMU_PLOT_BACKEND"
_WINDOWS_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "SimSun",
]
_MACOS_FONT_CANDIDATES = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
]
_LINUX_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Source Han Sans SC",
]
_COMMON_FONT_CANDIDATES = [
    "DejaVu Sans",
]


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


def _configure_matplotlib_fonts() -> None:
    """Set cross-platform Chinese font fallbacks for plot labels and titles."""
    preferred_fonts: list[str] = []
    system = platform.system()
    if system == "Windows":
        preferred_fonts.extend(_WINDOWS_FONT_CANDIDATES)
    elif system == "Darwin":
        preferred_fonts.extend(_MACOS_FONT_CANDIDATES)
    else:
        preferred_fonts.extend(_LINUX_FONT_CANDIDATES)
    preferred_fonts.extend(_COMMON_FONT_CANDIDATES)
    matplotlib.rcParams["font.sans-serif"] = preferred_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False


def _is_foreground_session() -> bool:
    return bool(os.environ.get("PYCHARM_HOSTED")) or sys.stdout.isatty()


def _has_tkinter() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


_configure_matplotlib_backend()
_configure_matplotlib_fonts()

import matplotlib.pyplot as plt
import numpy as np

from nr_phy_simu.common.runtime_context import get_runtime_context
from nr_phy_simu.common.types import SimulationResult
from nr_phy_simu.common.types import PlotArtifact
from nr_phy_simu.config import SimulationConfig


def save_simulation_plots(
    result: SimulationResult,
    config: SimulationConfig,
    output_dir: str | Path,
    prefix: str,
    show: bool = False,
    block: bool = False,
) -> dict[str, Path]:
    """Render all enabled plot artifacts for one simulation result.

    Args:
        result: Simulation result whose TX/RX arrays follow ``SimulationResult``
            shape conventions.
        config: Full simulation configuration that supplies plot metadata.
        output_dir: Directory where PNG files are written.
        prefix: File-name prefix for generated PNG files.
        show: Whether to show generated images in the foreground.
        block: Whether foreground display should block the Python process.

    Returns:
        Mapping from artifact name to generated PNG path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots: dict[str, Path] = {}
    artifacts = _collect_plot_artifacts(result, config)
    for artifact in artifacts:
        figure = _build_artifact_figure(artifact)
        path = output_path / f"{prefix}_{artifact.name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plots[artifact.name] = path

    if show:
        _show_plots(list(plots.values()), block=block)

    return plots


def _collect_plot_artifacts(
    result: SimulationResult,
    config: SimulationConfig,
) -> tuple[PlotArtifact, ...]:
    """Collect standard and custom plot artifacts from a result.

    Args:
        result: Simulation result. Relevant arrays include equalized symbols
            ``(num_data_symbols,)``, channel estimate
            ``(num_rx_ant, num_subcarriers, num_symbols)``, RX waveform
            ``(num_rx_ant, slot_samples)`` and RX grid
            ``(num_rx_ant, num_subcarriers, num_symbols)``.
        config: Full simulation configuration used to attach axis metadata.

    Returns:
        Tuple of plot artifacts. Each artifact carries its own data shape and metadata.
    """
    artifacts: list[PlotArtifact] = []
    if result.rx.equalized_symbols.size:
        artifacts.append(
            PlotArtifact(
                name="constellation",
                values=result.rx.equalized_symbols,
                plot_type="constellation",
                metadata={"snr_db": result.snr_db},
            )
        )
    if result.rx.channel_estimation.channel_estimate.size:
        artifacts.append(
            PlotArtifact(
                name="pilot_estimates",
                values={
                    "channel_estimation": result.rx.channel_estimation,
                    "dmrs_mask": result.tx.dmrs_mask,
                },
                plot_type="pilot_estimates",
            )
        )
    if result.rx.rx_waveform.size:
        artifacts.append(
            PlotArtifact(
                name="rx_time",
                values=result.rx.rx_waveform,
                plot_type="rx_time",
                metadata={
                    "cp_lengths": config.carrier.cyclic_prefix_lengths,
                    "fft_size": config.carrier.fft_size_effective,
                },
            )
        )
    if result.rx.rx_grid.size:
        artifacts.append(
            PlotArtifact(
                name="rx_freq",
                values=result.rx.rx_grid,
                plot_type="rx_freq",
                metadata={
                    "n_subcarriers": config.carrier.n_subcarriers,
                    "symbols_per_slot": config.carrier.symbols_per_slot,
                },
            )
        )
    artifacts.extend(replace(artifact, name=f"artifact_{artifact.name}") for artifact in result.rx.plot_artifacts)
    artifacts.extend(
        replace(artifact, name=f"context_{artifact.name}")
        for artifact in get_runtime_context().plot_artifacts
    )
    return tuple(artifacts)


def _build_constellation_figure(artifact: PlotArtifact) -> object:
    """Build an I/Q constellation figure.

    Args:
        artifact: Plot artifact whose ``values`` are complex symbols with shape
            ``(num_data_symbols,)``.

    Returns:
        Matplotlib figure object.
    """
    symbols = _as_plot_array(artifact.values)
    snr_db = float((artifact.metadata or {}).get("snr_db", 0.0))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, s=10, alpha=0.7)
    ax.set_title(f"Equalized Constellation (SNR={snr_db:.2f} dB)")
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _build_pilot_estimate_figure(artifact: PlotArtifact) -> object:
    """Build per-antenna pilot channel-estimate magnitude/phase figures.

    Args:
        artifact: Plot artifact containing ``channel_estimation`` and ``dmrs_mask``.
            The channel estimate has shape
            ``(num_rx_ant, num_subcarriers, num_symbols)`` and ``dmrs_mask`` has
            shape ``(num_subcarriers, num_symbols)``.

    Returns:
        Matplotlib figure object.
    """
    values = artifact.values
    channel_estimation = values["channel_estimation"]
    channel_estimate = _as_plot_array(channel_estimation.channel_estimate)
    if channel_estimate.ndim == 2:
        channel_estimate = channel_estimate[np.newaxis, ...]
    num_ant = channel_estimate.shape[0]
    dmrs_mask = _as_plot_array(values["dmrs_mask"])
    dmrs_symbols = np.where(np.any(dmrs_mask, axis=0))[0]
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
            pilot_estimates = _as_plot_array(channel_estimation.pilot_estimates)
            pilot_symbol_indices = _as_plot_array(channel_estimation.pilot_symbol_indices)
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
    return fig


def _build_rx_time_domain_figure(artifact: PlotArtifact) -> object:
    """Build received time-domain magnitude plots.

    Args:
        artifact: Plot artifact whose ``values`` have shape ``(slot_samples,)`` or
            ``(num_rx_ant, slot_samples)``; last axis is time-sample index.

    Returns:
        Matplotlib figure object with one subplot per RX antenna.
    """
    waveform = _as_plot_array(artifact.values)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    metadata = artifact.metadata or {}
    cp_lengths = metadata["cp_lengths"]
    fft_size = int(metadata["fft_size"])
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
    return fig


def _build_rx_frequency_domain_figure(artifact: PlotArtifact) -> object:
    """Build received frequency-domain magnitude plots.

    Args:
        artifact: Plot artifact whose ``values`` have shape
            ``(num_subcarriers, num_symbols)`` or
            ``(num_rx_ant, num_subcarriers, num_symbols)``.

    Returns:
        Matplotlib figure object with one subplot per RX antenna.
    """
    rx_grid = _as_plot_array(artifact.values)
    if rx_grid.ndim == 2:
        rx_grid = rx_grid[np.newaxis, ...]
    metadata = artifact.metadata or {}
    n_sc = int(metadata["n_subcarriers"])
    symbols_per_slot = int(metadata["symbols_per_slot"])
    num_ant = rx_grid.shape[0]
    fig, axes = plt.subplots(num_ant, 1, figsize=(12, max(3.5 * num_ant, 4)), sharex=True)
    axes = np.atleast_1d(axes)
    for ant_idx, ax in enumerate(axes):
        concatenated = np.abs(rx_grid[ant_idx, :n_sc, :]).reshape(-1, order="F")
        x = np.arange(concatenated.size)
        ax.plot(x, concatenated, linewidth=0.9)
        for symbol_idx in range(symbols_per_slot + 1):
            ax.axvline(symbol_idx * n_sc, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
        ymax = max(float(np.max(concatenated)), 1e-6)
        for symbol_idx in range(symbols_per_slot):
            center = symbol_idx * n_sc + n_sc / 2
            ax.text(center, 0.98 * ymax, f"sym {symbol_idx}", ha="center", va="top", fontsize=8)
        ax.set_title(f"Received Frequency-Domain Magnitude RX{ant_idx} (Full Cell BW)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Flattened Subcarrier Index Across Symbols")
    fig.tight_layout()
    return fig


def _build_artifact_figure(artifact: PlotArtifact) -> object:
    """Dispatch one plot artifact to the matching renderer.

    Args:
        artifact: Plot artifact. For generic plots, ``values`` may be scalar-like,
            one-dimensional ``(num_points,)`` data, or two-dimensional
            ``(num_series, num_points)``/image data.

    Returns:
        Matplotlib figure object.
    """
    if artifact.plot_type == "constellation":
        return _build_constellation_figure(artifact)
    if artifact.plot_type == "pilot_estimates":
        return _build_pilot_estimate_figure(artifact)
    if artifact.plot_type == "rx_time":
        return _build_rx_time_domain_figure(artifact)
    if artifact.plot_type == "rx_freq":
        return _build_rx_frequency_domain_figure(artifact)

    values = _as_plot_array(artifact.values)
    x_values = None if artifact.x is None else _as_plot_array(artifact.x)
    plot_type = artifact.plot_type.lower()
    title = artifact.title or artifact.name

    if plot_type == "image" or values.ndim == 2 and plot_type == "auto":
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(np.abs(values), aspect="auto", origin="lower")
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(artifact.xlabel)
        ax.set_ylabel(artifact.ylabel or "Row Index")
        fig.tight_layout()
        return fig

    series = values.reshape(-1) if values.ndim == 1 else values.reshape(values.shape[0], -1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if series.ndim == 1:
        _plot_artifact_series(ax, series, x_values, plot_type, label=None)
    else:
        for row_idx, row in enumerate(series):
            _plot_artifact_series(ax, row, x_values, plot_type, label=f"series {row_idx}")
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(artifact.xlabel)
    ax.set_ylabel(artifact.ylabel or _artifact_default_ylabel(plot_type))
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def _plot_artifact_series(ax, values: np.ndarray, x_values: np.ndarray | None, plot_type: str, label: str | None) -> None:
    """Plot one generic artifact series on an existing axis.

    Args:
        ax: Matplotlib axis object.
        values: One-dimensional series with shape ``(num_points,)``.
        x_values: Optional x-axis values with shape ``(num_points,)`` or longer.
        plot_type: Series projection type such as magnitude, phase, real or imag.
        label: Optional legend label for this series.
    """
    y_values = _artifact_y_values(values, plot_type)
    x = np.arange(y_values.size) if x_values is None else x_values[: y_values.size]
    ax.plot(x, y_values, linewidth=1.0, label=label)


def _artifact_y_values(values: np.ndarray, plot_type: str) -> np.ndarray:
    """Project a generic artifact array to plottable real y-values.

    Args:
        values: Numeric array with arbitrary shape; all axes are preserved.
        plot_type: Projection type such as magnitude, phase, real or imag.

    Returns:
        Real-valued array with the same shape as ``values``.
    """
    if plot_type in {"phase", "angle"}:
        return np.angle(values)
    if plot_type in {"real", "i"}:
        return values.real
    if plot_type in {"imag", "q"}:
        return values.imag
    return np.abs(values)


def _artifact_default_ylabel(plot_type: str) -> str:
    if plot_type in {"phase", "angle"}:
        return "Phase (rad)"
    if plot_type in {"real", "i"}:
        return "Real"
    if plot_type in {"imag", "q"}:
        return "Imag"
    return "Magnitude"


def _as_plot_array(value) -> np.ndarray:
    """Convert list, numpy array, or torch-like tensor values for matplotlib.

    Args:
        value: Array-like object with arbitrary shape. Torch-like tensors are moved
            through ``detach().cpu().numpy()`` when available.

    Returns:
        NumPy array preserving the input shape and axis order.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            pass
    return np.asarray(value)


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
    return platform.system() in {"Darwin", "Windows"} and _is_foreground_session()


def _open_with_system_viewer(paths: list[Path]) -> None:
    if platform.system() == "Windows":
        for path in paths:
            os.startfile(str(path))  # type: ignore[attr-defined]
        return

    subprocess.Popen(
        ["open", *[str(path) for path in paths]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
