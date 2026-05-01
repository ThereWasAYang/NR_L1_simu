from __future__ import annotations

from pathlib import Path

import numpy as np

from nr_phy_simu.config import SimulationConfig

_TEXT_FILE_READ_ENCODING = "utf-8-sig"


def load_text_waveform(path: str | Path, config: SimulationConfig) -> np.ndarray:
    """Load one TTI of time-domain IQ samples from text.

    Args:
        path: Text file path. File samples are antenna-major: all samples for RX0,
            then all samples for RX1, and so on.
        config: Full simulation configuration that defines ``num_rx_ant`` and the
            expected TTI sample count.

    Returns:
        Complex waveform with shape ``(slot_samples,)`` for one RX antenna or
        ``(num_rx_ant, slot_samples)`` for multiple RX antennas. Axis 0 is RX
        antenna when present, last axis is time-sample index.
    """
    resolved = Path(path).expanduser().resolve()
    values = [
        _parse_complex_line(line)
        for line in resolved.read_text(encoding=_TEXT_FILE_READ_ENCODING).splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"No waveform samples found in '{resolved}'.")

    num_rx_ant = int(config.link.num_rx_ant)
    if len(values) % num_rx_ant != 0:
        raise ValueError(
            f"Waveform sample count {len(values)} is not divisible by num_rx_ant={num_rx_ant}."
        )

    num_samples = len(values) // num_rx_ant
    expected_samples = config.waveform_input.num_samples_per_tti or config.carrier.slot_length_samples
    if num_samples != expected_samples:
        raise ValueError(
            f"Waveform samples per antenna ({num_samples}) do not match expected TTI length ({expected_samples})."
        )

    waveform = np.asarray(values, dtype=np.complex128).reshape(num_rx_ant, num_samples)
    return waveform[0] if num_rx_ant == 1 else waveform


def _parse_complex_line(line: str) -> complex:
    """Parse one IQ text line into a complex sample.

    Args:
        line: One input line representing a scalar sample. Supported forms are
            ``I Q``, ``I,Q`` and Python-style ``I+Qj``.

    Returns:
        One scalar complex time-domain sample.
    """
    stripped = line.strip().strip("()")
    comma_parts = [part.strip() for part in stripped.split(",")]
    if len(comma_parts) == 2:
        return complex(float(comma_parts[0]), float(comma_parts[1]))

    space_parts = stripped.split()
    if len(space_parts) == 2:
        return complex(float(space_parts[0]), float(space_parts[1]))

    normalized = stripped.replace("i", "j").replace("I", "j")
    return complex(normalized)
