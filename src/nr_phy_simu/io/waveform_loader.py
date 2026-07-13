from __future__ import annotations

from pathlib import Path

import numpy as np

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.io._complex_text import parse_complex_value

_TEXT_FILE_READ_ENCODING = "utf-8-sig"


def load_text_waveform(path: str | Path, config: SimulationConfig) -> np.ndarray:
    """Load one TTI of time-domain IQ samples from text.

    Args:
        path: Text file path. File samples are antenna-major: all samples for RX0,
            then all samples for RX1, and so on.
        config: Full simulation configuration that defines ``num_rx_ant`` and the
            expected TTI sample count.

    Returns:
        Complex waveform with shape ``(num_rx_ant, slot_samples)``. Axis 0 is RX
        antenna and axis 1 is time-sample index; the antenna axis is never omitted.
    """
    resolved = Path(path).expanduser().resolve()
    values = [
        parse_complex_value(line)
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
    expected_samples = (
        config.waveform_input.num_samples_per_tti
        or config.carrier.slot_length_samples_for_slot(config.slot_index)
    )
    if num_samples != expected_samples:
        raise ValueError(
            f"Waveform samples per antenna ({num_samples}) do not match expected TTI length ({expected_samples})."
        )

    waveform = np.asarray(values, dtype=np.complex128).reshape(num_rx_ant, num_samples)
    return waveform
