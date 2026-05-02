from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_TEXT_FILE_READ_ENCODING = "utf-8-sig"


def load_frequency_response(
    *,
    values: Any = None,
    path: str | Path | None = None,
) -> np.ndarray:
    """Load complex frequency-response coefficients from config values or a text file.

    Args:
        values: In-memory coefficient list. SISO input may be a one-dimensional
            list of complex values or ``[real, imag]`` pairs. MIMO input may be a
            nested array with shape ``(num_subcarriers, num_rx_ant, num_tx_ant)``,
            where each leaf is a complex value or ``[real, imag]`` pair.
        path: Optional text file path. For SISO, each non-empty line is one complex
            coefficient. For MIMO, each line may contain a semicolon-separated flat
            row of coefficients for one subcarrier, for example
            ``h00;h01;h10;h11``.

    Returns:
        Complex frequency-response array. SISO shape is ``(num_subcarriers,)``;
        MIMO shape is ``(num_subcarriers, num_rx_ant, num_tx_ant)``.
    """
    if values is None and path is None:
        raise ValueError("Either in-memory frequency_response values or frequency_response_path must be provided.")

    if path is not None:
        resolved = Path(path).expanduser().resolve()
        entries = [
            _parse_complex_line(line)
            for line in resolved.read_text(encoding=_TEXT_FILE_READ_ENCODING).splitlines()
            if line.strip()
        ]
    else:
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError("channel.params.frequency_response must be a sequence of complex coefficients.")
        parsed = _parse_complex_array(values)
        entries = np.asarray(parsed, dtype=np.complex128)

    if np.asarray(entries).size == 0:
        raise ValueError("Frequency-response input is empty.")
    return np.asarray(entries, dtype=np.complex128)


def _parse_complex_array(value: Any):
    """Parse nested complex coefficient arrays while preserving dimensions.

    Args:
        value: Scalar, scalar-like pair, or nested sequence of coefficients.

    Returns:
        Parsed scalar complex value or nested list of complex values.
    """
    if _is_complex_pair(value):
        return _parse_complex_value(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _parse_complex_value(value.item())
        return [_parse_complex_array(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_parse_complex_array(item) for item in value]
    return _parse_complex_value(value)


def _parse_complex_line(line: str):
    """Parse one SISO coefficient or one flat MIMO coefficient row from text."""
    stripped = line.strip()
    if ";" in stripped:
        return [_parse_complex_value(part.strip()) for part in stripped.split(";") if part.strip()]
    return _parse_complex_value(stripped)


def _is_complex_pair(value: Any) -> bool:
    """Return whether ``value`` is a scalar complex coefficient pair."""
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) == 2
        and all(isinstance(item, (int, float, np.integer, np.floating)) for item in value)
    )


def _parse_complex_value(value: Any) -> complex:
    """Parse one complex coefficient from config/file input.

    Args:
        value: Raw coefficient representation from config or text input.

    Returns:
        One scalar complex frequency-response coefficient.
    """
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return complex(float(value), 0.0)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, str):
        stripped = value.strip().strip("()")
        comma_parts = [part.strip() for part in stripped.split(",")]
        if len(comma_parts) == 2:
            return complex(float(comma_parts[0]), float(comma_parts[1]))
        space_parts = stripped.split()
        if len(space_parts) == 2:
            return complex(float(space_parts[0]), float(space_parts[1]))
        normalized = stripped.replace("i", "j").replace("I", "j")
        return complex(normalized)
    raise ValueError(f"Unsupported complex coefficient format: {value!r}")
