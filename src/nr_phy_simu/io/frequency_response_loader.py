from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE

_TEXT_FILE_READ_ENCODING = "utf-8-sig"


def load_frequency_response(
    *,
    values: Any = None,
    path: str | Path | None = None,
) -> torch.Tensor:
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
        Complex frequency-response tensor. SISO shape is ``(num_subcarriers,)``;
        MIMO text-file flat rows have shape ``(num_subcarriers, num_rx_ant*num_tx_ant)``;
        nested MIMO config values keep shape ``(num_subcarriers, num_rx_ant, num_tx_ant)``.
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
        if not isinstance(values, (list, tuple, torch.Tensor)):
            raise ValueError("channel.params.frequency_response must be a sequence of complex coefficients.")
        entries = _parse_complex_array(values)

    tensor = torch.as_tensor(entries, dtype=COMPLEX_DTYPE)
    if tensor.numel() == 0:
        raise ValueError("Frequency-response input is empty.")
    return tensor


def _parse_complex_array(value: Any):
    """Parse nested complex coefficient arrays while preserving dimensions."""
    if _is_complex_pair(value):
        return _parse_complex_value(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _parse_complex_value(value.item())
        return [_parse_complex_array(item) for item in value.detach().cpu().tolist()]
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
    if isinstance(value, torch.Tensor):
        if value.ndim == 1 and value.numel() == 2 and not torch.is_complex(value):
            return True
        return False
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _parse_complex_value(value: Any) -> complex:
    """Parse one complex coefficient from config/file input.

    Args:
        value: Raw coefficient representation from config or text input.

    Returns:
        Parsed complex coefficient.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            scalar = value.detach().cpu().reshape(-1)[0]
            if torch.is_complex(value):
                return complex(scalar.item())
            return complex(float(scalar.item()), 0.0)
        if value.numel() == 2:
            flat = value.detach().cpu().reshape(-1)
            return complex(float(flat[0].item()), float(flat[1].item()))
        raise ValueError(f"Unsupported tensor coefficient shape: {tuple(value.shape)!r}")
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
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
