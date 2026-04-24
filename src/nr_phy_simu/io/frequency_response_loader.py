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
        values: In-memory coefficient list. Each element may be a complex number,
            a real scalar, a string such as ``"1+0j"``, or a 2-item real/imag pair.
        path: Optional text file path. Each non-empty line follows the same formats
            accepted by :func:`_parse_complex_value`.

    Returns:
        One-dimensional complex frequency-response array.
    """
    if values is None and path is None:
        raise ValueError("Either in-memory frequency_response values or frequency_response_path must be provided.")

    if path is not None:
        resolved = Path(path).expanduser().resolve()
        entries = [
            _parse_complex_value(line)
            for line in resolved.read_text(encoding=_TEXT_FILE_READ_ENCODING).splitlines()
            if line.strip()
        ]
    else:
        if not isinstance(values, (list, tuple, torch.Tensor)):
            raise ValueError("channel.params.frequency_response must be a sequence of complex coefficients.")
        entries = [_parse_complex_value(item) for item in values]

    if not entries:
        raise ValueError("Frequency-response input is empty.")
    return torch.as_tensor(entries, dtype=COMPLEX_DTYPE).reshape(-1)


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
