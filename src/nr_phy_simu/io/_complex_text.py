from __future__ import annotations

from typing import Any

import numpy as np


def parse_complex_value(value: Any) -> complex:
    """Parse a scalar complex value from numeric, pair, or IQ text input."""
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
        return complex(stripped.replace("i", "j").replace("I", "j"))
    raise ValueError(f"Unsupported complex coefficient format: {value!r}")
