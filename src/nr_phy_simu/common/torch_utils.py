from __future__ import annotations

from typing import Any

import numpy as np
import torch


REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
INT_DTYPE = torch.int64
BIT_DTYPE = torch.int8


def as_real_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=REAL_DTYPE)
    return torch.as_tensor(value, dtype=REAL_DTYPE, device=device)


def as_complex_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=COMPLEX_DTYPE)
    return torch.as_tensor(value, dtype=COMPLEX_DTYPE, device=device)


def as_int_tensor(value: Any, *, device: torch.device | None = None, dtype: torch.dtype = INT_DTYPE) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def complex_randn(
    shape: tuple[int, ...] | list[int],
    *,
    generator: torch.Generator | None = None,
    std: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    real = torch.randn(*shape, generator=generator, dtype=REAL_DTYPE, device=device) * std
    imag = torch.randn(*shape, generator=generator, dtype=REAL_DTYPE, device=device) * std
    return torch.complex(real, imag)


def interp1d_complex(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = as_real_tensor(x)
    xp = as_real_tensor(xp, device=x.device)
    fp = as_complex_tensor(fp, device=x.device)

    if xp.numel() == 1:
        return fp.repeat(x.numel())

    idx = torch.bucketize(x, xp)
    idx = torch.clamp(idx, 1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    weight = (x - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + weight * (y1 - y0)


def ensure_antenna_axis(value: torch.Tensor) -> torch.Tensor:
    return value if value.ndim != 2 else value.unsqueeze(0)
