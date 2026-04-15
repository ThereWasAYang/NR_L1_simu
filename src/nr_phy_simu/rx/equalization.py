from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import MimoEqualizer
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import as_complex_tensor


class OneTapMmseEqualizer(MimoEqualizer):
    def equalize(
        self,
        rx_symbols: torch.Tensor,
        channel_estimate: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> torch.Tensor:
        del config
        rx_symbols = as_complex_tensor(rx_symbols)
        channel_estimate = as_complex_tensor(channel_estimate, device=rx_symbols.device)
        if rx_symbols.ndim == 2:
            numerator = torch.sum(torch.conj(channel_estimate) * rx_symbols, dim=0)
            denominator = torch.sum(torch.abs(channel_estimate) ** 2, dim=0) + noise_variance
            return numerator / torch.clamp(denominator, min=1e-12)

        denom = (torch.abs(channel_estimate) ** 2) + noise_variance
        weights = torch.conj(channel_estimate) / torch.clamp(denom, min=1e-12)
        return weights * rx_symbols
