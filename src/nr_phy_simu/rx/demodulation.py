from __future__ import annotations

import itertools

import torch

from nr_phy_simu.common.interfaces import Demodulator
from nr_phy_simu.common.mcs import bits_per_symbol
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.torch_utils import BIT_DTYPE, REAL_DTYPE, as_complex_tensor
from nr_phy_simu.tx.modulation import QamModulator


class QamDemodulator(Demodulator):
    def demap_symbols(
        self,
        symbols: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> torch.Tensor:
        symbols = as_complex_tensor(symbols)
        constellation, bit_labels = self._constellation(config.link.modulation)
        metric = torch.empty((symbols.numel(), constellation.numel()), dtype=REAL_DTYPE, device=symbols.device)
        flat_symbols = symbols.reshape(-1)
        for idx, point in enumerate(constellation):
            metric[:, idx] = torch.abs(flat_symbols - point) ** 2

        llrs = []
        variance = max(noise_variance, 1e-12)
        for bit_idx in range(bit_labels.shape[1]):
            zero_metric = torch.min(metric[:, bit_labels[:, bit_idx] == 0], dim=1).values
            one_metric = torch.min(metric[:, bit_labels[:, bit_idx] == 1], dim=1).values
            llrs.append((one_metric - zero_metric) / variance)
        return torch.stack(llrs, dim=1).reshape(-1)

    def _constellation(self, modulation: str) -> tuple[torch.Tensor, torch.Tensor]:
        bps = bits_per_symbol(modulation)
        bit_patterns = torch.tensor(list(itertools.product([0, 1], repeat=bps)), dtype=BIT_DTYPE)
        symbols = QamModulator.map_bits_for_modulation(bit_patterns.reshape(-1), modulation)
        return symbols, bit_patterns
