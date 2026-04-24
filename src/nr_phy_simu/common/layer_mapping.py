from __future__ import annotations

from dataclasses import dataclass

import torch

from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor


@dataclass(frozen=True)
class LayerMappingResult:
    layer_symbols: tuple[torch.Tensor, ...]
    serialized_symbols: torch.Tensor


class LayerMapper:
    """Explicit layer-mapping helper for shared-channel symbol streams.

    The current PHY chain still transmits a single codeword and equalizes a
    single serialized stream, but keeping this mapping step explicit makes the
    layer/codeword ownership visible and easier to extend later.
    """

    def map_symbols(self, symbols: torch.Tensor, num_layers: int) -> LayerMappingResult:
        """Partition a serialized symbol stream into per-layer views.

        Args:
            symbols: Serialized modulation symbol stream for one codeword.
            num_layers: Number of transmission layers requested by the link.

        Returns:
            A lightweight layer mapping bundle. The serialized stream is kept for
            the current single-stream resource mapper.
        """
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be a positive integer.")
        symbols = as_complex_tensor(symbols).reshape(-1)
        if symbols.numel() == 0:
            empty = tuple(torch.empty(0, dtype=COMPLEX_DTYPE, device=symbols.device) for _ in range(num_layers))
            return LayerMappingResult(layer_symbols=empty, serialized_symbols=symbols)

        layer_symbols = tuple(symbols[layer_index::num_layers].clone() for layer_index in range(num_layers))
        return LayerMappingResult(layer_symbols=layer_symbols, serialized_symbols=symbols)

    def unmap_symbols(self, symbols: torch.Tensor, num_layers: int) -> LayerMappingResult:
        """Build per-layer receive views from a serialized equalized stream."""
        return self.map_symbols(symbols, num_layers)
