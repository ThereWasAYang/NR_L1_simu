from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerMappingResult:
    layer_symbols: tuple[np.ndarray, ...]
    serialized_symbols: np.ndarray


class LayerMapper:
    """Explicit layer-mapping helper for shared-channel symbol streams.

    The current PHY chain still transmits a single codeword and equalizes a
    single serialized stream, but keeping this mapping step explicit makes the
    layer/codeword ownership visible and easier to extend later.
    """

    def map_symbols(self, symbols: np.ndarray, num_layers: int) -> LayerMappingResult:
        """Partition a serialized symbol stream into per-layer views.

        Args:
            symbols: One-dimensional complex symbol stream with shape
                ``(num_symbols_total,)``; axis 0 is serialized codeword symbol order.
            num_layers: Number of transmission layers requested by the link.

        Returns:
            A lightweight layer mapping bundle. The serialized stream is kept for
            the current single-stream resource mapper.
        """
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be a positive integer.")
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        if symbols.size == 0:
            return LayerMappingResult(layer_symbols=tuple(np.array([], dtype=np.complex128) for _ in range(num_layers)), serialized_symbols=symbols)

        layer_symbols = tuple(symbols[layer_index::num_layers].copy() for layer_index in range(num_layers))
        return LayerMappingResult(layer_symbols=layer_symbols, serialized_symbols=symbols)

    def unmap_symbols(self, symbols: np.ndarray, num_layers: int) -> LayerMappingResult:
        """Build per-layer receive views from a serialized equalized stream.

        Args:
            symbols: One-dimensional complex equalized stream with shape
                ``(num_symbols_total,)``; axis 0 is serialized receive symbol order.
            num_layers: Number of transmission layers requested by the link.

        Returns:
            Layer mapping bundle whose per-layer arrays have shape
            ``(ceil(num_symbols_total / num_layers),)`` or shorter for tail layers.
        """
        return self.map_symbols(symbols, num_layers)
