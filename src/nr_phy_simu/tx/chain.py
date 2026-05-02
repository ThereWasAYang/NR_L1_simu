from __future__ import annotations

from dataclasses import replace

import torch

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelCoder,
    DmrsSequenceGenerator,
    Modulator,
    ResourceMapper,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor
from nr_phy_simu.common.types import TxPayload
from nr_phy_simu.config import SimulationConfig


class Transmitter:
    def __init__(
        self,
        coder: ChannelCoder,
        modulator: Modulator,
        mapper: ResourceMapper,
        time_processor: TimeDomainProcessor,
        dmrs_generator: DmrsSequenceGenerator,
        scrambler: BitScrambler,
        layer_mapper: LayerMapper | None = None,
    ) -> None:
        self.coder = coder
        self.modulator = modulator
        self.mapper = mapper
        self.time_processor = time_processor
        self.dmrs_generator = dmrs_generator
        self.scrambler = scrambler
        self.layer_mapper = layer_mapper or LayerMapper()

    def build_slot_payload(self, transport_block: torch.Tensor, config: SimulationConfig) -> TxPayload:
        """Build all transmit-domain buffers up to the frequency-domain slot grid."""
        coded_bits = self.coder.encode(transport_block, config)
        scrambled_bits = self.scrambler.scramble(coded_bits, config)
        tx_symbols = self.modulator.map_bits(scrambled_bits, config)
        layer_mapping = self.layer_mapper.map_symbols(tx_symbols, config.link.num_layers)
        grid, dmrs_mask, data_mask, dmrs_symbols = self.mapper.map_to_grid(layer_mapping.serialized_symbols, config)
        grid = self._expand_tx_grid(grid, config)
        return TxPayload(
            transport_block=transport_block,
            coded_bits=coded_bits,
            tx_symbols=tx_symbols,
            resource_grid=grid,
            waveform=torch.empty((int(config.link.num_tx_ant), 0), dtype=COMPLEX_DTYPE, device=grid.device),
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            layer_symbols=layer_mapping.layer_symbols,
        )

    @staticmethod
    def _expand_tx_grid(grid: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Add an explicit transmit-antenna axis to the mapped resource grid."""
        tx_grid = as_complex_tensor(grid)
        if tx_grid.ndim == 3:
            return tx_grid
        if tx_grid.ndim != 2:
            raise ValueError("TX resource grid must have shape (num_subcarriers, num_symbols).")
        num_tx_ant = int(config.link.num_tx_ant)
        if num_tx_ant <= 1:
            return tx_grid.unsqueeze(0)
        scale = torch.sqrt(torch.tensor(float(num_tx_ant), dtype=torch.float64, device=tx_grid.device))
        return tx_grid.unsqueeze(0).repeat(num_tx_ant, 1, 1) / scale

    def transmit(self, transport_block: torch.Tensor, config: SimulationConfig) -> TxPayload:
        """Run the complete transmit chain for one slot."""
        payload = self.build_slot_payload(transport_block, config)
        waveform = self.time_processor.modulate(payload.resource_grid, config)
        return replace(payload, waveform=waveform)
