from __future__ import annotations

from dataclasses import replace

import numpy as np

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelCoder,
    DmrsSequenceGenerator,
    Modulator,
    ResourceMapper,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
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

    def build_slot_payload(self, transport_block: np.ndarray, config: SimulationConfig) -> TxPayload:
        """Build all transmit-domain buffers up to the frequency-domain slot grid.

        Args:
            transport_block: One-dimensional bit array with shape ``(tbs_bits,)``;
                axis 0 is the transport-block bit index before CRC/coding/scrambling.
            config: Full simulation configuration for waveform and link parameters.

        Returns:
            Structured TX payload with the frequency-domain grid populated and an
            empty waveform placeholder.
        """
        coded_bits = self.coder.encode(transport_block, config)
        scrambled_bits = self.scrambler.scramble(coded_bits, config)
        tx_symbols = self.modulator.map_bits(scrambled_bits, config)
        layer_mapping = self.layer_mapper.map_symbols(tx_symbols, config.link.num_layers)
        grid, dmrs_mask, data_mask, dmrs_symbols = self.mapper.map_to_grid(layer_mapping.serialized_symbols, config)
        return TxPayload(
            transport_block=transport_block,
            coded_bits=coded_bits,
            tx_symbols=tx_symbols,
            resource_grid=grid,
            waveform=np.array([], dtype=np.complex128),
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            layer_symbols=layer_mapping.layer_symbols,
        )

    def transmit(self, transport_block: np.ndarray, config: SimulationConfig) -> TxPayload:
        """Run the complete transmit chain for one slot.

        Args:
            transport_block: One-dimensional bit array with shape ``(tbs_bits,)``;
                axis 0 is the transport-block bit index before CRC/coding/scrambling.
            config: Full simulation configuration for waveform and link parameters.

        Returns:
            Structured TX payload containing intermediate buffers and final waveform.
        """
        payload = self.build_slot_payload(transport_block, config)
        waveform = self.time_processor.modulate(payload.resource_grid, config)
        return replace(payload, waveform=waveform)
