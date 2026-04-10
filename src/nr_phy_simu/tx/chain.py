from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelCoder,
    DmrsSequenceGenerator,
    Modulator,
    ResourceMapper,
    TimeDomainProcessor,
)
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
    ) -> None:
        self.coder = coder
        self.modulator = modulator
        self.mapper = mapper
        self.time_processor = time_processor
        self.dmrs_generator = dmrs_generator
        self.scrambler = scrambler

    def transmit(self, transport_block: np.ndarray, config: SimulationConfig) -> TxPayload:
        coded_bits = self.coder.encode(transport_block, config)
        scrambled_bits = self.scrambler.scramble(coded_bits, config)
        tx_symbols = self.modulator.map_bits(scrambled_bits, config)
        grid, dmrs_mask, data_mask, dmrs_symbols = self.mapper.map_to_grid(tx_symbols, config)
        waveform = self.time_processor.modulate(grid, config)
        return TxPayload(
            transport_block=transport_block,
            coded_bits=coded_bits,
            tx_symbols=tx_symbols,
            resource_grid=grid,
            waveform=waveform,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
        )
