from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.awgn import AwgnChannel
from nr_phy_simu.common.ofdm import OfdmProcessor
from nr_phy_simu.common.resource_grid import DataExtractor, NrResourceMapper
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.types import SimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.rx.channel_estimation import LeastSquaresEstimator
from nr_phy_simu.rx.decoding import HardDecisionRepetitionDecoder
from nr_phy_simu.rx.demodulation import QamDemodulator
from nr_phy_simu.rx.equalization import OneTapMmseEqualizer
from nr_phy_simu.tx.chain import Transmitter
from nr_phy_simu.tx.codec import RepetitionCoder
from nr_phy_simu.tx.modulation import QamModulator


class SharedChannelSimulation:
    def __init__(
        self,
        config: SimulationConfig,
        transmitter: Transmitter | None = None,
        receiver: Receiver | None = None,
        channel=None,
    ) -> None:
        self.config = config
        self.dmrs_generator = DmrsGenerator()
        mapper = NrResourceMapper(dmrs_generator=self.dmrs_generator)
        self.transmitter = transmitter or Transmitter(
            coder=RepetitionCoder(),
            modulator=QamModulator(),
            mapper=mapper,
            time_processor=OfdmProcessor(),
            dmrs_generator=self.dmrs_generator,
        )
        self.receiver = receiver or Receiver(
            time_processor=OfdmProcessor(),
            extractor=DataExtractor(),
            estimator=LeastSquaresEstimator(),
            equalizer=OneTapMmseEqualizer(),
            demodulator=QamDemodulator(),
            decoder=HardDecisionRepetitionDecoder(),
            dmrs_generator=self.dmrs_generator,
        )
        self.channel = channel or AwgnChannel(
            rng=np.random.default_rng(self.config.random_seed)
        )

    def run(self) -> SimulationResult:
        rng = np.random.default_rng(self.config.random_seed)
        transport_block = rng.integers(
            0,
            2,
            size=self.config.link.transport_block_size,
            dtype=np.int8,
        )
        tx_payload = self.transmitter.transmit(transport_block, self.config)
        rx_waveform, channel_info = self.channel.propagate(tx_payload.waveform, self.config)
        rx_payload = self.receiver.receive(
            rx_waveform=rx_waveform,
            dmrs_symbols=tx_payload.dmrs_symbols,
            dmrs_mask=tx_payload.dmrs_mask,
            data_mask=tx_payload.data_mask,
            noise_variance=float(channel_info["noise_variance"]),
            config=self.config,
        )
        decoded = rx_payload.decoded_bits[: transport_block.size]
        bit_errors = int(np.sum(decoded != transport_block))
        ber = bit_errors / transport_block.size
        return SimulationResult(
            tx=tx_payload,
            rx=rx_payload,
            bit_errors=bit_errors,
            bit_error_rate=ber,
            snr_db=self.config.snr_db,
        )
