from __future__ import annotations

import numpy as np

from nr_phy_simu.channels.awgn import AwgnChannel
from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.common.ofdm import OfdmProcessor
from nr_phy_simu.common.resource_grid import DataExtractor, NrResourceMapper
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.types import SimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.rx.channel_estimation import LeastSquaresEstimator
from nr_phy_simu.rx.decoding import NrLdpcDecoder
from nr_phy_simu.rx.demodulation import QamDemodulator
from nr_phy_simu.rx.equalization import OneTapMmseEqualizer
from nr_phy_simu.tx.chain import Transmitter
from nr_phy_simu.tx.codec import NrLdpcCoder
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
        self.mapper = NrResourceMapper(dmrs_generator=self.dmrs_generator)
        self.transmitter = transmitter or Transmitter(
            coder=NrLdpcCoder(),
            modulator=QamModulator(),
            mapper=self.mapper,
            time_processor=OfdmProcessor(),
            dmrs_generator=self.dmrs_generator,
        )
        self.receiver = receiver or Receiver(
            time_processor=OfdmProcessor(),
            extractor=DataExtractor(),
            estimator=LeastSquaresEstimator(),
            equalizer=OneTapMmseEqualizer(),
            demodulator=QamDemodulator(),
            decoder=NrLdpcDecoder(),
            dmrs_generator=self.dmrs_generator,
        )
        self.channel = channel or self._build_channel()

    def run(self) -> SimulationResult:
        apply_mcs_to_link(self.config)
        data_re = self.mapper.count_data_re(self.config)
        self.config.link.coded_bit_capacity = data_re * self._bits_per_symbol()
        if not self.config.link.transport_block_size:
            self.config.link.transport_block_size = resolve_transport_block_size(self.config, data_re)

        rng = np.random.default_rng(self.config.random_seed)
        transport_block = rng.integers(
            0,
            2,
            size=int(self.config.link.transport_block_size),
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
            snr_db=float(channel_info.get("snr_db", self.config.snr_db)),
        )

    def _build_channel(self):
        model = self.config.channel.model.upper()
        if model == "AWGN":
            return AwgnChannel(rng=np.random.default_rng(self.config.random_seed))
        raise NotImplementedError(f"Channel model '{self.config.channel.model}' is not implemented yet.")

    def _bits_per_symbol(self) -> int:
        modulation = self.config.link.modulation.upper()
        mapping = {"PI/2-BPSK": 1, "BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
        return mapping[modulation]
