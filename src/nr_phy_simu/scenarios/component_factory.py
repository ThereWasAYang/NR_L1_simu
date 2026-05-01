from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nr_phy_simu.channels.channel_factory import ChannelFactory, DefaultChannelFactory
from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelCoder,
    ChannelDecoder,
    ChannelEstimator,
    Demodulator,
    DmrsSequenceGenerator,
    FrequencyExtractor,
    MimoEqualizer,
    Modulator,
    ReceiverDataProcessor,
    ResourceMapper,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.ofdm import OfdmProcessor
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.sequences.scrambling import NrDataScrambler
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.channel_estimation import LeastSquaresEstimator
from nr_phy_simu.rx.decoding import HardDecisionBypassDecoder, NrLdpcDecoder
from nr_phy_simu.rx.demodulation import QamDemodulator
from nr_phy_simu.rx.equalization import OneTapMmseEqualizer
from nr_phy_simu.rx.frequency_extraction import FrequencyDomainExtractor
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.tx.chain import Transmitter
from nr_phy_simu.tx.codec import NrLdpcCoder, RandomBitCoder
from nr_phy_simu.tx.modulation import QamModulator
from nr_phy_simu.tx.resource_mapping import FrequencyDomainResourceMapper


@dataclass(frozen=True)
class TransmitterComponents:
    coder: ChannelCoder
    scrambler: BitScrambler
    modulator: Modulator
    mapper: ResourceMapper
    time_processor: TimeDomainProcessor


@dataclass(frozen=True)
class ReceiverComponents:
    time_processor: TimeDomainProcessor
    extractor: FrequencyExtractor
    estimator: ChannelEstimator
    equalizer: MimoEqualizer
    demodulator: Demodulator
    scrambler: BitScrambler
    decoder: ChannelDecoder
    data_processor: ReceiverDataProcessor | None = None


@dataclass(frozen=True)
class SharedComponents:
    dmrs_generator: DmrsSequenceGenerator


@dataclass(frozen=True)
class SimulationComponents:
    shared: SharedComponents
    transmitter: TransmitterComponents
    receiver: ReceiverComponents


class SimulationComponentFactory(ABC):
    @abstractmethod
    def create_components(self, config: SimulationConfig) -> SimulationComponents:
        raise NotImplementedError

    @abstractmethod
    def create_channel_factory(self) -> ChannelFactory:
        raise NotImplementedError


class DefaultSimulationComponentFactory(SimulationComponentFactory):
    """Default assembly for all PHY processing blocks."""

    def create_components(self, config: SimulationConfig) -> SimulationComponents:
        dmrs_generator = DmrsGenerator()
        scrambler = NrDataScrambler()
        time_processor = OfdmProcessor()
        bypass_channel_coding = bool(config.simulation.bypass_channel_coding)
        return SimulationComponents(
            shared=SharedComponents(dmrs_generator=dmrs_generator),
            transmitter=TransmitterComponents(
                coder=RandomBitCoder() if bypass_channel_coding else NrLdpcCoder(),
                scrambler=scrambler,
                modulator=QamModulator(),
                mapper=FrequencyDomainResourceMapper(dmrs_generator=dmrs_generator),
                time_processor=time_processor,
            ),
            receiver=ReceiverComponents(
                time_processor=time_processor,
                extractor=FrequencyDomainExtractor(),
                estimator=LeastSquaresEstimator(),
                equalizer=OneTapMmseEqualizer(),
                demodulator=QamDemodulator(),
                scrambler=scrambler,
                decoder=HardDecisionBypassDecoder() if bypass_channel_coding else NrLdpcDecoder(),
            ),
        )

    def create_channel_factory(self) -> ChannelFactory:
        return DefaultChannelFactory()


def build_transmitter(components: SimulationComponents) -> Transmitter:
    return Transmitter(
        coder=components.transmitter.coder,
        modulator=components.transmitter.modulator,
        mapper=components.transmitter.mapper,
        time_processor=components.transmitter.time_processor,
        dmrs_generator=components.shared.dmrs_generator,
        scrambler=components.transmitter.scrambler,
        layer_mapper=LayerMapper(),
    )


def build_receiver(components: SimulationComponents) -> Receiver:
    return Receiver(
        time_processor=components.receiver.time_processor,
        extractor=components.receiver.extractor,
        estimator=components.receiver.estimator,
        equalizer=components.receiver.equalizer,
        demodulator=components.receiver.demodulator,
        decoder=components.receiver.decoder,
        dmrs_generator=components.shared.dmrs_generator,
        scrambler=components.receiver.scrambler,
        layer_mapper=LayerMapper(),
        data_processor=components.receiver.data_processor,
    )
