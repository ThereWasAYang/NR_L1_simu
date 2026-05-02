from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelDecoder,
    ChannelEstimator,
    Demodulator,
    DmrsSequenceGenerator,
    FrequencyExtractor,
    MimoEqualizer,
    ReceiverDataProcessor,
    ReceiverProcessor,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.types import RxPayload
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.data_processing import ThreeStageReceiverDataProcessor
from nr_phy_simu.rx.receiver_processing import DefaultReceiverProcessor


class Receiver:
    def __init__(
        self,
        time_processor: TimeDomainProcessor,
        extractor: FrequencyExtractor,
        estimator: ChannelEstimator,
        equalizer: MimoEqualizer,
        demodulator: Demodulator,
        decoder: ChannelDecoder,
        dmrs_generator: DmrsSequenceGenerator,
        scrambler: BitScrambler,
        layer_mapper: LayerMapper | None = None,
        data_processor: ReceiverDataProcessor | None = None,
        receiver_processor: ReceiverProcessor | None = None,
    ) -> None:
        self.time_processor = time_processor
        self.extractor = extractor
        self.estimator = estimator
        self.equalizer = equalizer
        self.demodulator = demodulator
        self.decoder = decoder
        self.dmrs_generator = dmrs_generator
        self.scrambler = scrambler
        self.layer_mapper = layer_mapper or LayerMapper()
        self.data_processor = data_processor or ThreeStageReceiverDataProcessor(
            extractor=extractor,
            estimator=estimator,
            equalizer=equalizer,
            demodulator=demodulator,
            layer_mapper=self.layer_mapper,
        )
        self.receiver_processor = receiver_processor or DefaultReceiverProcessor()

    def receive(
        self,
        rx_waveform: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> RxPayload:
        """Run the complete receive chain for one slot.

        Args:
            rx_waveform: Time-domain waveform with shape
                ``(num_rx_ant, slot_samples)``; axis 0 is RX antenna and axis 1 is
                time-sample index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            noise_variance: Receiver noise variance used for equalization and demodulation.
            config: Full simulation configuration for waveform and link parameters.

        Returns:
            Structured RX payload containing intermediate buffers and decoded bits.
        """
        return self.receiver_processor.receive(
            receiver=self,
            rx_waveform=rx_waveform,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )

    def receive_from_grid(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
        rx_waveform: torch.Tensor | None = None,
    ) -> RxPayload:
        """Run the receive chain starting from an already demodulated grid.

        Args:
            rx_grid: Frequency-domain grid with shape
                ``(num_rx_ant, num_subcarriers, num_symbols)``; axes are RX antenna,
                cell subcarrier index, and OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            noise_variance: Receiver noise variance used for equalization and demodulation.
            config: Full simulation configuration for waveform and link parameters.
            rx_waveform: Optional waveform with shape ``(num_rx_ant, slot_samples)``.
                Use ``None`` when bypassing time domain.

        Returns:
            Structured RX payload containing intermediate buffers and decoded bits.
        """
        return self.receiver_processor.receive_from_grid(
            receiver=self,
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
            rx_waveform=rx_waveform,
        )
