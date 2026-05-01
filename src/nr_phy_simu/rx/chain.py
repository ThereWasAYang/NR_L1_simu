from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelDecoder,
    ChannelEstimator,
    Demodulator,
    DmrsSequenceGenerator,
    FrequencyExtractor,
    MimoEqualizer,
    ReceiverDataProcessor,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.types import ChannelEstimateResult, RxPayload
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.data_processing import ThreeStageReceiverDataProcessor


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

    def receive(
        self,
        rx_waveform: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        data_mask: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> RxPayload:
        """Run the complete receive chain for one slot.

        Args:
            rx_waveform: Time-domain waveform with shape ``(slot_samples,)`` for
                one RX antenna or ``(num_rx_ant, slot_samples)`` for multiple RX
                antennas; last axis is time-sample index.
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
        rx_grid = self.time_processor.demodulate(rx_waveform, config)
        return self.receive_from_grid(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
            rx_waveform=rx_waveform,
        )

    def receive_from_grid(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        data_mask: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
        rx_waveform: np.ndarray | None = None,
    ) -> RxPayload:
        """Run the receive chain starting from an already demodulated grid.

        Args:
            rx_grid: Frequency-domain grid with shape ``(num_subcarriers, num_symbols)``
                or ``(num_rx_ant, num_subcarriers, num_symbols)``; axes are RX
                antenna when present, cell subcarrier index, and OFDM symbol index.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            noise_variance: Receiver noise variance used for equalization and demodulation.
            config: Full simulation configuration for waveform and link parameters.
            rx_waveform: Optional waveform with shape ``(slot_samples,)`` or
                ``(num_rx_ant, slot_samples)``. Use ``None`` when bypassing time domain.

        Returns:
            Structured RX payload containing intermediate buffers and decoded bits.
        """
        if rx_grid.ndim == 2:
            rx_grid = rx_grid[np.newaxis, ...]
        processing = self.data_processor.process(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )
        channel_estimation = processing.channel_estimation or ChannelEstimateResult(
            channel_estimate=np.array([], dtype=np.complex128),
            pilot_estimates=np.array([], dtype=np.complex128),
            pilot_symbol_indices=np.array([], dtype=int),
            plot_artifacts=(),
        )
        descrambled_llrs = self.scrambler.descramble_llrs(processing.llrs, config)
        decoded_bits = self.decoder.decode(descrambled_llrs, config)
        crc_ok = getattr(self.decoder, "last_crc_ok", None)
        return RxPayload(
            rx_waveform=np.asarray([], dtype=np.complex128) if rx_waveform is None else rx_waveform,
            rx_grid=rx_grid,
            channel_estimation=channel_estimation,
            equalized_symbols=processing.equalized_symbols,
            layer_symbols=processing.layer_symbols,
            llrs=descrambled_llrs,
            decoded_bits=decoded_bits,
            crc_ok=crc_ok,
            dmrs_symbols=dmrs_symbols,
            plot_artifacts=processing.plot_artifacts,
        )
