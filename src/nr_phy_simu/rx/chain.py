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
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.types import RxPayload
from nr_phy_simu.config import SimulationConfig


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
        channel_estimation = self.estimator.estimate(rx_grid, dmrs_symbols, dmrs_mask, config)

        rx_data_symbols = self.extractor.extract(rx_grid, data_mask, config, despread=False)
        data_channel = self.extractor.extract(channel_estimation.channel_estimate, data_mask, config, despread=False)
        equalized_symbols = self.equalizer.equalize(
            rx_data_symbols,
            data_channel,
            noise_variance=noise_variance,
            config=config,
        )
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            equalized_symbols = self._despread_equalized(equalized_symbols, data_mask, config)
        layer_mapping = self.layer_mapper.unmap_symbols(equalized_symbols, config.link.num_layers)
        llrs = self.demodulator.demap_symbols(equalized_symbols, noise_variance, config)
        descrambled_llrs = self.scrambler.descramble_llrs(llrs, config)
        decoded_bits = self.decoder.decode(descrambled_llrs, config)
        crc_ok = getattr(self.decoder, "last_crc_ok", None)
        return RxPayload(
            rx_waveform=np.asarray([], dtype=np.complex128) if rx_waveform is None else rx_waveform,
            rx_grid=rx_grid,
            channel_estimation=channel_estimation,
            equalized_symbols=equalized_symbols,
            layer_symbols=layer_mapping.layer_symbols,
            llrs=descrambled_llrs,
            decoded_bits=decoded_bits,
            crc_ok=crc_ok,
            dmrs_symbols=dmrs_symbols,
            plot_artifacts=channel_estimation.plot_artifacts,
        )

    @staticmethod
    def _despread_equalized(
        equalized_symbols: np.ndarray,
        data_mask: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Undo DFT spreading on equalized PUSCH symbols.

        Args:
            equalized_symbols: One-dimensional complex equalized stream with shape
                ``(num_data_re,)`` before DFT-s-OFDM de-spreading.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``;
                axis 0 is cell subcarrier index and axis 1 is OFDM symbol index.
            config: Full simulation configuration that defines scheduled symbols.

        Returns:
            One-dimensional de-spread complex symbol sequence with shape
            ``(num_data_symbols_after_despreading,)``.
        """
        despread = []
        cursor = 0
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            count = int(np.count_nonzero(data_mask[:, symbol_idx]))
            if count == 0:
                continue
            symbol_values = equalized_symbols[cursor : cursor + count]
            cursor += count
            despread.append(np.fft.ifft(symbol_values, n=count) * np.sqrt(count))
        return np.concatenate(despread) if despread else np.array([], dtype=np.complex128)
