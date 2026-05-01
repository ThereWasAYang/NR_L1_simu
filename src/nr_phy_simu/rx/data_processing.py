from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import (
    ChannelEstimator,
    Demodulator,
    FrequencyExtractor,
    MimoEqualizer,
    ReceiverDataProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.types import ReceiverDataProcessingResult
from nr_phy_simu.config import SimulationConfig


class ThreeStageReceiverDataProcessor(ReceiverDataProcessor):
    """Default processor that keeps the classic estimate/equalize/demod stages."""

    def __init__(
        self,
        extractor: FrequencyExtractor,
        estimator: ChannelEstimator,
        equalizer: MimoEqualizer,
        demodulator: Demodulator,
        layer_mapper: LayerMapper | None = None,
    ) -> None:
        self.extractor = extractor
        self.estimator = estimator
        self.equalizer = equalizer
        self.demodulator = demodulator
        self.layer_mapper = layer_mapper or LayerMapper()

    def process(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        data_mask: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> ReceiverDataProcessingResult:
        """Run channel estimation, equalization, and demodulation.

        Args:
            rx_grid: Received frequency-domain grid with shape
                ``(num_rx_ant, num_subcarriers, num_symbols)``.
            dmrs_symbols: One-dimensional transmitted DMRS sequence with shape
                ``(num_dmrs_re,)`` in mapper RE order.
            dmrs_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.
            noise_variance: Scalar receiver noise variance.
            config: Full simulation configuration for receiver-side processing.

        Returns:
            Data-processing result whose LLRs are still scrambled and must be
            passed through the configured data descrambler.
        """
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
        return ReceiverDataProcessingResult(
            llrs=llrs,
            channel_estimation=channel_estimation,
            equalized_symbols=equalized_symbols,
            layer_symbols=layer_mapping.layer_symbols,
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
            data_mask: Boolean mask with shape ``(num_subcarriers, num_symbols)``.
            config: Full simulation configuration that defines scheduled symbols.

        Returns:
            One-dimensional de-spread complex symbol sequence.
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
