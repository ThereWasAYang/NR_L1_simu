from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import (
    ChannelEstimator,
    Demodulator,
    FrequencyExtractor,
    MimoEqualizer,
    ReceiverDataProcessor,
    ReceiverProcessingStage,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE
from nr_phy_simu.common.types import ReceiverDataProcessingResult, ReceiverProcessingContext
from nr_phy_simu.config import SimulationConfig


class ReceiverDataProcessorPipeline(ReceiverDataProcessor):
    """Composable receiver processor made from arbitrary user-defined stages."""

    def __init__(self, stages: tuple[ReceiverProcessingStage, ...] | list[ReceiverProcessingStage]) -> None:
        if not stages:
            raise ValueError("ReceiverDataProcessorPipeline requires at least one stage.")
        self.stages = tuple(stages)

    def process(
        self,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> ReceiverDataProcessingResult:
        """Run all configured stages and return the final LLR bundle.

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
            Data-processing result collected from the final context.
        """
        context = ReceiverProcessingContext(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )
        for stage in self.stages:
            context = stage.process(context)
        if context.llrs.numel() == 0:
            raise ValueError("ReceiverDataProcessorPipeline finished without producing LLRs.")
        return ReceiverDataProcessingResult(
            llrs=context.llrs,
            channel_estimation=context.channel_estimation,
            equalized_symbols=context.equalized_symbols,
            layer_symbols=context.layer_symbols,
            plot_artifacts=context.plot_artifacts,
        )


class ChannelEstimationStage(ReceiverProcessingStage):
    """Pipeline stage that runs a ``ChannelEstimator``."""

    def __init__(self, estimator: ChannelEstimator) -> None:
        self.estimator = estimator

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        context.channel_estimation = self.estimator.estimate(
            context.rx_grid,
            context.dmrs_symbols,
            context.dmrs_mask,
            context.config,
        )
        context.plot_artifacts = context.plot_artifacts + context.channel_estimation.plot_artifacts
        return context


class DataExtractionStage(ReceiverProcessingStage):
    """Pipeline stage that extracts data REs from RX grid and channel estimate."""

    def __init__(self, extractor: FrequencyExtractor, despread: bool = False) -> None:
        self.extractor = extractor
        self.despread = despread

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        context.rx_data_symbols = self.extractor.extract(
            context.rx_grid,
            context.data_mask,
            context.config,
            despread=self.despread,
        )
        if context.channel_estimation is not None:
            context.data_channel = self.extractor.extract(
                context.channel_estimation.channel_estimate,
                context.data_mask,
                context.config,
                despread=self.despread,
            )
        return context


class EqualizationStage(ReceiverProcessingStage):
    """Pipeline stage that runs a ``MimoEqualizer``."""

    def __init__(self, equalizer: MimoEqualizer) -> None:
        self.equalizer = equalizer

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        context.equalized_symbols = self.equalizer.equalize(
            context.rx_data_symbols,
            context.data_channel,
            noise_variance=context.noise_variance,
            config=context.config,
        )
        return context


class TransformPrecodingDespreadStage(ReceiverProcessingStage):
    """Pipeline stage that de-spreads DFT-s-OFDM PUSCH equalized symbols."""

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        config = context.config
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            context.equalized_symbols = ThreeStageReceiverDataProcessor._despread_equalized(
                context.equalized_symbols,
                context.data_mask,
                config,
            )
        return context


class LayerDemappingStage(ReceiverProcessingStage):
    """Pipeline stage that builds per-layer views from equalized symbols."""

    def __init__(self, layer_mapper: LayerMapper | None = None) -> None:
        self.layer_mapper = layer_mapper or LayerMapper()

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        layer_mapping = self.layer_mapper.unmap_symbols(
            context.equalized_symbols,
            context.config.link.num_layers,
        )
        context.layer_symbols = layer_mapping.layer_symbols
        return context


class DemodulationStage(ReceiverProcessingStage):
    """Pipeline stage that runs a ``Demodulator``."""

    def __init__(self, demodulator: Demodulator) -> None:
        self.demodulator = demodulator

    def process(self, context: ReceiverProcessingContext) -> ReceiverProcessingContext:
        context.llrs = self.demodulator.demap_symbols(
            context.equalized_symbols,
            context.noise_variance,
            context.config,
        )
        return context


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
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
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
        equalized_symbols: torch.Tensor,
        data_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> torch.Tensor:
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
            count = int(torch.count_nonzero(data_mask[:, symbol_idx]).item())
            if count == 0:
                continue
            symbol_values = equalized_symbols[cursor : cursor + count]
            cursor += count
            despread.append(
                torch.fft.ifft(symbol_values, n=count)
                * torch.sqrt(torch.tensor(float(count), dtype=torch.float64, device=symbol_values.device))
            )
        return torch.cat(despread) if despread else torch.empty(0, dtype=COMPLEX_DTYPE, device=equalized_symbols.device)
