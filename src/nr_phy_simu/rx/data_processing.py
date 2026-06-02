from __future__ import annotations

import numpy as np

from nr_phy_simu.common.bwp import allocated_subcarriers
from nr_phy_simu.common.interfaces import (
    ChannelEstimator,
    Demodulator,
    FrequencyExtractor,
    MimoEqualizer,
    ReceiverDataProcessor,
    ReceiverProcessingStage,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
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
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        data_mask: np.ndarray,
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
        _ensure_user_allocation(context)
        for stage in self.stages:
            context = stage.process(context)
        if context.llrs.size == 0:
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
        _ensure_user_allocation(context)
        context.channel_estimation = self.estimator.estimate(
            context.rx_user_grid,
            context.dmrs_symbols,
            context.dmrs_mask_user,
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
        _ensure_user_allocation(context)
        context.rx_data_symbols = self.extractor.extract(
            context.rx_user_grid,
            context.data_mask_user,
            context.config,
            despread=self.despread,
        )
        if context.channel_estimation is not None:
            context.data_channel = self.extractor.extract(
                context.channel_estimation.channel_estimate,
                context.data_mask_user,
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
            _ensure_user_allocation(context)
            context.equalized_symbols = ThreeStageReceiverDataProcessor._despread_equalized(
                context.equalized_symbols,
                context.data_mask_user,
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
        rx_user_grid, dmrs_mask_user, data_mask_user, _user_subcarriers = extract_user_allocation(
            rx_grid,
            dmrs_mask,
            data_mask,
            config,
        )
        channel_estimation = self.estimator.estimate(rx_user_grid, dmrs_symbols, dmrs_mask_user, config)
        rx_data_symbols = self.extractor.extract(rx_user_grid, data_mask_user, config, despread=False)
        data_channel = self.extractor.extract(channel_estimation.channel_estimate, data_mask_user, config, despread=False)
        equalized_symbols = self.equalizer.equalize(
            rx_data_symbols,
            data_channel,
            noise_variance=noise_variance,
            config=config,
        )
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            equalized_symbols = self._despread_equalized(equalized_symbols, data_mask_user, config)
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


def extract_user_allocation(
    rx_grid: np.ndarray,
    dmrs_mask: np.ndarray,
    data_mask: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop full-cell RX data and masks to the scheduled user PRB allocation.

    Args:
        rx_grid: Full-cell received grid with shape
            ``(num_rx_ant, num_subcarriers, num_symbols)``.
        dmrs_mask: Full-cell DMRS mask with shape ``(num_subcarriers, num_symbols)``.
        data_mask: Full-cell data mask with shape ``(num_subcarriers, num_symbols)``.
        config: Full simulation configuration that provides PRB start and width.

    Returns:
        Tuple ``(rx_user_grid, dmrs_mask_user, data_mask_user, user_subcarriers)``.
        Shapes are ``(num_rx_ant, num_user_subcarriers, num_symbols)``,
        ``(num_user_subcarriers, num_symbols)``,
        ``(num_user_subcarriers, num_symbols)`` and ``(num_user_subcarriers,)``.
    """
    if rx_grid.ndim != 3:
        raise ValueError(
            "Receiver data processing expects rx_grid shape "
            "(num_rx_ant, num_subcarriers, num_symbols)."
        )
    if dmrs_mask.ndim != 2 or data_mask.ndim != 2:
        raise ValueError("DMRS and data masks must have shape (num_subcarriers, num_symbols).")

    user_subcarriers = allocated_subcarriers(config)
    return (
        rx_grid[:, user_subcarriers, :],
        dmrs_mask[user_subcarriers, :],
        data_mask[user_subcarriers, :],
        user_subcarriers,
    )


def _ensure_user_allocation(context: ReceiverProcessingContext) -> None:
    """Populate user-allocation views in a receiver processing context.

    Args:
        context: Mutable receiver-processing context. On entry, ``rx_grid`` has
            shape ``(num_rx_ant, num_subcarriers, num_symbols)`` and masks are
            full-cell masks. On return, the user-allocation fields are populated.

    Returns:
        None. The context is mutated in place.
    """
    if context.rx_user_grid.size and context.dmrs_mask_user.size and context.data_mask_user.size:
        return
    (
        context.rx_user_grid,
        context.dmrs_mask_user,
        context.data_mask_user,
        context.user_subcarriers,
    ) = extract_user_allocation(
        context.rx_grid,
        context.dmrs_mask,
        context.data_mask,
        context.config,
    )
