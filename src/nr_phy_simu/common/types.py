from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ComplexArray = np.ndarray
RealArray = np.ndarray
BitArray = np.ndarray


@dataclass
class PlotArtifact:
    name: str
    values: Any
    title: str | None = None
    plot_type: str = "magnitude"
    x: Any = None
    xlabel: str = "Index"
    ylabel: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class TxPayload:
    """Transmitter-side buffers for one TTI.

    Shape conventions:
        ``transport_block``: ``(tbs_bits,)``.
        ``coded_bits``: ``(coded_bit_capacity,)``.
        ``tx_symbols``: ``(num_data_symbols,)``.
        ``resource_grid``: ``(num_subcarriers, num_symbols)``.
        ``waveform``: ``(slot_samples,)`` or ``(num_tx_ant, slot_samples)``.
        ``dmrs_symbols``: ``(num_dmrs_re,)``.
        ``dmrs_mask`` and ``data_mask``: ``(num_subcarriers, num_symbols)``.
    """
    transport_block: BitArray
    coded_bits: BitArray
    tx_symbols: ComplexArray
    resource_grid: ComplexArray
    waveform: ComplexArray
    dmrs_symbols: ComplexArray
    dmrs_mask: np.ndarray
    data_mask: np.ndarray
    layer_symbols: tuple[ComplexArray, ...] = ()


@dataclass
class ChannelEstimateResult:
    """Channel-estimation buffers for one TTI.

    Shape conventions:
        ``channel_estimate``: ``(num_rx_ant, num_subcarriers, num_symbols)``.
        ``pilot_estimates``: ``(num_rx_ant, num_dmrs_re)``.
        ``pilot_symbol_indices``: ``(num_dmrs_re,)``.
    """
    channel_estimate: ComplexArray
    pilot_estimates: ComplexArray
    pilot_symbol_indices: np.ndarray
    plot_artifacts: tuple[PlotArtifact, ...] = ()


@dataclass
class RxPayload:
    """Receiver-side buffers for one TTI.

    Shape conventions:
        ``rx_waveform``: ``(slot_samples,)`` or ``(num_rx_ant, slot_samples)``.
        ``rx_grid``: ``(num_rx_ant, num_subcarriers, num_symbols)``.
        ``equalized_symbols``: ``(num_data_symbols,)``.
        ``llrs``: ``(coded_bit_capacity,)``.
        ``decoded_bits``: ``(tbs_bits,)``.
        ``dmrs_symbols``: ``(num_dmrs_re,)``.
    """
    rx_waveform: ComplexArray
    rx_grid: ComplexArray
    channel_estimation: ChannelEstimateResult
    equalized_symbols: ComplexArray
    llrs: RealArray
    decoded_bits: BitArray
    crc_ok: bool | None
    dmrs_symbols: ComplexArray
    layer_symbols: tuple[ComplexArray, ...] = ()
    plot_artifacts: tuple[PlotArtifact, ...] = ()


@dataclass
class SimulationResult:
    tx: TxPayload
    rx: RxPayload
    bit_errors: int
    bit_error_rate: float
    snr_db: float
    transport_plan: Any | None = None
    crc_ok: bool | None = None
    evm_percent: float | None = None
    evm_snr_linear: float | None = None
    harq_process_id: int | None = None
    harq_rv: int | None = None
    harq_retransmission: bool | None = None
    interference_reports: tuple[Any, ...] = ()


@dataclass
class MultiTtiSimulationResult:
    num_ttis: int
    packet_errors: int
    block_error_rate: float
    average_evm_percent: float | None
    average_evm_snr_linear: float | None
    tti_results: tuple[SimulationResult, ...]
    final_config: Any

    @property
    def last_result(self) -> SimulationResult | None:
        """Return the final TTI result.

        Args:
            None.

        Returns:
            Last ``SimulationResult`` in ``tti_results`` or ``None`` when empty.
        """
        if not self.tti_results:
            return None
        return self.tti_results[-1]
