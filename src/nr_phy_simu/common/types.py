from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ComplexArray = torch.Tensor
RealArray = torch.Tensor
BitArray = torch.Tensor


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
    transport_block: BitArray
    coded_bits: BitArray
    tx_symbols: ComplexArray
    resource_grid: ComplexArray
    waveform: ComplexArray
    dmrs_symbols: ComplexArray
    dmrs_mask: torch.Tensor
    data_mask: torch.Tensor
    layer_symbols: tuple[ComplexArray, ...] = ()


@dataclass
class ChannelEstimateResult:
    channel_estimate: ComplexArray
    pilot_estimates: ComplexArray
    pilot_symbol_indices: torch.Tensor
    plot_artifacts: tuple[PlotArtifact, ...] = ()


@dataclass
class RxPayload:
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
        if not self.tti_results:
            return None
        return self.tti_results[-1]
