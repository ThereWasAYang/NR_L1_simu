from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ComplexArray = np.ndarray
RealArray = np.ndarray
BitArray = np.ndarray


@dataclass
class TxPayload:
    transport_block: BitArray
    coded_bits: BitArray
    tx_symbols: ComplexArray
    resource_grid: ComplexArray
    waveform: ComplexArray
    dmrs_symbols: ComplexArray
    dmrs_mask: np.ndarray
    data_mask: np.ndarray


@dataclass
class ChannelEstimateResult:
    channel_estimate: ComplexArray
    pilot_estimates: ComplexArray
    pilot_symbol_indices: np.ndarray


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


@dataclass
class SimulationResult:
    tx: TxPayload
    rx: RxPayload
    bit_errors: int
    bit_error_rate: float
    snr_db: float
    crc_ok: bool | None = None
    interference_reports: tuple[Any, ...] = ()


@dataclass
class MultiTtiSimulationResult:
    num_ttis: int
    packet_errors: int
    block_error_rate: float
    tti_results: tuple[SimulationResult, ...]
    final_config: Any

    @property
    def last_result(self) -> SimulationResult | None:
        if not self.tti_results:
            return None
        return self.tti_results[-1]
