from __future__ import annotations

from dataclasses import dataclass

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
class RxPayload:
    rx_waveform: ComplexArray
    rx_grid: ComplexArray
    channel_estimate: ComplexArray
    equalized_symbols: ComplexArray
    llrs: RealArray
    decoded_bits: BitArray
    dmrs_symbols: ComplexArray
    pilot_estimates: ComplexArray


@dataclass
class SimulationResult:
    tx: TxPayload
    rx: RxPayload
    bit_errors: int
    bit_error_rate: float
    snr_db: float
