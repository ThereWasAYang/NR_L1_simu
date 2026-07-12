from __future__ import annotations

import numpy as np

from nr_phy_simu.config import SimulationConfig

SUBCARRIERS_PER_RB = 12


def active_bwp_num_rbs(config: SimulationConfig) -> int:
    """Resolve the active BWP bandwidth.

    Args:
        config: Full simulation configuration. ``config.bwp.num_rbs`` may be
            ``None``, in which case the active BWP spans the full carrier grid.

    Returns:
        Active BWP width in RB.
    """
    return int(config.carrier.cell_bandwidth_rbs if config.bwp.num_rbs is None else config.bwp.num_rbs)


def bwp_start_subcarrier(config: SimulationConfig) -> int:
    """Return active BWP start in cell-grid subcarrier coordinates.

    Args:
        config: Full simulation configuration containing ``bwp.start_rb``.

    Returns:
        Scalar subcarrier index relative to the full cell grid.
    """
    return int(config.bwp.start_rb) * SUBCARRIERS_PER_RB


def bwp_stop_subcarrier(config: SimulationConfig) -> int:
    """Return active BWP stop in cell-grid subcarrier coordinates.

    Args:
        config: Full simulation configuration containing active BWP settings.

    Returns:
        Exclusive scalar subcarrier index relative to the full cell grid.
    """
    return bwp_start_subcarrier(config) + active_bwp_num_rbs(config) * SUBCARRIERS_PER_RB


def user_allocation_start_subcarrier(config: SimulationConfig) -> int:
    """Return scheduled user allocation start in cell-grid subcarrier coordinates.

    Args:
        config: Full simulation configuration. ``link.prb_start`` is interpreted
            as a BWP-relative PRB index.

    Returns:
        Scalar subcarrier index relative to the full cell grid.
    """
    return (int(config.bwp.start_rb) + int(config.link.prb_start)) * SUBCARRIERS_PER_RB


def allocated_subcarriers(config: SimulationConfig) -> np.ndarray:
    """Build scheduled user subcarrier indices in full cell-grid coordinates.

    Args:
        config: Full simulation configuration containing BWP and link allocation.

    Returns:
        One-dimensional integer array with shape ``(link.num_prbs * 12,)``. Axis
        0 enumerates scheduled RE frequency positions in cell-grid coordinates.
    """
    start = user_allocation_start_subcarrier(config)
    stop = start + int(config.link.num_prbs) * SUBCARRIERS_PER_RB
    return np.arange(start, stop, dtype=int)


def bwp_center_frequency_hz(config: SimulationConfig) -> float:
    """Derive the active BWP center frequency.

    Args:
        config: Full simulation configuration containing carrier center frequency,
            SCS, cell bandwidth, and active BWP location.

    Returns:
        Active BWP center frequency in Hz. The value is derived and should not be
        configured independently.
    """
    scs_hz = float(config.carrier.subcarrier_spacing_khz) * 1e3
    cell_rbs = int(config.carrier.cell_bandwidth_rbs)
    bwp_rbs = active_bwp_num_rbs(config)
    rb_offset_from_cell_center = int(config.bwp.start_rb) + 0.5 * bwp_rbs - 0.5 * cell_rbs
    return float(config.carrier.center_frequency_hz) + rb_offset_from_cell_center * SUBCARRIERS_PER_RB * scs_hz


def ofdm_phase_compensation_frequency_hz(config: SimulationConfig) -> float:
    """Return the frequency used by OFDM phase-compensation phasors.

    Args:
        config: Full simulation configuration.

    Returns:
        RF carrier frequency ``f0`` in Hz from 38.211 clause 5.4.
    """
    return float(config.carrier.center_frequency_hz)


def ofdm_phase_compensation_vector(
    config: SimulationConfig,
    symbol_start_sample: int,
    cp_length: int,
    symbol_length: int,
    *,
    inverse: bool,
) -> np.ndarray:
    """Build the per-sample OFDM phase-compensation vector for one symbol.

    Args:
        config: Full simulation configuration containing carrier/BWP timing and
            frequency settings.
        symbol_start_sample: CP-extended symbol start sample inside the slot.
        cp_length: Cyclic-prefix length in samples for this OFDM symbol.
        symbol_length: CP-extended symbol length in samples.
        inverse: ``False`` for TX modulation and ``True`` for RX compensation.

    Returns:
        Constant complex phasor vector with shape ``(symbol_length,)``. The
        constant is referenced to the useful-symbol start as required for a
        global complex-envelope representation of 38.211 clause 5.4.
    """
    if not bool(config.bwp.phase_compensation_enabled):
        return np.ones(symbol_length, dtype=np.complex128)

    sample_rate_hz = float(config.carrier.sample_rate_effective_hz)
    slots_per_subframe = 2 ** int(config.carrier.numerology)
    slot_in_subframe = int(config.slot_index) % slots_per_subframe
    slot_start_sample = slot_in_subframe * int(config.carrier.slot_length_samples)
    useful_symbol_start = slot_start_sample + int(symbol_start_sample) + int(cp_length)
    cycles = np.remainder(
        ofdm_phase_compensation_frequency_hz(config) * useful_symbol_start / sample_rate_hz,
        1.0,
    )
    sign = 1.0 if inverse else -1.0
    phasor = np.exp(1j * sign * 2.0 * np.pi * cycles)
    return np.full(symbol_length, phasor, dtype=np.complex128)
