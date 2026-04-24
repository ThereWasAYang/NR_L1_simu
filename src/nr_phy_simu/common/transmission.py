from __future__ import annotations

from dataclasses import dataclass

from nr_phy_simu.common.mcs import McsEntry, apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.config import SimulationConfig


@dataclass(frozen=True)
class CodewordPlan:
    index: int
    rv: int
    modulation: str
    bits_per_symbol: int
    target_code_rate: float
    coded_bit_capacity: int


@dataclass(frozen=True)
class TransportBlockPlan:
    size_bits: int
    num_layers: int
    num_codewords: int
    data_re_count: int
    codewords: tuple[CodewordPlan, ...]
    mcs: McsEntry


def build_transport_block_plan(config: SimulationConfig, data_re_count: int) -> TransportBlockPlan:
    """Resolve MCS/TBS/codeword bookkeeping for one scheduled transmission.

    Args:
        config: Full simulation configuration for the scheduled link.
        data_re_count: Number of RE available for data symbols in the slot.

    Returns:
        Structured transmission plan containing MCS, codeword, and TB metadata.
    """
    mcs = apply_mcs_to_link(config)
    coded_bit_capacity = int(data_re_count * mcs.bits_per_symbol)
    config.link.coded_bit_capacity = coded_bit_capacity
    if not config.link.transport_block_size:
        config.link.transport_block_size = resolve_transport_block_size(config, data_re_count)

    num_codewords = int(config.link.num_codewords)
    if num_codewords != 1:
        raise NotImplementedError("Current PHY chain supports one active codeword per transmission.")

    codeword = CodewordPlan(
        index=int(config.scrambling.codeword_index),
        rv=int(config.link.mcs.rv),
        modulation=mcs.modulation,
        bits_per_symbol=mcs.bits_per_symbol,
        target_code_rate=mcs.target_code_rate,
        coded_bit_capacity=coded_bit_capacity,
    )
    return TransportBlockPlan(
        size_bits=int(config.link.transport_block_size),
        num_layers=int(config.link.num_layers),
        num_codewords=num_codewords,
        data_re_count=int(data_re_count),
        codewords=(codeword,),
        mcs=mcs,
    )
