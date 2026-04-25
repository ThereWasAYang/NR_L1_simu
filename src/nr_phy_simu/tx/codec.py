from __future__ import annotations

import torch
from py3gpp import nrCRCEncode, nrCodeBlockSegmentLDPC

from nr_phy_simu.common.interfaces import ChannelCoder
from nr_phy_simu.common.torch_utils import BIT_DTYPE, as_int_tensor, to_numpy
from nr_phy_simu.common.ulsch_ldpc import (
    encode_ldpc_codeblocks,
    get_ulsch_ldpc_info,
    rate_match_ulsch_ldpc,
)
from nr_phy_simu.config import SimulationConfig


class NrLdpcCoder(ChannelCoder):
    """
    NR LDPC coding chain:
    TB CRC -> code block segmentation -> LDPC encode -> rate matching.
    """

    def encode(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Encode one transport block with the NR UL-SCH LDPC chain."""
        if config.link.coded_bit_capacity is None:
            raise ValueError("coded_bit_capacity must be resolved before LDPC encoding.")

        bits = as_int_tensor(bits, dtype=BIT_DTYPE)
        tbs = int(bits.numel())
        info = get_ulsch_ldpc_info(tbs, config.link.code_rate)
        tb_crc = nrCRCEncode(to_numpy(bits), info.crc)[:, 0].astype("int8")
        cbs = nrCodeBlockSegmentLDPC(tb_crc, info.base_graph)
        coded_cbs = encode_ldpc_codeblocks(cbs, info.base_graph)
        coded = rate_match_ulsch_ldpc(
            coded_cbs,
            out_length=int(config.link.coded_bit_capacity),
            rv=int(config.link.mcs.rv),
            modulation=config.link.modulation,
            num_layers=config.link.num_layers,
        )
        return coded.to(dtype=BIT_DTYPE)


class RandomBitCoder(ChannelCoder):
    """
    Bypass mode for link-level EVM or waveform studies.

    When channel coding is disabled, generate a deterministic pseudo-random
    bitstream with the same length as the coded-bit capacity and feed it
    directly into scrambling/modulation.
    """

    def encode(self, bits: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Generate deterministic pseudo-random coded bits in bypass mode."""
        del bits
        if config.link.coded_bit_capacity is None:
            raise ValueError("coded_bit_capacity must be resolved before bypass coding.")

        generator = torch.Generator().manual_seed(config.random_seed + 1000)
        return torch.randint(0, 2, (int(config.link.coded_bit_capacity),), dtype=BIT_DTYPE, generator=generator)
