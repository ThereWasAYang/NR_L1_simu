from __future__ import annotations

import numpy as np
from py3gpp import (
    nrCRCEncode,
    nrCodeBlockSegmentLDPC,
    nrDLSCHInfo,
    nrLDPCEncode,
    nrRateMatchLDPC,
)

from nr_phy_simu.common.interfaces import ChannelCoder
from nr_phy_simu.config import SimulationConfig


class NrLdpcCoder(ChannelCoder):
    """
    NR LDPC coding chain:
    TB CRC -> code block segmentation -> LDPC encode -> rate matching.
    """

    def encode(self, bits: np.ndarray, config: SimulationConfig) -> np.ndarray:
        if config.link.coded_bit_capacity is None:
            raise ValueError("coded_bit_capacity must be resolved before LDPC encoding.")

        tbs = int(bits.size)
        info = nrDLSCHInfo(tbs, config.link.code_rate)
        tb_crc = nrCRCEncode(bits.astype(np.int8), info["CRC"])[:, 0].astype(np.int8)
        cbs = nrCodeBlockSegmentLDPC(tb_crc, info["BGN"])
        try:
            coded_cbs = nrLDPCEncode(cbs, info["BGN"])
        except ValueError as exc:
            if "dimension mismatch" not in str(exc):
                raise
            coded_cbs = nrLDPCEncode(cbs, info["BGN"], algo="thangaraj")
        return nrRateMatchLDPC(
            coded_cbs,
            outlen=int(config.link.coded_bit_capacity),
            rv=int(config.link.mcs.rv),
            mod=config.link.modulation,
            nLayers=config.link.num_layers,
        ).astype(np.int8)
