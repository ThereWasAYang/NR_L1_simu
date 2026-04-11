from __future__ import annotations

import numpy as np
from py3gpp import (
    nrCRCDecode,
    nrCodeBlockDesegmentLDPC,
    nrDLSCHInfo,
    nrLDPCDecode,
    nrRateRecoverLDPC,
)

from nr_phy_simu.common.interfaces import ChannelDecoder
from nr_phy_simu.config import SimulationConfig


class NrLdpcDecoder(ChannelDecoder):
    def __init__(self) -> None:
        self.last_crc_ok: bool | None = None

    def decode(self, llrs: np.ndarray, config: SimulationConfig) -> np.ndarray:
        tbs = int(config.link.transport_block_size or 0)
        if tbs <= 0:
            raise ValueError("transport_block_size must be resolved before LDPC decoding.")

        info = nrDLSCHInfo(tbs, config.link.code_rate)
        recovered = nrRateRecoverLDPC(
            llrs,
            trblklen=tbs,
            R=config.link.code_rate,
            rv=int(config.link.mcs.rv),
            mod=config.link.modulation,
            nLayers=config.link.num_layers,
        )
        decoded_cbs, _ = nrLDPCDecode(recovered, info["BGN"], maxNumIter=25)
        tb_with_crc, _ = nrCodeBlockDesegmentLDPC(decoded_cbs, info["BGN"], tbs + info["L"])
        decoded, crc_error = nrCRCDecode(tb_with_crc.astype(np.int8), info["CRC"])
        self.last_crc_ok = bool(crc_error == 0)
        return np.asarray(decoded).reshape(-1)[:tbs].astype(np.int8)
