from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nr_phy_simu.config import HarqConfig


@dataclass
class HarqProcessState:
    process_id: int
    rv_index: int = 0
    retransmissions: int = 0
    active_transport_block: np.ndarray | None = None


@dataclass(frozen=True)
class HarqTransmission:
    process_id: int
    transport_block: np.ndarray
    rv: int
    is_retransmission: bool


@dataclass
class HarqManager:
    config: HarqConfig
    processes: dict[int, HarqProcessState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.processes = {
            process_id: HarqProcessState(process_id=process_id)
            for process_id in range(int(self.config.num_processes))
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def schedule(self, tti_index: int, tbs_bits: int, rng: np.random.Generator) -> HarqTransmission:
        """Schedule one HARQ process for the current TTI.

        Args:
            tti_index: Scalar TTI index.
            tbs_bits: Scalar transport-block size in bits.
            rng: Random generator used to create a new transport block.

        Returns:
            HARQ transmission descriptor. ``transport_block`` is a one-dimensional
            bit array with shape ``(tbs_bits,)``; axis 0 is TB bit index.
        """
        process_id = int(tti_index % int(self.config.num_processes))
        state = self.processes[process_id]
        rv_sequence = tuple(int(rv) for rv in self.config.rv_sequence)

        if state.active_transport_block is None:
            transport_block = rng.integers(0, 2, size=int(tbs_bits), dtype=np.int8)
            state.active_transport_block = transport_block
            state.rv_index = 0
            state.retransmissions = 0
            return HarqTransmission(
                process_id=process_id,
                transport_block=transport_block,
                rv=rv_sequence[0],
                is_retransmission=False,
            )

        return HarqTransmission(
            process_id=process_id,
            transport_block=state.active_transport_block.copy(),
            rv=rv_sequence[min(state.rv_index, len(rv_sequence) - 1)],
            is_retransmission=True,
        )

    def update(self, process_id: int, crc_ok: bool | None) -> None:
        """Update HARQ process state after one transmission result.

        Args:
            process_id: Scalar HARQ process identifier.
            crc_ok: Scalar CRC result. ``None`` clears the process in bypass mode.
        """
        if crc_ok is None:
            self.processes[int(process_id)].active_transport_block = None
            self.processes[int(process_id)].rv_index = 0
            self.processes[int(process_id)].retransmissions = 0
            return

        state = self.processes[int(process_id)]
        if crc_ok:
            state.active_transport_block = None
            state.rv_index = 0
            state.retransmissions = 0
            return

        state.retransmissions += 1
        if state.retransmissions > int(self.config.max_retransmissions):
            state.active_transport_block = None
            state.rv_index = 0
            state.retransmissions = 0
            return

        state.rv_index = min(state.rv_index + 1, len(self.config.rv_sequence) - 1)
