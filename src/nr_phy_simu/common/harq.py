from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nr_phy_simu.common.torch_utils import BIT_DTYPE, as_int_tensor
from nr_phy_simu.config import HarqConfig


@dataclass
class HarqProcessState:
    process_id: int
    rv_index: int = 0
    retransmissions: int = 0
    active_transport_block: torch.Tensor | None = None


@dataclass(frozen=True)
class HarqTransmission:
    process_id: int
    transport_block: torch.Tensor
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

    def schedule(
        self,
        tti_index: int,
        tbs_bits: int,
        generator: object | None = None,
        rng: object | None = None,
    ) -> HarqTransmission:
        """Schedule one HARQ process for the current TTI."""
        generator = generator if generator is not None else rng
        process_id = int(tti_index % int(self.config.num_processes))
        state = self.processes[process_id]
        rv_sequence = tuple(int(rv) for rv in self.config.rv_sequence)

        if state.active_transport_block is None:
            torch_generator = generator if isinstance(generator, torch.Generator) else None
            transport_block = torch.randint(
                0,
                2,
                (int(tbs_bits),),
                dtype=BIT_DTYPE,
                generator=torch_generator,
            )
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
            transport_block=as_int_tensor(state.active_transport_block, dtype=BIT_DTYPE).clone(),
            rv=rv_sequence[min(state.rv_index, len(rv_sequence) - 1)],
            is_retransmission=True,
        )

    def update(self, process_id: int, crc_ok: bool | None) -> None:
        """Update HARQ process state after one transmission result."""
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
