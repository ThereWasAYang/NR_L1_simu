from __future__ import annotations

from dataclasses import replace

import torch

from nr_phy_simu.common.runtime_context import SimulationRuntimeContext, set_runtime_context
from nr_phy_simu.common.torch_utils import BIT_DTYPE, REAL_DTYPE, as_complex_tensor, as_int_tensor
from nr_phy_simu.common.transmission import TransportBlockPlan, build_transport_block_plan
from nr_phy_simu.common.types import SimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.scenarios.component_factory import (
    DefaultSimulationComponentFactory,
    SimulationComponentFactory,
    build_receiver,
    build_transmitter,
)
from nr_phy_simu.scenarios.interference import InterferenceMixer
from nr_phy_simu.tx.chain import Transmitter


class SharedChannelSimulation:
    def __init__(
        self,
        config: SimulationConfig,
        transmitter: Transmitter | None = None,
        receiver: Receiver | None = None,
        channel=None,
        component_factory: SimulationComponentFactory | None = None,
        runtime_context: SimulationRuntimeContext | None = None,
    ) -> None:
        self.config = config
        self.runtime_context = runtime_context or SimulationRuntimeContext()
        self.component_factory = component_factory or DefaultSimulationComponentFactory()
        self.components = self.component_factory.create_components(config)
        self.dmrs_generator = self.components.shared.dmrs_generator
        self.mapper = self.components.transmitter.mapper
        self.transmitter = transmitter or build_transmitter(self.components)
        self.receiver = receiver or build_receiver(self.components)
        self.channel = channel or self.component_factory.create_channel_factory().create(config)
        self.interference_mixer = InterferenceMixer(self.component_factory)

    def run(
        self,
        transport_block_override: torch.Tensor | None = None,
        harq_process_id: int | None = None,
        harq_retransmission: bool | None = None,
    ) -> SimulationResult:
        """Run one end-to-end shared-channel simulation for a single TTI."""
        self.runtime_context.clear()
        set_runtime_context(self.runtime_context)
        if harq_process_id is not None:
            self.runtime_context.set("harq", "process_id", harq_process_id)
            self.runtime_context.set("harq", "is_retransmission", bool(harq_retransmission))

        data_re = self.mapper.count_data_re(self.config)
        transport_plan = build_transport_block_plan(self.config, data_re)
        self.runtime_context.set("transmission", "transport_plan", transport_plan)
        self.runtime_context.set("harq", "rv", int(transport_plan.codewords[0].rv))

        if transport_block_override is not None:
            transport_block = as_int_tensor(transport_block_override, dtype=BIT_DTYPE).reshape(-1)
        else:
            generator = torch.Generator().manual_seed(self.config.random_seed)
            transport_block = torch.randint(
                0,
                2,
                (int(transport_plan.size_bits),),
                dtype=BIT_DTYPE,
                generator=generator,
            )

        if self._uses_frequency_domain_channel():
            if self.config.interference.enabled:
                raise NotImplementedError("Frequency-domain direct channel does not currently support interference injection.")
            tx_payload = self.transmitter.build_slot_payload(transport_block, self.config)
            rx_grid, channel_info = self.channel.propagate_grid(tx_payload.resource_grid, self.config)
            rx_payload = self.receiver.receive_from_grid(
                rx_grid=rx_grid,
                dmrs_symbols=tx_payload.dmrs_symbols,
                dmrs_mask=tx_payload.dmrs_mask,
                data_mask=tx_payload.data_mask,
                noise_variance=float(channel_info["noise_variance"]),
                config=self.config,
                rx_waveform=None,
            )
            tx_payload = replace(
                tx_payload,
                waveform=torch.empty(
                    (int(self.config.link.num_tx_ant), 0),
                    dtype=tx_payload.resource_grid.dtype,
                    device=tx_payload.resource_grid.device,
                ),
            )
            return self._build_result(
                tx_payload=tx_payload,
                rx_payload=rx_payload,
                transport_block=transport_block,
                transport_plan=transport_plan,
                channel_info=channel_info,
                interference_reports=(),
                harq_process_id=harq_process_id,
                harq_retransmission=harq_retransmission,
            )

        tx_payload = self.transmitter.transmit(transport_block, self.config)
        rx_waveform, channel_info = self.channel.propagate(tx_payload.waveform, self.config)
        rx_waveform = as_complex_tensor(rx_waveform)
        rx_waveform, interference_reports = self.interference_mixer.apply(
            rx_waveform,
            noise_variance=float(channel_info["noise_variance"]),
            config=self.config,
        )
        rx_payload = self.receiver.receive(
            rx_waveform=rx_waveform,
            dmrs_symbols=tx_payload.dmrs_symbols,
            dmrs_mask=tx_payload.dmrs_mask,
            data_mask=tx_payload.data_mask,
            noise_variance=float(channel_info["noise_variance"]),
            config=self.config,
        )
        return self._build_result(
            tx_payload=tx_payload,
            rx_payload=rx_payload,
            transport_block=transport_block,
            transport_plan=transport_plan,
            channel_info=channel_info,
            interference_reports=interference_reports,
            harq_process_id=harq_process_id,
            harq_retransmission=harq_retransmission,
        )

    def _build_result(
        self,
        tx_payload,
        rx_payload,
        transport_block: torch.Tensor,
        transport_plan: TransportBlockPlan,
        channel_info: dict,
        interference_reports: tuple,
        harq_process_id: int | None,
        harq_retransmission: bool | None,
    ) -> SimulationResult:
        """Build the final simulation result object from chain outputs."""
        if self.config.simulation.bypass_channel_coding:
            reference_bits = tx_payload.coded_bits
            decoded = rx_payload.decoded_bits[: reference_bits.numel()]
        else:
            reference_bits = transport_block
            decoded = rx_payload.decoded_bits[: reference_bits.numel()]
        bit_errors = int(torch.sum(decoded != reference_bits).item())
        ber = bit_errors / reference_bits.numel()
        evm_percent, evm_snr_linear = self._compute_evm_metrics(tx_payload.tx_symbols, rx_payload.equalized_symbols)
        return SimulationResult(
            tx=tx_payload,
            rx=rx_payload,
            bit_errors=bit_errors,
            bit_error_rate=ber,
            snr_db=float(channel_info.get("snr_db", self.config.snr_db)),
            transport_plan=transport_plan,
            crc_ok=rx_payload.crc_ok,
            evm_percent=evm_percent,
            evm_snr_linear=evm_snr_linear,
            harq_process_id=harq_process_id,
            harq_rv=int(transport_plan.codewords[0].rv),
            harq_retransmission=harq_retransmission,
            interference_reports=interference_reports,
        )

    def _uses_frequency_domain_channel(self) -> bool:
        """Return whether the configured channel bypasses time-domain processing."""
        return self.config.channel.model.upper() == "EXTERNAL_FREQRESP_FD"

    @staticmethod
    def _compute_evm_metrics(
        reference_symbols: torch.Tensor,
        equalized_symbols: torch.Tensor,
    ) -> tuple[float | None, float | None]:
        """Compute mean EVM and derived EVM-SNR from equalized symbols."""
        if reference_symbols.numel() == 0 or equalized_symbols.numel() == 0:
            return None, None

        count = min(reference_symbols.numel(), equalized_symbols.numel())
        reference = as_complex_tensor(reference_symbols[:count])
        measured = as_complex_tensor(equalized_symbols[:count], device=reference.device)
        symbol_magnitude = torch.clamp(torch.abs(reference), min=1e-12)
        error_vector = torch.abs(measured - reference)
        evm_ratio = error_vector / symbol_magnitude
        mean_evm_ratio = float(torch.mean(evm_ratio.to(dtype=REAL_DTYPE)).item())
        evm_percent = mean_evm_ratio * 100.0
        evm_snr_linear = float(1.0 / max(mean_evm_ratio**2, 1e-24))
        return evm_percent, evm_snr_linear
