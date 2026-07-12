from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from nr_phy_simu.common.bwp import allocated_subcarriers, bwp_center_frequency_hz
from nr_phy_simu.common.ofdm import time_to_frequency_noise_variance
from nr_phy_simu.common.runtime_context import SimulationRuntimeContext, set_runtime_context
from nr_phy_simu.common.transmission import TransportBlockPlan, build_transport_block_plan
from nr_phy_simu.common.types import SimulationResult
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.scenarios.interference import InterferenceMixer
from nr_phy_simu.scenarios.component_factory import (
    DefaultSimulationComponentFactory,
    SimulationComponentFactory,
    build_receiver,
    build_transmitter,
)
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
        self.last_run_config: SimulationConfig | None = None
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
        transport_block_override: np.ndarray | None = None,
        harq_process_id: int | None = None,
        harq_retransmission: bool | None = None,
    ) -> SimulationResult:
        """Run one end-to-end shared-channel simulation for a single TTI.

        Args:
            transport_block_override: Optional one-dimensional bit array with shape
                ``(tbs_bits,)``. When provided, axis 0 is transport-block bit index
                and replaces the random TB generator.
            harq_process_id: Optional scalar HARQ process identifier stored in runtime context.
            harq_retransmission: Optional scalar flag indicating whether this TTI is a retransmission.

        Returns:
            Structured simulation result containing TX/RX buffers and KPIs.
        """
        # Keep the caller-owned configuration immutable.  The PHY chain still
        # consumes resolved MCS/TBS fields through a private per-run view.
        config = deepcopy(self.config)
        self.last_run_config = config
        self.runtime_context.clear()
        set_runtime_context(self.runtime_context)
        self.runtime_context.set("bwp", "center_frequency_hz", bwp_center_frequency_hz(config))
        self.runtime_context.set("bwp", "allocated_subcarriers", allocated_subcarriers(config))
        if harq_process_id is not None:
            self.runtime_context.set("harq", "process_id", harq_process_id)
            self.runtime_context.set("harq", "is_retransmission", bool(harq_retransmission))
        data_re = self.mapper.count_data_re(config)
        transport_plan = build_transport_block_plan(config, data_re)
        self.runtime_context.set("transmission", "transport_plan", transport_plan)
        self.runtime_context.set("harq", "rv", int(transport_plan.codewords[0].rv))

        rng = np.random.default_rng(config.random_seed)
        transport_block = (
            np.asarray(transport_block_override, dtype=np.int8).reshape(-1)
            if transport_block_override is not None
            else rng.integers(0, 2, size=int(transport_plan.size_bits), dtype=np.int8)
        )
        if self._uses_frequency_domain_channel(config):
            if config.interference.enabled:
                raise NotImplementedError("Frequency-domain direct channel does not currently support interference injection.")
            tx_payload = self.transmitter.build_slot_payload(transport_block, config)
            rx_grid, channel_info = self.channel.propagate_grid(tx_payload.resource_grid, config)
            rx_payload = self.receiver.receive_from_grid(
                rx_grid=rx_grid,
                dmrs_symbols=tx_payload.dmrs_symbols,
                dmrs_mask=tx_payload.dmrs_mask,
                data_mask=tx_payload.data_mask,
                noise_variance=float(channel_info["noise_variance"]),
                config=config,
                rx_waveform=None,
            )
            tx_payload = replace(
                tx_payload,
                waveform=np.empty((int(config.link.num_tx_ant), 0), dtype=np.complex128),
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
                config=config,
            )

        tx_payload = self.transmitter.transmit(transport_block, config)
        rx_waveform, channel_info = self.channel.propagate(tx_payload.waveform, config)
        rx_waveform, interference_reports = self.interference_mixer.apply(
            rx_waveform,
            noise_variance=float(channel_info["noise_variance"]),
            config=config,
        )
        receiver_noise_variance = time_to_frequency_noise_variance(
            float(channel_info["noise_variance"]),
            config,
        )
        rx_payload = self.receiver.receive(
            rx_waveform=rx_waveform,
            dmrs_symbols=tx_payload.dmrs_symbols,
            dmrs_mask=tx_payload.dmrs_mask,
            data_mask=tx_payload.data_mask,
            noise_variance=receiver_noise_variance,
            config=config,
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
            config=config,
        )

    def _build_result(
        self,
        tx_payload,
        rx_payload,
        transport_block: np.ndarray,
        transport_plan: TransportBlockPlan,
        channel_info: dict,
        interference_reports: tuple,
        harq_process_id: int | None,
        harq_retransmission: bool | None,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Build the final simulation result object from chain outputs.

        Args:
            tx_payload: TX payload whose arrays follow ``TxPayload`` shape conventions.
            rx_payload: RX payload whose arrays follow ``RxPayload`` shape conventions.
            transport_block: One-dimensional reference TB bit array with shape
                ``(tbs_bits,)``.
            transport_plan: Transport-block plan with scalar TBS/codeword metadata.
            channel_info: Channel metadata dictionary, including scalar noise variance.
            interference_reports: Tuple of per-interferer scalar report objects.
            harq_process_id: Optional scalar HARQ process identifier.
            harq_retransmission: Optional scalar retransmission flag.

        Returns:
            Structured simulation result containing payload arrays and scalar KPIs.
        """
        if config.simulation.bypass_channel_coding:
            reference_bits = tx_payload.coded_bits
            decoded = rx_payload.decoded_bits[: reference_bits.size]
        else:
            reference_bits = transport_block
            decoded = rx_payload.decoded_bits[: reference_bits.size]
        bit_errors = int(np.sum(decoded != reference_bits))
        ber = bit_errors / reference_bits.size
        evm_percent, evm_snr_linear = self._compute_evm_metrics(tx_payload.tx_symbols, rx_payload.equalized_symbols)
        return SimulationResult(
            tx=tx_payload,
            rx=rx_payload,
            bit_errors=bit_errors,
            bit_error_rate=ber,
            snr_db=float(channel_info.get("snr_db", config.snr_db)),
            transport_plan=transport_plan,
            crc_ok=rx_payload.crc_ok,
            evm_percent=evm_percent,
            evm_snr_linear=evm_snr_linear,
            harq_process_id=harq_process_id,
            harq_rv=int(transport_plan.codewords[0].rv),
            harq_retransmission=harq_retransmission,
            interference_reports=interference_reports,
            ldpc_iterations=self.runtime_context.get("decoder", "ldpc_iterations"),
            ldpc_decoder_path=self.runtime_context.get("decoder", "ldpc_decoder_path"),
        )

    @staticmethod
    def _uses_frequency_domain_channel(config: SimulationConfig) -> bool:
        """Return whether the configured channel bypasses time-domain processing."""
        return config.channel.model.upper() == "EXTERNAL_FREQRESP_FD"

    @staticmethod
    def _compute_evm_metrics(
        reference_symbols: np.ndarray,
        equalized_symbols: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Compute mean EVM and derived EVM-SNR from equalized symbols.

        Args:
            reference_symbols: One-dimensional ideal data-symbol array with shape
                ``(num_data_symbols,)``; axis 0 is mapper/modulator symbol order.
            equalized_symbols: One-dimensional equalized data-symbol array with shape
                ``(num_data_symbols,)`` or longer; axis 0 is receiver extraction order.

        Returns:
            Tuple of ``(evm_percent, evm_snr_linear)`` for the compared symbols.
        """
        if reference_symbols.size == 0 or equalized_symbols.size == 0:
            return None, None

        count = min(reference_symbols.size, equalized_symbols.size)
        reference = np.asarray(reference_symbols[:count], dtype=np.complex128)
        measured = np.asarray(equalized_symbols[:count], dtype=np.complex128)
        reference_power = float(np.mean(np.abs(reference) ** 2))
        error_power = float(np.mean(np.abs(measured - reference) ** 2))
        if reference_power <= 0.0:
            return None, None
        evm_ratio = float(np.sqrt(error_power / max(reference_power, 1e-24)))
        evm_percent = evm_ratio * 100.0
        evm_snr_linear = float(reference_power / max(error_power, 1e-24))
        return evm_percent, evm_snr_linear
