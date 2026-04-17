from __future__ import annotations

from dataclasses import replace

import numpy as np

from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
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
    ) -> None:
        self.config = config
        self.component_factory = component_factory or DefaultSimulationComponentFactory()
        self.components = self.component_factory.create_components(config)
        self.dmrs_generator = self.components.shared.dmrs_generator
        self.mapper = self.components.transmitter.mapper
        self.transmitter = transmitter or build_transmitter(self.components)
        self.receiver = receiver or build_receiver(self.components)
        self.channel = channel or self.component_factory.create_channel_factory().create(config)
        self.interference_mixer = InterferenceMixer(self.component_factory)

    def run(self) -> SimulationResult:
        """Run one end-to-end shared-channel simulation for a single TTI.

        Args:
            None.

        Returns:
            Structured simulation result containing TX/RX buffers and KPIs.
        """
        apply_mcs_to_link(self.config)
        data_re = self.mapper.count_data_re(self.config)
        self.config.link.coded_bit_capacity = data_re * self._bits_per_symbol()
        if not self.config.link.transport_block_size:
            self.config.link.transport_block_size = resolve_transport_block_size(self.config, data_re)

        rng = np.random.default_rng(self.config.random_seed)
        transport_block = rng.integers(
            0,
            2,
            size=int(self.config.link.transport_block_size),
            dtype=np.int8,
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
        else:
            tx_payload = self.transmitter.transmit(transport_block, self.config)
            rx_waveform, channel_info = self.channel.propagate(tx_payload.waveform, self.config)
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
                channel_info=channel_info,
                interference_reports=interference_reports,
            )

        tx_payload = replace(tx_payload, waveform=np.asarray([], dtype=np.complex128))
        return self._build_result(
            tx_payload=tx_payload,
            rx_payload=rx_payload,
            transport_block=transport_block,
            channel_info=channel_info,
            interference_reports=(),
        )

    def _build_result(
        self,
        tx_payload,
        rx_payload,
        transport_block: np.ndarray,
        channel_info: dict,
        interference_reports: tuple,
    ) -> SimulationResult:
        """Build the final simulation result object from chain outputs."""
        if self.config.simulation.bypass_channel_coding:
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
            snr_db=float(channel_info.get("snr_db", self.config.snr_db)),
            crc_ok=rx_payload.crc_ok,
            evm_percent=evm_percent,
            evm_snr_linear=evm_snr_linear,
            interference_reports=interference_reports,
        )

    def _uses_frequency_domain_channel(self) -> bool:
        """Return whether the configured channel bypasses time-domain processing."""
        return self.config.channel.model.upper() == "EXTERNAL_FREQRESP_FD"

    def _bits_per_symbol(self) -> int:
        """Resolve bits per modulation symbol from the configured modulation name.

        Args:
            None.

        Returns:
            Number of coded bits carried by one modulation symbol.
        """
        modulation = self.config.link.modulation.upper()
        mapping = {"PI/2-BPSK": 1, "BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
        return mapping[modulation]

    @staticmethod
    def _compute_evm_metrics(
        reference_symbols: np.ndarray,
        equalized_symbols: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Compute mean EVM and derived EVM-SNR from equalized symbols.

        Args:
            reference_symbols: Ideal transmitted data symbols before the channel.
            equalized_symbols: Equalized received data symbols after the receiver.

        Returns:
            Tuple of ``(evm_percent, evm_snr_linear)`` for the compared symbols.
        """
        if reference_symbols.size == 0 or equalized_symbols.size == 0:
            return None, None

        count = min(reference_symbols.size, equalized_symbols.size)
        reference = np.asarray(reference_symbols[:count], dtype=np.complex128)
        measured = np.asarray(equalized_symbols[:count], dtype=np.complex128)
        symbol_magnitude = np.maximum(np.abs(reference), 1e-12)
        error_vector = np.abs(measured - reference)
        evm_ratio = error_vector / symbol_magnitude
        mean_evm_ratio = float(np.mean(evm_ratio))
        evm_percent = mean_evm_ratio * 100.0
        evm_snr_linear = float(1.0 / max(mean_evm_ratio**2, 1e-24))
        return evm_percent, evm_snr_linear
