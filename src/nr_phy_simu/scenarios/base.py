from __future__ import annotations

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
        decoded = rx_payload.decoded_bits[: transport_block.size]
        bit_errors = int(np.sum(decoded != transport_block))
        ber = bit_errors / transport_block.size
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

    def _bits_per_symbol(self) -> int:
        modulation = self.config.link.modulation.upper()
        mapping = {"PI/2-BPSK": 1, "BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
        return mapping[modulation]

    @staticmethod
    def _compute_evm_metrics(
        reference_symbols: np.ndarray,
        equalized_symbols: np.ndarray,
    ) -> tuple[float | None, float | None]:
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
