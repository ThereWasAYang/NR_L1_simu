from __future__ import annotations

import torch

from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.common.types import SimulationResult, TxPayload
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.io.waveform_loader import load_text_waveform
from nr_phy_simu.rx.chain import Receiver
from nr_phy_simu.scenarios.interference import InterferenceMixer
from nr_phy_simu.scenarios.component_factory import (
    DefaultSimulationComponentFactory,
    SimulationComponentFactory,
    build_receiver,
)
from nr_phy_simu.common.torch_utils import BIT_DTYPE, COMPLEX_DTYPE, as_complex_tensor


class WaveformReplaySimulation:
    """Replay captured time-domain waveform directly into the receiver chain."""

    def __init__(
        self,
        config: SimulationConfig,
        receiver: Receiver | None = None,
        component_factory: SimulationComponentFactory | None = None,
    ) -> None:
        if not config.waveform_input.enabled:
            raise ValueError("waveform_input.waveform_path must be configured for waveform replay simulation.")
        self.config = config
        self.component_factory = component_factory or DefaultSimulationComponentFactory()
        self.components = self.component_factory.create_components(config)
        self.mapper = self.components.transmitter.mapper
        self.dmrs_generator = self.components.shared.dmrs_generator
        self.receiver = receiver or build_receiver(self.components)
        self.interference_mixer = InterferenceMixer(self.component_factory)

    def run(self) -> SimulationResult:
        apply_mcs_to_link(self.config)
        data_re = self.mapper.count_data_re(self.config)
        self.config.link.coded_bit_capacity = data_re * self._bits_per_symbol()
        if not self.config.link.transport_block_size:
            self.config.link.transport_block_size = resolve_transport_block_size(self.config, data_re)

        waveform = load_text_waveform(self.config.waveform_input.waveform_path, self.config)
        dmrs_mask, data_mask, dmrs_symbols = self._build_reference_masks()
        noise_variance = self._resolve_noise_variance(waveform)
        waveform, interference_reports = self.interference_mixer.apply(
            waveform,
            noise_variance=noise_variance,
            config=self.config,
        )

        rx_payload = self.receiver.receive(
            rx_waveform=waveform,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=self.config,
        )
        tx_placeholder = TxPayload(
            transport_block=torch.zeros(0, dtype=BIT_DTYPE),
            coded_bits=torch.zeros(0, dtype=BIT_DTYPE),
            tx_symbols=torch.zeros(0, dtype=COMPLEX_DTYPE),
            resource_grid=torch.zeros(
                (self.config.carrier.n_subcarriers, self.config.carrier.symbols_per_slot),
                dtype=COMPLEX_DTYPE,
            ),
            waveform=as_complex_tensor(waveform),
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
        )
        return SimulationResult(
            tx=tx_placeholder,
            rx=rx_payload,
            bit_errors=-1,
            bit_error_rate=float("nan"),
            snr_db=float(self.config.channel.params.get("snr_db", self.config.snr_db)),
            crc_ok=rx_payload.crc_ok,
            evm_percent=None,
            evm_snr_linear=None,
            interference_reports=interference_reports,
        )

    def _build_reference_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dummy_symbols = torch.zeros(self.mapper.count_data_re(self.config), dtype=COMPLEX_DTYPE)
        _, dmrs_mask, data_mask, dmrs_symbols = self.mapper.map_to_grid(dummy_symbols, self.config)
        return dmrs_mask, data_mask, dmrs_symbols

    def _resolve_noise_variance(self, waveform: torch.Tensor) -> float:
        if self.config.waveform_input.noise_variance is not None:
            return float(self.config.waveform_input.noise_variance)

        snr_db = float(self.config.channel.params.get("snr_db", self.config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = float(torch.mean(torch.abs(as_complex_tensor(waveform)) ** 2).item())
        return float(signal_power / max(snr_linear, 1e-12))

    def _bits_per_symbol(self) -> int:
        modulation = self.config.link.modulation.upper()
        mapping = {"PI/2-BPSK": 1, "BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
        return mapping[modulation]
