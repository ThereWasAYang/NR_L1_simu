from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import ReceiverProcessor
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor
from nr_phy_simu.common.types import ChannelEstimateResult, RxPayload
from nr_phy_simu.config import SimulationConfig


class DefaultReceiverProcessor(ReceiverProcessor):
    """Default receiver processor that preserves the standard RX chain."""

    def receive(
        self,
        receiver,
        rx_waveform: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> RxPayload:
        """Run time-domain processing, data processing, descrambling and decoding."""
        rx_grid = receiver.time_processor.demodulate(rx_waveform, config)
        return self.receive_from_grid(
            receiver=receiver,
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
            rx_waveform=rx_waveform,
        )

    def receive_from_grid(
        self,
        receiver,
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
        rx_waveform: torch.Tensor | None = None,
    ) -> RxPayload:
        """Run data processing, descrambling and decoding from a frequency grid."""
        rx_grid = as_complex_tensor(rx_grid)
        if rx_grid.ndim == 2:
            rx_grid = rx_grid.unsqueeze(0)
        processing = receiver.data_processor.process(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )
        channel_estimation = processing.channel_estimation or ChannelEstimateResult(
            channel_estimate=torch.empty(0, dtype=COMPLEX_DTYPE, device=rx_grid.device),
            pilot_estimates=torch.empty(0, dtype=COMPLEX_DTYPE, device=rx_grid.device),
            pilot_symbol_indices=torch.empty(0, dtype=torch.int64, device=rx_grid.device),
            plot_artifacts=(),
        )
        descrambled_llrs = receiver.scrambler.descramble_llrs(processing.llrs, config)
        decoded_bits = receiver.decoder.decode(descrambled_llrs, config)
        crc_ok = getattr(receiver.decoder, "last_crc_ok", None)
        rx_waveform_out = (
            torch.empty((int(config.link.num_rx_ant), 0), dtype=COMPLEX_DTYPE, device=rx_grid.device)
            if rx_waveform is None
            else as_complex_tensor(rx_waveform, device=rx_grid.device)
        )
        if rx_waveform_out.ndim == 1:
            rx_waveform_out = rx_waveform_out.unsqueeze(0)
        return RxPayload(
            rx_waveform=rx_waveform_out,
            rx_grid=rx_grid,
            channel_estimation=channel_estimation,
            equalized_symbols=processing.equalized_symbols,
            layer_symbols=processing.layer_symbols,
            llrs=descrambled_llrs,
            decoded_bits=decoded_bits,
            crc_ok=crc_ok,
            dmrs_symbols=dmrs_symbols,
            plot_artifacts=processing.plot_artifacts,
        )
