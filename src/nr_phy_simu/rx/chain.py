from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import (
    BitScrambler,
    ChannelDecoder,
    ChannelEstimator,
    Demodulator,
    DmrsSequenceGenerator,
    FrequencyExtractor,
    MimoEqualizer,
    TimeDomainProcessor,
)
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor
from nr_phy_simu.common.types import RxPayload
from nr_phy_simu.config import SimulationConfig


class Receiver:
    def __init__(
        self,
        time_processor: TimeDomainProcessor,
        extractor: FrequencyExtractor,
        estimator: ChannelEstimator,
        equalizer: MimoEqualizer,
        demodulator: Demodulator,
        decoder: ChannelDecoder,
        dmrs_generator: DmrsSequenceGenerator,
        scrambler: BitScrambler,
        layer_mapper: LayerMapper | None = None,
    ) -> None:
        self.time_processor = time_processor
        self.extractor = extractor
        self.estimator = estimator
        self.equalizer = equalizer
        self.demodulator = demodulator
        self.decoder = decoder
        self.dmrs_generator = dmrs_generator
        self.scrambler = scrambler
        self.layer_mapper = layer_mapper or LayerMapper()

    def receive(
        self,
        rx_waveform: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
    ) -> RxPayload:
        """Run the complete receive chain for one slot."""
        rx_waveform = as_complex_tensor(rx_waveform)
        rx_grid = self.time_processor.demodulate(rx_waveform, config)
        return self.receive_from_grid(
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
        rx_grid: torch.Tensor,
        dmrs_symbols: torch.Tensor,
        dmrs_mask: torch.Tensor,
        data_mask: torch.Tensor,
        noise_variance: float,
        config: SimulationConfig,
        rx_waveform: torch.Tensor | None = None,
    ) -> RxPayload:
        """Run the receive chain starting from an already demodulated grid."""
        rx_grid = as_complex_tensor(rx_grid)
        if rx_grid.ndim == 2:
            rx_grid = rx_grid.unsqueeze(0)
        dmrs_symbols = as_complex_tensor(dmrs_symbols, device=rx_grid.device)
        channel_estimation = self.estimator.estimate(rx_grid, dmrs_symbols, dmrs_mask, config)

        rx_data_symbols = self.extractor.extract(rx_grid, data_mask, config, despread=False)
        data_channel = self.extractor.extract(channel_estimation.channel_estimate, data_mask, config, despread=False)
        equalized_symbols = self.equalizer.equalize(
            rx_data_symbols,
            data_channel,
            noise_variance=noise_variance,
            config=config,
        )
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            equalized_symbols = self._despread_equalized(equalized_symbols, data_mask, config)
        layer_mapping = self.layer_mapper.unmap_symbols(equalized_symbols, config.link.num_layers)
        llrs = self.demodulator.demap_symbols(equalized_symbols, noise_variance, config)
        descrambled_llrs = self.scrambler.descramble_llrs(llrs, config)
        decoded_bits = self.decoder.decode(descrambled_llrs, config)
        crc_ok = getattr(self.decoder, "last_crc_ok", None)
        if rx_waveform is None:
            rx_waveform = torch.zeros(0, dtype=COMPLEX_DTYPE, device=rx_grid.device)
        return RxPayload(
            rx_waveform=rx_waveform,
            rx_grid=rx_grid,
            channel_estimation=channel_estimation,
            equalized_symbols=equalized_symbols,
            llrs=descrambled_llrs,
            decoded_bits=decoded_bits,
            crc_ok=crc_ok,
            dmrs_symbols=dmrs_symbols,
            layer_symbols=layer_mapping.layer_symbols,
            plot_artifacts=channel_estimation.plot_artifacts,
        )

    @staticmethod
    def _despread_equalized(
        equalized_symbols: torch.Tensor,
        data_mask: torch.Tensor,
        config: SimulationConfig,
    ) -> torch.Tensor:
        """Undo DFT spreading on equalized PUSCH symbols."""
        despread = []
        cursor = 0
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            count = int(torch.count_nonzero(data_mask[:, symbol_idx]).item())
            if count == 0:
                continue
            symbol_values = equalized_symbols[cursor : cursor + count]
            cursor += count
            despread.append(
                torch.fft.ifft(symbol_values, n=count)
                * torch.sqrt(torch.tensor(float(count), dtype=torch.float64, device=symbol_values.device))
            )
        return torch.cat(despread) if despread else torch.zeros(0, dtype=COMPLEX_DTYPE, device=equalized_symbols.device)
