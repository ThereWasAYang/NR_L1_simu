from __future__ import annotations

import math

import torch

from nr_phy_simu.common.interfaces import DmrsSequenceGenerator, ResourceMapper
from nr_phy_simu.common.torch_utils import COMPLEX_DTYPE, as_complex_tensor, as_int_tensor
from nr_phy_simu.config import SimulationConfig


class FrequencyDomainResourceMapper(ResourceMapper):
    """TX-side frequency-domain mapper for data and DMRS."""

    def __init__(self, dmrs_generator: DmrsSequenceGenerator) -> None:
        self.dmrs_generator = dmrs_generator

    def map_to_grid(
        self,
        data_symbols: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data_symbols = as_complex_tensor(data_symbols)
        n_sc = config.carrier.n_subcarriers
        n_sym = config.carrier.symbols_per_slot
        device = data_symbols.device
        grid = torch.zeros((n_sc, n_sym), dtype=COMPLEX_DTYPE, device=device)
        dmrs_mask = torch.zeros((n_sc, n_sym), dtype=torch.bool, device=device)
        data_mask = torch.zeros((n_sc, n_sym), dtype=torch.bool, device=device)
        allocated = self.allocated_subcarriers(config).to(device=device)
        dmrs_info = self.dmrs_generator.get_dmrs_info(config)
        source_symbols = data_symbols
        if source_symbols.numel() == 0:
            raise ValueError("No data symbols available for resource mapping.")

        data_ptr = 0
        dmrs_sequence = []
        dmrs_symbol_set = {int(v) for v in dmrs_info.symbol_indices}
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            symbol_dmrs_offsets = torch.zeros(0, dtype=torch.int64, device=device)
            is_dmrs_symbol = symbol_idx in dmrs_symbol_set
            is_transform_precoded_dmrs_symbol = (
                config.link.channel_type.upper() == "PUSCH"
                and config.link.waveform.upper() == "DFT-S-OFDM"
                and is_dmrs_symbol
            )
            skip_data_on_dmrs_symbol = is_transform_precoded_dmrs_symbol or (
                config.link.waveform.upper() == "CP-OFDM"
                and is_dmrs_symbol
                and not config.dmrs.data_mux_enabled
            )
            if is_dmrs_symbol:
                symbol_dmrs_offsets = self.symbol_dmrs_offsets(config, dmrs_info).to(device=device)
                dmrs_subcarriers = allocated[symbol_dmrs_offsets]
                dmrs_values = as_complex_tensor(self.dmrs_generator.generate_for_symbol(symbol_idx, config), device=device)
                dmrs_values = dmrs_values * self.dmrs_power_scale(config)
                grid[dmrs_subcarriers, symbol_idx] = dmrs_values
                dmrs_mask[dmrs_subcarriers, symbol_idx] = True
                dmrs_sequence.append(dmrs_values)

            if skip_data_on_dmrs_symbol:
                continue

            available_subcarriers = allocated
            if symbol_dmrs_offsets.numel():
                symbol_mask = torch.ones(allocated.numel(), dtype=torch.bool, device=device)
                symbol_mask[symbol_dmrs_offsets] = False
                available_subcarriers = allocated[symbol_mask]

            if available_subcarriers.numel() == 0:
                continue

            symbol_data = data_symbols[data_ptr : data_ptr + available_subcarriers.numel()]
            if symbol_data.numel() < available_subcarriers.numel():
                remaining = available_subcarriers.numel() - symbol_data.numel()
                repeats = int(math.ceil(remaining / source_symbols.numel()))
                extra = source_symbols.repeat(repeats)[:remaining]
                symbol_data = torch.cat([symbol_data, extra])
            data_ptr += available_subcarriers.numel()

            mapped_symbol = self.map_allocated_symbol(symbol_data, config)
            grid[available_subcarriers, symbol_idx] = mapped_symbol
            data_mask[available_subcarriers, symbol_idx] = True

        dmrs_symbols = torch.cat(dmrs_sequence) if dmrs_sequence else torch.zeros(0, dtype=COMPLEX_DTYPE, device=device)
        return grid, dmrs_mask, data_mask, dmrs_symbols

    def count_data_re(self, config: SimulationConfig) -> int:
        allocated = self.allocated_subcarriers(config)
        dmrs_info = self.dmrs_generator.get_dmrs_info(config)
        dmrs_symbol_set = {int(v) for v in dmrs_info.symbol_indices}
        total = 0
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            is_dmrs_symbol = symbol_idx in dmrs_symbol_set
            if (
                config.link.channel_type.upper() == "PUSCH"
                and config.link.waveform.upper() == "DFT-S-OFDM"
                and is_dmrs_symbol
            ) or (
                config.link.waveform.upper() == "CP-OFDM"
                and is_dmrs_symbol
                and not config.dmrs.data_mux_enabled
            ):
                continue
            symbol_count = allocated.numel()
            if is_dmrs_symbol:
                symbol_count -= self.symbol_dmrs_offsets(config, dmrs_info).numel()
            total += symbol_count
        return total

    @staticmethod
    def allocated_subcarriers(config: SimulationConfig) -> torch.Tensor:
        start = config.link.prb_start * 12
        stop = start + config.link.num_prbs * 12
        return torch.arange(start, stop, dtype=torch.int64)

    @staticmethod
    def symbol_dmrs_offsets(config: SimulationConfig, dmrs_info) -> torch.Tensor:
        per_prb = []
        re_offsets = as_int_tensor(dmrs_info.re_offsets, dtype=torch.int64)
        for prb in range(config.link.num_prbs):
            base = prb * 12
            per_prb.extend((base + re_offsets).tolist())
        return torch.tensor(per_prb, dtype=torch.int64)

    @staticmethod
    def map_allocated_symbol(symbol_data: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            size = symbol_data.numel()
            return torch.fft.fft(symbol_data, n=size) / torch.sqrt(
                torch.tensor(float(size), dtype=torch.float64, device=symbol_data.device)
            )
        return symbol_data

    @classmethod
    def dmrs_power_scale(cls, config: SimulationConfig) -> float:
        beta_db = cls.dmrs_epre_boost_db(config)
        return 10.0 ** (beta_db / 20.0)

    @classmethod
    def dmrs_epre_boost_db(cls, config: SimulationConfig) -> float:
        num_cdm_groups = cls._resolved_num_cdm_groups_without_data(config)
        table = cls._power_boost_table_db(config.dmrs.config_type)
        return table.get(num_cdm_groups, 0.0)

    @staticmethod
    def _power_boost_table_db(config_type: int) -> dict[int, float]:
        if config_type == 1:
            return {1: 0.0, 2: 10.0 * math.log10(2.0)}
        if config_type == 2:
            return {1: 0.0, 2: 10.0 * math.log10(2.0), 3: 10.0 * math.log10(3.0)}
        raise ValueError(f"Unsupported DMRS configuration type: {config_type}")

    @staticmethod
    def _resolved_num_cdm_groups_without_data(config: SimulationConfig) -> int:
        if config.dmrs.num_cdm_groups_without_data is not None:
            return int(config.dmrs.num_cdm_groups_without_data)

        no_data_on_dmrs_symbol = (
            config.link.waveform.upper() == "DFT-S-OFDM"
            or not config.dmrs.data_mux_enabled
        )
        if no_data_on_dmrs_symbol:
            return 2
        return 1
