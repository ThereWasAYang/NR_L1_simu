from __future__ import annotations

import math

import numpy as np

from nr_phy_simu.common.interfaces import DmrsSequenceGenerator, ResourceMapper
from nr_phy_simu.config import SimulationConfig


class FrequencyDomainResourceMapper(ResourceMapper):
    """TX-side frequency-domain mapper for data and DMRS."""

    def __init__(self, dmrs_generator: DmrsSequenceGenerator) -> None:
        self.dmrs_generator = dmrs_generator

    def map_to_grid(
        self,
        data_symbols: np.ndarray,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map serialized data symbols and DMRS into the scheduled slot grid.

        Args:
            data_symbols: Serialized complex data symbols from the modulator.
            config: Full simulation configuration that defines allocation and DMRS rules.

        Returns:
            Tuple of ``(grid, dmrs_mask, data_mask, dmrs_symbols)`` for one slot.
        """
        n_sc = config.carrier.n_subcarriers
        n_sym = config.carrier.symbols_per_slot
        grid = np.zeros((n_sc, n_sym), dtype=np.complex128)
        dmrs_mask = np.zeros((n_sc, n_sym), dtype=bool)
        data_mask = np.zeros((n_sc, n_sym), dtype=bool)
        allocated = self.allocated_subcarriers(config)
        dmrs_info = self.dmrs_generator.get_dmrs_info(config)
        source_symbols = data_symbols
        if source_symbols.size == 0:
            raise ValueError("No data symbols available for resource mapping.")

        data_ptr = 0
        dmrs_sequence = []
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            symbol_dmrs_offsets = np.array([], dtype=int)
            is_dmrs_symbol = symbol_idx in dmrs_info.symbol_indices
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
                symbol_dmrs_offsets = self.symbol_dmrs_offsets(config, dmrs_info)
                dmrs_subcarriers = allocated[symbol_dmrs_offsets]
                dmrs_values = self.dmrs_generator.generate_for_symbol(symbol_idx, config)
                dmrs_values = dmrs_values * self.dmrs_power_scale(config)
                grid[dmrs_subcarriers, symbol_idx] = dmrs_values
                dmrs_mask[dmrs_subcarriers, symbol_idx] = True
                dmrs_sequence.append(dmrs_values)

            if skip_data_on_dmrs_symbol:
                continue

            available_subcarriers = allocated
            if symbol_dmrs_offsets.size:
                symbol_mask = np.ones(allocated.size, dtype=bool)
                symbol_mask[symbol_dmrs_offsets] = False
                available_subcarriers = allocated[symbol_mask]

            if available_subcarriers.size == 0:
                continue

            symbol_data = data_symbols[data_ptr : data_ptr + available_subcarriers.size]
            if symbol_data.size < available_subcarriers.size:
                remaining = available_subcarriers.size - symbol_data.size
                extra = np.tile(source_symbols, int(np.ceil(remaining / source_symbols.size)))[:remaining]
                symbol_data = np.concatenate([symbol_data, extra])
            data_ptr += available_subcarriers.size

            mapped_symbol = self.map_allocated_symbol(symbol_data, config)
            grid[available_subcarriers, symbol_idx] = mapped_symbol
            data_mask[available_subcarriers, symbol_idx] = True

        dmrs_symbols = np.concatenate(dmrs_sequence) if dmrs_sequence else np.array([], dtype=np.complex128)
        return grid, dmrs_mask, data_mask, dmrs_symbols

    def count_data_re(self, config: SimulationConfig) -> int:
        """Count how many REs in the scheduled slot can carry data.

        Args:
            config: Full simulation configuration that defines allocation and DMRS rules.

        Returns:
            Number of scheduled data REs inside the allocation.
        """
        allocated = self.allocated_subcarriers(config)
        dmrs_info = self.dmrs_generator.get_dmrs_info(config)
        total = 0
        for symbol_idx in range(config.link.start_symbol, config.link.start_symbol + config.link.num_symbols):
            is_dmrs_symbol = symbol_idx in dmrs_info.symbol_indices
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
            symbol_count = allocated.size
            if is_dmrs_symbol:
                symbol_count -= self.symbol_dmrs_offsets(config, dmrs_info).size
            total += symbol_count
        return total

    @staticmethod
    def allocated_subcarriers(config: SimulationConfig) -> np.ndarray:
        """Build absolute subcarrier indices for the scheduled PRB allocation.

        Args:
            config: Full simulation configuration that provides PRB start and width.

        Returns:
            Flat array of scheduled subcarrier indices within the cell bandwidth.
        """
        start = config.link.prb_start * 12
        stop = start + config.link.num_prbs * 12
        return np.arange(start, stop, dtype=int)

    @staticmethod
    def symbol_dmrs_offsets(config: SimulationConfig, dmrs_info) -> np.ndarray:
        """Expand per-PRB DMRS offsets across all scheduled PRBs.

        Args:
            config: Full simulation configuration that provides allocation width.
            dmrs_info: DMRS metadata object returned by the DMRS generator.

        Returns:
            Absolute DMRS offsets inside the allocated subcarrier set.
        """
        per_prb = []
        for prb in range(config.link.num_prbs):
            base = prb * 12
            per_prb.extend((base + dmrs_info.re_offsets).tolist())
        return np.array(per_prb, dtype=int)

    @staticmethod
    def map_allocated_symbol(symbol_data: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Apply optional transform precoding before grid mapping.

        Args:
            symbol_data: Complex data symbols scheduled onto one OFDM symbol.
            config: Full simulation configuration that defines waveform behavior.

        Returns:
            Symbols after optional DFT spreading for transform-precoded PUSCH.
        """
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            return np.fft.fft(symbol_data, n=symbol_data.size) / np.sqrt(symbol_data.size)
        return symbol_data

    @classmethod
    def dmrs_power_scale(cls, config: SimulationConfig) -> float:
        """Convert DMRS EPRE boost from dB to linear amplitude scale.

        Args:
            config: Full simulation configuration that defines DMRS power settings.

        Returns:
            Linear amplitude scale applied to generated DMRS symbols.
        """
        beta_db = cls.dmrs_epre_boost_db(config)
        return 10.0 ** (beta_db / 20.0)

    @classmethod
    def dmrs_epre_boost_db(cls, config: SimulationConfig) -> float:
        """Resolve the DMRS-to-data EPRE offset in dB.

        Args:
            config: Full simulation configuration that defines DMRS format choices.

        Returns:
            EPRE offset in dB applied to DMRS symbols.
        """
        num_cdm_groups = cls._resolved_num_cdm_groups_without_data(config)
        table = cls._power_boost_table_db(config.dmrs.config_type)
        return table.get(num_cdm_groups, 0.0)

    @staticmethod
    def _power_boost_table_db(config_type: int) -> dict[int, float]:
        """Return the standard DMRS power-boost lookup for one config type.

        Args:
            config_type: DMRS configuration type defined by the standard.

        Returns:
            Mapping from ``numCdmGroupsWithoutData`` to EPRE offset in dB.
        """
        if config_type == 1:
            return {1: 0.0, 2: 10.0 * math.log10(2.0)}
        if config_type == 2:
            return {1: 0.0, 2: 10.0 * math.log10(2.0), 3: 10.0 * math.log10(3.0)}
        raise ValueError(f"Unsupported DMRS configuration type: {config_type}")

    @staticmethod
    def _resolved_num_cdm_groups_without_data(config: SimulationConfig) -> int:
        """Resolve ``numCdmGroupsWithoutData`` from explicit or implicit rules.

        Args:
            config: Full simulation configuration that defines waveform and DMRS muxing.

        Returns:
            Effective number of CDM groups that do not carry data.
        """
        if config.dmrs.num_cdm_groups_without_data is not None:
            return int(config.dmrs.num_cdm_groups_without_data)

        no_data_on_dmrs_symbol = (
            config.link.waveform.upper() == "DFT-S-OFDM"
            or not config.dmrs.data_mux_enabled
        )
        if no_data_on_dmrs_symbol:
            return 2
        return 1
