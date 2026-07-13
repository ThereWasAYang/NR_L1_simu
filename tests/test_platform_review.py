from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from py3gpp import nrOFDMModulate

from nr_phy_simu import MultiTtiSimulationRunner, PuschSimulation, load_simulation_config
from nr_phy_simu.channels.channel_factory import DefaultChannelFactory
from nr_phy_simu.common.bwp import ofdm_phase_compensation_vector
from nr_phy_simu.common.ofdm import OfdmProcessor
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.sequences.dmrs_tables import resolve_dmrs_symbol_indices
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation


ROOT = Path(__file__).resolve().parents[1]


def _awgn_config():
    config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
    config.plotting.enabled = False
    return config


def test_fixed_channel_seed_defines_reproducible_non_repeating_tti_sequence():
    config = _awgn_config()
    config.link.num_tx_ant = 1
    config.link.num_rx_ant = 1
    config.channel.seed = 1007
    config.simulation.num_ttis = 2

    first = MultiTtiSimulationRunner(config).run()
    second = MultiTtiSimulationRunner(config).run()

    first_noise = [item.rx.rx_waveform - item.tx.waveform for item in first.tti_results]
    second_noise = [item.rx.rx_waveform - item.tx.waveform for item in second.tti_results]
    assert not np.allclose(first_noise[0], first_noise[1])
    assert np.allclose(first_noise[0], second_noise[0])
    assert np.allclose(first_noise[1], second_noise[1])


def test_continuous_tdl_evolution_preserves_slot_boundary_continuity():
    config = _awgn_config()
    config.channel.model = "TDL"
    config.channel.seed = 1007
    config.channel.params = {
        "profile": "TDL-C",
        "delay_spread_ns": 30.0,
        "max_doppler_hz": 5.0,
        "num_sinusoids": 8,
        "add_noise": False,
        "tti_evolution": "continuous",
    }
    waveform = np.ones(
        (config.link.num_tx_ant, config.carrier.slot_length_samples),
        dtype=np.complex128,
    )
    channel = DefaultChannelFactory().create(config)

    config.slot_index = 0
    _, first_info = channel.propagate(waveform, config)
    config.slot_index = 1
    _, second_info = channel.propagate(waveform, config)

    boundary_step = np.max(
        np.abs(second_info["path_coefficients"][..., 0] - first_info["path_coefficients"][..., -1])
    )
    assert boundary_step < 1e-3


def test_replay_rejects_more_than_one_tti():
    config = load_simulation_config(ROOT / "configs" / "pusch_replay_template.yaml")
    config.simulation.num_ttis = 2
    with pytest.raises(ValueError, match="exactly one TTI"):
        WaveformReplaySimulation(config)


def test_replay_builds_components_from_its_private_config_copy():
    class RecordingFactory(DefaultSimulationComponentFactory):
        def __init__(self):
            self.component_config = None

        def create_components(self, config):
            self.component_config = config
            return super().create_components(config)

    config = load_simulation_config(ROOT / "configs" / "pusch_replay_template.yaml")
    factory = RecordingFactory()
    simulation = WaveformReplaySimulation(config, component_factory=factory)

    assert simulation.config is not config
    assert factory.component_config is simulation.config


def test_simulation_does_not_write_derived_fields_back_to_input_config():
    config = _awgn_config()
    original = (
        config.link.modulation,
        config.link.code_rate,
        config.link.transport_block_size,
        config.link.coded_bit_capacity,
    )
    first = PuschSimulation(config).run()
    assert (
        config.link.modulation,
        config.link.code_rate,
        config.link.transport_block_size,
        config.link.coded_bit_capacity,
    ) == original

    config.link.num_prbs //= 2
    second = PuschSimulation(config).run()
    assert second.transport_plan.data_re_count != first.transport_plan.data_re_count
    assert second.transport_plan.size_bits != first.transport_plan.size_bits


def test_base_config_path_recursively_merges_and_resolves_declaring_file_paths(tmp_path: Path):
    base_dir = tmp_path / "base"
    child_dir = tmp_path / "child"
    base_dir.mkdir()
    child_dir.mkdir()
    (base_dir / "base.yaml").write_text(
        "\n".join(
            [
                "link:",
                "  num_prbs: 12",
                "  mcs:",
                "    table: qam64",
                "    index: 2",
                "channel:",
                "  model: AWGN",
                "  params:",
                "    snr_db: 9",
                "simulation:",
                "  result_output_path: base-output.csv",
            ]
        ),
        encoding="utf-8",
    )
    (child_dir / "child.yaml").write_text(
        "\n".join(
            [
                "base_config_path: ../base/base.yaml",
                "link:",
                "  num_prbs: 6",
                "channel:",
                "  params:",
                "    snr_db: 15",
            ]
        ),
        encoding="utf-8",
    )

    config = load_simulation_config(child_dir / "child.yaml")
    assert config.link.num_prbs == 6
    assert config.link.mcs.table == "qam64"
    assert config.link.mcs.index == 2
    assert config.channel.params["snr_db"] == 15
    assert Path(config.simulation.result_output_path) == base_dir / "base-output.csv"


def test_ldpc_metadata_is_exposed_without_changing_input_config():
    config = _awgn_config()
    config.channel.params["snr_db"] = 30.0
    result = PuschSimulation(config).run()
    assert result.ldpc_decoder_path in {"min_sum", "gf2_direct", "py3gpp", "hard"}
    assert result.ldpc_iterations is not None
    assert result.ldpc_iterations >= 0


@pytest.mark.parametrize(
    ("channel", "mapping", "duration", "add_pos", "max_length", "type_a_pos", "expected"),
    [
        ("PDSCH", "A", 12, 1, 1, 3, (3, 9)),
        ("PDSCH", "A", 14, 2, 1, 3, (3, 7, 11)),
        ("PDSCH", "B", 11, 3, 1, 2, (2, 5, 8, 11)),
        ("PUSCH", "B", 14, 2, 1, 0, (0, 5, 10)),
        ("PDSCH", "A", 13, 1, 2, 2, (2, 3, 10, 11)),
        ("PUSCH", "B", 12, 1, 2, 1, (1, 2, 10, 11)),
    ],
)
def test_dmrs_table_edges_match_38211(
    channel: str,
    mapping: str,
    duration: int,
    add_pos: int,
    max_length: int,
    type_a_pos: int,
    expected: tuple[int, ...],
):
    assert resolve_dmrs_symbol_indices(
        channel_type=channel,
        start_symbol=0 if mapping == "A" else type_a_pos,
        num_symbols=duration,
        mapping_type=mapping,
        additional_positions=add_pos,
        max_length=max_length,
        type_a_position=type_a_pos,
    ) == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"additional_positions": 3, "type_a_position": 3}, "only supported"),
        ({"num_symbols": 3, "type_a_position": 3}, "must fall inside"),
        ({"start_symbol": 3, "num_symbols": 11}, "must fall inside"),
        ({"max_length": 2, "additional_positions": 2}, "only supports"),
    ],
)
def test_dmrs_rejects_protocol_invalid_position_combinations(overrides, message):
    arguments = {
        "channel_type": "PDSCH",
        "start_symbol": 0,
        "num_symbols": 14,
        "mapping_type": "A",
        "additional_positions": 0,
        "max_length": 1,
        "type_a_position": 2,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        resolve_dmrs_symbol_indices(**arguments)


def test_gold_dmrs_sequence_offset_uses_common_resource_block_reference():
    first = _awgn_config()
    first.link.num_prbs = 4
    first.bwp.start_rb = 5
    first.bwp.num_rbs = 40
    first.link.prb_start = 3

    second = _awgn_config()
    second.link.num_prbs = 4
    second.bwp.start_rb = 0
    second.bwp.num_rbs = None
    second.link.prb_start = 8

    generator = DmrsGenerator()
    symbol = generator.get_dmrs_info(first).symbol_indices[0]
    assert np.allclose(
        generator.generate_for_symbol(symbol, first),
        generator.generate_for_symbol(symbol, second),
    )


def test_ofdm_phase_compensation_is_symbol_constant_and_uses_carrier_f0():
    config = _awgn_config()
    config.bwp.start_rb = 7
    config.bwp.num_rbs = 20
    config.slot_index = 1
    cp_length = config.carrier.cyclic_prefix_lengths[0]
    symbol_length = config.carrier.fft_size_effective + cp_length
    vector = ofdm_phase_compensation_vector(
        config,
        symbol_start_sample=0,
        cp_length=cp_length,
        symbol_length=symbol_length,
        inverse=False,
    )
    useful_start = config.carrier.slot_length_samples + cp_length
    expected = np.exp(
        -1j
        * 2.0
        * np.pi
        * config.carrier.center_frequency_hz
        * useful_start
        / config.carrier.sample_rate_effective_hz
    )
    assert np.allclose(vector, expected)
    assert np.allclose(vector, vector[0])


def test_ofdm_slot_zero_matches_py3gpp_reference_waveform():
    config = _awgn_config()
    config.slot_index = 0
    rng = np.random.default_rng(17)
    grid = rng.normal(size=(config.carrier.n_subcarriers, config.carrier.symbols_per_slot)) + 1j * rng.normal(
        size=(config.carrier.n_subcarriers, config.carrier.symbols_per_slot)
    )
    actual = OfdmProcessor()._modulate_single(grid.copy(), config)
    # Keep this interoperability check at slot 0: py3gpp 0.6.0 treats
    # initialNSlot as a symbol offset, so its non-zero-slot waveform is not a
    # valid 38.211 reference.  Multi-slot timing is covered independently below.
    expected, _ = nrOFDMModulate(
        grid=grid.copy(),
        scs=config.carrier.subcarrier_spacing_khz,
        initialNSlot=0,
        CyclicPrefix="normal",
        Nfft=config.carrier.fft_size_effective,
        SampleRate=int(config.carrier.sample_rate_effective_hz),
        CarrierFrequency=config.carrier.center_frequency_hz,
    )
    assert np.allclose(actual, expected)


def test_mu2_normal_cp_and_timeline_follow_subframe_symbol_positions():
    config = _awgn_config()
    config.carrier.subcarrier_spacing_khz = 60
    processor = OfdmProcessor()
    rng = np.random.default_rng(23)
    grid = rng.normal(
        size=(config.carrier.n_subcarriers, config.carrier.symbols_per_slot)
    ) + 1j * rng.normal(
        size=(config.carrier.n_subcarriers, config.carrier.symbols_per_slot)
    )

    cp_by_slot = [
        config.carrier.cyclic_prefix_lengths_for_slot(slot_index)
        for slot_index in range(config.carrier.slots_per_subframe)
    ]
    long_cp = cp_by_slot[0][0]
    regular_cp = cp_by_slot[1][0]
    assert long_cp > regular_cp
    assert [lengths[0] for lengths in cp_by_slot] == [
        long_cp,
        regular_cp,
        long_cp,
        regular_cp,
    ]
    assert all(length == regular_cp for lengths in cp_by_slot for length in lengths[1:])

    cumulative_samples = 0
    for slot_index, cp_lengths in enumerate(cp_by_slot):
        config.slot_index = slot_index
        assert config.carrier.slot_start_sample_in_subframe(slot_index) == cumulative_samples
        waveform = processor._modulate_single(grid, config)
        assert waveform.size == config.carrier.slot_length_samples_for_slot(slot_index)
        assert np.allclose(processor._demodulate_single(waveform, config), grid)
        cumulative_samples += sum(cp_lengths) + (
            config.carrier.symbols_per_slot * config.carrier.fft_size_effective
        )
    assert cumulative_samples == config.carrier.subframe_length_samples
    assert config.carrier.slot_start_sample(config.carrier.slots_per_subframe + 1) == (
        config.carrier.subframe_length_samples
        + config.carrier.slot_length_samples_for_slot(0)
    )

    config.slot_index = 1
    cp_length = cp_by_slot[1][0]
    symbol_length = config.carrier.fft_size_effective + cp_length
    vector = ofdm_phase_compensation_vector(
        config,
        symbol_start_sample=0,
        cp_length=cp_length,
        symbol_length=symbol_length,
        inverse=False,
    )
    useful_start = config.carrier.slot_start_sample_in_subframe(1) + cp_length
    expected = np.exp(
        -1j
        * 2.0
        * np.pi
        * config.carrier.center_frequency_hz
        * useful_start
        / config.carrier.sample_rate_effective_hz
    )
    assert np.allclose(vector, expected)
