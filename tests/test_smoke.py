from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from py3gpp import (
    nrCRCDecode,
    nrCRCEncode,
    nrCodeBlockDesegmentLDPC,
    nrCodeBlockSegmentLDPC,
    nrDLSCHInfo,
    nrLDPCDecode,
    nrLDPCEncode,
    nrRateMatchLDPC,
    nrRateRecoverLDPC,
)

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.io.frequency_response_loader import load_frequency_response
from nr_phy_simu.io.multi_tti_report import append_multi_tti_report
from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.common.bwp import allocated_subcarriers, bwp_center_frequency_hz
from nr_phy_simu.common.runtime_context import SimulationRuntimeContext, get_runtime_context
from nr_phy_simu.common.ofdm import OfdmProcessor, time_to_frequency_noise_variance
from nr_phy_simu.common.harq import HarqManager
from nr_phy_simu.common.layer_mapping import LayerMapper
from nr_phy_simu.common.interfaces import ReceiverDataProcessor, ReceiverProcessingStage, ReceiverProcessor
from nr_phy_simu.common.types import ChannelEstimateResult, PlotArtifact, ReceiverDataProcessingResult, RxPayload
from nr_phy_simu.common.transmission import build_transport_block_plan
from nr_phy_simu.common.ulsch_ldpc import (
    _get_z_array,
    _lifting_set_index_from_zc,
    decode_ulsch_ldpc,
    encode_ldpc_codeblocks,
    get_ulsch_ldpc_info,
    rate_match_ulsch_ldpc,
    rate_recover_ulsch_ldpc,
)
from nr_phy_simu.common.sequences.dmrs_tables import resolve_dmrs_symbol_indices
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.base import SharedChannelSimulation
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
from nr_phy_simu.scenarios.multi_tti import MultiTtiSimulationRunner
from nr_phy_simu.tx.resource_mapping import FrequencyDomainResourceMapper
from nr_phy_simu.rx.frequency_extraction import FrequencyDomainExtractor
from nr_phy_simu.rx.data_processing import ReceiverDataProcessorPipeline
from nr_phy_simu.rx.channel_estimation import LeastSquaresEstimator
from nr_phy_simu.visualization import save_simulation_plots
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.mcs import resolve_mcs
from nr_phy_simu.channels.channel_factory import DefaultChannelFactory
from nr_phy_simu.channels.tdl import TdlChannel
from nr_phy_simu.channels.cdl import CdlChannel
from nr_phy_simu.channels.profile_tables import CDL_LOS_K_DB, CDL_PROFILES, TDL_LOS_K_DB, TDL_PROFILES
from nr_phy_simu.channels.external_frequency_response import ExternalFrequencyResponseFrequencyDomainChannel
from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation
from nr_phy_simu.scenarios.sweep import run_snr_sweep, write_snr_sweep_csv
from nr_phy_simu.scenarios.component_factory import build_transmitter
from nr_phy_simu.scenarios.interference import InterferenceMixer
from nr_phy_simu.tx.codec import RandomBitCoder


class PuschAwgnSmokeTest(unittest.TestCase):
    def test_pusch_multi_tti_bler_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.simulation.num_ttis = 20
        config.plotting.enabled = False
        config.channel.params["snr_db"] = 30.0
        batch_result = MultiTtiSimulationRunner(config).run()
        self.assertEqual(batch_result.num_ttis, 20)
        self.assertEqual(len(batch_result.tti_results), 20)
        self.assertEqual(batch_result.packet_errors, 0)
        self.assertEqual(batch_result.block_error_rate, 0.0)
        self.assertIsNotNone(batch_result.average_evm_percent)
        self.assertIsNotNone(batch_result.average_evm_snr_linear)
        self.assertIsNotNone(batch_result.last_result)

    def test_multi_tti_report_file_append(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_multi_tti.yaml")
        config.channel.params["snr_db"] = 30.0
        config.simulation.num_ttis = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "bler_report.csv"
            config.simulation.result_output_path = str(report_path)
            first_result = MultiTtiSimulationRunner(config).run()
            append_multi_tti_report(report_path, first_result, first_result.final_config)
            second_result = MultiTtiSimulationRunner(config).run()
            append_multi_tti_report(report_path, second_result, second_result.final_config)
            lines = report_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "信噪比,BLER,EVM,EVM_SNR,RB位置,MCS阶数,总TTI数,误包数,码率,调制阶数,TBsize")
            self.assertEqual(len(lines), 3)

    def test_pusch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.channel_estimation.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.channel_estimation.channel_estimate.ndim, 3)
        self.assertEqual(result.rx.rx_grid.shape[0], config.link.num_rx_ant)
        self.assertEqual(result.rx.rx_grid.shape[1], config.carrier.n_subcarriers)
        self.assertEqual(result.rx.channel_estimation.channel_estimate.shape[1], config.link.num_prbs * 12)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.ndim, 2)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.shape[0], config.link.num_rx_ant)
        self.assertIsNotNone(result.transport_plan.size_bits)
        self.assertIs(result.crc_ok, True)
        self.assertIsNotNone(result.evm_percent)
        self.assertIsNotNone(result.evm_snr_linear)

    def test_pusch_dfts_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.channel_estimation.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.channel_estimation.channel_estimate.shape[1], config.link.num_prbs * 12)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.ndim, 2)
        self.assertIs(result.crc_ok, True)

    def test_pusch_awgn_multi_rx_branches(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        config.link.num_rx_ant = 4
        result = PuschSimulation(config).run()
        self.assertEqual(result.rx.rx_waveform.ndim, 2)
        self.assertEqual(result.rx.rx_waveform.shape[0], 4)
        self.assertEqual(result.rx.rx_grid.shape[0], 4)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.shape[0], 4)
        self.assertEqual(result.rx.channel_estimation.channel_estimate.shape, (4, config.link.num_prbs * 12, config.carrier.symbols_per_slot))
        self.assertIs(result.crc_ok, True)

    def test_channel_estimator_requires_explicit_rx_antenna_axis(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        user_sc = config.link.num_prbs * 12
        rx_grid_2d = np.ones((user_sc, config.carrier.symbols_per_slot), dtype=np.complex128)
        dmrs_mask = np.zeros((user_sc, config.carrier.symbols_per_slot), dtype=bool)
        with self.assertRaisesRegex(ValueError, "num_rx_ant"):
            LeastSquaresEstimator().estimate(
                rx_grid_2d,
                np.array([], dtype=np.complex128),
                dmrs_mask,
                config,
            )

    def test_channel_estimator_runs_ls_frequency_and_time_steps_separately(self):
        estimator = LeastSquaresEstimator()
        rx_grid = np.zeros((2, 4, 3), dtype=np.complex128)
        dmrs_mask = np.zeros((4, 3), dtype=bool)
        dmrs_mask[[0, 2], 1] = True
        reference_dmrs = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex128)
        expected_pilot_ls = np.array(
            [
                [2.0 + 1.0j, 4.0 + 3.0j],
                [-1.0 + 2.0j, 1.0 - 2.0j],
            ],
            dtype=np.complex128,
        )
        rx_grid[:, [0, 2], 1] = reference_dmrs[np.newaxis, :] * expected_pilot_ls

        pilot_ls_result = estimator.estimate_pilot_re_ls(rx_grid, reference_dmrs, dmrs_mask)
        frequency_result = estimator.interpolate_frequency(
            pilot_ls_result,
            num_subcarriers=rx_grid.shape[1],
        )
        channel_estimate = estimator.interpolate_time(
            frequency_result,
            num_symbols=rx_grid.shape[2],
        )

        self.assertTrue(np.array_equal(pilot_ls_result.dmrs_symbol_indices, np.array([1])))
        self.assertTrue(
            np.array_equal(pilot_ls_result.pilot_subcarriers_by_symbol[0], np.array([0, 2]))
        )
        self.assertTrue(
            np.allclose(pilot_ls_result.pilot_estimates_by_symbol[0], expected_pilot_ls)
        )
        self.assertEqual(frequency_result.channel_estimates.shape, (2, 1, 4))
        self.assertTrue(
            np.allclose(
                frequency_result.channel_estimates[0, 0],
                np.array([2.0 + 1.0j, 3.0 + 2.0j, 4.0 + 3.0j, 4.0 + 3.0j]),
            )
        )
        self.assertEqual(channel_estimate.shape, rx_grid.shape)
        self.assertTrue(
            np.allclose(channel_estimate[0, :, 0], frequency_result.channel_estimates[0, 0])
        )
        self.assertTrue(
            np.allclose(channel_estimate[1, :, 2], frequency_result.channel_estimates[1, 0])
        )

    def test_pusch_awgn_with_interference_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertEqual(len(result.interference_reports), 2)
        self.assertTrue(all(report.scale >= 0.0 for report in result.interference_reports))
        self.assertEqual(result.interference_reports[0].channel_model, "AWGN")
        self.assertEqual(result.interference_reports[1].channel_model, "TDL")
        self.assertIsNotNone(result.interference_reports[1].config_path)
        self.assertIs(result.crc_ok, True)

    def test_file_based_interference_uses_referenced_user_config(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        source = config.interference.sources[1]
        self.assertNotIn("channel_model", source.explicit_fields)

        interferer_cfg = InterferenceMixer(DefaultSimulationComponentFactory())._build_interferer_config(
            config,
            source,
            index=1,
        )

        self.assertEqual(interferer_cfg.channel.model, "TDL")
        self.assertEqual(interferer_cfg.scrambling.rnti, 18002)
        self.assertEqual(interferer_cfg.scrambling.n_id, 202)
        self.assertEqual(interferer_cfg.dmrs.scrambling_id0, 202)
        self.assertEqual(interferer_cfg.dmrs.n_pusch_identity, 202)
        self.assertEqual(interferer_cfg.link.num_rx_ant, config.link.num_rx_ant)
        self.assertEqual(interferer_cfg.carrier.cell_bandwidth_rbs, config.carrier.cell_bandwidth_rbs)
        self.assertEqual(interferer_cfg.slot_index, config.slot_index)
        self.assertEqual(interferer_cfg.link.prb_start, 12)
        self.assertEqual(interferer_cfg.link.num_prbs, 8)
        self.assertEqual(interferer_cfg.interference.sources, ())
        self.assertTrue(interferer_cfg.simulation.bypass_channel_coding)
        self.assertFalse(interferer_cfg.plotting.enabled)
        self.assertFalse(interferer_cfg.channel.params["add_noise"])
        components = DefaultSimulationComponentFactory().create_components(interferer_cfg)
        self.assertIsInstance(components.transmitter.coder, RandomBitCoder)

    def test_interference_prb_allocation_must_fit_main_carrier(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        source = config.interference.sources[0]
        source.prb_start = config.carrier.cell_bandwidth_rbs - 1
        source.num_prbs = 2
        source.explicit_fields = frozenset(set(source.explicit_fields) | {"prb_start", "num_prbs"})
        with self.assertRaisesRegex(ValueError, "exceeds the active BWP"):
            InterferenceMixer(DefaultSimulationComponentFactory())._build_interferer_config(
                config,
                source,
                index=0,
            )

    def test_pusch_cp_ofdm_low_mcs_with_dmrs_no_data_symbol(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.link.waveform = "CP-OFDM"
        config.link.num_tx_ant = 1
        config.link.num_rx_ant = 1
        config.link.mcs.table = "qam64"
        config.link.mcs.index = 0
        config.dmrs.data_mux_enabled = False
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertEqual(result.bit_error_rate, 0.0)
        self.assertIs(result.crc_ok, True)

    def test_bypass_channel_coding_uses_random_coded_bits_without_crc(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        config.simulation.bypass_channel_coding = True
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertIsNone(result.crc_ok)
        self.assertEqual(result.tx.coded_bits.size, result.transport_plan.codewords[0].coded_bit_capacity)
        self.assertEqual(result.rx.decoded_bits.size, result.tx.coded_bits.size)

    def test_bypass_channel_coding_multi_tti_reports_bler_as_nan(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.simulation.num_ttis = 3
        config.simulation.bypass_channel_coding = True
        config.plotting.enabled = False
        result = MultiTtiSimulationRunner(config).run()
        self.assertEqual(result.packet_errors, 0)
        self.assertTrue(np.isnan(result.block_error_rate))
        self.assertTrue(all(tti_result.crc_ok is None for tti_result in result.tti_results))

    def test_external_frequency_response_time_domain_channel_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.plotting.enabled = False
        config.link.num_tx_ant = 1
        config.link.num_rx_ant = 1
        config.channel.model = "EXTERNAL_FREQRESP_TD"
        config.channel.params = {
            "frequency_response": [[1.0, 0.0]] * config.carrier.n_subcarriers,
            "add_noise": False,
        }
        result = PuschSimulation(config).run()
        self.assertIs(result.crc_ok, True)
        self.assertEqual(result.bit_error_rate, 0.0)
        self.assertEqual(result.tx.resource_grid.shape, (1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot))
        self.assertEqual(result.tx.waveform.shape[0], 1)
        self.assertEqual(result.rx.rx_waveform.shape[0], 1)
        self.assertEqual(result.rx.rx_grid.ndim, 3)

    def test_external_frequency_response_frequency_domain_channel_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.plotting.enabled = False
        config.link.num_tx_ant = 1
        config.link.num_rx_ant = 1
        config.channel.model = "EXTERNAL_FREQRESP_FD"
        config.channel.params = {
            "frequency_response": [[1.0, 0.0]] * config.carrier.n_subcarriers,
            "add_noise": False,
        }
        result = PuschSimulation(config).run()
        self.assertIs(result.crc_ok, True)
        self.assertEqual(result.bit_error_rate, 0.0)
        self.assertEqual(result.tx.waveform.size, 0)
        self.assertEqual(result.tx.waveform.shape, (1, 0))
        self.assertEqual(result.rx.rx_waveform.shape, (1, 0))
        self.assertEqual(result.rx.rx_waveform.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)

    def test_external_frequency_response_frequency_domain_mimo_matrix_multiply(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.link.num_tx_ant = 2
        config.link.num_rx_ant = 3
        config.channel.params = {"add_noise": False}
        num_sc = config.carrier.n_subcarriers
        num_symbols = 4
        frequency_response = np.zeros((num_sc, 3, 2), dtype=np.complex128)
        frequency_response[:, 0, 0] = 1.0
        frequency_response[:, 0, 1] = 2.0
        frequency_response[:, 1, 0] = 0.5j
        frequency_response[:, 1, 1] = -1.0
        frequency_response[:, 2, 0] = 0.25
        frequency_response[:, 2, 1] = 0.75j
        config.channel.params["frequency_response"] = frequency_response
        tx_grid = np.zeros((2, num_sc, num_symbols), dtype=np.complex128)
        tx_grid[0] = 1.0 + 0.5j
        tx_grid[1] = -0.25 + 2.0j

        rx_grid, info = ExternalFrequencyResponseFrequencyDomainChannel().propagate_grid(tx_grid, config)
        expected = np.einsum("krt,tks->rks", frequency_response, tx_grid)

        self.assertEqual(rx_grid.shape, (3, num_sc, num_symbols))
        self.assertTrue(np.allclose(rx_grid, expected))
        self.assertEqual(info["noise_variance"], 0.0)

    def test_external_frequency_response_noise_rng_advances_on_same_channel_instance(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.link.num_tx_ant = 1
        config.link.num_rx_ant = 1
        config.channel.model = "EXTERNAL_FREQRESP_FD"
        config.channel.seed = 123
        config.channel.params = {
            "frequency_response": [[1.0, 0.0]] * config.carrier.n_subcarriers,
            "snr_db": 10.0,
        }
        tx_grid = np.ones((1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot), dtype=np.complex128)
        channel = DefaultChannelFactory().create(config)
        first, _ = channel.propagate_grid(tx_grid, config)
        second, _ = channel.propagate_grid(tx_grid, config)
        self.assertFalse(np.allclose(first, second))

    def test_external_frequency_response_fd_noise_power_uses_active_res(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.link.num_tx_ant = 1
        config.link.num_rx_ant = 1
        config.channel.model = "EXTERNAL_FREQRESP_FD"
        config.channel.seed = 123
        config.channel.params = {
            "frequency_response": [[1.0, 0.0]] * config.carrier.n_subcarriers,
            "snr_db": 0.0,
        }
        tx_grid = np.zeros((1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot), dtype=np.complex128)
        tx_grid[0, 0, 0] = 1.0
        _, info = DefaultChannelFactory().create(config).propagate_grid(tx_grid, config)
        self.assertAlmostEqual(info["noise_variance"], 1.0)

    def test_external_frequency_response_sample_config_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_external_freqresp_fd.yaml")
        config.plotting.enabled = False
        result = PuschSimulation(config).run()
        self.assertIs(result.crc_ok, True)
        self.assertEqual(result.bit_error_rate, 0.0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)

    def test_bwp_allocation_is_relative_to_active_bwp(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.bwp.start_rb = 10
        config.bwp.num_rbs = 32
        config.link.prb_start = 4
        config.link.num_prbs = 6
        expected_start = (10 + 4) * 12
        allocated = allocated_subcarriers(config)
        self.assertEqual(allocated[0], expected_start)
        self.assertEqual(allocated[-1], expected_start + 6 * 12 - 1)

    def test_bwp_center_frequency_is_derived_from_carrier_and_bwp(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.carrier.center_frequency_hz = 3.5e9
        config.carrier.cell_bandwidth_rbs = 52
        config.bwp.start_rb = 10
        config.bwp.num_rbs = 20
        expected = 3.5e9 + (10 + 20 / 2 - 52 / 2) * 12 * 30e3
        self.assertAlmostEqual(bwp_center_frequency_hz(config), expected)

    def test_ofdm_phase_compensation_round_trip_with_nonzero_bwp(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.bwp.start_rb = 8
        config.bwp.num_rbs = 32
        config.link.prb_start = 2
        config.link.num_prbs = 8
        rng = np.random.default_rng(123)
        grid = (
            rng.normal(size=(1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot))
            + 1j * rng.normal(size=(1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot))
        )
        processor = OfdmProcessor()
        waveform = processor.modulate(grid, config)
        recovered = processor.demodulate(waveform, config)
        self.assertTrue(np.allclose(recovered, grid, atol=1e-10))

    def test_ofdm_phase_compensation_can_be_disabled(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        grid = np.zeros((1, config.carrier.n_subcarriers, config.carrier.symbols_per_slot), dtype=np.complex128)
        grid[:, config.carrier.n_subcarriers // 2, :] = 1.0
        processor = OfdmProcessor()
        config.bwp.phase_compensation_enabled = False
        disabled = processor.modulate(grid, config)
        config.bwp.phase_compensation_enabled = True
        enabled = processor.modulate(grid, config)
        self.assertFalse(np.allclose(enabled, disabled))

    def test_time_domain_noise_variance_converts_to_frequency_domain_variance(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        self.assertEqual(
            time_to_frequency_noise_variance(0.25, config),
            0.25 * config.carrier.fft_size_effective,
        )

    def test_evm_metrics_use_rms_definition(self):
        reference = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
        measured = np.array([1.1 + 0.0j, 1.8 + 0.0j], dtype=np.complex128)
        evm_percent, evm_snr_linear = SharedChannelSimulation._compute_evm_metrics(reference, measured)
        expected_evm = np.sqrt((0.1**2 + 0.2**2) / (1.0**2 + 2.0**2))
        self.assertAlmostEqual(evm_percent, expected_evm * 100.0)
        self.assertAlmostEqual(evm_snr_linear, 1.0 / expected_evm**2)

    def test_resource_mapper_rejects_insufficient_data_symbols(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        mapper = FrequencyDomainResourceMapper(dmrs_generator=DmrsGenerator())
        data_re_count = mapper.count_data_re(config)
        with self.assertRaisesRegex(ValueError, "Insufficient data symbols"):
            mapper.map_to_grid(np.ones(data_re_count - 1, dtype=np.complex128), config)

    def test_pusch_cp_and_dfts_ofdm_run_with_nonzero_bwp_offset(self):
        for waveform in ("CP-OFDM", "DFT-s-OFDM"):
            with self.subTest(waveform=waveform):
                config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
                config.plotting.enabled = False
                config.channel.params["snr_db"] = 35.0
                config.link.waveform = waveform
                config.link.num_tx_ant = 1
                config.link.num_rx_ant = 1
                config.bwp.start_rb = 8
                config.bwp.num_rbs = 32
                config.link.prb_start = 2
                config.link.num_prbs = 8
                config.link.mcs.table = "qam256"
                config.link.mcs.index = 0
                config.dmrs.config_type = 1
                config.dmrs.data_mux_enabled = False
                result = PuschSimulation(config).run()
                self.assertIs(result.crc_ok, True)
                self.assertEqual(result.rx.rx_grid.shape[1], config.carrier.n_subcarriers)
                self.assertEqual(result.rx.channel_estimation.channel_estimate.shape[1], config.link.num_prbs * 12)


@pytest.mark.slow
class BaselineRegressionTest(unittest.TestCase):
    def test_pusch_baseline_cases(self):
        cases = [
            ("pusch_cp_ofdm_qam256_mcs0_awgn_snr0.yaml", 0.0, None),
            ("pusch_dfts_ofdm_qam256_mcs0_awgn_snr0.yaml", 0.0, None),
            ("pusch_cp_ofdm_qam256_mcs27_awgn_snr50.yaml", 0.0, None),
            ("pusch_dfts_ofdm_qam256_mcs27_awgn_snr50.yaml", 0.0, 2.0),
        ]
        baseline_dir = ROOT / "configs" / "baseline"
        for file_name, expected_bler, evm_upper_bound in cases:
            config = load_simulation_config(baseline_dir / file_name)
            result = MultiTtiSimulationRunner(config).run()
            self.assertEqual(
                result.block_error_rate,
                expected_bler,
                msg=f"{file_name} should satisfy BLER={expected_bler}",
            )
            if evm_upper_bound is not None:
                self.assertIsNotNone(result.average_evm_percent, msg=f"{file_name} should report average EVM")
                self.assertLess(
                    result.average_evm_percent,
                    evm_upper_bound,
                    msg=f"{file_name} should satisfy average EVM < {evm_upper_bound}%",
                )


class PdschAwgnSmokeTest(unittest.TestCase):
    def test_pdsch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PdschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.channel_estimation.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.ndim, 2)
        self.assertIs(result.crc_ok, True)

    def test_pdsch_qam1024_non_bypass_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.plotting.enabled = False
        config.channel.params["snr_db"] = 60.0
        config.link.mcs.table = "qam1024"
        config.link.mcs.index = 23
        result = PdschSimulation(config).run()
        self.assertEqual(result.transport_plan.mcs.modulation, "1024QAM")
        self.assertIs(result.crc_ok, True)


class VisualizationSmokeTest(unittest.TestCase):
    def test_save_simulation_plots(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 20.0
        result = PuschSimulation(config).run()
        paths = save_simulation_plots(result, config, ROOT / "outputs" / "tests", "smoke")
        self.assertTrue(paths["constellation"].exists())
        self.assertTrue(paths["pilot_estimates"].exists())
        self.assertTrue(paths["rx_time"].exists())
        self.assertTrue(paths["rx_freq"].exists())

    def test_save_simulation_plots_includes_generic_artifacts(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.plotting.enabled = False
        result = PuschSimulation(config).run()
        result.rx.plot_artifacts = (
            PlotArtifact(
                name="estimator_debug_metric",
                values=np.array([1.0, 0.5, 0.25]),
                title="Estimator Debug Metric",
                plot_type="magnitude",
            ),
        )
        paths = save_simulation_plots(result, config, ROOT / "outputs" / "tests", "artifact")
        self.assertTrue(paths["artifact_estimator_debug_metric"].exists())

    def test_save_simulation_plots_includes_runtime_context_artifacts(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.plotting.enabled = False
        runtime_context = SimulationRuntimeContext()
        result = PuschSimulation(config, runtime_context=runtime_context).run()
        get_runtime_context().set("channel_estimation", "debug_scalar", 1.0)
        get_runtime_context().add_plot_artifact(
            PlotArtifact(
                name="runtime_debug_metric",
                values=np.array([0.2, 0.4, 0.6]),
                title="Runtime Debug Metric",
                plot_type="magnitude",
            )
        )
        paths = save_simulation_plots(result, config, ROOT / "outputs" / "tests", "runtime")
        self.assertEqual(get_runtime_context().get("channel_estimation", "debug_scalar"), 1.0)
        self.assertTrue(paths["context_runtime_debug_metric"].exists())


class DmrsSequenceTest(unittest.TestCase):
    def test_type_a_single_symbol_dmrs_table_additional_position_zero(self):
        cases = [
            (8, 0, (2,)),
            (8, 1, (2, 7)),
            (9, 0, (2,)),
            (9, 1, (2, 7)),
            (10, 0, (2,)),
            (10, 1, (2, 9)),
        ]
        for num_symbols, additional_position, expected in cases:
            with self.subTest(num_symbols=num_symbols, additional_position=additional_position):
                positions = resolve_dmrs_symbol_indices(
                    channel_type="PDSCH",
                    start_symbol=0,
                    num_symbols=num_symbols,
                    mapping_type="A",
                    additional_positions=additional_position,
                    max_length=1,
                    type_a_position=2,
                )
                self.assertEqual(positions, expected)

    def test_transform_precoded_pusch_dmrs_short_lengths(self):
        generator = DmrsGenerator()
        for num_prbs in (1, 2, 3, 4):
            config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
            config.link.num_prbs = num_prbs
            symbols = generator.generate_for_symbol(symbol=2, config=config)
            self.assertEqual(symbols.size, num_prbs * 6)
            self.assertTrue(np.allclose(np.abs(symbols), 1.0))

    def test_low_papr_type1_length_30_uses_spec_closed_form(self):
        generator = DmrsGenerator()
        n = np.arange(30, dtype=np.float64)
        for group in (0, 7, 29):
            actual = generator._low_papr_type1(u=group, v=0, length=30)
            expected = np.exp(-1j * np.pi * (group + 1) * (n + 1) * (n + 2) / 31.0)
            self.assertTrue(np.allclose(actual, expected))

    def test_transform_precoded_pusch_rejects_dmrs_config_type2(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.dmrs.config_type = 2
        with self.assertRaisesRegex(ValueError, "only supports DMRS configuration type 1"):
            config._validate_protocol_constraints()

    def test_transform_precoded_pusch_rejects_data_dmrs_symbol_multiplexing(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.dmrs.data_mux_enabled = True
        with self.assertRaisesRegex(ValueError, "does not support data/DMRS symbol multiplexing"):
            config._validate_protocol_constraints()

    def test_transform_precoded_pusch_requires_two_cdm_groups_without_data(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.dmrs.num_cdm_groups_without_data = 1
        with self.assertRaisesRegex(ValueError, "requires num_cdm_groups_without_data = 2"):
            config._validate_protocol_constraints()

    def test_cp_ofdm_dmrs_can_disable_data_multiplexing(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.dmrs.data_mux_enabled = False
        mapper = FrequencyDomainResourceMapper(dmrs_generator=DmrsGenerator())
        grid, dmrs_mask, data_mask, _ = mapper.map_to_grid(np.ones(mapper.count_data_re(config), dtype=np.complex128), config)
        dmrs_symbols = np.where(np.any(dmrs_mask, axis=0))[0]
        self.assertGreater(dmrs_symbols.size, 0)
        for symbol_idx in dmrs_symbols:
            self.assertEqual(int(np.count_nonzero(data_mask[:, symbol_idx])), 0)

    def test_type1_dmrs_power_boost_without_data_multiplexing(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.dmrs.config_type = 1
        config.dmrs.data_mux_enabled = False
        config.link.mcs.table = None
        config.link.mcs.index = None
        config.link.modulation = "QPSK"
        mapper = FrequencyDomainResourceMapper(dmrs_generator=DmrsGenerator())
        grid, dmrs_mask, data_mask, _ = mapper.map_to_grid(
            np.ones(mapper.count_data_re(config), dtype=np.complex128),
            config,
        )
        dmrs_power = float(np.mean(np.abs(grid[dmrs_mask]) ** 2))
        data_power = float(np.mean(np.abs(grid[data_mask]) ** 2))
        self.assertAlmostEqual(dmrs_power / data_power, 2.0, places=6)


class ConfigLoaderTest(unittest.TestCase):
    def test_load_yaml_and_json_config(self):
        yaml_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        yaml_multi_tti_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn_multi_tti.yaml")
        yaml_pdsch_cfg = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        yaml_interference_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        yaml_replay_cfg = load_simulation_config(ROOT / "configs" / "pusch_replay_template.yaml")
        self.assertEqual(yaml_cfg.link.channel_type, "PUSCH")
        self.assertEqual(yaml_multi_tti_cfg.simulation.num_ttis, 20)
        self.assertEqual(yaml_pdsch_cfg.link.channel_type, "PDSCH")
        self.assertGreater(yaml_cfg.carrier.fft_size_effective, 0)
        self.assertEqual(yaml_cfg.carrier.cyclic_prefix_mode, "NORMAL")
        self.assertEqual(len(yaml_cfg.carrier.cyclic_prefix_lengths), yaml_cfg.carrier.symbols_per_slot)
        self.assertEqual(yaml_cfg.scrambling.rnti, 4660)
        self.assertEqual(yaml_pdsch_cfg.scrambling.effective_data_scrambling_id, 1)
        self.assertEqual(len(yaml_interference_cfg.interference.sources), 2)
        self.assertEqual(yaml_interference_cfg.interference.sources[0].channel_model, "AWGN")
        self.assertEqual(
            Path(yaml_interference_cfg.interference.sources[0].config_path),
            (ROOT / "configs" / "interferers" / "pusch_interferer_awgn.yaml").resolve(),
        )
        self.assertFalse(yaml_cfg.harq.enabled)
        self.assertEqual(yaml_cfg.decoder.ldpc_max_iterations, 25)
        self.assertEqual(
            Path(yaml_replay_cfg.waveform_input.waveform_path),
            (ROOT / "inputs" / "pusch_capture.txt").resolve(),
        )

    def test_resolve_frequency_response_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            response_path = tmpdir_path / "freq_resp.txt"
            response_path.write_text("\n".join(["1.0 0.0"] * 624), encoding="utf-8")
            config_path = tmpdir_path / "freq_resp.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "carrier:",
                        "  cell_bandwidth_rbs: 52",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "channel:",
                        "  model: EXTERNAL_FREQRESP_TD",
                        "  params:",
                        "    frequency_response_path: ./freq_resp.txt",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_simulation_config(config_path)
            self.assertEqual(
                Path(cfg.channel.params["frequency_response_path"]),
                response_path.resolve(),
            )

    def test_channel_config_path_merges_external_file_and_inline_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            response_path = tmpdir_path / "freq_resp.txt"
            response_path.write_text("\n".join(["1.0 0.0"] * 624), encoding="utf-8")
            channel_path = tmpdir_path / "awgn_channel.yaml"
            channel_path.write_text(
                "\n".join(
                    [
                        "model: AWGN",
                        "seed: 11",
                        "geometry:",
                        "  tx_position_m: [0.0, 0.0, 25.0]",
                        "  rx_position_m: [100.0, 0.0, 1.5]",
                        "  ue_velocity_vector_mps: [3.0, 0.0, 4.0]",
                        "params:",
                        "  snr_db: 1.0",
                        "  frequency_response_path: ./freq_resp.txt",
                    ]
                ),
                encoding="utf-8",
            )
            config_path = tmpdir_path / "sim.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "channel:",
                        "  config_path: ./awgn_channel.yaml",
                        "  seed: 12",
                        "  geometry:",
                        "    rx_position_m: [50.0, 0.0, 1.5]",
                        "  params:",
                        "    snr_db: 7.0",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_simulation_config(config_path)

            self.assertEqual(cfg.channel.model, "AWGN")
            self.assertEqual(cfg.channel.seed, 12)
            self.assertEqual(Path(cfg.channel.config_path), channel_path.resolve())
            self.assertEqual(cfg.channel.params.snr_db, 7.0)
            self.assertEqual(Path(cfg.channel.params.frequency_response_path), response_path.resolve())
            self.assertEqual(cfg.channel.geometry.tx_position_m, [0.0, 0.0, 25.0])
            self.assertEqual(cfg.channel.geometry.rx_position_m, [50.0, 0.0, 1.5])
            self.assertEqual(cfg.channel.geometry.ue_velocity_vector_mps, [3.0, 0.0, 4.0])

    def test_legacy_channel_carrier_frequency_migrates_to_carrier_center_frequency(self):
        cfg = SimulationConfig.from_mapping(
            {
                "carrier": {"cell_bandwidth_rbs": 24},
                "link": {"channel_type": "PUSCH", "waveform": "CP-OFDM", "num_prbs": 12},
                "channel": {
                    "model": "TDL",
                    "params": {"carrier_frequency_hz": 2.6e9, "profile": "TDL-C"},
                },
            }
        )
        self.assertEqual(cfg.carrier.center_frequency_hz, 2.6e9)

    def test_bwp_validation_rejects_out_of_bounds_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeds the active BWP"):
            SimulationConfig.from_mapping(
                {
                    "carrier": {"cell_bandwidth_rbs": 24, "center_frequency_hz": 3.5e9},
                    "bwp": {"start_rb": 4, "num_rbs": 8},
                    "link": {"channel_type": "PUSCH", "waveform": "CP-OFDM", "prb_start": 4, "num_prbs": 8},
                    "channel": {"model": "AWGN"},
                }
            )

    def test_config_rejects_unimplemented_multi_layer_and_multi_codeword(self):
        with self.assertRaisesRegex(NotImplementedError, "one transmission layer"):
            SimulationConfig.from_mapping(
                {
                    "link": {"channel_type": "PUSCH", "waveform": "CP-OFDM", "num_layers": 2},
                    "channel": {"model": "AWGN"},
                }
            )
        with self.assertRaisesRegex(NotImplementedError, "one active codeword"):
            SimulationConfig.from_mapping(
                {
                    "link": {"channel_type": "PDSCH", "waveform": "CP-OFDM", "num_codewords": 2},
                    "channel": {"model": "AWGN"},
                }
            )

    def test_channel_seed_controls_rng(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.channel.seed = 123
        waveform = np.ones((1, 64), dtype=np.complex128)
        first, _ = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
        second, _ = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
        self.assertTrue(np.allclose(first, second))

        cfg.channel.seed = "auto"
        third, _ = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
        fourth, _ = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
        self.assertFalse(np.allclose(third, fourth))

    def test_repository_mimo_frequency_response_example_shape(self):
        response = load_frequency_response(path=ROOT / "inputs" / "mimo_frequency_response_24rb_2rx2tx.txt")
        self.assertEqual(response.shape, (24 * 12, 4))

        tap_rows = load_frequency_response(path=ROOT / "inputs" / "mimo_time_domain_taps_2rx2tx_8tap.txt")
        self.assertEqual(tap_rows.shape, (8, 4))

    def test_load_yaml_with_utf8_bom_and_chinese_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_path = tmpdir_path / "bom_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "# 中文注释：这是一个 Windows 兼容性回归测试",
                        "carrier:",
                        "  cell_bandwidth_rbs: 52",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "channel:",
                        "  model: AWGN",
                        "  params:",
                        "    snr_db: 30.0",
                    ]
                ),
                encoding="utf-8-sig",
            )
            cfg = load_simulation_config(config_path)
            self.assertEqual(cfg.link.channel_type, "PUSCH")

    def test_dynamic_config_fields_in_known_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dynamic_known_sections.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "carrier:",
                        "  cell_bandwidth_rbs: 16",
                        "  n_subcarriers: 999",
                        "  custom_carrier:",
                        "    nested_value: 123",
                        "dmrs:",
                        "  symbol_positions: [2]",
                        "  custom_dmrs_value: enabled",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "  num_prbs: 16",
                        "  mcs:",
                        "    table: qam256",
                        "    index: 0",
                        "    custom_mcs_flag: true",
                        "channel:",
                        "  model: AWGN",
                        "  params:",
                        "    snr_db: 12.5",
                        "    nested:",
                        "      points:",
                        "        - 1",
                        "        - name: two",
                        "    items: method-name-collision",
                        "interference:",
                        "  monitor_label: monitor-a",
                        "  sources:",
                        "    - label: i0",
                        "      inr_db: 0",
                        "      custom_source:",
                        "        gain: 3",
                        "      channel_params:",
                        "        custom_param:",
                        "          delay: 4",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_simulation_config(config_path)

            self.assertEqual(cfg.carrier.n_subcarriers, 16 * 12)
            self.assertEqual(cfg.carrier.extras.n_subcarriers, 999)
            self.assertEqual(cfg.carrier.custom_carrier.nested_value, 123)
            self.assertEqual(cfg.dmrs.custom_dmrs_value, "enabled")
            self.assertTrue(cfg.link.mcs.custom_mcs_flag)
            self.assertEqual(cfg.channel.params.snr_db, 12.5)
            self.assertEqual(cfg.channel.params["nested"].points[1].name, "two")
            self.assertEqual(cfg.channel.params["items"], "method-name-collision")
            self.assertEqual(cfg.interference.monitor_label, "monitor-a")
            self.assertEqual(cfg.interference.sources[0].custom_source.gain, 3)
            self.assertEqual(cfg.interference.sources[0].channel_params.custom_param.delay, 4)
            self.assertEqual(cfg.interference.sources[0].explicit_fields, frozenset({"label", "inr_db", "custom_source", "channel_params"}))

    def test_file_based_interference_ignores_nested_interference_and_supports_inline_dmrs_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            interferer_path = tmpdir_path / "interferer.yaml"
            interferer_path.write_text(
                "\n".join(
                    [
                        "carrier:",
                        "  cell_bandwidth_rbs: 52",
                        "  subcarrier_spacing_khz: 30",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "  num_rx_ant: 1",
                        "  prb_start: 3",
                        "  num_prbs: 4",
                        "  mcs:",
                        "    table: qam256",
                        "    index: 2",
                        "channel:",
                        "  model: TDL",
                        "  params:",
                        "    profile: TDL-C",
                        "    delay_spread_ns: 300",
                        "    snr_db: 9",
                        "scrambling:",
                        "  rnti: 200",
                        "  n_id: 201",
                        "dmrs:",
                        "  config_type: 1",
                        "  data_mux_enabled: false",
                        "  scrambling_id0: 201",
                        "  n_pusch_identity: 201",
                        "interference:",
                        "  sources:",
                        "    - label: nested",
                        "      enabled: true",
                    ]
                ),
                encoding="utf-8",
            )
            main_path = tmpdir_path / "main.yaml"
            main_path.write_text(
                "\n".join(
                    [
                        "carrier:",
                        "  cell_bandwidth_rbs: 52",
                        "  subcarrier_spacing_khz: 30",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "  num_rx_ant: 4",
                        "channel:",
                        "  model: AWGN",
                        "interference:",
                        "  sources:",
                        "    - label: file-user",
                        "      config_path: ./interferer.yaml",
                        "      inr_db: -3",
                        "      prb_start: 5",
                        "      mcs:",
                        "        index: 7",
                        "      dmrs:",
                        "        scrambling_id0: 301",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_simulation_config(main_path)
            source = cfg.interference.sources[0]
            interferer_cfg = InterferenceMixer(DefaultSimulationComponentFactory())._build_interferer_config(
                cfg,
                source,
                index=0,
            )

            self.assertEqual(Path(source.config_path), interferer_path.resolve())
            self.assertEqual(interferer_cfg.channel.model, "TDL")
            self.assertEqual(interferer_cfg.channel.params.profile, "TDL-C")
            self.assertEqual(interferer_cfg.link.num_rx_ant, 4)
            self.assertEqual(interferer_cfg.link.prb_start, 5)
            self.assertEqual(interferer_cfg.link.num_prbs, 4)
            self.assertEqual(interferer_cfg.link.mcs.table, "qam256")
            self.assertEqual(interferer_cfg.link.mcs.index, 7)
            self.assertEqual(interferer_cfg.scrambling.rnti, 200)
            self.assertEqual(interferer_cfg.scrambling.n_id, 201)
            self.assertEqual(interferer_cfg.dmrs.scrambling_id0, 301)
            self.assertEqual(interferer_cfg.dmrs.n_pusch_identity, 201)
            self.assertEqual(interferer_cfg.interference.sources, ())
            self.assertTrue(interferer_cfg.simulation.bypass_channel_coding)

    def test_dynamic_config_top_level_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dynamic_top_level.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "my_receiver:",
                        "  algorithm: neural_mmse",
                        "  hidden_size: 128",
                        "  debug:",
                        "    dump_llr: true",
                        "  stages:",
                        "    - name: front",
                        "      enabled: true",
                        "my-receiver:",
                        "  algorithm: invalid_identifier_name",
                        "link:",
                        "  channel_type: PUSCH",
                        "  waveform: CP-OFDM",
                        "channel:",
                        "  model: AWGN",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_simulation_config(config_path)

            self.assertEqual(cfg.my_receiver.algorithm, "neural_mmse")
            self.assertEqual(cfg.my_receiver.hidden_size, 128)
            self.assertTrue(cfg.my_receiver.debug.dump_llr)
            self.assertEqual(cfg.my_receiver.stages[0].name, "front")
            self.assertEqual(cfg.extras.my_receiver.algorithm, "neural_mmse")
            self.assertEqual(cfg.extras["my-receiver"].algorithm, "invalid_identifier_name")
            with self.assertRaises(AttributeError):
                getattr(cfg, "my-receiver")
            self.assertEqual(cfg.channel.model, "AWGN")


class TransmissionPlanningTest(unittest.TestCase):
    def test_transport_block_plan_builds_codeword_metadata(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        mapper = FrequencyDomainResourceMapper(dmrs_generator=DmrsGenerator())
        data_re_count = mapper.count_data_re(config)
        plan = build_transport_block_plan(config, data_re_count)
        self.assertEqual(plan.num_codewords, 1)
        self.assertEqual(plan.codewords[0].rv, config.link.mcs.rv)
        self.assertEqual(plan.codewords[0].coded_bit_capacity, config.link.coded_bit_capacity)
        self.assertEqual(plan.size_bits, config.link.transport_block_size)


class LayerMappingTest(unittest.TestCase):
    def test_layer_mapper_exposes_per_layer_views(self):
        mapper = LayerMapper()
        symbols = np.arange(12, dtype=np.float64).astype(np.complex128)
        result = mapper.map_symbols(symbols, num_layers=3)
        self.assertEqual(len(result.layer_symbols), 3)
        self.assertTrue(np.array_equal(result.layer_symbols[0], symbols[0::3]))
        self.assertTrue(np.array_equal(result.layer_symbols[1], symbols[1::3]))
        self.assertTrue(np.array_equal(result.layer_symbols[2], symbols[2::3]))


class HarqManagerTest(unittest.TestCase):
    def test_harq_rv_progression_and_reset(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.harq.enabled = True
        manager = HarqManager(config.harq)
        rng = np.random.default_rng(7)

        first = manager.schedule(tti_index=0, tbs_bits=128, rng=rng)
        self.assertEqual(first.rv, 0)
        self.assertFalse(first.is_retransmission)

        manager.update(first.process_id, False)
        second = manager.schedule(tti_index=0 + config.harq.num_processes, tbs_bits=128, rng=rng)
        self.assertTrue(second.is_retransmission)
        self.assertEqual(second.rv, config.harq.rv_sequence[1])
        self.assertTrue(np.array_equal(first.transport_block, second.transport_block))

        manager.update(second.process_id, True)
        third = manager.schedule(tti_index=0 + 2 * config.harq.num_processes, tbs_bits=128, rng=rng)
        self.assertFalse(third.is_retransmission)
        self.assertEqual(third.rv, 0)
        self.assertFalse(np.array_equal(second.transport_block, third.transport_block))


class McsTableTest(unittest.TestCase):
    def test_pdsch_all_mcs_tables(self):
        cases = [
            ("qam64", 10, "16QAM"),
            ("qam256", 20, "256QAM"),
            ("qam64lowse", 21, "64QAM"),
            ("qam1024", 23, "1024QAM"),
        ]
        for table_name, index, modulation in cases:
            cfg = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
            cfg.link.mcs.table = table_name
            cfg.link.mcs.index = index
            entry = resolve_mcs(cfg)
            self.assertEqual(entry.modulation, modulation)
            self.assertGreater(entry.target_code_rate, 0.0)

    def test_pusch_transform_precoding_tables(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        cfg.link.mcs.table = "tp64qam"
        cfg.link.mcs.index = 0
        cfg.link.mcs.tp_pi2bpsk = True
        entry = resolve_mcs(cfg)
        self.assertEqual(entry.modulation, "PI/2-BPSK")

        cfg.link.mcs.table = "tp64lowse"
        cfg.link.mcs.index = 24
        cfg.link.mcs.tp_pi2bpsk = False
        entry = resolve_mcs(cfg)
        self.assertEqual(entry.modulation, "64QAM")

    def test_invalid_mcs_table_combination_raises(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.link.mcs.table = "qam1024"
        cfg.link.mcs.index = 23
        with self.assertRaises(ValueError):
            resolve_mcs(cfg)


class UlschLdpcRegressionTest(unittest.TestCase):
    def test_ldpc_lifting_set_index_is_selected_from_zc_table(self):
        for expected_index, lifting_sizes in enumerate(_get_z_array()):
            for zc in lifting_sizes:
                with self.subTest(zc=zc):
                    self.assertEqual(_lifting_set_index_from_zc(zc), expected_index)

    def test_ldpc_lifting_set_index_regression_for_bg2_small_blocks(self):
        self.assertEqual(_lifting_set_index_from_zc(10), 2)
        self.assertEqual(_lifting_set_index_from_zc(11), 5)
        self.assertEqual(_lifting_set_index_from_zc(12), 1)

    def test_rate_matching_accepts_1024qam_modulation_order(self):
        encoded = np.tile(np.array([[0], [1]], dtype=np.int8), (50, 1))
        matched = rate_match_ulsch_ldpc(
            encoded,
            out_length=100,
            rv=0,
            modulation="1024QAM",
            num_layers=1,
        )
        self.assertEqual(matched.size, 100)

    def test_local_ulsch_ldpc_chain_decodes_low_rate_cp_ofdm_case(self):
        tbs = 552
        target_code_rate = 120 / 1024
        info = get_ulsch_ldpc_info(tbs, target_code_rate)
        transport_block = np.random.default_rng(7).integers(0, 2, size=tbs, dtype=np.int8)
        tb_with_crc = nrCRCEncode(transport_block, info.crc)[:, 0].astype(np.int8)
        code_blocks = nrCodeBlockSegmentLDPC(tb_with_crc, info.base_graph)
        encoded = encode_ldpc_codeblocks(code_blocks, info.base_graph)
        rate_matched = rate_match_ulsch_ldpc(encoded, out_length=4608, rv=0, modulation="QPSK", num_layers=1)
        llrs = (1 - 2 * rate_matched.astype(np.float64)) * 50.0
        recovered = rate_recover_ulsch_ldpc(
            llrs,
            trblklen=tbs,
            target_code_rate=target_code_rate,
            rv=0,
            modulation="QPSK",
            num_layers=1,
        )
        decoded_cbs = decode_ulsch_ldpc(recovered, info, max_num_iter=25)
        tb_with_crc_hat, _ = nrCodeBlockDesegmentLDPC(decoded_cbs, info.base_graph, tbs + info.tb_crc_bits)
        decoded, crc_error = nrCRCDecode(tb_with_crc_hat.astype(np.int8), info.crc)
        self.assertEqual(int(np.asarray(crc_error).reshape(-1)[0]), 0)
        self.assertTrue(np.array_equal(np.asarray(decoded).reshape(-1)[:tbs].astype(np.int8), transport_block))

    def test_py3gpp_rate_recover_and_decode_contract_is_inconsistent(self):
        tbs = 552
        target_code_rate = 120 / 1024
        info = nrDLSCHInfo(tbs, target_code_rate)
        transport_block = np.random.default_rng(7).integers(0, 2, size=tbs, dtype=np.int8)
        tb_with_crc = nrCRCEncode(transport_block, info["CRC"])[:, 0].astype(np.int8)
        code_blocks = nrCodeBlockSegmentLDPC(tb_with_crc, info["BGN"])
        encoded = nrLDPCEncode(code_blocks, info["BGN"], algo="thangaraj")
        rate_matched = nrRateMatchLDPC(encoded, outlen=4608, rv=0, mod="QPSK", nLayers=1).astype(np.int8)
        llrs = (1 - 2 * rate_matched.astype(np.float64)) * 50.0
        recovered = nrRateRecoverLDPC(
            llrs,
            trblklen=tbs,
            R=target_code_rate,
            rv=0,
            mod="QPSK",
            nLayers=1,
        )
        self.assertEqual(recovered.shape[0], int(info["N"]))
        decoded_cbs, _ = nrLDPCDecode(recovered, info["BGN"], maxNumIter=25)
        tb_with_crc_hat, _ = nrCodeBlockDesegmentLDPC(decoded_cbs, info["BGN"], tbs + info["L"])
        _, crc_error = nrCRCDecode(tb_with_crc_hat.astype(np.int8), info["CRC"])
        self.assertNotEqual(int(np.asarray(crc_error).reshape(-1)[0]), 0)

    def test_local_decoder_recovers_low_snr_cp_ofdm_multi_tti_case(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.plotting.enabled = False
        config.simulation.num_ttis = 20
        config.link.waveform = "CP-OFDM"
        config.link.num_prbs = 16
        config.link.mcs.table = "qam64"
        config.link.mcs.index = 0
        config.dmrs.data_mux_enabled = False
        config.channel.params["snr_db"] = 0.0

        result = MultiTtiSimulationRunner(config).run()
        self.assertEqual(result.packet_errors, 0)
        self.assertEqual(result.block_error_rate, 0.0)


class ComponentAbstractionTest(unittest.TestCase):
    def test_default_component_factory_builds_independent_stage_classes(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        factory = DefaultSimulationComponentFactory()
        components = factory.create_components(cfg)
        self.assertIsInstance(components.transmitter.mapper, FrequencyDomainResourceMapper)
        self.assertIsInstance(components.receiver.extractor, FrequencyDomainExtractor)
        self.assertIsNotNone(factory.create_channel_factory().create(cfg))

    def test_receiver_can_replace_estimation_equalization_demod_with_one_processor(self):
        class DirectLlrProcessor(ReceiverDataProcessor):
            def __init__(self):
                self.called = False
                self.rx_grid_shape = None

            def process(self, rx_grid, dmrs_symbols, dmrs_mask, data_mask, noise_variance, config):
                self.called = True
                self.rx_grid_shape = rx_grid.shape
                llr_count = int(config.link.coded_bit_capacity or 0)
                return ReceiverDataProcessingResult(
                    llrs=np.ones(llr_count, dtype=np.float64),
                )

        class DirectProcessorFactory(DefaultSimulationComponentFactory):
            def __init__(self, processor):
                self.processor = processor

            def create_components(self, config):
                components = super().create_components(config)
                return replace(
                    components,
                    receiver=replace(
                        components.receiver,
                        data_processor=self.processor,
                    ),
                )

        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.simulation.bypass_channel_coding = True
        cfg.plotting.enabled = False
        processor = DirectLlrProcessor()
        result = PuschSimulation(
            cfg,
            component_factory=DirectProcessorFactory(processor),
        ).run()

        self.assertTrue(processor.called)
        self.assertEqual(len(processor.rx_grid_shape), 3)
        self.assertEqual(result.rx.llrs.size, int(result.transport_plan.codewords[0].coded_bit_capacity))
        self.assertEqual(result.rx.channel_estimation.channel_estimate.size, 0)
        self.assertEqual(result.rx.equalized_symbols.size, 0)
        self.assertIsNone(result.crc_ok)

    def test_receiver_data_processor_pipeline_allows_arbitrary_stage_composition(self):
        class FeatureStage(ReceiverProcessingStage):
            def process(self, context):
                context.metadata["feature_shape"] = context.rx_grid.shape
                return context

        class DirectLlrStage(ReceiverProcessingStage):
            def process(self, context):
                context.llrs = np.ones(int(context.config.link.coded_bit_capacity or 0), dtype=np.float64)
                return context

        class PipelineFactory(DefaultSimulationComponentFactory):
            def __init__(self, pipeline):
                self.pipeline = pipeline

            def create_components(self, config):
                components = super().create_components(config)
                return replace(
                    components,
                    receiver=replace(
                        components.receiver,
                        data_processor=self.pipeline,
                    ),
                )

        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.simulation.bypass_channel_coding = True
        cfg.plotting.enabled = False
        pipeline = ReceiverDataProcessorPipeline([FeatureStage(), DirectLlrStage()])
        result = PuschSimulation(
            cfg,
            component_factory=PipelineFactory(pipeline),
        ).run()

        self.assertEqual(result.rx.llrs.size, int(result.transport_plan.codewords[0].coded_bit_capacity))
        self.assertEqual(result.rx.channel_estimation.channel_estimate.size, 0)
        self.assertIsNone(result.crc_ok)

    def test_receiver_processor_can_replace_arbitrary_receiver_steps(self):
        class DirectReceiverProcessor(ReceiverProcessor):
            def __init__(self):
                self.called_from_waveform = False

            def receive(self, receiver, rx_waveform, dmrs_symbols, dmrs_mask, data_mask, noise_variance, config):
                self.called_from_waveform = True
                rx_grid = receiver.time_processor.demodulate(rx_waveform, config)
                return self.receive_from_grid(
                    receiver,
                    rx_grid,
                    dmrs_symbols,
                    dmrs_mask,
                    data_mask,
                    noise_variance,
                    config,
                    rx_waveform,
                )

            def receive_from_grid(self, receiver, rx_grid, dmrs_symbols, dmrs_mask, data_mask, noise_variance, config, rx_waveform=None):
                if rx_grid.ndim != 3:
                    raise ValueError("Custom receiver processor expects rx_grid shape (num_rx_ant, num_subcarriers, num_symbols).")
                llrs = np.ones(int(config.link.coded_bit_capacity or 0), dtype=np.float64)
                decoded_bits = receiver.decoder.decode(llrs, config)
                return RxPayload(
                    rx_waveform=(
                        np.empty((int(config.link.num_rx_ant), 0), dtype=np.complex128)
                        if rx_waveform is None
                        else rx_waveform
                    ),
                    rx_grid=rx_grid,
                    channel_estimation=ChannelEstimateResult(
                        channel_estimate=np.array([], dtype=np.complex128),
                        pilot_estimates=np.array([], dtype=np.complex128),
                        pilot_symbol_indices=np.array([], dtype=int),
                    ),
                    equalized_symbols=np.array([], dtype=np.complex128),
                    llrs=llrs,
                    decoded_bits=decoded_bits,
                    crc_ok=getattr(receiver.decoder, "last_crc_ok", None),
                    dmrs_symbols=dmrs_symbols,
                )

        class DirectReceiverFactory(DefaultSimulationComponentFactory):
            def __init__(self, processor):
                self.processor = processor

            def create_components(self, config):
                components = super().create_components(config)
                return replace(
                    components,
                    receiver=replace(
                        components.receiver,
                        receiver_processor=self.processor,
                    ),
                )

        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.simulation.bypass_channel_coding = True
        cfg.plotting.enabled = False
        processor = DirectReceiverProcessor()
        result = PuschSimulation(
            cfg,
            component_factory=DirectReceiverFactory(processor),
        ).run()

        self.assertTrue(processor.called_from_waveform)
        self.assertEqual(result.rx.llrs.size, int(result.transport_plan.codewords[0].coded_bit_capacity))
        self.assertEqual(result.rx.channel_estimation.channel_estimate.size, 0)
        self.assertIsNone(result.crc_ok)


class FadingChannelSmokeTest(unittest.TestCase):
    def test_tdl_cdl_profile_tables_match_38901_sanity_points(self):
        self.assertEqual({name: len(taps) for name, taps in TDL_PROFILES.items()}, {
            "TDL-A": 23,
            "TDL-B": 23,
            "TDL-C": 24,
            "TDL-D": 14,
            "TDL-E": 15,
        })
        self.assertEqual({name: len(profile.clusters) for name, profile in CDL_PROFILES.items()}, {
            "CDL-A": 23,
            "CDL-B": 23,
            "CDL-C": 24,
            "CDL-D": 14,
            "CDL-E": 15,
        })
        self.assertAlmostEqual(TDL_PROFILES["TDL-A"][0].normalized_delay, 0.0)
        self.assertAlmostEqual(TDL_PROFILES["TDL-A"][-1].normalized_delay, 9.6586)
        self.assertAlmostEqual(TDL_PROFILES["TDL-C"][5].power_db, 0.0)
        self.assertEqual(TDL_PROFILES["TDL-D"][0].fading, "LOS")
        self.assertEqual(TDL_PROFILES["TDL-E"][0].fading, "LOS")
        self.assertEqual(TDL_LOS_K_DB, {"TDL-D": 13.3, "TDL-E": 22.0})
        self.assertAlmostEqual(CDL_PROFILES["CDL-A"].xpr_db, 10.0)
        self.assertAlmostEqual(CDL_PROFILES["CDL-B"].c_asa_deg, 22.0)
        self.assertEqual(CDL_PROFILES["CDL-D"].clusters[0].fading, "LOS")
        self.assertEqual(CDL_PROFILES["CDL-E"].clusters[0].fading, "LOS")
        self.assertEqual(CDL_LOS_K_DB, {"CDL-D": 13.3, "CDL-E": 22.0})

    def test_tdl_channel_propagates(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl.yaml")
        waveform = np.ones((cfg.link.num_tx_ant, 2048), dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        self.assertIsInstance(channel, TdlChannel)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (1, waveform.shape[1]))
        self.assertGreater(info["path_delays_s"].size, 0)
        self.assertGreaterEqual(info["noise_variance"], 0.0)

    def test_tdl_channel_supports_multi_tx_multi_rx_and_path_overrides(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 4
        cfg.channel.params["path_delays_ns"] = [0.0, 70.0, 190.0]
        cfg.channel.params["path_powers_db"] = [0.0, -2.5, -7.0]
        waveform = np.ones((cfg.link.num_tx_ant, 2048), dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (4, waveform.shape[1]))
        self.assertEqual(info["path_coefficients"].shape[:3], (4, 2, 3))
        self.assertTrue(np.allclose(info["path_delays_s"], np.array([0.0, 70.0, 190.0]) * 1e-9))

    def test_channel_geometry_derives_distance_and_velocity_angles(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.channel.geometry.tx_position_m = [0.0, 0.0, 0.0]
        cfg.channel.geometry.rx_position_m = [3.0, 4.0, 0.0]
        cfg.channel.geometry.ue_velocity_vector_mps = [0.0, 3.0, 4.0]

        info = TdlChannel._channel_geometry_info(cfg)

        self.assertAlmostEqual(info["tx_rx_distance_m"], 5.0)
        self.assertTrue(np.allclose(info["los_unit_vector_tx_to_rx"], np.array([0.6, 0.8, 0.0])))
        self.assertAlmostEqual(info["ue_speed_mps"], 5.0)
        self.assertAlmostEqual(info["ue_azimuth_deg"], 90.0)
        self.assertAlmostEqual(info["ue_zenith_deg"], np.rad2deg(np.arccos(4.0 / 5.0)))

    def test_channel_profile_config_files_can_propagate(self):
        channel_files = [
            ROOT / "configs" / "channels" / "awgn.yaml",
            *(ROOT / "configs" / "channels" / f"tdl_{name}.yaml" for name in "abcde"),
            *(ROOT / "configs" / "channels" / f"cdl_{name}.yaml" for name in "abcde"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for channel_path in channel_files:
                config_path = tmpdir_path / f"{channel_path.stem}_sim.yaml"
                config_path.write_text(
                    "\n".join(
                        [
                            "carrier:",
                            "  cell_bandwidth_rbs: 24",
                            "link:",
                            "  channel_type: PUSCH",
                            "  waveform: CP-OFDM",
                            "  num_tx_ant: 1",
                            "  num_rx_ant: 1",
                            "channel:",
                            f"  config_path: {channel_path}",
                            "  params:",
                            "    add_noise: false",
                        ]
                    ),
                    encoding="utf-8",
                )
                cfg = load_simulation_config(config_path)
                waveform = np.ones((1, 256), dtype=np.complex128)
                rx_waveform, info = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
                self.assertEqual(rx_waveform.shape, (1, waveform.shape[1]), msg=channel_path.name)
                self.assertIn("snr_db", info, msg=channel_path.name)

    def test_tdl_iid_mimo_is_not_single_outer_product(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 2
        cfg.channel.params["add_noise"] = False
        cfg.channel.params["ue_speed_mps"] = 0.0
        cfg.channel.params["tdl_mimo_method"] = "iid"
        waveform = np.ones((2, 512), dtype=np.complex128)
        _rx_waveform, info = TdlChannel(rng=np.random.default_rng(1)).propagate(waveform, cfg)
        first_path_matrix = info["path_coefficients"][:, :, 0, 0]
        self.assertGreater(abs(np.linalg.det(first_path_matrix)), 1e-6)

    def test_tdl_correlation_matrices_are_applied(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 2
        cfg.channel.params["add_noise"] = False
        cfg.channel.params["tdl_mimo_method"] = "correlated"
        cfg.channel.params["tdl_rx_correlation"] = [[1.0, 0.7], [0.7, 1.0]]
        cfg.channel.params["tdl_tx_correlation"] = [[1.0, 0.3], [0.3, 1.0]]
        waveform = np.ones((2, 256), dtype=np.complex128)
        _rx_waveform, info = TdlChannel(rng=np.random.default_rng(2)).propagate(waveform, cfg)
        self.assertEqual(info["path_coefficients"].shape[:2], (2, 2))

    def test_tdl_explicit_spatial_filter_is_applied(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 2
        cfg.channel.params["add_noise"] = False
        cfg.channel.params["spatial_filter"] = [[1.0, 0.0], [0.0, 1.0]]
        waveform = np.ones((2, 256), dtype=np.complex128)
        _rx_waveform, info = TdlChannel(rng=np.random.default_rng(7)).propagate(waveform, cfg)
        self.assertTrue(np.allclose(info["path_coefficients"][0, 1], 0.0))
        self.assertTrue(np.allclose(info["path_coefficients"][1, 0], 0.0))

    def test_cdl_dual_polarized_xpr_controls_cross_polar_leakage(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 2
        cfg.channel.params["add_noise"] = False
        cfg.channel.params["profile"] = "CDL-A"
        cfg.channel.params["tx_array"] = {"polarization": "dual"}
        cfg.channel.params["rx_array"] = {"polarization": "dual"}
        waveform = np.ones((2, 256), dtype=np.complex128)

        high_xpr_cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        high_xpr_cfg.link.num_tx_ant = 2
        high_xpr_cfg.link.num_rx_ant = 2
        high_xpr_cfg.channel.params.update(cfg.channel.params)
        high_xpr_cfg.channel.params["xpr_db"] = 100.0
        _rx_high, high_info = CdlChannel(rng=np.random.default_rng(3)).propagate(waveform, high_xpr_cfg)

        low_xpr_cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        low_xpr_cfg.link.num_tx_ant = 2
        low_xpr_cfg.link.num_rx_ant = 2
        low_xpr_cfg.channel.params.update(cfg.channel.params)
        low_xpr_cfg.channel.params["xpr_db"] = 0.0
        _rx_low, low_info = CdlChannel(rng=np.random.default_rng(3)).propagate(waveform, low_xpr_cfg)

        high_cross = np.mean(np.abs(high_info["path_coefficients"][0, 1]))
        low_cross = np.mean(np.abs(low_info["path_coefficients"][0, 1]))
        self.assertLess(high_cross, low_cross)

    def test_cdl_doppler_scales_with_carrier_frequency(self):
        waveform = np.ones((1, 256), dtype=np.complex128)
        low_cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        low_cfg.channel.params["add_noise"] = False
        low_cfg.channel.params["ue_speed_mps"] = 10.0
        low_cfg.carrier.center_frequency_hz = 1.0e9
        _rx_low, low_info = CdlChannel(rng=np.random.default_rng(4)).propagate(waveform, low_cfg)

        high_cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        high_cfg.channel.params["add_noise"] = False
        high_cfg.channel.params["ue_speed_mps"] = 10.0
        high_cfg.carrier.center_frequency_hz = 10.0e9
        _rx_high, high_info = CdlChannel(rng=np.random.default_rng(4)).propagate(waveform, high_cfg)

        self.assertAlmostEqual(high_info["max_doppler_hz"] / low_info["max_doppler_hz"], 10.0)

    def test_cdl_angle_scaling_applies_requested_mean_and_spread(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        cfg.channel.params["angle_scaling_enabled"] = True
        cfg.channel.params["desired_mean_aoa_deg"] = 15.0
        cfg.channel.params["desired_asa_deg"] = 5.0
        profile = CDL_PROFILES["CDL-A"]
        powers = CdlChannel._normalize_powers_db(np.array([cluster.power_db for cluster in profile.clusters]))
        angles = CdlChannel(rng=np.random.default_rng(5))._resolve_cluster_angles(cfg, profile, powers)
        radians = np.deg2rad(angles["aoa"])
        mean_aoa = np.rad2deg(np.angle(np.sum(powers * np.exp(1j * radians))))
        centered = np.rad2deg(np.angle(np.exp(1j * (radians - np.deg2rad(mean_aoa)))))
        spread = np.sqrt(np.sum(powers * centered**2))
        self.assertAlmostEqual(mean_aoa, 15.0, places=6)
        self.assertAlmostEqual(spread, 5.0, places=6)

    def test_tdl_cdl_reject_out_of_range_carrier_frequency(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl.yaml")
        cfg.carrier.center_frequency_hz = 200.0e9
        waveform = np.ones((cfg.link.num_tx_ant, 128), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "0.5 GHz and 100 GHz"):
            TdlChannel(rng=np.random.default_rng(6)).propagate(waveform, cfg)

    def test_all_tdl_and_cdl_profiles_can_be_instantiated(self):
        for model, profiles in (
            ("TDL", ("TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E")),
            ("CDL", ("CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E")),
        ):
            for profile in profiles:
                cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
                cfg.plotting.enabled = False
                cfg.channel.model = model
                cfg.channel.params = {"profile": profile, "delay_spread_ns": 300.0, "add_noise": False}
                waveform = np.ones((cfg.link.num_tx_ant, 1024), dtype=np.complex128)
                rx_waveform, info = DefaultChannelFactory().create(cfg).propagate(waveform, cfg)
                self.assertGreater(np.asarray(info["path_delays_s"]).size, 0, msg=f"{model}/{profile}")
                self.assertGreater(np.mean(np.abs(rx_waveform)), 0.0, msg=f"{model}/{profile}")


class WaveformReplaySmokeTest(unittest.TestCase):
    def test_replay_repository_example_config(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_replay_template.yaml")
        result = WaveformReplaySimulation(cfg).run()
        self.assertTrue(np.isnan(result.bit_error_rate))
        self.assertGreater(result.rx.decoded_bits.size, 0)
        self.assertIs(result.crc_ok, True)


class SweepSmokeTest(unittest.TestCase):
    def test_run_snr_sweep_and_write_csv(self):
        cfg = load_simulation_config(ROOT / "configs" / "baseline" / "pusch_cp_ofdm_qam256_mcs0_awgn_snr0.yaml")
        cfg.plotting.enabled = False
        points = run_snr_sweep(cfg, [0.0, 5.0])
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].snr_db, 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = write_snr_sweep_csv(Path(tmpdir) / "curve.csv", points)
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)

    def test_replay_waveform_file_into_receiver(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        cfg.link.num_rx_ant = 1
        factory = DefaultSimulationComponentFactory()
        components = factory.create_components(cfg)
        transmitter = build_transmitter(components)

        mcs_entry = apply_mcs_to_link(cfg)
        data_re = components.transmitter.mapper.count_data_re(cfg)
        cfg.link.coded_bit_capacity = data_re * mcs_entry.bits_per_symbol
        cfg.link.transport_block_size = resolve_transport_block_size(cfg, data_re)
        transport_block = np.random.default_rng(0).integers(
            0, 2, size=int(cfg.link.transport_block_size), dtype=np.int8
        )
        tx_payload = transmitter.transmit(transport_block, cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            waveform_path = Path(tmpdir) / "capture.txt"
            waveform_path.write_text(
                "\n".join(f"{sample.real:.12e} {sample.imag:.12e}" for sample in tx_payload.waveform[0])
            )
            replay_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
            replay_cfg.link.num_rx_ant = 1
            replay_cfg.waveform_input.waveform_path = str(waveform_path)
            replay_cfg.channel.params["snr_db"] = 50.0
            replay_cfg.link.transport_block_size = cfg.link.transport_block_size
            replay_cfg.link.coded_bit_capacity = cfg.link.coded_bit_capacity
            result = WaveformReplaySimulation(replay_cfg).run()
            self.assertTrue(np.isnan(result.bit_error_rate))
            self.assertTrue(np.array_equal(result.rx.decoded_bits[: transport_block.size], transport_block))
            self.assertIs(result.crc_ok, True)

    def test_cdl_channel_propagates(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        waveform = np.ones((cfg.link.num_tx_ant, 2048), dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        self.assertIsInstance(channel, CdlChannel)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (1, waveform.shape[1]))
        self.assertGreater(info["path_delays_s"].size, 0)
        self.assertGreaterEqual(info["noise_variance"], 0.0)

    def test_cdl_channel_supports_multi_tx_multi_rx(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 4
        waveform = np.ones((cfg.link.num_tx_ant, 2048), dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (4, waveform.shape[1]))
        self.assertEqual(info["path_coefficients"].shape[0], 4)
        self.assertEqual(info["path_coefficients"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
