from pathlib import Path
import sys
import tempfile
import unittest

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
from nr_phy_simu.io.multi_tti_report import append_multi_tti_report
from nr_phy_simu.common.ulsch_ldpc import (
    decode_ulsch_ldpc,
    encode_ldpc_codeblocks,
    get_ulsch_ldpc_info,
    rate_match_ulsch_ldpc,
    rate_recover_ulsch_ldpc,
)
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
from nr_phy_simu.scenarios.multi_tti import MultiTtiSimulationRunner
from nr_phy_simu.tx.resource_mapping import FrequencyDomainResourceMapper
from nr_phy_simu.rx.frequency_extraction import FrequencyDomainExtractor
from nr_phy_simu.visualization import save_simulation_plots
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.common.mcs import resolve_mcs
from nr_phy_simu.channels.channel_factory import DefaultChannelFactory
from nr_phy_simu.channels.tdl import TdlChannel
from nr_phy_simu.channels.cdl import CdlChannel
from nr_phy_simu.common.mcs import apply_mcs_to_link, resolve_transport_block_size
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation
from nr_phy_simu.scenarios.component_factory import build_transmitter


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
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.ndim, 2)
        self.assertEqual(result.rx.channel_estimation.pilot_estimates.shape[0], config.link.num_rx_ant)
        self.assertIsNotNone(config.link.transport_block_size)
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
        self.assertIs(result.crc_ok, True)

    def test_pusch_awgn_with_interference_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertEqual(len(result.interference_reports), 2)
        self.assertTrue(all(report.scale >= 0.0 for report in result.interference_reports))
        self.assertIs(result.crc_ok, True)

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


class DmrsSequenceTest(unittest.TestCase):
    def test_transform_precoded_pusch_dmrs_short_lengths(self):
        generator = DmrsGenerator()
        for num_prbs in (1, 2, 3, 4):
            config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
            config.link.num_prbs = num_prbs
            symbols = generator.generate_for_symbol(symbol=2, config=config)
            self.assertEqual(symbols.size, num_prbs * 6)
            self.assertTrue(np.allclose(np.abs(symbols), 1.0))

    def test_transform_precoded_pusch_rejects_dmrs_config_type2(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.dmrs.config_type = 2
        with self.assertRaisesRegex(ValueError, "only supports DMRS configuration type 1"):
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
            Path(yaml_replay_cfg.waveform_input.waveform_path),
            (ROOT / "inputs" / "pusch_capture.txt").resolve(),
        )


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


class FadingChannelSmokeTest(unittest.TestCase):
    def test_tdl_channel_propagates(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl.yaml")
        waveform = np.ones(2048, dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        self.assertIsInstance(channel, TdlChannel)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, waveform.shape)
        self.assertGreater(info["path_delays_s"].size, 0)
        self.assertGreaterEqual(info["noise_variance"], 0.0)

    def test_tdl_channel_supports_multi_tx_multi_rx_and_path_overrides(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_tdl_c.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 4
        cfg.channel.params["path_delays_ns"] = [0.0, 70.0, 190.0]
        cfg.channel.params["path_powers_db"] = [0.0, -2.5, -7.0]
        waveform = np.ones(2048, dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (4, waveform.size))
        self.assertEqual(info["path_coefficients"].shape[:3], (4, 2, 3))
        self.assertTrue(np.allclose(info["path_delays_s"], np.array([0.0, 70.0, 190.0]) * 1e-9))


class WaveformReplaySmokeTest(unittest.TestCase):
    def test_replay_repository_example_config(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_replay_template.yaml")
        result = WaveformReplaySimulation(cfg).run()
        self.assertTrue(np.isnan(result.bit_error_rate))
        self.assertGreater(result.rx.decoded_bits.size, 0)
        self.assertIs(result.crc_ok, True)

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
                "\n".join(f"{sample.real:.12e} {sample.imag:.12e}" for sample in tx_payload.waveform)
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
        waveform = np.ones(2048, dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        self.assertIsInstance(channel, CdlChannel)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, waveform.shape)
        self.assertGreater(info["path_delays_s"].size, 0)
        self.assertGreaterEqual(info["noise_variance"], 0.0)

    def test_cdl_channel_supports_multi_tx_multi_rx(self):
        cfg = load_simulation_config(ROOT / "configs" / "pusch_cdl.yaml")
        cfg.link.num_tx_ant = 2
        cfg.link.num_rx_ant = 4
        waveform = np.ones(2048, dtype=np.complex128)
        channel = DefaultChannelFactory().create(cfg)
        rx_waveform, info = channel.propagate(waveform, cfg)
        self.assertEqual(rx_waveform.shape, (4, waveform.size))
        self.assertEqual(info["path_coefficients"].shape[0], 4)
        self.assertEqual(info["path_coefficients"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
