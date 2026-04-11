from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
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
    def test_pusch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.channel_estimate.ndim, 3)
        self.assertEqual(result.rx.rx_grid.shape[0], config.link.num_rx_ant)
        self.assertEqual(result.rx.pilot_estimates.ndim, 2)
        self.assertEqual(result.rx.pilot_estimates.shape[0], config.link.num_rx_ant)
        self.assertIsNotNone(config.link.transport_block_size)

    def test_pusch_dfts_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.pilot_estimates.ndim, 2)

    def test_pusch_awgn_multi_rx_branches(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        config.link.num_rx_ant = 4
        result = PuschSimulation(config).run()
        self.assertEqual(result.rx.rx_waveform.ndim, 2)
        self.assertEqual(result.rx.rx_waveform.shape[0], 4)
        self.assertEqual(result.rx.rx_grid.shape[0], 4)
        self.assertEqual(result.rx.pilot_estimates.shape[0], 4)

    def test_pusch_awgn_with_interference_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertEqual(len(result.interference_reports), 2)
        self.assertTrue(all(report.scale >= 0.0 for report in result.interference_reports))


class PdschAwgnSmokeTest(unittest.TestCase):
    def test_pdsch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PdschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)
        self.assertEqual(result.rx.rx_grid.ndim, 3)
        self.assertEqual(result.rx.pilot_estimates.ndim, 2)


class VisualizationSmokeTest(unittest.TestCase):
    def test_save_simulation_plots(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 20.0
        result = PuschSimulation(config).run()
        paths = save_simulation_plots(result, config, ROOT / "outputs" / "tests", "smoke")
        self.assertTrue(paths["constellation"].exists())
        self.assertTrue(paths["pilot_estimates"].exists())
        self.assertTrue(paths["rx_time_ant0"].exists())
        self.assertTrue(paths["rx_freq_ant0"].exists())


class DmrsSequenceTest(unittest.TestCase):
    def test_transform_precoded_pusch_dmrs_short_lengths(self):
        generator = DmrsGenerator()
        for num_prbs in (1, 2, 3, 4):
            config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
            config.link.num_prbs = num_prbs
            symbols = generator.generate_for_symbol(symbol=2, config=config)
            self.assertEqual(symbols.size, num_prbs * 6)
            self.assertTrue(np.allclose(np.abs(symbols), 1.0))


class ConfigLoaderTest(unittest.TestCase):
    def test_load_yaml_and_json_config(self):
        yaml_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        yaml_pdsch_cfg = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        yaml_interference_cfg = load_simulation_config(ROOT / "configs" / "pusch_awgn_with_interference.yaml")
        self.assertEqual(yaml_cfg.link.channel_type, "PUSCH")
        self.assertEqual(yaml_pdsch_cfg.link.channel_type, "PDSCH")
        self.assertGreater(yaml_cfg.carrier.fft_size_effective, 0)
        self.assertEqual(yaml_cfg.carrier.cyclic_prefix_mode, "NORMAL")
        self.assertEqual(len(yaml_cfg.carrier.cyclic_prefix_lengths), yaml_cfg.carrier.symbols_per_slot)
        self.assertEqual(yaml_cfg.scrambling.rnti, 4660)
        self.assertEqual(yaml_pdsch_cfg.scrambling.effective_data_scrambling_id, 1)
        self.assertEqual(len(yaml_interference_cfg.interference.sources), 2)
        self.assertEqual(yaml_interference_cfg.interference.sources[0].channel_model, "AWGN")


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
