from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.visualization import save_simulation_plots
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator


class PuschAwgnSmokeTest(unittest.TestCase):
    def test_pusch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)
        self.assertIsNotNone(config.link.transport_block_size)

    def test_pusch_dfts_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_dfts_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)


class PdschAwgnSmokeTest(unittest.TestCase):
    def test_pdsch_cp_ofdm_awgn_smoke(self):
        config = load_simulation_config(ROOT / "configs" / "pdsch_awgn.yaml")
        config.channel.params["snr_db"] = 30.0
        result = PdschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)


class VisualizationSmokeTest(unittest.TestCase):
    def test_save_simulation_plots(self):
        config = load_simulation_config(ROOT / "configs" / "pusch_awgn.yaml")
        config.channel.params["snr_db"] = 20.0
        result = PuschSimulation(config).run()
        paths = save_simulation_plots(result, ROOT / "outputs" / "tests", "smoke")
        self.assertTrue(paths["constellation"].exists())
        self.assertTrue(paths["pilot_estimates"].exists())


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
        self.assertEqual(yaml_cfg.link.channel_type, "PUSCH")
        self.assertEqual(yaml_pdsch_cfg.link.channel_type, "PDSCH")
        self.assertGreater(yaml_cfg.carrier.fft_size_effective, 0)
        self.assertEqual(yaml_cfg.carrier.cyclic_prefix_mode, "NORMAL")
        self.assertEqual(len(yaml_cfg.carrier.cyclic_prefix_lengths), yaml_cfg.carrier.symbols_per_slot)
        self.assertEqual(yaml_cfg.scrambling.rnti, 4660)
        self.assertEqual(yaml_pdsch_cfg.scrambling.effective_data_scrambling_id, 1)


if __name__ == "__main__":
    unittest.main()
