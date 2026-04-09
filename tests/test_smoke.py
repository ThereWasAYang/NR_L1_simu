from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nr_phy_simu.config import LinkConfig, SimulationConfig
from nr_phy_simu.common.sequences.dmrs import DmrsGenerator
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.visualization import save_simulation_plots


class PuschAwgnSmokeTest(unittest.TestCase):
    def test_pusch_cp_ofdm_awgn_smoke(self):
        config = SimulationConfig(
            snr_db=30.0,
            link=LinkConfig(
                channel_type="PUSCH",
                waveform="CP-OFDM",
                modulation="QPSK",
                transport_block_size=128,
                code_rate=0.5,
                prb_start=0,
                num_prbs=12,
            ),
        )
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)

    def test_pusch_dfts_ofdm_awgn_smoke(self):
        config = SimulationConfig(
            snr_db=30.0,
            link=LinkConfig(
                channel_type="PUSCH",
                waveform="DFT-s-OFDM",
                modulation="QPSK",
                transport_block_size=128,
                code_rate=0.5,
                prb_start=0,
                num_prbs=12,
            ),
        )
        result = PuschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)


class PdschAwgnSmokeTest(unittest.TestCase):
    def test_pdsch_cp_ofdm_awgn_smoke(self):
        config = SimulationConfig(
            snr_db=30.0,
            link=LinkConfig(
                channel_type="PDSCH",
                waveform="CP-OFDM",
                modulation="QPSK",
                transport_block_size=128,
                code_rate=0.5,
                prb_start=0,
                num_prbs=12,
            ),
        )
        result = PdschSimulation(config).run()
        self.assertTrue(0.0 <= result.bit_error_rate <= 1.0)
        self.assertGreater(result.rx.pilot_estimates.size, 0)


class VisualizationSmokeTest(unittest.TestCase):
    def test_save_simulation_plots(self):
        config = SimulationConfig(
            snr_db=20.0,
            link=LinkConfig(
                channel_type="PUSCH",
                waveform="CP-OFDM",
                modulation="QPSK",
                transport_block_size=64,
                code_rate=0.5,
                prb_start=0,
                num_prbs=12,
            ),
        )
        result = PuschSimulation(config).run()
        paths = save_simulation_plots(result, ROOT / "outputs" / "tests", "smoke")
        self.assertTrue(paths["constellation"].exists())
        self.assertTrue(paths["pilot_estimates"].exists())


class DmrsSequenceTest(unittest.TestCase):
    def test_transform_precoded_pusch_dmrs_short_lengths(self):
        generator = DmrsGenerator()
        for num_prbs in (1, 2, 3, 4):
            config = SimulationConfig(
                link=LinkConfig(
                    channel_type="PUSCH",
                    waveform="DFT-s-OFDM",
                    modulation="QPSK",
                    num_prbs=num_prbs,
                ),
            )
            symbols = generator.generate_for_symbol(symbol=2, config=config)
            self.assertEqual(symbols.size, num_prbs * 6)
            self.assertTrue(np.allclose(np.abs(symbols), 1.0))


if __name__ == "__main__":
    unittest.main()
