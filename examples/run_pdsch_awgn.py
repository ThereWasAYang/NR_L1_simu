from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nr_phy_simu.config import LinkConfig, SimulationConfig
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.visualization import save_simulation_plots


def main() -> None:
    config = SimulationConfig(
        snr_db=12.0,
        link=LinkConfig(
            channel_type="PDSCH",
            waveform="CP-OFDM",
            modulation="QPSK",
            transport_block_size=1024,
            code_rate=0.5,
            prb_start=4,
            num_prbs=24,
        ),
    )
    result = PdschSimulation(config).run()
    plots = save_simulation_plots(result, ROOT / "outputs", "pdsch_awgn")
    print("NR PDSCH AWGN simulation")
    print(f"SNR: {result.snr_db:.2f} dB")
    print(f"BER: {result.bit_error_rate:.6f}")
    print(f"Bit errors: {result.bit_errors}")
    print(f"Constellation plot: {plots['constellation']}")
    print(f"Pilot estimate plot: {plots['pilot_estimates']}")


if __name__ == "__main__":
    main()
