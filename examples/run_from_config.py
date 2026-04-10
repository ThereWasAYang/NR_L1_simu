from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.visualization import save_simulation_plots


def main(config_relpath: str = "configs/pusch_awgn.yaml") -> None:
    config_path = ROOT / config_relpath
    config = load_simulation_config(config_path)
    simulation = PuschSimulation(config) if config.link.channel_type.upper() == "PUSCH" else PdschSimulation(config)
    result = simulation.run()
    prefix = config_path.stem
    plots = save_simulation_plots(result, ROOT / "outputs", prefix, show=True)

    print(f"Config: {config_path}")
    print(f"Channel type: {config.link.channel_type}")
    print(f"Waveform: {config.link.waveform}")
    print(f"MCS table/index: {config.link.mcs.table}/{config.link.mcs.index}")
    print(f"Modulation: {config.link.modulation}")
    print(f"Code rate: {config.link.code_rate:.6f}")
    print(f"TBS: {config.link.transport_block_size}")
    print(f"SNR: {result.snr_db:.2f} dB")
    print(f"Sample rate: {config.carrier.sample_rate_effective_hz:.2f} Hz")
    print(f"FFT size: {config.carrier.fft_size_effective}")
    print(f"BER: {result.bit_error_rate:.6f}")
    print(f"Bit errors: {result.bit_errors}")
    print(f"Constellation plot: {plots['constellation']}")
    print(f"Pilot estimate plot: {plots['pilot_estimates']}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/pusch_awgn.yaml"
    main(arg)
