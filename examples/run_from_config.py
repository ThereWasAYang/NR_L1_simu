from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation
from nr_phy_simu.visualization import save_simulation_plots


def main(config_relpath: str = "configs/pusch_awgn.yaml") -> None:
    config_path = ROOT / config_relpath
    config = load_simulation_config(config_path)
    if config.waveform_input.enabled:
        simulation = WaveformReplaySimulation(config)
    else:
        simulation = PuschSimulation(config) if config.link.channel_type.upper() == "PUSCH" else PdschSimulation(config)
    result = simulation.run()
    prefix = config_path.stem
    plots = {}
    if config.plotting.enabled:
        plots = save_simulation_plots(result, config, ROOT / "outputs", prefix, show=True, block=False)

    print(f"Config: {config_path}")
    if config.waveform_input.enabled:
        print(f"Waveform input: {config.waveform_input.waveform_path}")
    print(f"Channel type: {config.link.channel_type}")
    print(f"Waveform: {config.link.waveform}")
    print(f"MCS table/index: {config.link.mcs.table}/{config.link.mcs.index}")
    print(f"Modulation: {config.link.modulation}")
    print(f"Code rate: {config.link.code_rate:.6f}")
    print(f"TBS: {config.link.transport_block_size}")
    print(f"SNR: {result.snr_db:.2f} dB")
    if result.interference_reports:
        print(f"Interference sources: {len(result.interference_reports)}")
        for report in result.interference_reports:
            prb_start = report.prb_start if report.prb_start >= 0 else config.link.prb_start
            num_prbs = report.num_prbs if report.num_prbs >= 0 else config.link.num_prbs
            print(
                f"  - {report.label}: channel={report.channel_model}, INR={report.inr_db:.2f} dB, "
                f"RB=[{prb_start}, {prb_start + num_prbs - 1}]"
            )
    print(f"Sample rate: {config.carrier.sample_rate_effective_hz:.2f} Hz")
    print(f"FFT size: {config.carrier.fft_size_effective}")
    if result.crc_ok is not None:
        print(f"CRC OK: {result.crc_ok}")
    if np.isfinite(result.bit_error_rate):
        print(f"BER: {result.bit_error_rate:.6f}")
        print(f"Bit errors: {result.bit_errors}")
    else:
        print("BER: N/A (waveform replay mode has no transmitted reference bits)")
    if config.plotting.enabled:
        for name, path in plots.items():
            print(f"Plot [{name}]: {path}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/pusch_awgn.yaml"
    main(arg)
