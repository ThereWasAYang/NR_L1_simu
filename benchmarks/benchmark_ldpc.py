from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nr_phy_simu import MultiTtiSimulationRunner, load_simulation_config


def main() -> None:
    config = load_simulation_config(
        ROOT / "configs" / "baseline" / "pusch_cp_ofdm_qam256_mcs27_awgn_snr50.yaml"
    )
    config.plotting.enabled = False
    started = time.perf_counter()
    result = MultiTtiSimulationRunner(config).run()
    elapsed = time.perf_counter() - started
    last = result.last_result
    print(f"TTIs: {result.num_ttis}")
    print(f"Elapsed: {elapsed:.6f} s")
    print(f"Per TTI: {elapsed / result.num_ttis:.6f} s")
    if last is not None:
        print(f"Decoder path: {last.ldpc_decoder_path}")
        print(f"LDPC iterations: {last.ldpc_iterations}")
    print(f"BLER: {result.block_error_rate:.6f}")


if __name__ == "__main__":
    main()
