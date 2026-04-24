from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
from nr_phy_simu.scenarios.sweep import run_snr_sweep, write_snr_sweep_csv


def _parse_snr_points(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BER/BLER curve sweep for the NR PHY simulator.")
    parser.add_argument("config_path", help="Base YAML/JSON/XML config path.")
    parser.add_argument(
        "--snr-points",
        default="-5,0,5,10,15,20,25,30",
        help="Comma-separated SNR points in dB.",
    )
    parser.add_argument(
        "--output-prefix",
        default="link_curve",
        help="Output file prefix inside ./outputs.",
    )
    args = parser.parse_args()

    config = load_simulation_config(args.config_path)
    config.plotting.enabled = False
    snr_points = _parse_snr_points(args.snr_points)
    sweep_points = run_snr_sweep(config, snr_points, component_factory=DefaultSimulationComponentFactory())

    outputs_dir = Path("outputs")
    csv_path = write_snr_sweep_csv(outputs_dir / f"{args.output_prefix}.csv", sweep_points)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].semilogy([point.snr_db for point in sweep_points], [max(point.bler, 1e-6) for point in sweep_points], marker="o")
    axes[0].set_ylabel("BLER")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].set_title("Link-Level BLER Curve")

    axes[1].semilogy([point.snr_db for point in sweep_points], [max(point.ber, 1e-6) for point in sweep_points], marker="o")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("BER")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    png_path = outputs_dir / f"{args.output_prefix}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"SNR sweep CSV: {csv_path.resolve()}")
    print(f"SNR sweep plot: {png_path.resolve()}")


if __name__ == "__main__":
    main()
