from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from nr_phy_simu.common.runtime_context import SimulationRuntimeContext
from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.io.multi_tti_report import append_multi_tti_report
from nr_phy_simu.scenarios.pdsch import PdschSimulation
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory
from nr_phy_simu.scenarios.multi_tti import MultiTtiSimulationRunner
from nr_phy_simu.scenarios.waveform_replay import WaveformReplaySimulation
from nr_phy_simu.visualization import save_simulation_plots


def _format_evm_snr_db(evm_snr_linear: float | None) -> str | None:
    if evm_snr_linear is None:
        return None
    return f"{10.0 * np.log10(max(evm_snr_linear, 1e-24)):.6f}"


def main(config_relpath: str = "configs/pusch_awgn.yaml") -> None:
    config_path = ROOT / config_relpath
    config = load_simulation_config(config_path)
    component_factory = DefaultSimulationComponentFactory()
    runtime_context = SimulationRuntimeContext()
    if config.simulation.num_ttis > 1:
        batch_result = MultiTtiSimulationRunner(
            config,
            component_factory=component_factory,
            runtime_context=runtime_context,
        ).run()
        result = batch_result.last_result
        if result is None:
            raise RuntimeError("Multi-TTI simulation did not produce any TTI result.")
        effective_config = batch_result.final_config
        report_path = None
        if effective_config.simulation.result_output_path:
            report_path = append_multi_tti_report(
                effective_config.simulation.result_output_path,
                batch_result,
                effective_config,
            )
    else:
        batch_result = None
        report_path = None
        if config.waveform_input.enabled:
            simulation = WaveformReplaySimulation(
                config,
                component_factory=component_factory,
                runtime_context=runtime_context,
            )
        else:
            simulation = (
                PuschSimulation(config, component_factory=component_factory, runtime_context=runtime_context)
                if config.link.channel_type.upper() == "PUSCH"
                else PdschSimulation(config, component_factory=component_factory, runtime_context=runtime_context)
            )
        result = simulation.run()
        effective_config = config
    prefix = config_path.stem
    plots = {}
    if effective_config.plotting.enabled:
        plots = save_simulation_plots(result, effective_config, ROOT / "outputs", prefix, show=True, block=False)

    print(f"Config: {config_path}")
    if effective_config.waveform_input.enabled:
        print(f"Waveform input: {effective_config.waveform_input.waveform_path}")
    print(f"Channel type: {effective_config.link.channel_type}")
    print(f"Waveform: {effective_config.link.waveform}")
    print(f"MCS table/index: {effective_config.link.mcs.table}/{effective_config.link.mcs.index}")
    print(f"Modulation: {effective_config.link.modulation}")
    print(f"Code rate: {effective_config.link.code_rate:.6f}")
    print(f"TBS: {effective_config.link.transport_block_size}")
    print(f"Layers/codewords: {effective_config.link.num_layers}/{effective_config.link.num_codewords}")
    if effective_config.harq.enabled:
        print(
            f"HARQ: enabled, processes={effective_config.harq.num_processes}, "
            f"max_retx={effective_config.harq.max_retransmissions}, rv_sequence={effective_config.harq.rv_sequence}"
        )
    if effective_config.simulation.bypass_channel_coding:
        print("Channel coding: bypassed (random coded-bit sequence, no decoding, no CRC)")
    print(f"SNR: {result.snr_db:.2f} dB")
    if result.evm_percent is not None:
        print(f"EVM (last TTI): {result.evm_percent:.6f} %")
    evm_snr_db = _format_evm_snr_db(result.evm_snr_linear)
    if evm_snr_db is not None:
        print(f"EVM_SNR (last TTI): {evm_snr_db} dB")
    if batch_result is not None:
        print(f"TTIs: {batch_result.num_ttis}")
        print(f"Packet errors: {batch_result.packet_errors}")
        if np.isfinite(batch_result.block_error_rate):
            print(f"BLER: {batch_result.block_error_rate:.6f}")
        else:
            print("BLER: N/A (CRC disabled in bypass-channel-coding mode)")
        if batch_result.average_evm_percent is not None:
            print(f"Average EVM: {batch_result.average_evm_percent:.6f} %")
        average_evm_snr_db = _format_evm_snr_db(batch_result.average_evm_snr_linear)
        if average_evm_snr_db is not None:
            print(f"Average EVM_SNR: {average_evm_snr_db} dB")
        if report_path is not None:
            print(f"Multi-TTI report: {report_path}")
    if result.interference_reports:
        print(f"Interference sources: {len(result.interference_reports)}")
        for report in result.interference_reports:
            prb_start = report.prb_start if report.prb_start >= 0 else effective_config.link.prb_start
            num_prbs = report.num_prbs if report.num_prbs >= 0 else effective_config.link.num_prbs
            print(
                f"  - {report.label}: channel={report.channel_model}, INR={report.inr_db:.2f} dB, "
                f"RB=[{prb_start}, {prb_start + num_prbs - 1}]"
            )
    print(f"Sample rate: {effective_config.carrier.sample_rate_effective_hz:.2f} Hz")
    print(f"FFT size: {effective_config.carrier.fft_size_effective}")
    if result.crc_ok is not None:
        print(f"CRC OK: {result.crc_ok}")
    if result.harq_process_id is not None:
        print(
            f"HARQ result: process={result.harq_process_id}, rv={result.harq_rv}, "
            f"retransmission={result.harq_retransmission}"
        )
    if np.isfinite(result.bit_error_rate):
        print(f"BER: {result.bit_error_rate:.6f}")
        print(f"Bit errors: {result.bit_errors}")
    else:
        print("BER: N/A (waveform replay mode has no transmitted reference bits)")
    if effective_config.plotting.enabled:
        for name, path in plots.items():
            print(f"Plot [{name}]: {path}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/pusch_awgn.yaml"
    main(arg)
