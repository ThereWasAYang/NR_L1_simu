from __future__ import annotations

import torch

from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.common.torch_utils import as_complex_tensor, complex_randn
from nr_phy_simu.config import SimulationConfig


class AwgnChannel(ChannelModel):
    def __init__(self, rng: torch.Generator | None = None) -> None:
        self.rng = rng or torch.Generator().manual_seed(0)

    def propagate(
        self,
        waveform: torch.Tensor,
        config: SimulationConfig,
    ) -> tuple[torch.Tensor, dict]:
        """Apply receive-branch expansion and AWGN impairment."""
        tx_waveform = self._expand_receive_branches(as_complex_tensor(waveform), config)
        if not bool(config.channel.params.get("add_noise", True)):
            return tx_waveform, {"noise_variance": 0.0, "snr_db": float("inf")}

        snr_db = float(config.channel.params.get("snr_db", config.snr_db))
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = float(torch.mean(torch.abs(tx_waveform) ** 2).item())
        noise_variance = signal_power / max(snr_linear, 1e-12)
        noise = complex_randn(
            tuple(tx_waveform.shape),
            generator=self.rng,
            std=(noise_variance / 2.0) ** 0.5,
            device=tx_waveform.device,
        )
        return tx_waveform + noise, {"noise_variance": noise_variance, "snr_db": snr_db}

    @staticmethod
    def _expand_receive_branches(waveform: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
        """Replicate a single received stream across configured receive branches."""
        num_rx_ant = int(config.link.num_rx_ant)
        if waveform.ndim == 1:
            return waveform.unsqueeze(0).repeat(num_rx_ant, 1)
        if waveform.ndim != 2:
            raise ValueError("AWGN channel expects waveform shape (slot_samples,) or (num_ant, slot_samples).")
        if waveform.shape[0] == num_rx_ant:
            return waveform
        if waveform.shape[0] == 1:
            return waveform.repeat(num_rx_ant, 1)
        reference_stream = torch.sum(waveform, dim=0) / torch.sqrt(
            torch.tensor(float(waveform.shape[0]), dtype=torch.float64, device=waveform.device)
        )
        return reference_stream.unsqueeze(0).repeat(num_rx_ant, 1)
