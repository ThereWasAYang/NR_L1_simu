# 性能与规模预期

## 推荐运行方式

- 单 TTI 调试优先使用 AWGN、小 PRB 和低 MCS。
- BLER/SNR 曲线先用 10～100 TTI 做链路检查，再提高到统计所需规模。
- `MultiTtiSimulationRunner` 内部按顺序运行；多个 SNR 点或独立 seed 可在调用方使用多进程并行。
- 开启 `logging.INFO` 可查看长跑进度；不要在每个 TTI 保存全部绘图。

LDPC normalized min-sum 的 check-node 和 variable-node 更新已向量化。实际速度仍随 base graph、lifting size、码块数、SNR 和最大迭代数变化，建议在目标机器上用 baseline 配置测量后再估算大批量任务。

可重复的高 MCS 基准入口：

```bash
python benchmarks/benchmark_ldpc.py
```

## TDL/CDL 内存

时变路径系数使用 `complex128`，主要数组形状为：

```text
(num_rx_ant, num_tx_ant, num_paths_or_clusters, slot_samples)
```

CDL 约 23 clusters、30 kHz SCS、2048 FFT 时，4×4 MIMO 单 TTI 的系数数组约 180 MB，8×8 约 720 MB；还应为波形、临时 cluster matrix 和绘图留出空间。大天线规模建议降低并发数、避免保留所有 TTI 的信道系数，或使用自定义按符号采样/低精度信道后端。

## 测试分层

- 快测：`python -m pytest -m "not slow"`
- 全量：`python -m pytest`
- 覆盖率：`python -m pytest -m "not slow" --cov=nr_phy_simu`

CI 在 Ubuntu/Windows、Python 3.10/3.12 上执行 compileall、快测和覆盖率统计；发布前仍应在目标科学计算环境运行全量 baseline。
