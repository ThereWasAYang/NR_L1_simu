# FAQ / Troubleshooting

## 固定 seed 后多 TTI 是否会重复？

不会。整数 `channel.seed` 定义可复现的 realization 序列；同一 runner 会复用信道对象并推进 RNG。`tti_evolution: independent` 逐 TTI 独立抽样，`continuous` 在绝对时间轴上连续演进。

## 为什么回放模式不能设置多个 TTI？

当前文本文件只承载一个 TTI。重复读取同一波形却递增 slot 会让本地 DMRS 与录制导频不匹配，因此平台会拒绝 `waveform_input` 与 `num_ttis > 1` 的组合。多 TTI 回放需要先定义带分段和 slot 元数据的新文件格式。

## 为什么运行后 config 里的 TBS 仍是 None？

这是预期行为。config 是调用方输入，运行时派生的 MCS、TBS 和 coded-bit capacity 不再回写。单 TTI 从 `result.transport_plan` 读取，多 TTI 从 `result.final_config` 读取。

## 如何知道使用了哪个 LDPC 译码路径？

检查 `SimulationResult.ldpc_decoder_path` 和 `ldpc_iterations`。fallback 会写 `WARNING`；应用通过 `logging.basicConfig(level=logging.INFO)` 开启日志。

## SNR 应修改哪一处？

使用 `channel.params.snr_db`。顶层 `snr_db` 只保留作旧代码兼容镜像。

## 中文配置或文本在 Windows 上乱码

配置、波形和频响读取均使用 `utf-8-sig`，可接受普通 UTF-8 和带 BOM 的 UTF-8。请勿用系统 ANSI 编码保存文件。

## py3gpp 升级后结果变化

项目依赖范围限制为 `py3gpp>=0.6.0,<0.7`。升级该边界前必须运行全量协议和 baseline 测试；仓库已知 DMRS 互通问题见项目相关开发文档与测试。

## CDL 运行内存过高

路径系数内存随 `Nrx × Ntx × clusters × slot_samples` 线性增长。估算和建议见 [performance.md](performance.md)。
