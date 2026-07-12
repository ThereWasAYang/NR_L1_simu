# Changelog

本文件记录面向使用者的显著变化。完整开发历史仍保留在 Git 提交中。

## Unreleased

- 修复固定 channel seed 下多 TTI 重复同一信道与噪声 realization 的统计错误。
- 多 TTI 改为复用组件、收发链和信道，并支持 TDL/CDL `independent` / `continuous` 时间演进口径。
- 配置输入不再被运行期 MCS、TBS 和 coded-bit capacity 回写。
- 回放模式明确拒绝多 TTI。
- LDPC min-sum 按 check-row degree 向量化，并公开译码路径与迭代次数。
- 引入库级 logging、多 TTI 进度和 fallback 警告。
- 消除衰落信道按 RX 天线重复做分数延迟卷积的问题。
- 增加顶层配置继承、最小配置样例、公共包入口、版本号、CI 和测试覆盖率门禁。
- 清理死代码、重复复数解析和 matplotlib 导入副作用。
- 增加 MIT License 与对应打包元数据。
- 按 38.211 R18.8 补齐 PUSCH/PDSCH 单/双符号 DMRS 位置表、type-A pos3、CRB 序列参考点和长度 30 低 PAPR 闭式序列。
- 修正 OFDM clause 5.4 复包络相位为 symbol 常量、RF carrier 频率和 subframe 时间参考，并加入 py3gpp 波形互通测试。

## 0.1.0 - 2026-06-03

- 建立 PUSCH/PDSCH 端到端 NR PHY 仿真框架。
- 实现 AWGN、TDL/CDL、外部频响、波形回放、多 TTI、HARQ 框架和链路扫描。
- 完成 MCS/TBS、UL-SCH LDPC、DMRS、OFDM/BWP 相位补偿及协议回归基线。
