# 2026-07-12 推广前评审闭环记录

本记录逐项复核 `REVIEW_REPORT_2026-07-12_platform.md`。结论以当前工作区代码、官方 3GPP TS 38.211 V18.8.0 表格、py3gpp 0.6.0 互通对拍和全量测试为依据。

## R-1 到 R-13

| 项 | 判断 | 闭环 |
|---|---|---|
| R-1 固定 seed 多 TTI 重复 | 真实 | runner 每次 run 只创建一个信道对象；整数 seed 现在定义可复现且逐 TTI 不重复的序列，新增回归测试。 |
| R-2 无时间连续性 | 真实，原建议只加时间偏移不充分 | 新增 `channel.params.tti_evolution: independent/continuous`；continuous 同时固定随机几何、射线耦合/初相并使用绝对 slot 时间，测试 TDL 边界连续性。 |
| R-3 每 TTI 重建与 HARQ probe | 真实 | 组件、TX/RX、主信道及干扰信道跨 TTI 复用；HARQ 直接复用 mapper 计数。 |
| R-4 LDPC Python 热点 | 真实 | check row 按度数分桶，check/variable node 更新 NumPy 向量化；20 TTI 高 MCS baseline 实测约 2.2 s，约 0.11 s/TTI。 |
| R-5 重复分数延迟卷积 | 真实 | 延迟波形按 `(tx,path)` 缓存后供所有 RX 分支复用；CDL 内存上限写入性能文档。 |
| R-6 无 logging/fallback 静默 | 真实 | 包根 NullHandler、fallback WARNING、长跑 INFO 进度、py3gpp stdout 转 DEBUG；结果增加 decoder path/iterations。 |
| R-7 多 TTI 回放错误 | 真实 | 回放构造时拒绝 `num_ttis > 1`，并新增回归测试。 |
| R-8 卫生项 | 真实 | 删除死函数/死模块/auto 分支，抽共享复数解析，整理场景控制流，修复曲线脚本路径，matplotlib 改惰性初始化。 |
| R-9 包出口 | 真实 | 顶层导出常用仿真、配置加载和绘图入口，增加 `__version__`。 |
| R-10 文档 | 真实 | 修正文档旧机制和目录树，增加 MIT LICENSE、CHANGELOG、FAQ、性能文档及统计口径。 |
| R-11 依赖/CI | 真实 | `py3gpp>=0.6.0,<0.7`；Ubuntu/Windows × Python 3.10/3.12 快测 CI；补 authors/classifiers/urls/test extra。 |
| R-12 配置 UX | 真实 | 调用方 config 不再被运行派生值回写；保留顶层 SNR 兼容并声明 channel params 为权威入口；实现递归 `base_config_path` 和最小配置。 |
| R-13 测试组织 | 部分是维护建议 | 增加独立 `test_platform_review.py`、slow marker、CI coverage 和全部针对性用例。未机械拆散原 smoke 文件，避免无行为收益的大规模 blame/cherry-pick 冲突。 |

## 上轮项目

| 项 | 判断与闭环 |
|---|---|
| M-5 多层 MIMO | 是已声明的产品能力边界，不是静默缺陷。当前配置守卫、单层接口和文档收口合理；真实多层预编码/双码字属于独立产品特性，不在推广缺陷修复中伪装实现。 |
| L-1 DMRS 频率参考 | 已修复：Gold DMRS 序列切片使用 `bwp.start_rb + link.prb_start` 的 CRB 参考。 |
| L-2 DMRS 位置表 | 已按 38.211 R18.8 分别实现 PUSCH/PDSCH、mapping A/B、单/双符号、type-A pos3 的表项和非法组合拒绝。 |
| L-3 OFDM 相位补偿 | 已修复：clause 5.4 symbol 常量复包络相位、RF carrier `f0`、subframe 时间参考；slot 0 与 py3gpp 逐样点对拍。 |
| L-5 长度 30 低 PAPR | 已按 clause 5.2.2.2 的闭式公式实现；原“补专表”建议不准确。 |

## 验证结果

- `compileall src tests examples benchmarks`: 通过。
- `pytest`: 102 passed / 59 subtests，13.75 s。
- baseline slow suite: 通过。
- `pip check`: 无依赖冲突。
- wheel (`--no-build-isolation`): 构建成功。
- 顶层包导入不加载或配置 matplotlib；首次绘图时才初始化。
