# 2026-07-13 复审闭环记录

本记录逐项核对 `REVIEW_REPORT_2026-07-13_recheck.md`。结论来自当前代码、py3gpp 0.6.0 实测、py3gpp upstream main `a4cc7cff`（2026-07-03）源码复核和新增回归测试。

| 项 | 判断 | 闭环 |
|---|---|---|
| F-1 DMRS 非法组合未拒绝 | 真实 | mapping A 的 addPos3/type-A pos3、前置 DMRS 不在 allocation 内、解析位置越界和空结果均显式 `ValueError`；mapping B 不误用无语义的 type-A position 做校验。 |
| F-2 双符号 DMRS 静默钳位 | 真实 | 删除 `min(add_pos, 1)`；`max_length=2` 只接受 addPos0/1，addPos2/3 显式拒绝。 |
| F-3 py3gpp 上游问题 | 真实，非本仓库功能缺陷 | py3gpp 0.6.0 与 2026-07-03 upstream main 均保留这两处实现；项目根目录新增合并报告 `PY3GPP_DMRS_BUG_REPORT.md`，slot-0 互通测试补充边界注释。未用错误上游结果改动本地协议实现。 |
| F-4 replay 使用调用方 config | 真实，当前影响有限 | component factory 改为接收 `self.config` 私有副本，并用身份断言回归测试锁定。 |
| F-5 μ≥2 CP 布局 | 真实，原“仅补文档”不足 | 直接实现 slot-aware normal CP；OFDM 调制/解调、clause 5.4 相位、连续 TDL/CDL 绝对时间、波形回放长度和绘图元数据统一使用精确时间轴。新增 μ=2 四 slot CP、长度、OFDM 往返和相位测试。 |

## 独立实证

- py3gpp 0.6.0：同一随机网格 slot 0 与本地实现最大绝对误差约 `8.9e-11`，slot 1 为 `0.177`；源码确认 `initialNSlot` 被直接加到 symbol index。
- py3gpp 0.6.0：PDSCH mapping-A、`ld=12` 时 addPos1/2 返回 `[2,11]` / `[2,7,11]`，官方表项应为 `[2,9]` / `[2,6,9]`。
- 专项测试：`tests/test_platform_review.py` 21 passed。
- 全量测试：108 passed + 59 subtests。

## 设计取舍

DMRS 校验放在 `resolve_dmrs_symbol_indices` 的协议解析边界，而非只放在 dataclass 构造阶段。配置对象允许调用方在构造后调整字段，运行时解析边界才能保证任何入口都不会让非法组合静默进入信道估计。

`CarrierConfig.cyclic_prefix_lengths` 和 `slot_length_samples` 保留为 slot 0 兼容属性；新增的 `*_for_slot` 与 `slot_start_sample*` API 承载精确多 slot 语义，避免破坏已有外部脚本。
