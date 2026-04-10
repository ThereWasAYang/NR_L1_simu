# NR PHY Simulation Project

这个工程提供一个面向 5G NR 物理层仿真的 Python 骨架，核心目标是：

- 发射机、接收机、信道三大块明确分层
- 模块之间通过接口解耦，方便替换算法
- 同时覆盖 `PUSCH` 与 `PDSCH` 两类下行/上行共享信道
- 为 `AWGN`、`TDL`、`CDL` 信道留出统一接入方式
- 将 `DMRS` 等协议相关序列生成聚合到公共模块
- 用外部 `YAML/JSON/XML` 文件统一管理仿真参数

## 当前状态

当前版本已经提供：

- 可运行的端到端仿真框架
- 面向 `PUSCH/PDSCH` 的统一配置与链路封装
- `PUSCH` 的 `CP-OFDM` 与 `DFT-s-OFDM` 两种波形入口
- `AWGN` 信道实现
- `TDL/CDL` 信道接口骨架
- 可替换的编码、调制、映射、OFDM、估计、均衡、解调、译码模块
- 基于 `MCS table + MCS index` 自动解析调制方式、目标码率和 `TBS`
- 基于 `py3gpp` 的 NR `CRC/LDPC/rate matching/rate recovery`
- 基于协议参数建模的 DMRS 插入与 LS 信道估计
- 每次示例仿真自动输出信噪比、解调星座图、导频信道估计幅度/相位图

当前版本仍需进一步补齐：

- 继续扩展更多标准 `MCS table type`
- 完整的 PUSCH/PDSCH 映射规则与多天线预编码流程
- 继续补齐 DMRS/PTRS/CSI-RS 等协议表项，尤其是 transform-precoded PUSCH DMRS 的 clause 5.2.2/5.2.3 完整细节
- 3GPP TDL/CDL 精确参数集、时变多径与多普勒模型

因此，这个版本更适合作为“工程基础框架 + 可扩展参考实现”，而不是“已完成全部 R18 细节校准的协议级金模型”。协议严格化工作已经在结构上预留好了接入点。

## 目录结构

```text
src/nr_phy_simu/
  common/          公共类型、接口、资源栅格、OFDM、DMRS
  tx/              发射机：编码、调制、映射、时域处理
  rx/              接收机：时域处理、抽取、估计、均衡、解调、译码
  channels/        AWGN / TDL / CDL
  scenarios/       PUSCH / PDSCH 业务链路封装
  config.py        全局配置 dataclass
examples/
  run_pusch_awgn.py
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/run_from_config.py configs/pusch_awgn.yaml
python examples/run_from_config.py configs/pusch_dfts_awgn.yaml
python examples/run_from_config.py configs/pdsch_awgn.json
```

## 参数文件

当前支持：

- `YAML`
- `JSON`
- `XML`

样例文件位于：

- [configs/pusch_awgn.yaml](/Users/yang/Work/NR_L1_Simu/configs/pusch_awgn.yaml)
- [configs/pusch_dfts_awgn.yaml](/Users/yang/Work/NR_L1_Simu/configs/pusch_dfts_awgn.yaml)
- [configs/pdsch_awgn.json](/Users/yang/Work/NR_L1_Simu/configs/pdsch_awgn.json)

当前参数模型至少覆盖：

- `MCS table type`
- `MCS index`
- `waveform`
- `channel type`
- `channel params`
- `cell bandwidth RBs`
- `user bandwidth RBs`
- `subcarrier spacing`
- `sample rate`
- `DMRS config type`
- `DMRS symbol positions`
- `RV`
- `layer number`
- `antenna number`
- `slot index`

当 `fft_size` 或 `sample_rate_hz` 未显式提供时，工程会自动选择满足当前带宽要求的最小 `2` 的整数次幂 FFT，并据此计算采样率。

## 设计原则

### 1. 模块可插拔

所有核心算法都通过抽象基类定义：

- `ChannelCoder`
- `Modulator`
- `ResourceMapper`
- `TimeDomainProcessor`
- `ChannelModel`
- `FrequencyExtractor`
- `ChannelEstimator`
- `MimoEqualizer`
- `Demodulator`
- `ChannelDecoder`

### 2. 协议实现与算法实现分离

像 `DMRS`、资源位置、信道配置、码块处理等协议相关内容，建议持续沉淀在：

- `common/sequences/`
- `scenarios/`
- `config.py`

这样可以避免协议规则散落在各个算法模块里。

### 3. 场景对象负责装配

`PUSCHSimulation` / `PDSCHSimulation` 负责把 TX、RX、Channel、DMRS 这些模块组装起来；具体模块可以在构造时替换。

## DMRS 说明

当前 DMRS 实现在 [src/nr_phy_simu/common/sequences/dmrs.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/common/sequences/dmrs.py)：

- `PDSCH DMRS` 采用 38.211 中的 Gold 序列初始化形式
- `PUSCH DMRS` 在 `transform precoding disabled` 时走同类 Gold 序列初始化路径
- `PUSCH DFT-s-OFDM` 已拆出独立分支，并依据 38.211 中 transform precoding enabled 时的 low-PAPR type 1 路径生成序列

## 编码与调制

当前编码/调制相关实现集中在：

- [src/nr_phy_simu/tx/codec.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/tx/codec.py)
- [src/nr_phy_simu/rx/decoding.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/rx/decoding.py)
- [src/nr_phy_simu/tx/modulation.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/tx/modulation.py)
- [src/nr_phy_simu/rx/demodulation.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/rx/demodulation.py)
- [src/nr_phy_simu/common/mcs.py](/Users/yang/Work/NR_L1_Simu/src/nr_phy_simu/common/mcs.py)

当前版本已接入：

- NR `CRC`
- NR `LDPC`
- NR `rate matching / rate recovery`
- `MCS -> modulation/code rate/TBS` 自动解析

当前 `MCS table` 已实现 `qam64`，并保留了继续扩展其他标准表的接口。

## 后续建议

如果你接下来要把它做成“协议精确仿真平台”，建议按这个顺序推进：

1. 补齐 NR LDPC 编解码
2. 将 DMRS 序列和位置配置严格映射到 R18 参数表
3. 接入 38.901 的 TDL/CDL 参数集
4. 增加 HARQ、MCS、层映射、码字与传输块管理
5. 增加单元测试与链路级 BER/BLER 曲线脚本
