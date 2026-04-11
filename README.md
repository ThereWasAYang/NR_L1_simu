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
- 基于 3GPP TR 38.901 profile 的 `TDL/CDL` 信道实现
- 可替换的编码、调制、映射、OFDM、估计、均衡、解调、译码模块
- 基于 `MCS table + MCS index` 自动解析调制方式、目标码率和 `TBS`
- 基于 `py3gpp` 的 NR `CRC/LDPC/rate matching/rate recovery`
- 基于协议参数建模的 DMRS 插入与 LS 信道估计
- 每次示例仿真自动输出信噪比、解调星座图、导频信道估计幅度/相位图

当前版本仍需进一步补齐：

- 完整的 PUSCH/PDSCH 映射规则与多天线预编码流程
- 继续补齐 DMRS/PTRS/CSI-RS 等协议表项，尤其是 transform-precoded PUSCH DMRS 的 clause 5.2.2/5.2.3 完整细节
- `TDL/CDL` 的完整 MIMO 天线阵列、极化与空间一致性增强

因此，这个版本更适合作为“工程基础框架 + 可扩展参考实现”，而不是“已完成全部 R18 细节校准的协议级金模型”。协议严格化工作已经在结构上预留好了接入点。

## 目录结构

```text
src/nr_phy_simu/
  common/          公共类型、接口、资源栅格、OFDM、DMRS
  tx/              发射机：编码、扰码、调制、频域资源映射
  rx/              接收机：频域抽取、信道估计、均衡、解调、译码
  channels/        AWGN / TDL / CDL / profile tables / channel factory
  scenarios/       PUSCH / PDSCH 场景封装与组件工厂
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
python examples/run_from_config.py configs/pdsch_awgn.yaml
# 仓库内置灌数样例，可直接运行
python examples/run_from_config.py configs/pusch_replay_template.yaml
```

前台运行示例脚本时，星座图和导频信道估计图会直接显示；同时仍会保存到 `outputs/`。在 macOS 前台运行时，默认会直接调用系统图片查看器打开生成的 PNG，因此不会阻塞 Python 主进程，也不会依赖 `matplotlib` 的 GUI 事件循环。
如果当前环境没有图形显示能力，则会自动退化为仅保存图片。若需要手动指定 `matplotlib` backend，可设置环境变量 `NR_PHY_SIMU_PLOT_BACKEND`，例如 `macosx` 或 `TkAgg`。

绘图调用链和如何新增绘图节点的开发说明见：

- [docs/plotting_development.md](docs/plotting_development.md)

## 参数文件

当前支持：

- `YAML`
- `JSON`
- `XML`

样例文件位于：

- [configs/pusch_awgn.yaml](configs/pusch_awgn.yaml)
- [configs/pusch_dfts_awgn.yaml](configs/pusch_dfts_awgn.yaml)
- [configs/pdsch_awgn.yaml](configs/pdsch_awgn.yaml)
- [configs/pusch_replay_template.yaml](configs/pusch_replay_template.yaml)
- [inputs/pusch_capture.txt](inputs/pusch_capture.txt)

当前参数模型至少覆盖：

- `MCS table type`
- `MCS index`
- `waveform`
- `channel type`
- `channel params`
- `cell bandwidth RBs`
- `user bandwidth RBs`
- `subcarrier spacing`
- `cyclic prefix type (NormalCP / ECP)`
- `sample rate`
- `RNTI`
- `N_id`
- `data scrambling ID`
- `DMRS config type`
- `DMRS symbol positions`
- `DMRS scrambling IDs / n_SCID / hopping`
- `RV`
- `layer number`
- `antenna number`
- `slot index`
- `waveform_input.waveform_path`
- `waveform_input.num_samples_per_tti`
- `waveform_input.noise_variance`

对于 `TDL/CDL`，当前已支持的典型信道参数包括：

- `profile`
- `delay_spread_ns`
- `carrier_frequency_hz`
- `ue_speed_mps`
- `max_doppler_hz`
- `ue_azimuth_deg` / `ue_zenith_deg`（CDL）
- `num_sinusoids`
- `k_factor_db`（LOS profile 可选覆盖）

当 `fft_size` 或 `sample_rate_hz` 未显式提供时，工程会自动选择满足当前带宽要求的最小 `2` 的整数次幂 FFT，并据此计算采样率。
循环前缀长度不再由配置文件手动输入，而是根据 `cyclic_prefix + subcarrier spacing + sample rate` 自动推导。

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
- `BitScrambler`
- `DmrsSequenceGenerator`

### 2. 协议实现与算法实现分离

像 `DMRS`、资源位置、信道配置、码块处理等协议相关内容，建议持续沉淀在：

- `common/sequences/`
- `scenarios/`
- `config.py`

这样可以避免协议规则散落在各个算法模块里。

### 3. 场景对象负责装配

`PUSCHSimulation` / `PDSCHSimulation` 负责把 TX、RX、Channel、DMRS 这些模块组装起来；具体模块可以在构造时替换。

### 4. 组件工厂负责默认实现装配

默认装配逻辑位于 [component_factory.py](src/nr_phy_simu/scenarios/component_factory.py)：

- 发射机各处理块分别以独立类存在于 `tx/`
- 接收机各处理块分别以独立类存在于 `rx/`
- `DMRS` 与数据扰码位于 `common/sequences/`
- 信道实例创建位于 [channel_factory.py](src/nr_phy_simu/channels/channel_factory.py)

如果你后续要引入新算法，推荐做法是：

- 派生对应抽象接口的新类
- 自定义一个 `SimulationComponentFactory`
- 在 `PuschSimulation(..., component_factory=...)` 或 `PdschSimulation(..., component_factory=...)` 中注入

## 灌数仿真

当前已支持“灌数仿真”模式：直接从文本文件读取时域 IQ，跳过发射机和信道模型，把数据送入接收机处理链。入口在 [waveform_replay.py](src/nr_phy_simu/scenarios/waveform_replay.py)。

文本文件格式如下：

- 每一行一个复数样点
- 可以写成 `I Q`
- 也可以写成 `I,Q`
- 也支持 Python 复数字面量形式，例如 `0.12+0.34j`
- 若一个 TTI 每根接收天线有 `M` 个样点、接收天线数为 `N`
- 则前 `M` 行对应第 1 根天线
- 第 `M+1` 到第 `2M` 行对应第 2 根天线
- 以此类推

灌数仿真所需配置位于 `waveform_input` 段：

- `waveform_path`：文本文件路径
- `num_samples_per_tti`：每根天线每个 TTI 的样点数；为空时按当前 numerology 自动推导
- `noise_variance`：已知噪声方差；为空时按 `snr_db` 和灌入波形功率估计

仓库当前已内置一份可直接运行的示例：

- 配置文件：[configs/pusch_replay_template.yaml](configs/pusch_replay_template.yaml)
- 波形文件：[inputs/pusch_capture.txt](inputs/pusch_capture.txt)

在该模式下：

- 发射机和信道模型不会参与运行
- `DMRS`、资源映射位置、解调与译码仍按当前配置生成和执行
- 因为没有本地发端参考比特，`BER` 会显示为 `N/A`

## DMRS 说明

当前 DMRS 实现在 [src/nr_phy_simu/common/sequences/dmrs.py](src/nr_phy_simu/common/sequences/dmrs.py)：

- `PDSCH DMRS` 采用 38.211 中的 Gold 序列初始化形式
- `PUSCH DMRS` 在 `transform precoding disabled` 时走同类 Gold 序列初始化路径
- `PUSCH DFT-s-OFDM` 已拆出独立分支，并依据 38.211 中 transform precoding enabled 时的 low-PAPR type 1 路径生成序列

数据扰码实现在 [src/nr_phy_simu/common/sequences/scrambling.py](src/nr_phy_simu/common/sequences/scrambling.py)：

- 发端按 `RNTI + codeword index + data scrambling ID/N_id` 生成共享信道数据扰码
- 收端对 LLR 执行对应解扰，再进入 LDPC 译码

## 信道说明

当前信道实现位于：

- [awgn.py](src/nr_phy_simu/channels/awgn.py)
- [tdl.py](src/nr_phy_simu/channels/tdl.py)
- [cdl.py](src/nr_phy_simu/channels/cdl.py)
- [profile_tables.py](src/nr_phy_simu/channels/profile_tables.py)

当前版本已支持：

- `TDL-A/B/C/D/E`
- `CDL-A/B/C/D/E`
- profile 归一化时延按目标 `delay spread` 缩放
- 基于 UE 速度/载频的多普勒频移建模
- `LOS` profile 的 Rician 分量

当前这版 `TDL/CDL` 先面向现有链路接口实现为 `SISO`。如果后续你要把 `num_tx_ant / num_rx_ant > 1` 的阵列、极化、角域响应也纳入同一套仿真，我建议下一步把当前 `ChannelModel` 接口扩成显式 MIMO 通道矩阵/多分支波形接口。

## 接收数据维度

接收机内部当前统一采用带天线维的数据组织方式：

- `rx_grid` 统一为 `num_rx_ant x num_subcarrier x num_symbol`
- `channel_estimate` 统一为 `num_rx_ant x num_subcarrier x num_symbol`
- 即使是单接收天线，天线维也会保留，长度为 `1`
- `pilot_estimates` 统一为 `num_rx_ant x num_dmrs_re`

这样做是为了保证单天线和多天线场景下的数据维度一致，方便后续替换估计器、均衡器和多天线算法。

## 编码与调制

当前编码/调制相关实现集中在：

- [src/nr_phy_simu/tx/codec.py](src/nr_phy_simu/tx/codec.py)
- [src/nr_phy_simu/rx/decoding.py](src/nr_phy_simu/rx/decoding.py)
- [src/nr_phy_simu/tx/modulation.py](src/nr_phy_simu/tx/modulation.py)
- [src/nr_phy_simu/rx/demodulation.py](src/nr_phy_simu/rx/demodulation.py)
- [src/nr_phy_simu/common/mcs.py](src/nr_phy_simu/common/mcs.py)

当前版本已接入：

- NR `CRC`
- NR `LDPC`
- NR `rate matching / rate recovery`
- `MCS -> modulation/code rate/TBS` 自动解析

当前已实现的 `MCS table` 包括：

- `qam64`
- `qam256`
- `qam64lowse`
- `qam1024`（PDSCH）
- `tp64qam`（transform-precoded PUSCH）
- `tp64lowse`（transform-precoded PUSCH）

## 后续建议

如果你接下来要把它做成“协议精确仿真平台”，建议按这个顺序推进：

1. 补齐 NR LDPC 编解码
2. 将 DMRS 序列和位置配置严格映射到 R18 参数表
3. 接入 38.901 的 TDL/CDL 参数集
4. 增加 HARQ、MCS、层映射、码字与传输块管理
5. 增加单元测试与链路级 BER/BLER 曲线脚本
