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
- 本地可控的 `UL-SCH LDPC / rate matching / rate recovery / decoding` 主路径
- 基于协议参数建模的 DMRS 插入与 LS 信道估计
- `transform-precoded PUSCH DMRS` 的 38.211 clause `5.2.2 / 5.2.3` 双分支实现
- 支持多发多收的 `TDL/CDL` 基础 MIMO 传播
- 每次示例仿真自动输出信噪比、解调星座图、导频信道估计幅度/相位图

当前版本仍需进一步补齐：

- 完整的 PUSCH/PDSCH 映射规则与多天线预编码流程
- `PTRS / CSI-RS` 等协议信号与更完整的表项覆盖
- `TDL/CDL` 的完整阵列、极化、XPR 与空间一致性增强

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
python examples/run_from_config.py configs/pusch_awgn_multi_tti.yaml
# 仓库内置灌数样例，可直接运行
python examples/run_from_config.py configs/pusch_replay_template.yaml
```

### Windows 11 一键环境配置

对于不熟悉 Python 环境配置的使用者，仓库提供了 Windows 一键安装脚本：

```bat
scripts\setup_windows.bat
```

该脚本会自动完成：

- 检测 `py` 或 `python`
- 创建 `.venv` 虚拟环境
- 升级 `pip / setuptools / wheel`
- 以 editable 模式安装本工程
- 校验关键依赖是否安装成功

如果你更习惯 PowerShell，也可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

安装完成后，在 Windows 下可这样运行：

```bat
.venv\Scripts\activate
python examples\run_from_config.py configs\pusch_awgn.yaml
```

### 绘图与中文显示

前台运行示例脚本时，星座图和导频信道估计图会直接显示；同时仍会保存到 `outputs/`。

- 在 macOS 前台运行时，默认会调用系统图片查看器打开 PNG，因此不会阻塞 Python 主进程，也不会依赖 `matplotlib` 的 GUI 事件循环。
- 在 Windows 前台运行时，默认会调用系统图片查看器打开 PNG。
- 如果当前环境没有图形显示能力，则会自动退化为仅保存图片。
- 若需要手动指定 `matplotlib` backend，可设置环境变量 `NR_PHY_SIMU_PLOT_BACKEND`，例如 `TkAgg`。

为了尽量避免中文乱码，工程会在绘图时自动配置常见中文字体回退：

- Windows：`Microsoft YaHei / SimHei / SimSun`
- macOS：`PingFang SC / Hiragino Sans GB`
- Linux：`Noto Sans CJK SC / WenQuanYi Zen Hei`

如果你在 Windows 上仍然遇到中文方框或乱码，通常说明系统缺少上述字体之一；优先建议安装或启用 `Microsoft YaHei`。

绘图调用链和如何新增绘图节点的开发说明见：

- [docs/plotting_development.md](docs/plotting_development.md)

当前所有绘图都会先统一整理成 `PlotArtifact`，再由 [visualization.py](src/nr_phy_simu/visualization.py) 渲染。标准图会由系统自动生成对应 artifact；如果你在开发新算法时只是想把某个中间变量快速画出来，也可以自己挂载 `PlotArtifact`，通常不需要直接修改绘图主流程。例如在信道估计结果中挂载：

```python
from nr_phy_simu.common.types import PlotArtifact

plot_artifacts = (
    PlotArtifact(
        name="my_estimator_metric",
        values=my_metric,
        title="My Estimator Metric",
        plot_type="magnitude",
    ),
)
```

只要该对象最终进入 `ChannelEstimateResult.plot_artifacts` 或 `RxPayload.plot_artifacts`，系统就会自动保存为：

```text
outputs/<prefix>_artifact_my_estimator_metric.png
```

当前支持的通用绘图类型包括 `magnitude`、`phase`、`real`、`imag` 和 `image`。更复杂、长期稳定的公共绘图节点仍建议在 [visualization.py](src/nr_phy_simu/visualization.py) 中新增专用 `plot_type` 渲染分支。

## 参数文件

当前支持：

- `YAML`
- `JSON`
- `XML`

样例文件位于：

- [configs/pusch_awgn.yaml](configs/pusch_awgn.yaml)
- [configs/pusch_awgn_multi_tti.yaml](configs/pusch_awgn_multi_tti.yaml)
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
- `DMRS uplink_transform_precoding / pi2bpsk scrambling IDs`
- `RV`
- `layer number`
- `antenna number`
- `slot index`
- `simulation.num_ttis`
- `simulation.result_output_path`
- `simulation.bypass_channel_coding`
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

当 `simulation.num_ttis > 1` 时，系统会连续运行多个 TTI，并统计最终 `BLER`。这里的误包定义为：`CRC` 错误的 TTI。
如果同时配置了 `simulation.result_output_path`，系统还会把多 TTI 结果按 CSV 风格追加写入文本文件；文件不存在时会自动创建，空文件会先写标题行。
结果文件当前列顺序为：`信噪比, BLER, EVM, EVM_SNR, RB位置, MCS阶数, 总TTI数, 误包数, 码率, 调制阶数, TBsize`。

当 `simulation.bypass_channel_coding = true` 时：

- 发端跳过 `CRC/LDPC/rate matching`，改为生成与 `coded_bit_capacity` 等长的伪随机比特序列
- 收端跳过译码，只对解扰后的 LLR 做硬判决
- `CRC` 不再校验，因此单 TTI 的 `crc_ok` 为 `None`
- 多 TTI 模式下 `BLER` 会显示为 `N/A`
- 该模式适合做调制、EVM、波形和接收前端处理链验证，不作为标准协议链路统计结果

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

注意：

- `waveform_input.waveform_path` 现在按“相对配置文件所在目录”解析
- 因此 `configs/pusch_replay_template.yaml` 中使用的是 `../inputs/pusch_capture.txt`

在该模式下：

- 发射机和信道模型不会参与运行
- `DMRS`、资源映射位置、解调与译码仍按当前配置生成和执行
- 因为没有本地发端参考比特，`BER` 会显示为 `N/A`

## DMRS 说明

当前 DMRS 实现在 [src/nr_phy_simu/common/sequences/dmrs.py](src/nr_phy_simu/common/sequences/dmrs.py)：

- `PDSCH DMRS` 采用 38.211 中的 Gold 序列初始化形式
- `PUSCH DMRS` 在 `transform precoding disabled` 时走同类 Gold 序列初始化路径
- `PUSCH DFT-s-OFDM` 已拆出独立分支：
  - 普通 transform-precoded DMRS 走 38.211 clause `5.2.2` low-PAPR sequence type 1
  - `pi/2-BPSK + dmrs-UplinkTransformPrecoding-r16` 场景走 clause `5.2.3` low-PAPR sequence type 2
  - `M_ZC < 30` 时使用协议短序列表，较长长度时按协议公式生成并做归一化 DFT

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
- 基于阵列响应的基础 `num_tx_ant / num_rx_ant` MIMO 传播

当前这版 `TDL/CDL` 已支持基础多发多收，但仍未补齐完整的 38.901 阵列、极化/XPR 与空间一致性模型。如果后续要进一步做协议级对齐，建议继续把当前 `ChannelModel` 接口往显式阵列参数与空间一致性状态扩展。

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
