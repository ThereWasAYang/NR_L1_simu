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
- 基于 3GPP TR 38.901 R18 Clause 7.7 的 link-level `TDL/CDL` 信道实现
- 可替换的编码、调制、映射、OFDM、估计、均衡、解调、译码模块
- 基于 `MCS table + MCS index` 自动解析调制方式、目标码率和 `TBS`
- 本地可控的 `UL-SCH LDPC / rate matching / rate recovery / decoding` 主路径
- 基于协议参数建模的 DMRS 插入与 LS 信道估计
- `transform-precoded PUSCH DMRS` 的 38.211 clause `5.2.2 / 5.2.3` 双分支实现
- 支持多发多收的 `TDL/CDL` link-level MIMO、CDL 双极化/XPR 与 TDL 相关矩阵传播
- 每次示例仿真自动输出信噪比、解调星座图、导频信道估计幅度/相位图

当前版本仍需进一步补齐：

- 完整的 PUSCH/PDSCH 映射规则与多天线预编码流程
- `PTRS / CSI-RS` 等协议信号与更完整的表项覆盖
- 38.901 系统级大尺度模型，例如路径损耗、阴影衰落、空间一致性、阻塞和氧吸收

因此，这个版本更适合作为“工程基础框架 + 可扩展参考实现”，而不是“已完成全部 R18 细节校准的协议级金模型”。协议严格化工作已经在结构上预留好了接入点。

本项目采用 [MIT License](LICENSE)。

## 信号维度约定

所有信号流数组都必须保留天线、流等物理维度，不能因为 `SISO`、单天线或单流场景省略维度，避免二次开发时误解各轴含义。

- 发射频域网格：`(num_tx_ant, num_subcarriers, num_symbols)`，`SISO` 时为 `(1, num_subcarriers, num_symbols)`。
- 发射时域波形：`(num_tx_ant, slot_samples)`，`SISO` 时为 `(1, slot_samples)`。
- 接收时域波形：`(num_rx_ant, slot_samples)`，`SISO` 时为 `(1, slot_samples)`。
- 接收频域网格：`(num_rx_ant, num_subcarriers, num_symbols)`，`SISO` 时为 `(1, num_subcarriers, num_symbols)`。
- 频域 MIMO 信道系数：`(num_subcarriers, num_rx_ant, num_tx_ant)`，`SISO` 时为 `(num_subcarriers, 1, 1)`。

部分底层函数仍会兼容旧的一维/二维输入，便于单元测试和外部脚本迁移；但主链路模块的输出必须遵守上述固定维度。

## 目录结构

```text
src/nr_phy_simu/
  common/          公共类型、接口、OFDM、BWP、MCS、LDPC、HARQ、序列
  tx/              发射机：编码、扰码、调制、频域资源映射
  rx/              接收机：频域抽取、信道估计、均衡、解调、译码
  channels/        AWGN / TDL / CDL / profile tables / channel factory
  scenarios/       PUSCH / PDSCH / 多 TTI / 回放 / 扫描与组件工厂
  io/              配置、波形、频响与多 TTI 报告读写
  config.py        全局配置 dataclass
  visualization.py 统一 PlotArtifact 渲染入口
configs/           完整、最小、baseline、信道与干扰配置样例
docs/              系统设计、扩展开发、绘图、性能与排障文档
examples/
  run_from_config.py
  run_link_curve.py
tests/             快速、专项与 baseline 回归测试
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

如果你希望先从整体上理解系统设计、模块职责、数据维度约定和主要函数调用链，建议先阅读：

- [docs/system_design.md](docs/system_design.md)

绘图调用链和如何新增绘图节点的开发说明见：

- [docs/plotting_development.md](docs/plotting_development.md)
- [docs/custom_intermediate_plotting.md](docs/custom_intermediate_plotting.md)

如果你要替换信道估计、均衡、信道模型等模块，并搭建一条新的发射机-信道-接收机链路，开发说明见：

- [docs/custom_link_development.md](docs/custom_link_development.md)

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

对于不适合放进 `config` 或 `result` 的运行期中间变量，可以使用全局生效的 `SimulationRuntimeContext`：

```python
from nr_phy_simu.common.runtime_context import get_runtime_context
from nr_phy_simu.common.types import PlotArtifact

context = get_runtime_context()
context.set("channel_estimation", "my_metric", my_metric)
context.add_plot_artifact(
    PlotArtifact(
        name="my_metric",
        values=my_metric,
        plot_type="magnitude",
    )
)
```

`context.set(...)` 适合保存其他模块可能读取的临时变量；`context.add_plot_artifact(...)` 适合保存绘图变量，输出文件名形如 `outputs/<prefix>_context_my_metric.png`。

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
- [configs/pusch_external_freqresp_fd.yaml](configs/pusch_external_freqresp_fd.yaml)
- [configs/pusch_replay_template.yaml](configs/pusch_replay_template.yaml)
- [configs/minimal_pusch_awgn.yaml](configs/minimal_pusch_awgn.yaml)
- [inputs/pusch_capture.txt](inputs/pusch_capture.txt)

当前参数模型至少覆盖：

- `MCS table type`
- `MCS index`
- `waveform`
- `channel type`
- `channel.config_path`
- `channel.seed`
- `channel.params.tti_evolution`
- `channel.geometry.tx_position_m / rx_position_m`
- `channel.geometry.ue_velocity_vector_mps`
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

完整配置也可以通过顶层 `base_config_path` 继承另一个 YAML/JSON/XML 配置；字典递归合并，当前文件覆盖基底，列表和标量整体替换。路径始终相对于声明它的文件解析，循环继承会直接报错：

```yaml
base_config_path: minimal_pusch_awgn.yaml
channel:
  params:
    snr_db: 15.0
simulation:
  num_ttis: 100
```

直接构造或加载后的 `SimulationConfig` 是调用方输入。仿真会在私有运行视图中解析 MCS、TBS 和 coded-bit capacity，不再把派生值回写到调用方对象；解析结果从 `SimulationResult.transport_plan` 或多 TTI 的 `final_config` 读取。

### 动态配置字段

配置文件支持自适应扩展。新增字段不需要同步修改 `config.py`，加载后会递归转换成可属性访问的配置节点：

```yaml
my_receiver:
  algorithm: neural_mmse
  debug:
    dump_llr: true
```

代码中可以读取为：

```python
config.my_receiver.algorithm
config.my_receiver.debug.dump_llr
```

已有配置段也支持新增字段，例如 `channel.params.snr_db` 既可以继续用 `config.channel.params["snr_db"]` 访问，也可以用 `config.channel.params.snr_db` 访问。

需要注意：

- 已有强类型字段仍优先解析，例如 `dmrs`、`link`、`channel` 的标准字段不会被动态字段覆盖。
- 与已有属性或方法重名的字段不会挂成属性，但会保存在 `extras` 或字典里，例如 `config.carrier.extras["n_subcarriers"]`。
- 非法 Python 标识符字段不能用点号访问，例如 `my-receiver` 只能通过 `config.extras["my-receiver"]` 访问。
- 只有既有路径字段会自动解析相对路径；新增动态字符串字段不会自动当作路径处理。

### 外部频域信道系数

工程支持从配置或文本文件输入每子载波频域信道系数。对应信道类型为：

- `EXTERNAL_FREQRESP_FD`：频域直通过信道，跳过发端 OFDM 调制和收端 OFDM 解调，直接对频域资源栅格做 `Y[k,l] = H[k] X[k,l] + N[k,l]`。
- `EXTERNAL_FREQRESP_TD`：先把外部频域响应 IFFT 成时域 FIR tap，再对时域波形滤波。当前该模式仅支持 `SISO`。

配置入口位于 `channel.params`：

- `frequency_response_path`：外部信道系数文本文件路径。
- `frequency_response`：直接写在 YAML/JSON/XML 中的信道系数数组。
- `time_domain_tap_length`：仅 `EXTERNAL_FREQRESP_TD` 使用，表示从 IFFT 后的 impulse response 中截取多少个时域 tap；`null` 表示保留完整 IFFT 长度。

无论从文件还是从配置数组读取，子载波维度长度都必须等于小区带宽子载波数：

```text
num_subcarriers = carrier.cell_bandwidth_rbs * 12
```

例如 `cell_bandwidth_rbs: 24` 时，文件必须有 `288` 行，每行对应一个小区带宽内的子载波。

#### 子载波顺序

文件第 `0` 行对应资源栅格的 `subcarrier index = 0`，第 `1` 行对应 `subcarrier index = 1`，依次递增，直到 `num_subcarriers - 1`。

这里的索引是工程内部 cell-band active subcarrier 的顺序，不需要在文件里写 FFT 负频率/正频率的移位顺序，也不需要包含保护子载波。外部系数只覆盖小区带宽内的 active subcarriers。

#### SISO 文本文件格式

`SISO` 时，虽然主链路内部统一使用 `(num_subcarriers, 1, 1)`，但文本文件可以简写为“每行一个复数”。每行就是当前子载波上的标量信道系数 `H[k, rx0, tx0]`。

支持的复数写法包括：

```text
1.0 0.0
0.98 -0.05
0.92+0.12j
0.87,-0.18
```

等价含义为：

```text
第0个子载波: H[0,0,0] = 1.0 + 0.0j
第1个子载波: H[1,0,0] = 0.98 - 0.05j
第2个子载波: H[2,0,0] = 0.92 + 0.12j
第3个子载波: H[3,0,0] = 0.87 - 0.18j
```

#### MIMO 文本文件格式

`MIMO` 时，每一行仍然对应一个子载波，但这一行需要写出该子载波上完整的 `Nrx x Ntx` 信道矩阵。当前文本格式使用分号 `;` 分隔矩阵元素，并按“先 RX、后 TX”的行优先顺序展开：

```text
H[k,rx0,tx0]; H[k,rx0,tx1]; ...; H[k,rx1,tx0]; H[k,rx1,tx1]; ...
```

也就是说，每行元素个数必须等于：

```text
num_rx_ant * num_tx_ant
```

例如 `num_rx_ant = 2`、`num_tx_ant = 2` 时，每行写 `4` 个复数：

```text
h00; h01; h10; h11
```

对应矩阵为：

```text
H[k] = [[h00, h01],
        [h10, h11]]
```

一个具体例子：

```text
1.0+0.0j; 0.2+0.1j; -0.1+0.05j; 0.9-0.2j
0.98-0.02j; 0.22+0.08j; -0.12+0.04j; 0.88-0.18j
```

含义为：

```text
第0个子载波:
  H[0,0,0] = 1.0+0.0j
  H[0,0,1] = 0.2+0.1j
  H[0,1,0] = -0.1+0.05j
  H[0,1,1] = 0.9-0.2j

第1个子载波:
  H[1,0,0] = 0.98-0.02j
  H[1,0,1] = 0.22+0.08j
  H[1,1,0] = -0.12+0.04j
  H[1,1,1] = 0.88-0.18j
```

如果是 `num_rx_ant = 4`、`num_tx_ant = 2`，则每行必须写 `8` 个复数，顺序为：

```text
H[k,0,0]; H[k,0,1]; H[k,1,0]; H[k,1,1]; H[k,2,0]; H[k,2,1]; H[k,3,0]; H[k,3,1]
```

#### 在 YAML 中直接配置

也可以不使用文本文件，直接在 `frequency_response` 中写数组。

`SISO` 可以写成长度为 `num_subcarriers` 的数组，每个元素可以是 `[real, imag]`：

```yaml
channel:
  model: EXTERNAL_FREQRESP_FD
  params:
    add_noise: false
    frequency_response_path: null
    frequency_response:
      - [1.0, 0.0]
      - [0.98, -0.05]
      - [0.92, 0.12]
```

`MIMO` 推荐写成形状为 `(num_subcarriers, num_rx_ant, num_tx_ant)` 的三维数组：

```yaml
channel:
  model: EXTERNAL_FREQRESP_FD
  params:
    add_noise: false
    frequency_response_path: null
    frequency_response:
      - # subcarrier 0
        - [[1.0, 0.0], [0.2, 0.1]]
        - [[-0.1, 0.05], [0.9, -0.2]]
      - # subcarrier 1
        - [[0.98, -0.02], [0.22, 0.08]]
        - [[-0.12, 0.04], [0.88, -0.18]]
```

其中每个 `[real, imag]` 表示一个复数系数。上述例子的第一层是子载波，第二层是 RX 天线，第三层是 TX 天线。

#### 典型用例

仓库中已有可运行的 `SISO` 外部频域信道示例：

- 配置文件：[configs/pusch_external_freqresp_fd.yaml](configs/pusch_external_freqresp_fd.yaml)
- 信道系数文件：[inputs/siso_frequency_response_24rb_flat.txt](inputs/siso_frequency_response_24rb_flat.txt)

运行方式：

```bash
python examples/run_from_config.py configs/pusch_external_freqresp_fd.yaml
```

`inputs/` 目录还提供了两个 MIMO 信道系数格式样例：

- [inputs/mimo_frequency_response_24rb_2rx2tx.txt](inputs/mimo_frequency_response_24rb_2rx2tx.txt)：`2Rx x 2Tx` 频域信道系数示例，共 `288` 行，对应 `24RB` 小区带宽。每行是一个子载波上的 `2 x 2` 信道矩阵，按 `H[k,0,0]; H[k,0,1]; H[k,1,0]; H[k,1,1]` 排列。
- [inputs/mimo_time_domain_taps_2rx2tx_8tap.txt](inputs/mimo_time_domain_taps_2rx2tx_8tap.txt)：`2Rx x 2Tx` 时域 tap 系数格式示例，共 `8` 行，每行是一个 tap 上的 `2 x 2` tap 矩阵，按 `h[tap,0,0]; h[tap,0,1]; h[tap,1,0]; h[tap,1,1]` 排列。当前内置 `EXTERNAL_FREQRESP_TD` 仍只消费外部频域响应并仅支持 `SISO`，该文件主要用于说明 MIMO 时域 tap 文件的推荐排布，便于后续自定义 MIMO 时域信道读取。

### 独立信道配置文件

系统仿真 YAML 可以只配置独立信道文件路径：

```yaml
channel:
  config_path: channels/tdl_c.yaml
```

信道文件位于 [configs/channels](configs/channels)，当前提供：

- `awgn.yaml`
- `tdl_a.yaml` 到 `tdl_e.yaml`
- `cdl_a.yaml` 到 `cdl_e.yaml`

外部信道文件格式示例：

```yaml
model: TDL
seed: 1007
geometry:
  tx_position_m: [0.0, 0.0, 25.0]
  rx_position_m: [100.0, 0.0, 1.5]
  ue_velocity_vector_mps: [3.0, 0.0, 0.0]
params:
  profile: TDL-C
  delay_spread_ns: 300.0
  snr_db: 12.0
```

合并规则：外部信道文件作为基础配置；系统 YAML 中的 `channel.model`、`channel.seed`、`channel.geometry`、`channel.params` 会覆盖外部文件。`carrier.center_frequency_hz` 是系统唯一载频配置，OFDM 相位补偿、TDL/CDL Doppler、波长和阵列相位都读取该字段；旧配置中的 `channel.params.carrier_frequency_hz` 仅作为兼容迁移入口，不建议继续使用。

随机与时间演进口径：

- `channel.seed: null`：回退到系统级 `random_seed`。
- `channel.seed: 整数`：固定一条可复现的多 TTI realization 序列；同一次多 TTI 运行中的各 TTI 不会重复。
- `channel.seed: auto`：每次运行使用非确定随机源。
- `channel.params.tti_evolution: independent`（默认）：逐 TTI 独立 drop，信道对象和 RNG 保持复用。
- `channel.params.tti_evolution: continuous`：TDL/CDL 固定几何、射线耦合和初始相位，并在由 `slot_index` 推导的绝对时间轴上连续演进。

SNR 的权威配置入口是 `channel.params.snr_db`。顶层 `snr_db` 仅为旧代码兼容镜像，不建议直接修改。

`geometry` 属于 link-level 辅助信息：系统会校验三维坐标，计算 TX/RX 距离、LOS 单位方向，并优先用 `ue_velocity_vector_mps` 推导 UE 速度大小和运动方向。当前不会用坐标引入路径损耗、阴影衰落、LOS 概率或空间一致性。

对于 `TDL/CDL`，当前已支持的典型信道参数包括：

- `profile`
- `delay_spread_ns`
- `ue_speed_mps`
- `max_doppler_hz`
- `ue_azimuth_deg` / `ue_zenith_deg`
- `num_sinusoids`
- `k_factor_db`（LOS profile 可选覆盖）
- `tdl_mimo_method`：`iid` / `correlated` / `spatial_filter`
- `tdl_tx_correlation` / `tdl_rx_correlation`
- `spatial_filter`（TDL 可选显式空间滤波矩阵）
- `tx_array` / `rx_array`：支持默认单极化 ULA，也可显式配置双极化端口
- `xpr_db` / `xpr_sigma_db`（CDL）
- `angle_scaling_enabled` 以及 `desired_*` 角度均值/扩展配置（CDL）

`TDL/CDL` link-level 模型按 38.901 的 `0.5 GHz` 到 `100 GHz` 载频范围校验配置，当前激活带宽上限为 `2 GHz`。Profile 表本身不按载频切换；载频影响波长、阵列相位和 Doppler。

### BWP 和载频相位补偿

完整仿真配置中使用顶层 `bwp` 描述单 active BWP：

```yaml
carrier:
  center_frequency_hz: 3500000000.0  # 系统唯一载频，单位 Hz

bwp:
  start_rb: 0
  num_rbs: null
  phase_compensation_enabled: true
```

`link.prb_start` 是 BWP 内 PRB 起点；实际映射到小区全带 grid 的子载波起点为 `(bwp.start_rb + link.prb_start) * 12`。当 `bwp.num_rbs: null` 时，BWP 等于小区带宽，旧配置行为保持不变。BWP 中心频率由 `carrier.center_frequency_hz`、小区带宽、BWP 起点/宽度和 SCS 推导，不需要单独配置。

OFDM 复包络按 38.211 clause 5.4 在每个 symbol 的有效起点使用常量相位，频率采用 RF carrier `carrier.center_frequency_hz`，时间参考在每个 subframe 重置；slot 0 波形与 py3gpp 参考实现逐样点对齐。CP-OFDM PUSCH/PDSCH Gold DMRS 的序列偏移使用 `(bwp.start_rb + link.prb_start)` 对应的 CRB 参考点，而不是仅用 BWP 内偏移。

当 `fft_size` 或 `sample_rate_hz` 未显式提供时，工程会自动选择满足当前带宽要求的最小 `2` 的整数次幂 FFT，并据此计算采样率。
循环前缀长度不再由配置文件手动输入，而是根据 `cyclic_prefix + subcarrier spacing + sample rate` 自动推导。

### 干扰源配置文件

干扰源可以直接引用另一个仿真配置文件，把干扰用户的 `link / dmrs / scrambling / channel / random_seed` 独立出来：

```yaml
interference:
  sources:
    - label: jammer_tdl_c
      enabled: true
      config_path: interferers/pusch_interferer_tdl_c.yaml
      inr_db: -3.0
      prb_start: 12
      num_prbs: 8
```

被引用的干扰配置文件可以像普通仿真 YAML 一样配置自己的 DMRS、小区扰码、RNTI、MCS、波形和信道。仓库示例位于：

- [configs/interferers/pusch_interferer_awgn.yaml](configs/interferers/pusch_interferer_awgn.yaml)
- [configs/interferers/pusch_interferer_tdl_c.yaml](configs/interferers/pusch_interferer_tdl_c.yaml)

合并和约束规则：

- `config_path` 文件作为干扰用户基础配置；`interference.sources[]` 中显式写出的 `prb_start / num_prbs / waveform / mcs / channel_params` 等字段会覆盖它。
- `label / enabled / inr_db` 只控制干扰注入，不从被引用文件读取。
- 主链路的 `carrier`、`bwp`、`slot_index` 和 `link.num_rx_ant` 会强制覆盖干扰用户配置，保证目标信号和干扰信号能逐样点相加。
- 被引用文件中的 `interference` 字段会被忽略，避免循环引用或嵌套干扰。
- 干扰信号强制使用 `simulation.bypass_channel_coding = true`，即不做 `CRC/LDPC/rate matching`，而是生成随机 coded bits 后进入扰码、调制、资源映射、DMRS 和时域处理。
- 干扰信道内部噪声会关闭，最终只按 `inr_db` 相对主链路底噪做功率缩放并叠加。
- 干扰信道与目标信道的相关性当前通过各自 `channel.seed` 和 `channel.geometry` 间接控制；暂不做联合多用户信道生成。

当 `simulation.num_ttis > 1` 时，系统会连续运行多个 TTI，并统计最终 `BLER`。这里的误包定义为：`CRC` 错误的 TTI。
多 TTI 运行器只按顺序执行，不提供内部线程/进程并行；调用方可以按 SNR 点或独立 seed 在进程级并行。回放文件格式当前只表示单个 TTI，因此 `waveform_input` 与 `num_ttis > 1` 的组合会明确报错。
如果同时配置了 `simulation.result_output_path`，系统还会把多 TTI 结果按 CSV 风格追加写入文本文件；文件不存在时会自动创建，空文件会先写标题行。
结果文件当前列顺序为：`信噪比, BLER, EVM, EVM_SNR, RB位置, MCS阶数, 总TTI数, 误包数, 码率, 调制阶数, TBsize`。
其中 `EVM` 使用 RMS 定义：`sqrt(mean(|equalized-reference|^2) / mean(|reference|^2))`；`EVM_SNR` 使用线性域 `1/EVM^2`，写文件时转换为 dB。

噪声方差口径：时域信道先按时域波形功率加噪，进入接收机均衡/解调前会按当前 FFT 点数换算为频域 RE 噪声方差。外部频域直通信道直接在频域加噪，因此保持频域噪声方差不再换算。

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
- DMRS 符号位置解析已经抽到 [src/nr_phy_simu/common/sequences/dmrs_tables.py](src/nr_phy_simu/common/sequences/dmrs_tables.py)，当前默认位置推导走显式表驱动入口，更方便后续继续补 R18 表项或做协议对表

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
- TDL 默认 `iid` zero-correlation MIMO，以及可选 TX/RX 相关矩阵
- CDL 20-ray cluster、随机 ray coupling、双极化/XPR、阵列相位和 LOS K-factor 合成
- 可选 CDL cluster 角度均值和角度扩展缩放

当前 `TDL/CDL` 范围是 38.901 R18 Clause 7.7 link-level 模型，不包含 38.901 系统级路径损耗、阴影衰落、空间一致性、阻塞、氧吸收或地图模型。

典型配置文件可参考：

- [configs/pusch_tdl_c.yaml](configs/pusch_tdl_c.yaml)
- [configs/pusch_tdl_c_explicit_paths.yaml](configs/pusch_tdl_c_explicit_paths.yaml)
- [configs/pusch_cdl.yaml](configs/pusch_cdl.yaml)

## 接收数据维度

接收机内部当前统一采用带天线维的数据组织方式：

- `rx_grid` 统一为 `num_rx_ant x num_subcarrier x num_symbol`
- `channel_estimate` 统一为 `num_rx_ant x num_user_subcarrier x num_symbol`，信道估计发生在用户 PRB 频域抽取之后
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
- `LDPC` 译码最大迭代次数、normalized min-sum 缩放因子与 `py3gpp` fallback 已开放为配置项，见 `decoder.*`

当前已实现的 `MCS table` 包括：

- `qam64`
- `qam256`
- `qam64lowse`
- `qam1024`（PDSCH）
- `tp64qam`（transform-precoded PUSCH）
- `tp64lowse`（transform-precoded PUSCH）

LDPC normalized min-sum 的 check-node 更新已按行度数分桶并用 NumPy 向量化。`SimulationResult.ldpc_decoder_path` 与 `ldpc_iterations` 会给出实际译码路径和迭代次数；min-sum 未收敛而切换到 GF(2)、py3gpp 或硬判决时记录 `WARNING`。作为库使用时默认安装 `NullHandler`，应用可自行启用日志：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

多 TTI runner 会周期性记录 `INFO` 进度。性能规模和 CDL 内存预期见 [docs/performance.md](docs/performance.md)，常见故障见 [docs/troubleshooting.md](docs/troubleshooting.md)，版本变化见 [CHANGELOG.md](CHANGELOG.md)。

## HARQ、层与传输块管理

当前版本新增了显式的传输计划与 HARQ 管理：

- [src/nr_phy_simu/common/transmission.py](src/nr_phy_simu/common/transmission.py)
  统一解析 `MCS`、`TBS`、码字容量与传输块计划
- [src/nr_phy_simu/common/harq.py](src/nr_phy_simu/common/harq.py)
  提供多 TTI 下的 HARQ 进程、RV 轮换与重传状态管理
- [src/nr_phy_simu/common/layer_mapping.py](src/nr_phy_simu/common/layer_mapping.py)
  将层映射显式抽象成独立步骤，便于后续扩到更完整的多层实现

当前主链路显式限制为 `num_layers = 1`、`num_codewords = 1`。多天线发射默认是单流复制并做总功率归一，不是真实 NR 多层空间复用或预编码。

典型 HARQ 配置可参考：

- [configs/pusch_awgn_harq.yaml](configs/pusch_awgn_harq.yaml)

## 曲线脚本

当前版本已新增链路级 BER/BLER sweep 脚本：

- [examples/run_link_curve.py](examples/run_link_curve.py)

一个典型调用方式如下：

```bash
python examples/run_link_curve.py configs/pusch_curve_awgn.yaml --snr-points -5,0,5,10,15
```

输出内容包括：

- `outputs/<prefix>.csv`
- `outputs/<prefix>.png`

其中 CSV 记录每个 `SNR` 点的 `BLER / BER / average EVM / average EVM_SNR`，PNG 则绘制链路级 BER/BLER 曲线。

## 当前状态

README 之前列出的 5 条“后续建议”，当前已经收敛为以下已实现能力：

1. 本地可控的 NR `UL-SCH LDPC` 编解码主路径，且 `i_LS` 由最终 `Zc` 按 38.212 lifting-size 集合查表
2. DMRS 生成与默认位置推导已拆成独立模块，type A single-symbol 的 `l_d=8/9, addPos=0` 边界已按协议修正
3. 38.901 R18 Clause 7.7 `TDL-A/B/C/D/E` 与 `CDL-A/B/C/D/E` link-level 模型已接入并纳入测试
4. HARQ、MCS、层映射、码字与传输块管理已有显式数据结构与运行流程
5. 单元测试、baseline 回归、以及链路级 BER/BLER 曲线脚本已经补齐

如果后续还要继续往“协议精确平台”推进，剩余更适合继续深挖的方向主要是：

- `PDSCH` 双码字与更完整的多层/预编码链路
- 更广泛的 MATLAB/商用仪表参考波形互通回归集
- 38.901 系统级大尺度模型与空间一致性
- 更完整的 R18 表驱动资源配置（包括尚未实现的 `PTRS/CSI-RS`）
