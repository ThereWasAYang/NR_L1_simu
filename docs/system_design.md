# NR L1 Simu 系统设计说明

本文面向希望读懂、维护、二次开发本工程的使用者。目标不是替代源码中的每一行代码，而是给出一条可以对照源码阅读的“地图”：读者可以从入口脚本开始，顺着配置、场景、发射机、信道、接收机、绘图、测试逐层进入，理解每个文件、每个类、每个函数为什么存在，输入输出是什么，以及和其他模块如何连接。

## 阅读方法

建议按下面顺序阅读代码：

1. 先看 [README.md](../README.md) 的使用方式、配置示例和信号维度约定。
2. 再看 [config.py](../src/nr_phy_simu/config.py)，理解所有参数最终会被规整成哪些 dataclass。
3. 进入 [scenarios/base.py](../src/nr_phy_simu/scenarios/base.py)，这是单 TTI 仿真的主控流程。
4. 进入 [scenarios/component_factory.py](../src/nr_phy_simu/scenarios/component_factory.py)，理解默认发射机、接收机、信道由哪些对象组成。
5. 分别阅读 `tx/`、`channels/`、`rx/` 下的处理模块。
6. 最后看 [visualization.py](../src/nr_phy_simu/visualization.py)、[tests/test_smoke.py](../tests/test_smoke.py) 和 `docs/` 中的扩展文档。

本文中的数组维度默认采用 `main` 分支的 NumPy 实现。`codex-torch-port` 分支保持相同设计和维度约定，只是核心矩阵/张量计算逐步替换为 `torch.Tensor`。

## 总体目标

工程的核心设计目标有四个：

- 将 NR 物理层共享信道仿真拆成发射机、信道、接收机三大块。
- 所有核心算法通过抽象接口解耦，方便替换单个模块或一组模块。
- 所有仿真参数由外部 YAML/JSON/XML 管理，代码只消费统一的 `SimulationConfig`。
- 所有信号流数组固定保留天线、流等物理维度，避免 SISO 时隐式降维造成二次开发误解。

因此，本工程不是把所有逻辑写进一个脚本，而是让每个处理块成为独立类，并通过场景对象装配成完整链路。

## 信号维度总约定

主链路中的信号流数组不因天线数、流数为 1 而省略维度：

- `TxPayload.resource_grid`: `(num_tx_ant, num_subcarriers, num_symbols)`。
- `TxPayload.waveform`: `(num_tx_ant, slot_samples)`。
- `RxPayload.rx_waveform`: `(num_rx_ant, slot_samples)`。
- `RxPayload.rx_grid`: `(num_rx_ant, num_subcarriers, num_symbols)`。
- 外部频域 MIMO 信道: `(num_subcarriers, num_rx_ant, num_tx_ant)`。
- 信道估计结果: `(num_rx_ant, num_subcarriers, num_symbols)`。
- MIMO 合并前的数据 RE: `(num_rx_ant, num_data_re)`。
- 均衡后数据符号: `(num_data_symbols,)`。
- LLR: `(coded_bit_capacity,)`。

部分底层函数仍允许旧式一维/二维输入，原因是测试和外部脚本迁移更方便；但模块输出应恢复到上述固定维度。

## 源码目录职责

### `config.py`

`config.py` 定义所有配置 dataclass。外部 YAML/JSON/XML 最终都会被加载成 `SimulationConfig`。

关键类：

- `CarrierConfig`: 载波、SCS、CP、FFT、采样率。
- `DmrsConfig`: DMRS config type、符号位置、扰码 ID、transform-precoded 相关字段。
- `ScramblingConfig`: 数据扰码的 RNTI、N_id、codeword index。
- `McsConfig`: MCS table、MCS index、调制方式、目标码率、RV。
- `ChannelConfig`: 信道模型名和信道参数字典。
- `InterferenceConfig` / `InterferenceSourceConfig`: 干扰源列表和每个干扰源的 INR、RB、信道类型等。
- `SimulationControlConfig`: TTI 数、结果文件路径、是否跳过编译码。
- `WaveformInputConfig`: 灌数仿真文件路径、样点数、噪声方差。
- `LinkConfig`: PUSCH/PDSCH、波形、层数、天线数、RB 分配、时域符号分配。
- `SimulationConfig`: 总配置对象。

关键函数和属性：

- `CarrierConfig.n_subcarriers`: 小区带宽子载波数，等于 `cell_bandwidth_rbs * 12`。
- `CarrierConfig.fft_size_effective`: 若配置未提供 FFT 点数，则自动选择不小于小区带宽的最小 2 的整数次幂。
- `CarrierConfig.sample_rate_effective_hz`: 若配置未提供采样率，则用 `fft_size * SCS` 自动计算。
- `CarrierConfig.cyclic_prefix_lengths`: 根据 CP 类型、SCS 和采样率推导每个 OFDM 符号 CP 长度。
- `SimulationConfig.from_mapping`: 把配置文件中的字典转成强类型 dataclass。
- `SimulationConfig.__post_init__`: 创建配置后立即执行协议约束检查。
- `_validate_protocol_constraints`: 检查 DFT-s-OFDM 只能使用 type1 DMRS、不支持数据/DMRS 共符号、需要 `num_cdm_groups_without_data = 2` 等约束。

设计意图：

- CP 长度不暴露为手工配置，避免不符合协议的配置进入链路。
- `ChannelConfig.params` 保持字典形式，是为了允许不同信道模型拥有不同参数。
- `SimulationConfig` 在构造时做协议约束，避免错误配置延迟到中间模块才报错。

### `io/config_loader.py`

配置加载器负责把文件读入并构造 `SimulationConfig`。

关键函数：

- `load_simulation_config(path)`: 根据文件后缀选择 YAML/JSON/XML 解析，然后调用 `SimulationConfig.from_mapping`。
- `_resolve_relative_paths`: 将 `waveform_input.waveform_path` 和 `channel.params.frequency_response_path` 从相对配置文件路径转换成绝对路径。
- `_read_text`: 使用 `utf-8-sig` 读取文本，兼容 Windows 下带 BOM 的 UTF-8 文件。

设计意图：

- Windows 默认编码可能不是 UTF-8，因此所有文本配置文件显式按 UTF-8 读取。
- 路径相对配置文件，而不是相对当前工作目录，便于从任意目录运行示例脚本。

### `common/interfaces.py`

这是全工程的抽象接口层。所有可替换算法都通过这里定义边界。

基础接口：

- `ChannelCoder.encode(bits, config)`: TB bits 到编码比特。
- `BitScrambler.scramble(bits, config)`: 编码比特扰码。
- `Modulator.map_bits(bits, config)`: 扰码比特到复调制符号。
- `ResourceMapper.map_to_grid(data_symbols, config)`: 数据和 DMRS 映射到频域资源网格。
- `TimeDomainProcessor.modulate(grid, config)`: 频域网格到时域波形。
- `ChannelModel.propagate(waveform, config)`: 时域信号过信道。
- `FrequencyExtractor.extract(grid, data_mask, config, despread)`: 从接收频域网格或信道估计网格抽取数据 RE。
- `ChannelEstimator.estimate(rx_grid, dmrs_symbols, dmrs_mask, config)`: 信道估计。
- `MimoEqualizer.equalize(rx_symbols, channel_estimate, noise_variance, config)`: MIMO 合并或单天线均衡。
- `Demodulator.demap_symbols(symbols, noise_variance, config)`: 星座点到 LLR。
- `ChannelDecoder.decode(llrs, config)`: LLR 到 TB bits。
- `DmrsSequenceGenerator.generate_for_symbol(symbol, config)`: 生成指定 OFDM symbol 的本地 DMRS。

高层接收机接口：

- `ReceiverDataProcessor.process(...)`: 替换“rx_grid 到 LLR”这一段。适合神经网络接收机、联合信道估计+均衡+解调模块。
- `ReceiverProcessingStage.process(context)`: 管线式处理中一个可组合 stage。
- `ReceiverProcessor.receive(...)` / `receive_from_grid(...)`: 替换整个接收机或任意更大范围接收流程。

设计意图：

- 低层接口适合替换单个算法模块。
- `ReceiverDataProcessor` 适合无法拆成信道估计、均衡、解调三步的算法。
- `ReceiverProcessor` 适合连时域处理、解扰、译码都想接管的高级用户。

### `common/types.py`

这里定义跨模块传递的结构体。它们比散乱的 tuple 更易读，也方便绘图和测试读取中间结果。

关键 dataclass：

- `TxPayload`: 发射机输出和中间变量。
- `ChannelEstimateResult`: 信道估计完整网格、导频处估计和绘图 artifact。
- `ReceiverDataProcessingResult`: `ReceiverDataProcessor` 输出，至少包含 LLR。
- `ReceiverProcessingContext`: 管线式接收机 stage 共享上下文。
- `RxPayload`: 接收机输出和中间变量。
- `SimulationResult`: 单 TTI 仿真结果。
- `MultiTtiSimulationResult`: 多 TTI 汇总结果。
- `PlotArtifact`: 通用绘图数据包。

设计意图：

- `TxPayload` 和 `RxPayload` 是绘图、误码统计、EVM 计算的统一数据来源。
- `ChannelEstimateResult` 既保存完整估计，也保存导频处抽取结果，避免绘图函数重新做一次信道估计。
- `PlotArtifact` 允许算法模块把任意中间变量交给统一绘图系统。

## 单 TTI 场景主流程

单 TTI 的主入口是 `SharedChannelSimulation.run`，PUSCH 和 PDSCH 都继承这一条主流程。

文件：

- [scenarios/pusch.py](../src/nr_phy_simu/scenarios/pusch.py)
- [scenarios/pdsch.py](../src/nr_phy_simu/scenarios/pdsch.py)
- [scenarios/base.py](../src/nr_phy_simu/scenarios/base.py)

### `PuschSimulation` / `PdschSimulation`

这两个类本身很薄，主要职责是：

- 设置或检查 `config.link.channel_type`。
- 复用 `SharedChannelSimulation` 的端到端流程。

设计意图：

- PUSCH/PDSCH 未来可以在自己的类中扩展差异化流程。
- 当前共享信道主流程相同，因此集中在 `SharedChannelSimulation` 中。

### `SharedChannelSimulation.__init__`

构造函数做五件事：

1. 保存 `config`。
2. 创建或接收外部注入的 `SimulationRuntimeContext`。
3. 通过 `component_factory.create_components(config)` 创建默认组件。
4. 调用 `build_transmitter` 和 `build_receiver` 装配 TX/RX。
5. 通过 `create_channel_factory().create(config)` 创建信道模型。

为什么使用 `component_factory`：

- 如果只传 `PuschSimulation(config)`，系统使用默认链路。
- 如果用户想替换模块，可以自定义 `SimulationComponentFactory`，只替换其中一个字段。
- 这避免用户手动 new 出完整发射机、接收机、信道对象。

### `SharedChannelSimulation.run`

这是最重要的主控函数。执行顺序如下：

1. 清空 runtime context，并设置为当前线程全局上下文。
2. 如果是 HARQ 重传，写入 `harq.process_id` 和 `harq.is_retransmission`。
3. 调用 `mapper.count_data_re(config)` 统计可承载数据的 RE 数。
4. 调用 `build_transport_block_plan(config, data_re)` 计算 TBS、码率、码字容量等。
5. 根据 `transport_block_override` 或随机种子生成 TB bits。
6. 判断是否为 `EXTERNAL_FREQRESP_FD` 频域直通信道。
7. 若是频域直通：
   - 调用 `transmitter.build_slot_payload` 只生成频域网格，不做 OFDM。
   - 调用 `channel.propagate_grid` 直接频域过信道。
   - 调用 `receiver.receive_from_grid` 从频域网格开始接收。
   - 将 `tx_payload.waveform` 置为空的 `(num_tx_ant, 0)`。
8. 若是普通时域信道：
   - 调用 `transmitter.transmit` 完成发射机和 OFDM。
   - 调用 `channel.propagate` 过时域信道。
   - 调用 `interference_mixer.apply` 叠加干扰源。
   - 调用 `receiver.receive` 完成接收机。
9. 调用 `_build_result` 形成 `SimulationResult`。

### `SharedChannelSimulation._build_result`

这个函数负责结果统计：

- 若 `bypass_channel_coding = true`，参考比特是发射端随机 coded bits。
- 否则参考比特是 transport block。
- `bit_errors` 由 `decoded != reference` 统计。
- `bit_error_rate = bit_errors / reference_bits.size`。
- `_compute_evm_metrics` 比较发射星座点和均衡后星座点，得到 EVM 和 EVM_SNR。
- 将 HARQ、CRC、干扰报告、transport plan 等一起写入 `SimulationResult`。

### `SharedChannelSimulation._uses_frequency_domain_channel`

只检查：

```python
config.channel.model.upper() == "EXTERNAL_FREQRESP_FD"
```

原因：

- 频域直通信道不应该经过 OFDM 调制/解调。
- 其他信道都走时域波形路径。

## 组件装配

文件：[scenarios/component_factory.py](../src/nr_phy_simu/scenarios/component_factory.py)

这个文件把抽象接口和默认实现绑定起来。

### 组件 dataclass

- `TransmitterComponents`: `coder/scrambler/modulator/mapper/time_processor`。
- `ReceiverComponents`: `time_processor/extractor/estimator/equalizer/demodulator/scrambler/decoder/data_processor/receiver_processor`。
- `SharedComponents`: 当前只放 `dmrs_generator`。
- `SimulationComponents`: 总容器，包含 shared、transmitter、receiver。

为什么 DMRS generator 是 shared：

- 发射端资源映射需要生成 DMRS。
- 接收端需要同样的本地 DMRS 作为信道估计参考。
- 放在 shared 中便于发射和接收使用同一个默认实现。

### `DefaultSimulationComponentFactory.create_components`

默认装配如下：

- `DmrsGenerator` 作为 DMRS 序列生成器。
- `NrDataScrambler` 作为数据扰码器和 LLR 解扰器。
- `OfdmProcessor` 同时作为发射和接收时域处理模块。
- 发射端：
  - `NrLdpcCoder` 或 `RandomBitCoder`。
  - `QamModulator`。
  - `FrequencyDomainResourceMapper`。
- 接收端：
  - `FrequencyDomainExtractor`。
  - `LeastSquaresEstimator`。
  - `OneTapMmseEqualizer`。
  - `QamDemodulator`。
  - `NrLdpcDecoder` 或 `HardDecisionBypassDecoder`。

当 `simulation.bypass_channel_coding = true`：

- 发射端使用 `RandomBitCoder`。
- 接收端使用 `HardDecisionBypassDecoder`。
- CRC 不参与 BLER 统计。

### `build_transmitter`

将 `TransmitterComponents` 转成一个 `Transmitter` 对象。这里显式传入 `LayerMapper`，便于未来替换多层映射逻辑。

### `build_receiver`

将 `ReceiverComponents` 转成一个 `Receiver` 对象。这里会把 `data_processor` 和 `receiver_processor` 传给 `Receiver`，因此自定义链路可以通过组件工厂注入高级接收机模块。

## 发射机设计

入口文件：[tx/chain.py](../src/nr_phy_simu/tx/chain.py)

### `Transmitter.build_slot_payload`

这是发射端频域网格生成函数。执行顺序：

1. `coder.encode(transport_block, config)`。
2. `scrambler.scramble(coded_bits, config)`。
3. `modulator.map_bits(scrambled_bits, config)`。
4. `layer_mapper.map_symbols(tx_symbols, num_layers)`。
5. `mapper.map_to_grid(serialized_symbols, config)`。
6. `_expand_tx_grid(grid, config)` 将 2D mapper 输出扩展成 `(num_tx_ant, K, L)`。
7. 返回 `TxPayload`，其中 `waveform` 是空的 `(num_tx_ant, 0)`。

为什么拆成 `build_slot_payload`：

- 频域直通信道不需要 OFDM 调制。
- 干扰源生成和普通链路都可以复用同一套频域构造逻辑。

### `Transmitter._expand_tx_grid`

默认 mapper 当前输出单流 2D 网格 `(K, L)`。这个函数保证主链路输出固定为 3D：

- 若输入已是 3D，直接返回。
- 若 `num_tx_ant == 1`，返回 `grid[np.newaxis, ...]`。
- 若 `num_tx_ant > 1`，复制到每根发射天线并除以 `sqrt(num_tx_ant)`，保持总功率不变。

注意：

- 这不是完整 NR 预编码，只是一个默认占位策略。
- 未来引入真正 precoder 时，应替换这一段或在 mapper 前后增加预编码模块。

### `Transmitter.transmit`

执行：

1. 调用 `build_slot_payload` 得到频域网格。
2. 调用 `time_processor.modulate(resource_grid, config)` 得到时域波形。
3. 用 `dataclasses.replace` 替换 `payload.waveform` 后返回。

## 编码和译码

发射端文件：[tx/codec.py](../src/nr_phy_simu/tx/codec.py)

接收端文件：[rx/decoding.py](../src/nr_phy_simu/rx/decoding.py)

底层 LDPC 文件：[common/ulsch_ldpc.py](../src/nr_phy_simu/common/ulsch_ldpc.py)

### `NrLdpcCoder.encode`

主要职责：

- 根据 `build_transport_block_plan` 的结果决定 TBS、码率、RV、调制阶数、码字容量。
- 添加 CRC。
- code block segmentation。
- LDPC encoding。
- rate matching。
- 输出长度为 `coded_bit_capacity` 的编码比特。

工程为什么实现本地 UL-SCH LDPC 主路径：

- 曾发现 py3gpp 在 BG2 K=640 等低码率场景下存在接口或实现问题。
- 本工程需要稳定通过低 MCS、低 SNR、多 TTI 基线用例。

### `RandomBitCoder.encode`

当 `simulation.bypass_channel_coding = true`：

- 不做 CRC/LDPC/rate matching。
- 根据 `coded_bit_capacity` 生成随机 coded bits。
- 适合验证调制、资源映射、信道、均衡、EVM 等功能。

### `NrLdpcDecoder.decode`

主要职责：

- LLR rate recovery。
- LDPC decoding。
- code block desegmentation。
- CRC decode。
- 将 CRC 结果保存到 `self.last_crc_ok`。

### `HardDecisionBypassDecoder.decode`

跳过译码，只做 LLR 硬判决：

```python
bit = llr < 0
```

并将 `last_crc_ok` 置为 `None`。

## 调制和解调

发射端文件：[tx/modulation.py](../src/nr_phy_simu/tx/modulation.py)

接收端文件：[rx/demodulation.py](../src/nr_phy_simu/rx/demodulation.py)

### `QamModulator.map_bits`

读取 `config.link.modulation`，调用静态函数 `map_bits_for_modulation`。

支持：

- `PI/2-BPSK`
- `BPSK`
- `QPSK`
- `16QAM`
- `64QAM`
- `256QAM`
- `1024QAM`

### `QamModulator._pad_bits`

若输入 bits 数不是每个调制符号 bit 数的整数倍，则补零。原因是调制映射必须按固定 bit group 切分。

### `QamDemodulator.demap_symbols`

解调流程：

1. 根据调制方式生成理想星座点和 bit label。
2. 计算每个接收符号到每个星座点的欧氏距离。
3. 对每个 bit 位，取 bit=0 星座集合与 bit=1 星座集合的最小距离。
4. 用 max-log 近似得到 LLR。

LLR 符号约定和硬判决保持一致：`llr < 0` 判为 1。

## MCS 和传输块规划

文件：

- [common/mcs.py](../src/nr_phy_simu/common/mcs.py)
- [common/transmission.py](../src/nr_phy_simu/common/transmission.py)

### `resolve_mcs`

根据 `config.link.mcs.table` 和 `index` 查表，得到：

- modulation。
- target code rate。
- transform-precoded PUSCH 是否可用。

若用户显式配置了 `modulation` 和 `target_code_rate`，则也支持绕过 MCS table。

### `apply_mcs_to_link`

将解析出的 MCS 结果写回 `config.link.modulation` 和 `config.link.code_rate`，让后续模块只读取统一字段。

### `build_transport_block_plan`

根据数据 RE 数、调制阶数、层数、目标码率、xOh 等计算：

- TBS。
- coded bit capacity。
- base graph 相关规划。
- codeword 级 RV 等信息。

该结果会被写入 runtime context，编码器和译码器可复用。

## 资源映射和 DMRS

资源映射文件：[tx/resource_mapping.py](../src/nr_phy_simu/tx/resource_mapping.py)

DMRS 文件：

- [common/sequences/dmrs.py](../src/nr_phy_simu/common/sequences/dmrs.py)
- [common/sequences/dmrs_tables.py](../src/nr_phy_simu/common/sequences/dmrs_tables.py)

### `FrequencyDomainResourceMapper.count_data_re`

用于在真正映射前统计 data RE 数。TBS 和 coded bit capacity 依赖这个值。

核心逻辑：

- 遍历用户 PRB 和调度 OFDM symbol。
- 根据 DMRS mask 排除导频 RE。
- 根据 CP-OFDM 的 `data_mux_enabled` 决定 DMRS symbol 内非导频 RE 是否可用于数据。
- DFT-s-OFDM 下遵守 transform-precoded PUSCH 约束。

### `FrequencyDomainResourceMapper.map_to_grid`

输出：

- `grid`: 频域资源网格。
- `dmrs_mask`: DMRS RE 布尔 mask。
- `data_mask`: data RE 布尔 mask。
- `dmrs_symbols`: 按 mapper 顺序串行化的 DMRS 序列。

执行顺序：

1. 创建全零 cell-band grid。
2. 推导 DMRS symbol 位置。
3. 根据 config type 生成 DMRS comb RE。
4. 调用 `DmrsGenerator` 生成本地 DMRS 序列。
5. 根据 DMRS 功率偏置写入导频。
6. 对数据符号进行 CP-OFDM 或 DFT-s-OFDM 映射。

### DMRS 功率偏置

当 type1 DMRS 且不做数据/导频共符号时，导频 RE 功率相对数据 RE 有协议规定的功率偏置。mapper 在写入 DMRS 时应用该缩放。

### DFT-s-OFDM 数据处理

DFT-s-OFDM 波形下：

- 数据进入频域映射前先做 DFT spreading。
- DMRS 仍按频域 comb 映射。
- 非导频 RE 根据 transform-precoded PUSCH 约束处理。

### `DmrsGenerator`

`DmrsGenerator` 负责生成 NR DMRS 序列，主要分两类：

- CP-OFDM PUSCH/PDSCH DMRS。
- transform-precoded PUSCH DMRS。

关键设计：

- Gold 序列、低 PAPR 序列、group hopping、sequence hopping 等协议细节集中在 `dmrs.py`。
- 资源位置由 mapper 管，序列值由 generator 管，二者解耦。
- 生成结果要求恒模特性满足 DMRS 典型属性。

## OFDM 时域处理

文件：[common/ofdm.py](../src/nr_phy_simu/common/ofdm.py)

### `OfdmProcessor.modulate`

输入：

- `(num_tx_ant, K, L)` 或兼容旧式 `(K, L)`。

输出：

- `(num_tx_ant, slot_samples)`。

执行逻辑：

1. 对每根 TX antenna 调用 `_modulate_single`。
2. `_modulate_single` 对每个 OFDM symbol：
   - 创建完整 FFT bin。
   - 将 cell-band active subcarriers 放入 FFT 中心。
   - `ifftshift` 后 IFFT。
   - 添加 cyclic prefix。
   - 串接所有 symbol。

### `OfdmProcessor.demodulate`

输入：

- `(num_rx_ant, slot_samples)` 或兼容旧式 `(slot_samples,)`。

输出：

- `(num_rx_ant, K, L)`。

执行逻辑：

1. 对每根 RX antenna 调用 `_demodulate_single`。
2. `_demodulate_single` 根据 CP 长度切片每个 OFDM symbol。
3. FFT 后 `fftshift`，抽取 active subcarriers。

设计意图：

- OFDM 只负责频域/时域转换，不关心 DMRS、数据、信道估计。
- CP 长度来自 `CarrierConfig`，不手写。

## 信道模型

目录：[channels/](../src/nr_phy_simu/channels)

### `channel_factory.py`

`DefaultChannelFactory.create(config)` 根据 `config.channel.model` 创建：

- `AWGN`
- `TDL`
- `CDL`
- `EXTERNAL_FREQRESP_TD`
- `EXTERNAL_FREQRESP_FD`

### `awgn.py`

`AwgnChannel.propagate`：

1. `_expand_receive_branches` 将输入扩展到 `(num_rx_ant, slot_samples)`。
2. 若 `add_noise = false`，直接返回。
3. 计算信号平均功率。
4. 根据 SNR 计算噪声方差。
5. 生成复高斯白噪声并相加。

注意：

- 若 TX 多分支输入进 AWGN，默认先合成参考流再复制到 RX 分支。这只是简单 AWGN 模型，不是 MIMO 信道。

### `fading_base.py`

TDL/CDL 共享的时变多径 MIMO 基类。

核心函数：

- `propagate`: 总入口，扩展 TX 分支，生成路径系数，卷积信道，加噪声。
- `_generate_path_coefficients`: 抽象函数，由 TDL/CDL 子类实现。
- `_apply_time_varying_channel`: 对每个 RX、TX、path 做延迟和时变系数相乘累加。
- `_fractional_delay`: 用 sinc 插值实现分数采样延迟。
- `_rayleigh_process`: 用正弦和近似生成 Rayleigh fading。
- `_rician_process`: LOS profile 中生成 Rician fading。
- `_normalize_powers_db`: dB path power 转线性归一化功率。
- `_resolve_path_parameters`: 支持用户覆盖每径时延和功率。
- `_expand_tx_branches`: 保证 TX 波形有显式天线轴。
- `_array_response`: 生成简化 ULA 空间响应。

数学形式近似：

```text
r_rx[n] = sum_tx sum_path h[rx,tx,path,n] * x_tx[n-delay_path] + noise[n]
```

### `tdl.py`

`TdlChannel._generate_path_coefficients`：

1. 从 `profile_tables.py` 读取 TDL-A/B/C/D/E 的默认 normalized delays 和 powers。
2. 调用 `_resolve_path_parameters` 应用 delay spread 或用户显式路径覆盖。
3. 对每条 path 生成 Rayleigh/Rician 时变过程。
4. 叠加简化的 TX/RX 空间响应，形成 `(num_rx_ant, num_tx_ant, num_paths, num_samples)`。

### `cdl.py`

`CdlChannel._generate_path_coefficients`：

1. 从 profile table 读取 CDL cluster。
2. 根据 cluster 角度、UE 速度、载频计算简化 Doppler。
3. 对 cluster 内 rays 叠加空间响应。
4. 对 CDL-D/E 等 LOS profile 叠加 Rician/specular 分量。

当前 CDL 是基础 MIMO profile 实现，尚未覆盖完整 38.901 阵列、极化、XPR 和空间一致性。

### `external_frequency_response.py`

外部信道系数模式。

`ExternalFrequencyResponseBase.load_frequency_response`：

- 从 `frequency_response` 或 `frequency_response_path` 读取复数数组。
- 检查第一维必须等于 `carrier.n_subcarriers`。

`_frequency_response_matrix`：

- SISO 1D 输入转成 `(K, 1, 1)`。
- MIMO 文件中每行 `Nrx*Ntx` 的 2D 输入 reshape 成 `(K, Nrx, Ntx)`。
- YAML 三维数组直接校验为 `(K, Nrx, Ntx)`。

`ExternalFrequencyResponseFrequencyDomainChannel.propagate_grid`：

1. 读取并规整 `H[k, rx, tx]`。
2. 将 TX grid 规整成 `(Ntx, K, L)`。
3. 执行：

```python
rx_grid = np.einsum("krt,tks->rks", channel_matrix, tx_grid)
```

等价于每个子载波、每个符号做 `Y = H X`。

4. 加 AWGN。
5. 输出 `(Nrx, K, L)`。

`ExternalFrequencyResponseTimeDomainChannel.propagate`：

- 当前只支持 SISO。
- 将频域响应嵌入 FFT bins。
- IFFT 得到 impulse response。
- 截取 `time_domain_tap_length`。
- 用 FIR tap 对时域波形滤波。

## 干扰源

文件：[scenarios/interference.py](../src/nr_phy_simu/scenarios/interference.py)

### `InterferenceMixer.apply`

输入是期望信号过信道后的 `rx_waveform`。

执行逻辑：

1. 若未配置干扰源，直接返回原波形。
2. 遍历每个 enabled source。
3. `_build_interferer_config` 基于主配置复制出干扰源配置。
4. `_generate_interferer_waveform` 为干扰源独立跑发射机和信道。
5. `_scale_to_inr` 根据噪声方差和目标 INR 缩放干扰源功率。
6. 将干扰波形加到主波形。

### `_scale_to_inr`

目标功率：

```text
target_power = noise_variance * 10^(INR/10)
```

缩放系数：

```text
scale = sqrt(target_power / measured_interference_power)
```

设计意图：

- 干扰源可以配置独立信道、RB 位置、MCS、波形。
- 当前频域直通信道模式不支持干扰注入，因为主链路跳过了时域信号。

## 接收机设计

入口文件：[rx/chain.py](../src/nr_phy_simu/rx/chain.py)

### `Receiver.__init__`

保存所有接收机子模块：

- `time_processor`
- `extractor`
- `estimator`
- `equalizer`
- `demodulator`
- `decoder`
- `scrambler`
- `data_processor`
- `receiver_processor`

若没有传入 `data_processor`，自动创建 `ThreeStageReceiverDataProcessor`。

若没有传入 `receiver_processor`，自动创建 `DefaultReceiverProcessor`。

### `Receiver.receive`

从时域波形开始接收。它不直接实现细节，而是委托给：

```python
self.receiver_processor.receive(...)
```

这使得用户可以替换整个接收机主流程。

### `Receiver.receive_from_grid`

从频域网格开始接收。用于：

- `EXTERNAL_FREQRESP_FD`
- 自定义频域灌入链路
- 用户跳过时域处理的实验

同样委托给 `receiver_processor.receive_from_grid`。

## 默认接收机主处理器

文件：[rx/receiver_processing.py](../src/nr_phy_simu/rx/receiver_processing.py)

### `DefaultReceiverProcessor.receive`

执行：

1. `receiver.time_processor.demodulate(rx_waveform, config)`。
2. 调用 `receive_from_grid`。

### `DefaultReceiverProcessor.receive_from_grid`

执行：

1. 若 `rx_grid` 是 2D，兼容性地扩展成 3D。
2. 调用 `receiver.data_processor.process(...)` 得到 LLR 和中间结果。
3. 如果没有显式信道估计结果，构造空的 `ChannelEstimateResult`。
4. `receiver.scrambler.descramble_llrs(processing.llrs, config)`。
5. `receiver.decoder.decode(descrambled_llrs, config)`。
6. 组装 `RxPayload`。

设计意图：

- 默认流程仍是传统接收机。
- 高级用户可以只替换 `data_processor`，保留解扰和译码。
- 更高级用户可以替换 `receiver_processor`，接管全部流程。

## 接收机中段处理器

文件：[rx/data_processing.py](../src/nr_phy_simu/rx/data_processing.py)

### `ThreeStageReceiverDataProcessor.process`

默认中段处理流程：

1. `estimator.estimate` 得到完整信道估计。
2. `extractor.extract(rx_grid, data_mask, despread=False)` 抽取数据 RE。
3. `extractor.extract(channel_estimate, data_mask, despread=False)` 抽取对应信道。
4. `equalizer.equalize` 做 MMSE 合并。
5. DFT-s-OFDM 时调用 `_despread_equalized`。
6. `layer_mapper.unmap_symbols` 构造 per-layer 视图。
7. `demodulator.demap_symbols` 得到 LLR。
8. 返回 `ReceiverDataProcessingResult`。

### `_despread_equalized`

对每个调度 OFDM symbol：

- 根据 `data_mask` 统计该 symbol 的数据 RE 数。
- 从均衡后串行符号中切出对应段。
- 做 IFFT 并乘 `sqrt(N)`，撤销发射端 DFT spreading。

### `ReceiverDataProcessorPipeline`

这是更灵活的组合机制。它维护一个 `ReceiverProcessingContext`，每个 stage 可以读写：

- `rx_grid`
- `channel_estimation`
- `rx_data_symbols`
- `data_channel`
- `equalized_symbols`
- `llrs`
- `metadata`
- `plot_artifacts`

内置 stage：

- `ChannelEstimationStage`
- `DataExtractionStage`
- `EqualizationStage`
- `TransformPrecodingDespreadStage`
- `LayerDemappingStage`
- `DemodulationStage`

设计意图：

- 用户可以任意组合传统模块和自定义模块。
- 不强迫一个新算法必须拆成固定三步。

## 信道估计

文件：[rx/channel_estimation.py](../src/nr_phy_simu/rx/channel_estimation.py)

### `LeastSquaresEstimator.estimate`

输入：

- `rx_grid`: `(Nrx, K, L)`。
- `dmrs_symbols`: 本地 DMRS 串行序列。
- `dmrs_mask`: DMRS RE mask。

执行：

1. 若输入是 2D，兼容性扩成 3D。
2. 对每根 RX antenna 调用 `_estimate_single`。
3. 堆叠成 `(Nrx, K, L)`。
4. `_extract_pilot_estimates` 抽取导频 RE 上的估计结果，用于绘图。
5. 返回 `ChannelEstimateResult`。

### `_estimate_single`

对单根天线：

1. 如果没有 DMRS，则返回全 1 信道。
2. `_estimate_dmrs_symbols` 先在每个 DMRS symbol 上做 LS 和频域插值。
3. `_interpolate_time` 将 DMRS symbol 的估计插值到全 slot。

### `_ls_estimate`

最简单的 LS：

```text
H_ls = Y_dmrs / X_dmrs
```

### `_interpolate_frequency`

对一个 DMRS symbol，已知 comb 导频处的 H，用 `np.interp` 对实部和虚部分别做线性插值，得到全带宽 H。

### `_interpolate_time`

跨 OFDM symbol 做线性插值。若只有一个 DMRS symbol，则直接复制到所有 symbol。

### `_extract_pilot_estimates`

只从 `channel_estimate` 中取 `dmrs_mask=True` 的位置，用于绘制导频估计幅相图。这样绘图不会重新实现信道估计逻辑。

## 频域抽取、均衡、解调

### `rx/frequency_extraction.py`

`FrequencyDomainExtractor.extract`：

- 若输入是 `(Nrx, K, L)`，对每根 RX antenna 单独抽取并 stack。
- 若输入是 `(K, L)`，按单天线处理。
- `_extract_single` 按调度 symbol 顺序读取 `data_mask=True` 的 RE。
- 若 `despread=True` 且是 PUSCH DFT-s-OFDM，则对每个 symbol 做 IFFT despreading。

默认接收机中 `despread=False`，因为 DFT-s-OFDM despreading 在均衡之后统一做。

### `rx/equalization.py`

`OneTapMmseEqualizer.equalize`：

- 若 `rx_symbols` 是 2D，按 RX antenna 做 MRC/MMSE 合并：

```text
x_hat = sum(conj(H) * Y) / (sum(|H|^2) + noise)
```

- 若是 1D，则做单天线 one-tap MMSE：

```text
w = conj(H) / (|H|^2 + noise)
x_hat = w * Y
```

### `rx/demodulation.py`

见前文调制解调说明。输出 LLR 后，默认接收机主流程会先解扰再译码。

## 数据扰码

文件：[common/sequences/scrambling.py](../src/nr_phy_simu/common/sequences/scrambling.py)

### `NrDataScrambler.scramble`

根据 RNTI、N_id、codeword index、slot 等生成 NR 数据扰码序列，并对编码比特做 XOR。

### `NrDataScrambler.descramble_llrs`

LLR 域解扰不是 XOR，而是根据扰码 bit 翻转 LLR 符号：

```text
scrambling bit = 0 -> LLR 不变
scrambling bit = 1 -> LLR 取反
```

设计意图：

- 发射和接收共用同一个 scrambler 对象即可保证种子逻辑一致。

## 多 TTI、HARQ 和结果文件

### `scenarios/multi_tti.py`

`MultiTtiSimulationRunner.run`：

- 连续运行 `simulation.num_ttis` 个 TTI。
- 每个 TTI 调用单 TTI 场景。
- 根据 `crc_ok` 统计 packet error。
- 计算 BLER。
- 对 EVM 和 EVM_SNR 做跨 TTI 平均。
- 若配置了 `result_output_path`，由上层脚本或报告函数写文件。

多 TTI 中误包定义：

```text
crc_ok is False
```

若 `bypass_channel_coding = true`，CRC 为 `None`，BLER 用 `NaN/N/A` 表示。

### `common/harq.py`

`HarqManager` 管理 HARQ process：

- process id。
- RV sequence。
- retransmission 次数。
- 是否达到最大重传。

当前 HARQ 逻辑是框架级实现，主要为后续软合并和真实 HARQ buffer 预留结构。

### `io/multi_tti_report.py`

`append_multi_tti_report`：

- 文件不存在时创建。
- 文件为空时写标题行。
- 文件非空时追加数据行。
- 列包括 SNR、BLER、EVM、EVM_SNR、RB 位置、MCS、TTI 数、误包数、码率、调制阶数、TBsize。

EVM_SNR 写文件时用 dB，但 TTI 间平均先在线性域平均。

## 灌数仿真

文件：

- [scenarios/waveform_replay.py](../src/nr_phy_simu/scenarios/waveform_replay.py)
- [io/waveform_loader.py](../src/nr_phy_simu/io/waveform_loader.py)

### `load_text_waveform`

读取时域 IQ 文本：

- 每行一个复数样点。
- 支持 `I Q`、`I,Q`、`I+Qj`。
- 文件按天线 major 排列：RX0 全部样点、RX1 全部样点，以此类推。
- 返回 `(num_rx_ant, num_samples)`。

### `WaveformReplaySimulation.run`

跳过发射机和信道：

1. 根据配置计算 data RE、coded bit capacity、TBS。
2. 读取外部时域波形。
3. 构造本地 DMRS 和 data mask。
4. 估计或读取噪声方差。
5. 可选叠加干扰。
6. 调用 `receiver.receive`。
7. 构造空 TX placeholder 和 `SimulationResult`。

用途：

- 接入外部仪表或其他仿真器生成的时域数据。
- 只验证接收机链路。

## 绘图系统

文件：[visualization.py](../src/nr_phy_simu/visualization.py)

入口函数：

- `save_simulation_plots(result, config, output_dir, prefix)`。

绘图包括：

- 解调星座图。
- 信道估计导频幅度/相位。
- 接收机入口时域波形幅值。
- OFDM 后频域网格幅值。
- `PlotArtifact` 通用绘图。

### 标准图如何产生

`save_simulation_plots` 从 `SimulationResult` 中读取：

- `result.rx.equalized_symbols` 画星座图。
- `result.rx.channel_estimation.pilot_estimates` 画导频估计。
- `result.rx.rx_waveform` 画时域图。
- `result.rx.rx_grid` 画频域图。
- `result.rx.plot_artifacts` 和 runtime context artifact 画自定义图。

### `PlotArtifact`

任何模块都可以构造：

```python
PlotArtifact(
    name="my_metric",
    values=my_array,
    plot_type="magnitude",
)
```

只要该 artifact 进入 `RxPayload.plot_artifacts`、`ChannelEstimateResult.plot_artifacts` 或 runtime context，就会被统一渲染。

设计意图：

- 新算法不需要改主绘图流程。
- 绘图和算法计算解耦。
- 绘图函数只消费已经保存的中间结果，不重复实现估计算法。

## Runtime Context

文件：[common/runtime_context.py](../src/nr_phy_simu/common/runtime_context.py)

`SimulationRuntimeContext` 是一份单次仿真的运行期全局上下文，用于保存：

- 不适合写入 config 的中间参数。
- 不适合写入 result 的调试变量。
- 自定义绘图 artifact。
- HARQ、transport plan 等跨模块共享信息。

典型调用：

```python
context = get_runtime_context()
context.set("channel_estimation", "my_metric", value)
context.add_plot_artifact(PlotArtifact(...))
```

`SharedChannelSimulation.run` 每个 TTI 开始时会清空 context，避免上一次仿真的变量污染当前结果。

## 外部信道系数文件

频域文件读取函数：[io/frequency_response_loader.py](../src/nr_phy_simu/io/frequency_response_loader.py)

### `load_frequency_response`

支持两种输入：

- `frequency_response`: 配置文件中直接给数组。
- `frequency_response_path`: 文本文件路径。

SISO 文件：

```text
1.0 0.0
0.98 -0.05
0.92+0.12j
```

MIMO 文件每行一个子载波，分号分隔 `Nrx*Ntx` 个复数：

```text
H[k,0,0]; H[k,0,1]; H[k,1,0]; H[k,1,1]
```

仓库示例：

- [inputs/siso_frequency_response_24rb_flat.txt](../inputs/siso_frequency_response_24rb_flat.txt)
- [inputs/mimo_frequency_response_24rb_2rx2tx.txt](../inputs/mimo_frequency_response_24rb_2rx2tx.txt)
- [inputs/mimo_time_domain_taps_2rx2tx_8tap.txt](../inputs/mimo_time_domain_taps_2rx2tx_8tap.txt)

## 测试体系

文件：[tests/test_smoke.py](../tests/test_smoke.py)

测试分几类：

- PUSCH/PDSCH AWGN smoke。
- TDL/CDL channel smoke。
- 外部频响 SISO/MIMO shape 和运行测试。
- DMRS 恒模和协议约束测试。
- MCS table 测试。
- LDPC 回归测试。
- 多 TTI BLER baseline 测试。
- 绘图 smoke。
- 自定义接收机处理器装配测试。
- 灌数仿真测试。
- SNR sweep 测试。

每次上库前必须跑：

```bash
conda run -n NRpy312 python -m unittest -v tests.test_smoke.BaselineRegressionTest
```

建议同时跑：

```bash
conda run -n NRpy312 python -m compileall -q src tests
conda run -n NRpy312 python -m pytest tests/test_smoke.py -q
```

baseline 用例位于 [configs/baseline](../configs/baseline)，覆盖：

- PUSCH CP-OFDM QAM256 table MCS0 SNR=0。
- PUSCH DFT-s-OFDM QAM256 table MCS0 SNR=0。
- PUSCH CP-OFDM QAM256 table MCS27 SNR=50。
- PUSCH DFT-s-OFDM QAM256 table MCS27 SNR=50 且平均 EVM < 2%。

## 扩展点总结

### 替换单个传统模块

派生对应接口并在自定义 `SimulationComponentFactory` 中替换：

- 编码器：`ChannelCoder`
- 调制器：`Modulator`
- 资源映射：`ResourceMapper`
- 信道模型：`ChannelModel`
- 信道估计：`ChannelEstimator`
- 均衡器：`MimoEqualizer`
- 解调器：`Demodulator`
- 译码器：`ChannelDecoder`

### 替换 rx_grid 到 LLR 的整段算法

实现 `ReceiverDataProcessor`，并注入到 `ReceiverComponents.data_processor`。

适合：

- 神经网络接收机。
- 联合信道估计+均衡+解调。
- 不显式输出信道估计的黑盒算法。

### 任意组合多个接收机 stage

使用 `ReceiverDataProcessorPipeline` 和多个 `ReceiverProcessingStage`。

适合：

- 保留部分默认处理块。
- 插入自定义中间变量。
- 逐步替换算法。

### 替换整个接收机流程

实现 `ReceiverProcessor`，并注入到 `ReceiverComponents.receiver_processor`。

适合：

- 替换时域处理。
- 自定义频域处理。
- 连解扰、译码也接管。

### 新增绘图

推荐优先使用 `PlotArtifact`：

1. 算法模块生成中间变量。
2. 构造 `PlotArtifact`。
3. 放进结果对象或 runtime context。
4. `visualization.py` 自动渲染。

只有当绘图逻辑长期稳定、需要成为公共图形类型时，才修改 `visualization.py`。

## 端到端调用链速查

普通 PUSCH AWGN 单 TTI：

```text
examples/run_from_config.py
  -> load_simulation_config
  -> PuschSimulation(config)
  -> SharedChannelSimulation.__init__
  -> DefaultSimulationComponentFactory.create_components
  -> build_transmitter / build_receiver
  -> SharedChannelSimulation.run
  -> mapper.count_data_re
  -> build_transport_block_plan
  -> Transmitter.transmit
     -> coder.encode
     -> scrambler.scramble
     -> modulator.map_bits
     -> layer_mapper.map_symbols
     -> mapper.map_to_grid
     -> ofdm.modulate
  -> channel.propagate
  -> interference_mixer.apply
  -> Receiver.receive
     -> DefaultReceiverProcessor.receive
     -> ofdm.demodulate
     -> DefaultReceiverProcessor.receive_from_grid
     -> ThreeStageReceiverDataProcessor.process
        -> estimator.estimate
        -> extractor.extract
        -> equalizer.equalize
        -> demodulator.demap_symbols
     -> scrambler.descramble_llrs
     -> decoder.decode
  -> SharedChannelSimulation._build_result
  -> save_simulation_plots
```

频域直通外部信道：

```text
SharedChannelSimulation.run
  -> Transmitter.build_slot_payload
  -> ExternalFrequencyResponseFrequencyDomainChannel.propagate_grid
  -> Receiver.receive_from_grid
  -> SharedChannelSimulation._build_result
```

灌数仿真：

```text
WaveformReplaySimulation.run
  -> load_text_waveform
  -> mapper.map_to_grid(dummy_symbols)
  -> receiver.receive
  -> SimulationResult
```

多 TTI：

```text
MultiTtiSimulationRunner.run
  -> for each TTI:
       PuschSimulation/PdschSimulation.run
  -> packet error count
  -> BLER / average EVM
  -> optional append_multi_tti_report
```

## 如何对照本文读源码

如果你想理解某一行代码，建议按下面问题定位：

1. 这行代码属于哪一层：配置、装配、TX、Channel、RX、绘图、测试？
2. 这个函数的输入输出维度是什么？先查本文维度约定，再查 dataclass 注释。
3. 这个函数是接口实现，还是主流程调度？
4. 如果是接口实现，它是否可以被替换？对应抽象类在哪？
5. 如果是主流程调度，它调用的下一个模块是谁？
6. 中间变量是否会进入 `TxPayload`、`RxPayload`、`ChannelEstimateResult` 或 runtime context？
7. 是否有 `tests/test_smoke.py` 中的测试覆盖这一行为？

按这个方式阅读，绝大多数代码都可以归入清晰的信号链条中，而不是孤立理解。

