# NR_L1_Simu 全项目评审报告

> 评审日期：2026-07-13
>
> 评审基线：main 分支，提交 8844952
>
> 评审性质：只读评审；除新增本报告外，未修改产品代码、测试、配置或既有文档

## 1. 执行摘要

NR_L1_Simu 已经形成了一个可运行、可测试、具有较清晰分层的 5G NR 链路级仿真框架。项目在 AWGN 主链路、DMRS 处理、配置驱动、多格式输入、平台基础测试和问题诊断文档方面有扎实基础。当前版本适合作为研发原型、算法接入骨架和教学参考，但尚不宜被定位为可直接充当协议一致性判据或物理层“金参考”的仿真器。

本次评审的总体判断如下：

| 维度 | 结论 | 风险 |
|---|---|---|
| 项目结构与模块划分 | 主体合理，领域目录清晰；接收端扩展层次偏多 | 中 |
| 架构设计 | 主流程清楚，但配置、上下文和信道接口存在跨层耦合；局部有过度抽象 | 中高 |
| 代码质量与简洁性 | 命名总体清楚，测试可读；热点文件过大，若干静默容错会掩盖错误 | 中 |
| 接口与数据结构 | 基础抽象齐全，但契约不闭合，结果类型和信道返回值弱类型 | 中高 |
| 依赖与兼容性 | 当前环境和 CI 覆盖的版本可运行；私有 py3gpp API、无约束环境和打包入口构成风险 | 中高 |
| 性能与资源 | 单次运行可接受；多 TTI 默认保存全部大数组，规模化运行会线性耗尽内存 | 高 |
| 可观测性 | 有基础日志和部分诊断字段；缺少统一运行元数据、阶段耗时和结构化事件 | 中 |
| 文档完整性 | 数量和主题覆盖较好；API、完整配置、开发/发布指南、标准追踪矩阵缺失 | 中 |
| 文档与代码一致性 | 存在可复现的不一致和失效示例 | 中高 |
| 使用者体验 | 仓库内示例友好；安装后缺少正式 CLI、配置校验和可发现的资源 | 中 |
| 需求与目标一致性 | 多数框架级目标部分或基本满足；TDL/CDL 与 HARQ 的实现语义尚未达到完整能力声明 | 高 |
| Git 跟踪与 ignore | 当前跟踪范围干净；共享 ignore 和跨平台文本策略仍可补强 | 低中 |
| 安全与供应链 | 本地仿真器暴露面较小；依赖、CI Action 固定策略和大输入限制可加强 | 低中 |

### 1.1 最重要的结论

1. **P0：衰落信道的正分数延迟方向错误。** FadingChannelBase._fractional_delay 对正整数延迟会把脉冲向前移动，足够大的延迟甚至会把有效样本完全裁掉。TDL/CDL 均通过该路径施加时延，因此现有端到端“能跑通”的测试不能证明衰落信道在时域上物理正确。修复前，不应把 TDL/CDL 结果用于可信的链路性能结论。
2. **P1：HARQ 当前更接近重传调度器，不是完整软合并 HARQ。** 接收端每次只对当前传输的 LLR 做速率恢复和译码，没有按 HARQ process 维护软缓冲；多 TTI 报表又把每次 CRC 失败计为 packet error。当前 BLER 是“传输尝试错误率”，不是“最大重传后的 TB 失败率”。
3. **P1：长批次运行的默认结果保留策略不可扩展。** MultiTtiSimulationRunner 保存每个 TTI 的完整波形、资源网格、信道估计和 LLR。一次默认 PUSCH 结果的数组净数据约 2.76 MB；10,000 TTI 粗略即约 27.6 GB，尚未计 Python 对象和临时内存。
4. **P1：配置校验过于宽松。** 不支持的 SCS、过小 FFT、越界符号、零天线数和拼错字段均可能被接受；这会把配置错误变成难以定位的数值异常。ExternalFrequencyResponse 的 time_domain_tap_length: null 还会直接触发 TypeError，与 README 明确描述相反。
5. **P1：信道统一接口未真正闭合。** ChannelModel 只定义 propagate(waveform)，但外部频响的频域信道实现会让该方法无条件抛错，主引擎必须按 model 字符串分支调用非接口方法 propagate_grid。这削弱了多态性，也使新信道接入必须修改引擎。

### 1.2 建议的发布门槛

在对外强化版本能力声明或用于正式对比数据前，至少应完成：

- 修复分数延迟方向和边界行为，并增加解析可验证的脉冲、纯延迟相位、能量与多径叠加测试；
- 明确 HARQ 是“仅调度”还是“含软合并”，同步更正指标名称、文档和测试；
- 对所有配置建立严格校验，特别是 SCS、FFT、采样率、时频资源、天线数、外部信道参数和未知字段；
- 为多 TTI 增加流式聚合与结果保留策略；
- 把衰落信道解析基线和至少一项慢速端到端基线纳入 CI；
- 修正文档中已确认的错误示例和能力表述。

## 2. 评审范围、方法与限制

### 2.1 覆盖范围

本次评审覆盖以下内容：

- src/nr_phy_simu 下的公共接口、配置、公共数据结构、收发链路、信道、场景、I/O、运行时上下文、可视化和报表；
- configs、examples、benchmarks、scripts；
- tests/test_smoke.py 与 tests/test_platform_review.py；
- pyproject.toml、setup.py、CI、Git 跟踪和 ignore 策略；
- README、系统设计、性能指南、自定义链路指南、绘图指南、故障排查、变更记录和已记录问题；
- PUSCH/PDSCH、AWGN、TDL/CDL、外部频响、HARQ、多 TTI、重放和 SNR sweep 等主要使用路径。

### 2.2 采用的方法

- 静态阅读公共接口、关键实现、配置路径、异常处理、测试和文档；
- 对目录、代码规模、依赖、打包内容和 Git 跟踪范围进行盘点；
- 运行全量测试、慢速基线、字节码编译、依赖一致性检查和 wheel 构建；
- 用最小输入针对分数延迟、null 配置、非法 SCS、资源越界、未知字段和结果内存进行诊断；
- 将 README/设计文档中的声明与实际符号、默认值、调用方式和测试覆盖交叉核对。

### 2.3 已执行验证

| 验证 | 结果 |
|---|---|
| Python 3.12.13 环境下全量 pytest | 108 passed，约 14.42 s |
| 标记为 slow 的协议基线 | 1 passed，约 7.09 s |
| src、tests、examples、benchmarks 字节码编译 | 通过 |
| pip check | 通过 |
| 无隔离、无依赖下载的 wheel 构建 | 通过 |
| wheel 内容检查 | 仅包含 Python 包和 dist-info；未包含示例、配置、脚本或文档 |
| 覆盖率本地复核 | 当前既有环境未安装 pytest-cov，未获得可复核覆盖率百分比 |

当前验证环境的主要依赖为 NumPy 2.4.4、SciPy 1.17.1、Matplotlib 3.10.8、PyYAML 6.0.3、py3gpp 0.6.0。该结果证明当前组合可运行，不等于所有声明版本范围均已验证。

### 2.4 评审限制

- 本次未将输出与商业仿真器、3GPP 官方向量或另一套独立 PHY 实现逐样本比对，因此无法对所有算法作标准一致性背书；
- 未进行长时间 CPU profiling、内存峰值 profiling 和多平台实机运行，只做了关键路径的静态分析与针对性估算；
- 本地 .git/info/exclude 排除了几份尚未跟踪的 py3gpp 调研草稿和 AGENTS.md，它们不是远程仓库的一部分，故不作为项目正式文档评审基线；
- “通过测试”只说明当前测试断言满足，不能替代物理正确性、协议一致性和跨版本兼容性验证。

## 3. 优先级与问题总表

优先级定义：P0 为阻断正确性或会导致核心结论错误；P1 为高影响、应在近期版本修复；P2 为中等影响、影响维护性或体验；P3 为低风险改进。

| ID | 优先级 | 问题 | 主要影响 |
|---|---|---|---|
| COR-01 | P0 | 分数延迟方向和裁剪错误 | TDL/CDL 物理结果不可信 |
| HARQ-01 | P1 | HARQ 无软合并，BLER 统计语义错误 | 误导链路增益与可靠性结论 |
| PERF-01 | P1 | 多 TTI 默认保存完整结果 | 大规模运行内存失控 |
| CFG-01 | P1 | 配置校验不完整且未知字段静默接受 | 错误配置被掩盖 |
| CFG-02 | P1 | time_domain_tap_length: null 与实现冲突 | 合法文档配置运行失败 |
| API-01 | P1 | 统一信道接口不满足替换原则 | 引擎按具体模型分支，扩展困难 |
| TEST-01 | P1 | 衰落信道缺乏解析 oracle，慢速基线未进 CI | 核心错误未被测试发现 |
| MET-01 | P1 | EVM/比特长度与聚合方式不严格 | 形状错误被掩盖，指标含义含混 |
| DEP-01 | P1 | 直接调用 py3gpp 私有 LDPC API | 小版本升级即可破坏兼容性 |
| DOC-01 | P1 | 多处文档示例和符号已失效 | 用户按文档操作失败 |
| ARCH-01 | P2 | 全量 mutable config 贯穿组件，解析与运行计划混合 | 隐式耦合、重复深拷贝 |
| ARCH-02 | P2 | 接收端存在多套重叠扩展机制 | 过度设计、选择成本高 |
| CTX-01 | P2 | ContextVar 生命周期未作用域化 | 嵌套/并发运行串扰与对象滞留 |
| DATA-01 | P2 | dict/Any/ndarray 别名过多，运行元数据不足 | 契约不可发现、复现困难 |
| UX-01 | P2 | 安装包无 CLI，wheel 不带示例配置 | 安装后首跑路径断裂 |
| OBS-01 | P2 | 缺少统一结构化运行观测模型 | 性能和失败定位成本高 |
| CI-01 | P2 | CI 无覆盖率阈值、lint/type、安装包 smoke | 变更门禁不完整 |
| DOC-02 | P2 | API/配置/开发/发布/标准追踪文档缺口 | 学习、扩展和审计成本高 |
| GIT-01 | P3 | ignore 与 EOL 策略可补强 | 工具产物和跨平台噪声风险 |
| SEC-01 | P3 | 依赖和 Action 未充分锁定，大输入无限制 | 供应链与资源滥用风险 |

## 4. 项目结构与模块划分

### 4.1 做得较好的部分

src/nr_phy_simu 按 common、tx、rx、channels、scenarios、io 划分，领域边界比按“工具函数大杂烩”更清晰。配置、接口、类型、仿真引擎、场景入口和外部数据加载都有独立位置，PUSCH/PDSCH 能复用共享链路。examples、configs、tests、benchmarks 和 docs 也按用途分开，仓库中没有已跟踪的 build、dist、缓存或输出目录。

这种结构适合当前项目：核心仍是单进程 Python 链路仿真，不需要进一步拆成多个包或服务。现在引入插件服务、依赖注入容器、消息总线或分布式执行都会显著增加复杂度，收益不足。

### 4.2 需要调整的边界

#### ARCH-02：接收端扩展面重复

现状：接口层同时存在细粒度解调/解扰/速率恢复/译码组件、ThreeStageReceiverDataProcessor、任意 ReceiverDataProcessorPipeline，以及可替换整个 ReceiverProcessor。Factory 又为这些组合提供多层嵌套配置。

问题：这些扩展面覆盖相近职责。新开发者需要先理解“替换一个算法”“插入一个阶段”“替换三阶段数据处理”“替换整个接收机”之间的优先级与组合规则。所有组件还普遍接收完整 SimulationConfig，抽象虽然形式上解耦，语义上仍被全局配置绑定。

建议：保留一套主扩展机制——有序的 ReceiverStage pipeline。每个 stage 使用结构化输入输出与只读 RunContext；现有单算法组件通过 adapter 包装成 stage；整接收机替换只保留为高级逃生口。ThreeStageReceiverDataProcessor 可以进入兼容期，文档明确弃用路线。

验收标准：默认接收机只由一份阶段列表定义；插入自定义均衡器或译码器无需改 Factory 和引擎；同一扩展点只有一个推荐入口；兼容层有明确警告和移除版本。

#### 文件规模与职责集中

fading_base.py、ulsch_ldpc.py、dmrs.py、config.py、visualization.py 和 interfaces.py 均是高认知负荷热点。大文件本身不是缺陷，但这些文件同时承担算法、校验、缓存/适配、诊断或展示职责。

建议按稳定职责拆分，而非机械按行数拆分：

- config.py：原始 schema、跨字段验证、运行计划解析；
- ulsch_ldpc.py：公共 LDPC 适配、基图/分段、编码、速率匹配；
- visualization.py：artifact 模型、数据准备、具体 renderer；
- interfaces.py：按 tx、channel、rx、reporting 划分协议，公共层只重导出稳定接口。

拆分必须以减少依赖和提高独立测试性为目标，不建议为每个类创建一个文件。

## 5. 架构设计与过度设计评估

### 5.1 当前架构的优点

- SharedChannelSimulation 给出了易理解的单次运行骨架；
- 场景类薄，PUSCH/PDSCH 复用公共流程，避免重复实现；
- TX、Channel、RX、Metrics 的阶段边界已经可辨识；
- 工厂允许接入定制处理器，符合算法实验平台需求；
- 配置驱动和中间产物可视化为调试提供便利。

### 5.2 当前架构的核心问题

#### ARCH-01：原始配置、派生状态和运行计划混在一起

SimulationConfig 中既有用户输入，也有运行期间派生或改写的 modulation、code_rate、transport_block_size、coded_bit_capacity 等状态。不同路径通过 deepcopy 规避污染，但 sweep 又会直接修改调用者传入的 SNR，并在结束后留下最后一个值。

影响：

- 很难判断某字段来自用户、默认值、标准推导还是运行时回写；
- 组件依赖整个配置对象，无法声明最小输入契约；
- sweep、multi-TTI、probe、replay 容易产生不一致的复制和修改规则；
- 配置 hash、结果复现、并发运行和缓存都变得困难。

建议采用三阶段配置模型：

1. RawConfig：只表达用户输入，严格 schema，未知扩展只能放在 extensions 命名空间；
2. validate：做字段和跨字段校验，收集并一次性报告所有错误；
3. resolve：产生不可变 ResolvedRunPlan，其中包含 CarrierPlan、AllocationPlan、DmrsPlan、TransportPlan、ChannelPlan 和 SeedPlan。

运行组件只接收所需 plan 或只读 RunContext，不再修改原始配置。SNR sweep 使用不可变 replace 产生每个点的 plan。

#### API-01：信道抽象没有覆盖实际输入域

ChannelModel 只声明时域 waveform 的 propagate，但 ExternalFrequencyResponseFrequencyDomainChannel 的 propagate 会无条件报错，实际需要 propagate_grid。SharedChannelSimulation 根据 model 名称走特殊分支。

这违反替换原则：一个宣称实现 ChannelModel 的对象不能用于 ChannelModel 契约；新增频域信道还必须修改中心引擎，形成开放/封闭原则的缺口。

推荐两种可接受方案，优先选择第一种：

1. 统一 frame：SignalFrame 携带 values、domain、axes 和采样/资源网格元数据；Channel.process(frame, plan, rng) 返回 ChannelOutput。信道声明支持的 domain，引擎按 capability 做一次通用转换或报清晰错误；
2. 分离接口：TimeDomainChannel 与 FrequencyDomainChannel 两个显式 Protocol，引擎通过已解析 ChannelPlan 选择标准适配器，不按 model 字符串硬编码。

不建议继续向 ChannelModel 增加多个可选方法并在运行时 hasattr 判断，这只会把契约不确定性扩散。

#### CTX-01：全局运行时上下文生命周期隐式

RuntimeContext 使用 ContextVar 是比普通模块全局变量更安全的选择，但 simulation 和 replay 设置上下文后没有按 token 恢复；可视化再从“当前上下文”隐式取 artifact。顺序单次运行通常正常，嵌套运行、同线程的多个引擎、异常中断或长期批次会出现上下文串扰和大对象滞留风险。

建议：RunContext 显式由引擎创建并传递，artifact collector 属于本次运行；若仍保留 ContextVar 兼容入口，应提供 contextmanager，在 finally 中 reset token。绘图 API 优先接收 result/artifacts，而不是依赖当前全局状态。

### 5.3 是否存在过度设计

结论是“局部存在，但整体没有”。目录和主流程没有过度设计；项目目标包含自定义算法和可扩展链路，抽象接口是必要的。过度主要出现在接收端多套重叠扩展层、较大的 ABC 接口集合，以及“每个抽象都接收整个 mutable config”造成的伪解耦。

不建议进行全面重写。合适策略是先修正确性与契约，再以 adapter 渐进统一 pipeline。为了追求纯架构而引入 DI 框架、事件总线或微服务，会超出项目规模和目标。

## 6. 正确性、代码质量与简洁性

### 6.1 COR-01：分数延迟实现方向错误

证据：FadingChannelBase._fractional_delay 计算 integer_delay = floor(delay)，卷积后却从 filter_half_len + integer_delay 开始截取。对输入第 4 点的单位脉冲做最小诊断时：

| 配置延迟 | 当前输出峰值 | 正确期望 |
|---|---|---|
| 0 | 4 | 4 |
| +2 | 2 | 6 |
| +5 | 有效脉冲被裁掉 | 9（若输出长度允许） |

正延迟被实现为提前。_apply_time_varying_channel 会对每条径调用该函数，因此 TDL/CDL 均受影响。

风险：时域多径的到达顺序、相位、ISI 和接收窗对齐会出错。现有测试主要验证 shape、profile 信息和系数连续性，没有用解析脉冲响应验证延迟方向，所以 108 项测试全部通过仍未捕获此问题。

优化方案：

- 明确定义 delay 的符号、单位、输出长度和边界规则；
- 修正卷积对齐，使正 delay 将样本移向更晚时刻；
- 明确负延迟是否禁止，若允许则定义裁剪规则；
- 对超过输出窗的路径给出可测试的截断行为，而不是偶然消失；
- 优先使用统一的分数延迟核和显式 padding/cropping 计算，避免用难以审计的 start 偏移暗含符号。

必需测试：整数延迟脉冲、0.25/0.5/0.75 样点延迟、单音相位斜率、能量守恒、多径叠加、delay 超窗、零延迟恒等、TDL/CDL 已知 profile 的离散冲激响应。至少一个测试应使用独立公式生成 oracle，而不是复用生产代码。

### 6.2 CFG-02：null 配置与文档冲突

README 和示例配置把 channel.params.time_domain_tap_length: null 定义为“保留完整 IFFT”。ExternalFrequencyResponse 实现却直接对取出的值执行 int(...)，因此显式 null 会产生 TypeError。只有完全省略字段才会走默认值；当前测试覆盖的是省略，而不是文档中的 null。

建议先读取 raw 值：None 表示 fft_size；整数必须位于 1..fft_size；0、负数、布尔值和非整数产生带完整配置路径的 ConfigError。为 null、缺省、合法整数和上下界错误分别加测试。

### 6.3 CFG-01：校验缺口与静默容错

以下异常配置已被最小诊断确认可通过构造或加载：

- subcarrier_spacing_khz = 20 被 round(log2(scs/15)) 解释为 numerology 0；
- start_symbol = -1 被接受，Python 负索引可能把错误变成“最后一个符号”；
- 52 RB 配合 fft_size = 128 被接受，活动子载波明显超出 FFT 容量；
- num_tx_antennas = 0 被接受；
- simulation.num_tti 拼写错误会成为动态属性，而真正的 num_ttis 仍保持默认值 1。

建议建立集中式跨字段验证：

- SCS 只接受项目实际支持的枚举；numerology 用精确映射而非 round；
- fft_size 必须容纳 BWP/PRB、满足实现要求，并与 sample_rate = fft_size × SCS 一致；
- start_symbol、num_symbols 必须落入 slot，PRB 范围不得越过 BWP；
- TX/RX 天线数、layers、codewords 和端口数必须为正且满足组合约束；
- decoder 迭代、噪声、seed、slot、TTI 数和抽头长度应有明确范围；
- 已知配置区默认 extra=forbid，实验扩展放入 extensions，必要时提供显式 strict=false 兼容模式；
- 一次报告所有校验错误，错误包含 YAML/JSON/XML 路径、收到的值、允许范围和关联字段。

### 6.4 MET-01：指标计算会掩盖形状问题

EVM 计算使用 min(reference.size, estimate.size) 静默截短；这会让映射、抽取或译码链路少/多一个符号时仍得到貌似有效的 EVM。比特比较则依赖长度自然匹配，没有一致的契约。多 TTI 对 EVM 百分比和线性 EVM-SNR 做逐 TTI 算术平均，其统计含义也不同于将所有 reference/error 能量聚合后计算总体 EVM。

建议：

- 正常路径要求数组长度完全相等，否则抛出含 stage 和 shape 的错误；
- 仅在明确标注 partial/diagnostic 模式时允许截短；
- 保存每 TTI 的 reference_energy 和 error_energy，用总能量比生成 aggregate EVM；
- 报表明确区分 mean_per_tti_evm、aggregate_rms_evm 和由 EVM 推导的 SNR；
- 所有指标写出单位、分母、是否排除未检查 CRC 的 TTI 和 NaN 规则。

### 6.5 其他代码质量建议

- 将裸 tuple 和无结构 dict 替换成命名类型，减少位置错误与魔法 key；
- 对预期不可达的 fallback 明确计数并暴露到 result，而不只记录 warning；
- 用领域异常层次区分 ConfigError、InputFormatError、CapabilityError、NumericalError 和 DecodeError；
- 错误消息携带 stage、配置路径、shape/dtype 和关键 plan ID，但不要自动打印大数组；
- 对科学计算中的静默裁剪、自动 round、隐式广播和容错 fallback 坚持“默认失败、显式选择容错”。

## 7. 接口与数据结构设计

### 7.1 当前优点

- SimulationResult、TransmitterResult、ReceiverResult 比直接返回多级 tuple 更可读；
- 抽象接口明确列出了收发链路可替换职责；
- 配置 dataclass 对 IDE 和测试较友好；
- 输入加载使用统一配置入口，YAML/JSON/XML 的上层语义基本一致。

### 7.2 DATA-01：结果和边界类型仍然过弱

当前信道返回 (ndarray, dict)，channel_info 中的内容缺少稳定 schema，且主结果没有完整保留信道摘要。多个结果字段使用 Any；TransportPlan、interference、final_config 的含义和必需字段无法从类型看出。Waveform、ResourceGrid 等只是 ndarray 别名，不能表达 dtype、shape、轴顺序和采样率。

建议的新数据结构：

- SignalFrame：values、domain、axis spec、sample_rate 或 grid metadata；
- ChannelOutput：frame、ChannelSummary、可选 DebugPayloadRef；
- RunMetadata：run_id、配置 hash、代码版本、依赖版本、原始/有效 seed、slot/TTI、起止时间、阶段耗时、warning/fallback；
- RunResult：稳定 KPI、typed plans、RunMetadata，以及受 CapturePolicy 控制的 debug buffers；
- CapturePolicy：none、metrics、last、sampled、all 或按 artifact 类型选择。

公共类型应写明数组 shape/dtype/轴定义，并在运行边界检查，而不是试图通过复杂泛型在每个 NumPy 算子中做静态证明。

### 7.3 API 稳定性

顶层包目前导出主要仿真与配置入口，但结果类型、artifact、sweep 等能力的可发现性不一致，也没有 public/experimental/internal 的稳定性说明。

建议：

- 在文档中列出唯一推荐的 public API；
- 内部模块和私有适配器不承诺兼容；
- 对 public API 使用语义化版本和弃用周期；
- 任何返回结构变化在 changelog 说明迁移方式；
- 增加 API import smoke，确保 wheel 安装后公共入口真实可用。

## 8. 依赖、打包与兼容性

### 8.1 DEP-01：依赖 py3gpp 私有实现

ulsch_ldpc.py 直接导入 py3gpp.nrLDPCEncode 中以下划线开头的 _encode、_gen_submat、_lift_basegraph、_load_basegraph。私有符号不受兼容承诺保护，而 pyproject 允许任何 >=0.6,<0.7 版本；即使只升级 patch，也可能在不违反对方公共 API 的情况下破坏本项目。

建议按优先级选择：

1. 把项目实际需要的最小 LDPC 适配层/矩阵逻辑纳入受控代码，并用标准向量与 py3gpp 双重测试；
2. 若暂时不能内置，则精确锁定已验证版本，集中到单一 compatibility adapter，启动时检查版本并给出明确错误；
3. 与上游协作形成公共 API，但在落地前仍需本地保护。

无论采用哪种方案，都应新增支持矩阵：Python × NumPy/SciPy × py3gpp 的最低和最高已测试组合。

### 8.2 依赖范围与可选功能

- Matplotlib 是硬依赖，但核心仿真可以不画图；建议变为 plot 可选 extra，CLI 的绘图命令按需检查；
- 当前无 lock/constraints，研究结果难以长期复现；建议发布 tested-constraints.txt 或 conda environment，并在结果元数据记录版本；
- 构建依赖 setuptools>=58 的下界较老，需在隔离构建和最低支持环境实际验证，或提高到项目明确维护的现代基线；
- setup.py 只保留一行兼容入口，功能与 pyproject 重叠；若不再支持旧工具可移除，否则文档说明用途。

### 8.3 wheel 与安装后体验

wheel 构建成功，但只包含 Python 包。README 的首个运行示例依赖仓库中的 examples/run_from_config.py 和 configs/*.yaml；普通 wheel 用户安装后没有这些路径，也没有 console script。

建议提供 project.scripts 入口，例如：

- nr-phy-simu run CONFIG；
- nr-phy-simu validate CONFIG；
- nr-phy-simu sweep CONFIG；
- nr-phy-simu show-config CONFIG --resolved。

示例配置可以放入包资源，或由 CLI 提供 init/example 命令复制到用户目录。CI 应创建干净虚拟环境安装 wheel，并在非源码目录执行 import、--help、validate 和最小运行 smoke。

### 8.4 平台兼容性

CI 覆盖 Ubuntu/Windows 与 Python 3.10/3.12，这是良好起点。建议补充：

- 一个最低依赖组合和一个接近上界组合；
- macOS 可按资源设为定期或 release job；
- Windows setup 脚本除 NumPy/SciPy/Matplotlib/YAML 外，还应验证 py3gpp 和 import nr_phy_simu；
- 明确 BLAS/线程环境对性能基线的影响；
- 文档列出 Python 与依赖的“声明支持”和“CI 实测”两张表。

## 9. 性能与资源使用

### 9.1 PERF-01：多 TTI 结果保留导致 O(N) 大内存

MultiTtiSimulationRunner 把每次完整 SimulationResult 追加到列表并最终返回 tuple。单个结果包含 TX/RX 波形、资源网格、信道估计、LLR 等大数组。对默认 PUSCH 配置做去重后的数组净字节估计约 2,756,224 bytes/TTI：

| 主要对象 | 约占用 |
|---|---:|
| RX waveform | 983 KB |
| RX grid | 559 KB |
| TX waveform | 492 KB |
| TX grid | 280 KB |
| channel estimate | 172 KB |
| 其他数组 | 余量 |

按线性估算，10,000 TTI 约 27.6 GB，仅计算数组有效数据。实际还包括对象、容器、临时数组、FFT workspace 和可视化 artifact。

建议默认采用在线聚合：

- 指标累加器保存比特数、错误数、CRC 事件、参考/误差能量和耗时，不保存全结果；
- result_retention 默认为 none 或 last，支持 sampled(k)、failures、all；
- 提供 callback/sink，将每 TTI 轻量结果写入 JSONL/CSV/Parquet 或用户数据库；
- debug buffer 由 CapturePolicy 独立控制；
- 记录峰值 RSS、每 TTI 平均/分位耗时和输出字节数。

验收标准：10,000 TTI 默认运行的常驻内存不随 TTI 数线性增长；all 模式清楚标为诊断用途并预估内存。

### 9.2 算法与实现层面

- FFT、卷积、信道矩阵和 LDPC 是主要计算热点，应先 profiler 后优化；
- benchmark 当前主要打印时长，没有阈值、基线版本、预热和噪声控制，无法作为回归门禁；
- 可对固定 channel/path 参数缓存不随 TTI 变化的滤波核和索引计划，但必须确保 seed/slot 变化不被误缓存；
- 避免在阶段间不必要 deepcopy 大 ndarray；配置小对象的 deepcopy 不是首要性能瓶颈，但应通过不可变 plan 消除语义负担；
- 不建议在没有 profiler 证据时全面引入 GPU/JIT；先解决结果保留、重复分配与解析 plan 复用。

### 9.3 建议的性能基线

至少固定三档：最小 smoke、典型 1×1 PUSCH、较重多天线/衰落配置。每档记录运行环境、依赖、线程数、TTI/s、p50/p95、峰值 RSS、输出保留策略。CI 可只对明显退化设置宽松门槛，详细基线放在定期 job，避免共享 runner 抖动导致误报。

## 10. HARQ 与批处理语义

### 10.1 HARQ-01：当前能力不是完整 HARQ

HarqManager 管理相同 TB、RV 轮转、重传次数和 reset，但接收链每次只对当前传输的 LLR 进行速率恢复/译码，没有按 process 和 code block 维护软缓冲或将不同 RV 映射位置的 LLR 合并。

MultiTtiSimulationRunner 把每一个 CRC false 的 TTI 计为 packet_errors，并以 crc_checked_ttis 为分母。启用 HARQ 后，同一个 TB 的初传失败和后续重传失败会被分别计为多个 packet error；最终成功前的失败也不会从“包失败”中撤销。

影响：用户可能把报表理解为 HARQ 后 TB BLER、吞吐增益或丢包率，实际得到的是 transmission-attempt CRC failure rate。

短期方案：明确改名为 HarqRetransmissionScheduler 或文档标记 scheduler-only；指标使用 transmission_bler，增加 new_tb_count、retransmission_count、ack_count、drop_count，禁止称为 post-HARQ BLER。

完整方案：

- HarqProcessState 持有 TB 标识、NDI、RV、每 code block 的软缓冲和生命周期；
- 每次传输先按照当前 RV 做速率恢复到相同 Ncb 位置，再按约定做 LLR 累加/饱和；
- 使用累计软缓冲译码，ACK 后释放，达到最大重传后计 drop；
- 区分 first-transmission BLER、attempt BLER、post-HARQ TB BLER、平均重传次数、goodput 和 latency；
- 添加多 RV 人工 LLR、先失败后成功、最大重传失败、多个 process 交错、相同 seed 可复现测试。

## 11. 可观测性、日志与错误追踪

### 11.1 现状

项目已有合理的 library logging 基础：使用 NullHandler，批次进度可输出 INFO，重放噪声、零干扰功率、LDPC fallback 等会发 warning/debug，接收结果保留部分译码路径与迭代信息。这些设计对本地算法库是合适的。

缺口是信息分散且不成体系：没有统一 run ID、配置 fingerprint、有效 seed、版本、阶段耗时、内存、信道摘要、warning 列表和 fallback 计数；日志也没有稳定事件 schema。发生“结果异常但没有异常抛出”时，很难复盘到底使用了哪个派生参数和算法路径。

### 11.2 OBS-01：建议的观测模型

每次运行创建 RunMetadata：

- identity：run_id、scenario、config hash、代码 commit/version；
- reproducibility：原始 seed、各阶段派生 seed、slot/TTI、依赖版本；
- plan summary：载波、分配、调制编码、信道、天线、HARQ；
- timing：TX/channel/RX/metrics/report 各阶段 wall/CPU time；
- resource：可选峰值 RSS、主要 buffer 字节；
- diagnostics：warnings、fallbacks、decoder path、iterations、数值异常；
- outcome：CRC、BER、EVM、吞吐和失败分类。

日志支持人类可读文本与可选 JSONL event sink。库不应默认联网或绑定 Sentry/Prometheus；远程错误追踪和监控属于上层应用，应提供 callback/export hook。这样既保持本地仿真器简洁，也能被批量实验平台接入。

### 11.3 错误处理

- 配置错误在运行前一次性发现，避免进入数值链路后才出现 ndarray shape 错误；
- 输入解析错误携带文件、字段/行位置和格式，不吞掉原异常；
- capability 错误明确说出“此信道只接受频域 frame”等支持范围；
- 浮点 NaN/Inf 在关键 stage 边界按可配置策略检查；
- 失败结果可输出轻量 reproducibility bundle，但默认不包含完整用户输入数据和大数组。

## 12. 测试与 CI

### 12.1 优点

- 当前全量 108 项测试均通过；
- 有 PUSCH/PDSCH、配置、平台、信道、重放、报表和协议表边界的广泛 smoke；
- CI 覆盖 Python 3.10/3.12、Ubuntu/Windows；
- 有显式 slow baseline 标记，说明项目意识到协议级验证成本；
- 依赖安装使用项目 test extra，基本构建路径规范。

### 12.2 TEST-01：测试的主要盲区

衰落信道测试偏向自洽性：验证输出 shape、profile 元信息、系数连续性和端到端可运行，但缺少独立解析 oracle，因而没有发现 P0 延迟方向错误。配置测试也缺少大量 invalid case、边界组合和未知字段测试。

此外，测试集中在两个大文件，定位、并行和领域所有权不够清晰。建议按 config、tx、rx、channels、scenarios、io、cli、integration 划分；公共 fixture 放在 conftest，不要简单按文件行数切割。

### 12.3 CI-01：门禁缺口

- CI 排除 slow 测试，核心协议基线不会在普通 PR 运行；可将一项确定性较强的基线纳入 PR，其余放夜间；
- 使用 pytest-cov 但没有 cov-fail-under，CHANGELOG 所称“覆盖率门禁”与实际不一致；
- 无 lint、format、type check；建议从 ruff 和渐进式 pyright/mypy 开始，不要求一次清零所有历史类型债务；
- 无 sdist/wheel 构建与干净安装 smoke；
- 无最低/最高依赖组合；
- GitHub Actions 只固定到主版本 tag，若组织要求更高供应链保证，可固定完整 commit SHA 并用更新机器人维护。

建议门禁顺序：先加入 wheel smoke、关键解析测试、覆盖率阈值和严格配置测试；再逐步引入 lint/type。覆盖率目标应以风险模块为主，不以单一全局数字替代关键行为测试。

## 13. 文档完整性评估

### 13.1 文档矩阵

| 文档类别 | 当前状态 | 结论与缺口 |
|---|---|---|
| 安装说明 | README、Windows setup | 基本存在；缺 wheel 安装后首跑和支持矩阵 |
| 使用说明 | README、examples、配置样例 | 较完整；缺正式 CLI 和安装包资源路径 |
| 配置说明 | README、configs | 部分存在；没有完整字段/类型/默认值/枚举/跨字段约束和 schema |
| 架构说明 | system_design.md | 存在且较详细；偏实现快照，缺 ADR、稳定边界和迁移原则 |
| API 文档 | 无集中 API reference | 缺失；公共/实验/内部边界不清 |
| 数据结构说明 | types 与系统设计零散说明 | 部分存在；数组 shape、axis、dtype、channel_info schema 不完整 |
| 开发指南 | README/测试片段 | 不完整；缺 CONTRIBUTING、环境、风格、分支、提交、release、兼容政策 |
| 测试说明 | README/system_design | 部分存在；未覆盖 test_platform_review、slow/CI 差异和 oracle 策略 |
| 部署/发布说明 | 基本无 | 对纯库可不谈服务部署，但仍需构建、发布、制品验证和版本说明 |
| 变更记录 | CHANGELOG.md | 存在；有一处能力声明与 CI 不符 |
| 常见问题 | troubleshooting.md | 有故障排查，但不是完整 FAQ；可扩展安装/性能/结果解释 |
| 性能说明 | performance_guide.md | 存在；未覆盖多 TTI 结果保留的主要内存风险 |
| 标准一致性追踪 | 缺失 | 应建立 38.211/38.212/38.901 条款到代码、测试、限制的矩阵 |
| 安全/输入信任边界 | 缺失 | 本地工具可简短说明，不需重型安全手册 |

### 13.2 建议的文档信息架构

1. README：定位、5 分钟首跑、能力/限制、入口导航；
2. installation.md：源码、wheel、Windows/Linux/macOS、支持矩阵；
3. configuration-reference.md：由 schema 自动生成字段、默认值、约束、示例；
4. api-reference：从公共 docstring 生成，区分 stable/experimental/internal；
5. architecture.md + ADR：只讲稳定边界和关键选择；
6. data-model.md：frame、plan、result、array axis 与单位；
7. development.md：环境、测试、lint/type、benchmark、release；
8. conformance-matrix.md：标准条款、实现、测试、已知偏差；
9. troubleshooting/FAQ：错误消息、性能、结果解释、重现模板；
10. changelog：遵循一致格式并关联 breaking change/migration。

配置和 API 文档应尽量从代码 schema/docstring 生成或至少在 CI 做示例执行，避免手工双份事实源继续漂移。

## 14. 文档与代码一致性

### 14.1 DOC-01：已确认的不一致

| 文档位置/声明 | 实际代码 | 影响 | 建议 |
|---|---|---|---|
| README 使用 time_domain_tap_length: null 表示完整 IFFT | 实现对 None 调用 int，抛 TypeError | 文档合法配置失败 | 修实现并加文档示例测试 |
| README 自定义示例引用 PUSCHSimulation/PDSCHSimulation | 实际类名为 PuschSimulation/PdschSimulation | import/复制代码失败 | 用真实符号，并在 CI 执行示例 |
| custom_intermediate_plotting 引用 LeastSquaresChannelEstimator | 实际导出 LeastSquaresEstimator | 示例不可运行 | 修名并增加 docs snippet test |
| CHANGELOG 声称测试覆盖率门禁 | CI 无 cov-fail-under | 发布质量声明不准确 | 加阈值或改为“采集覆盖率” |
| system_design 主要只列 test_smoke | 仓库另有 test_platform_review | 测试说明不完整 | 更新测试结构和职责 |
| README 首跑依赖 examples/configs | wheel 不包含这些资源且无 CLI | 安装后路径不成立 | 提供正式入口和包资源 |

### 14.2 能力表述需要更精确

- “HARQ”应注明是否含软合并、process 并发和 TB 级统计；
- “统一信道接口”应在接口重构前改为“统一配置入口，时域/频域有不同传播路径”；
- “TDL/CDL 支持”应在 P0 修复和独立 oracle 通过前标记 experimental/known limitation；
- “PDSCH 支持”应列出当前与 PUSCH 共享/不同的真实功能，不仅以场景类存在为依据；
- “参考实现、非金参考”的定位是准确且重要的，应保留并在结果页/文档明显展示。

## 15. 使用者体验

### 15.1 优点

- 仓库内有多个配置样例和最小配置，学习入口较直观；
- YAML/JSON/XML 支持方便不同系统接入；
- 输出报表、绘图 artifact、重放和 sweep 覆盖了常见实验工作流；
- 多数异常消息比直接暴露底层 NumPy 错误更友好；
- README 对定位和已知限制总体诚实。

### 15.2 UX-01：主要痛点与优化方案

1. **安装后不可直接运行。** 增加正式 CLI 和内置示例，不要求用户定位源码 examples。
2. **配置错误发现太晚。** 提供 validate 与 show-config --resolved，一次展示所有错误、默认值和推导值。
3. **自动绘图/输出行为不够可控。** CLI 提供 --output、--no-show、--capture、--log-level、--seed 和 --format；库 API 默认不主动打开 GUI。
4. **报表机器可读性弱。** CSV 使用标准 writer，字段有 schema/version、稳定英文 key 和单位；显示层可本地化中文标题。复杂结构另提供 JSON。
5. **结果解释不足。** 输出明确指标分母、CRC 未检查原因、HARQ 语义、信道域和 fallback；不要只给一个 BLER 数字。
6. **公共入口不易发现。** 顶层导出稳定结果/计划类型，并提供 docs API 索引和小型 recipes。

## 16. 需求与目标一致性

依据 README 中的项目定位和主要能力，对当前完成度作如下判断：

| 目标 | 结论 | 说明 |
|---|---|---|
| TX→Channel→RX→Metrics 分层 | 基本满足 | 主流程清楚，但频域信道特殊分支破坏统一边界 |
| 支持算法替换与自定义 | 部分满足 | 扩展点丰富；完整 config 耦合和多套 RX 抽象提高接入成本 |
| PUSCH/PDSCH 场景 | 部分满足 | 有可运行场景骨架；需明确两者功能差异和未实现协议特性 |
| AWGN 信道 | 基本满足 | 当前端到端路径和测试表现稳定 |
| TDL/CDL 信道 | 阻断性部分满足 | 有 profile 和时变实现，但分数延迟 P0 使物理结论不可信 |
| 外部频响注入 | 部分满足 | 频域路径可用；接口不统一且 null 配置失败 |
| DMRS 与信道估计 | 较好满足 | 复用和覆盖较好，仍需标准追踪矩阵 |
| 配置驱动 YAML/JSON/XML | 形式满足、语义部分满足 | 多格式可加载；严格 schema 和跨字段校验不足 |
| 多 TTI/HARQ | 部分满足 | 多 TTI 可运行；内存不可扩展，HARQ 仅调度语义 |
| 重放、扫描、可视化 | 基本满足 | 功能存在；上下文、安装后入口和非 GUI 控制需改善 |
| 作为金参考/一致性工具 | 未满足，且项目未宣称完全满足 | 缺独立标准向量覆盖，TDL/CDL 有阻断缺陷 |

总体上，项目的“可扩展链路级参考框架”目标已经具备骨架和相当多能力；“可信的全场景物理仿真器”目标尚未完成。下一阶段应优先提高正确性证据、契约严格度和批量运行能力，而不是继续快速增加新场景名或新抽象层。

## 17. Git 跟踪、ignore 与仓库卫生

### 17.1 当前情况

- 当前基线未发现已跟踪的 build、dist、缓存、测试输出或 IDE 文件；
- inputs/pusch_capture.txt 约 588 KB，作为示例采集数据规模尚合理，但应说明来源、格式、许可证/可再分发性和裁剪方式；
- .gitignore 已覆盖 Python 缓存、虚拟环境、构建产物、coverage、IDE 和 outputs；
- egg-info 规则存在轻微重复，不影响功能；
- .git/info/exclude 中的规则只对当前克隆有效，不应承载团队必须共享的忽略约定或正式知识。

### 17.2 GIT-01：建议

- 按实际采用工具补充 .ruff_cache、.tox、.nox、.hypothesis、coverage.xml、junit.xml、*.log；
- 谨慎处理 .env：可忽略 .env 和 .env.*，但保留可跟踪的 .env.example；
- 增加 .gitattributes，统一文本 EOL，特别是 .ps1/.bat/.sh 和样例输入；
- 对未来大采集文件设定阈值和存储策略，必要时使用外部下载/校验和，而不是无界加入 Git；
- CI 可检查新增大文件、秘密模式和生成物误提交；
- 若本地排除的 py3gpp 调研文档会影响设计决策，应整理后正式跟踪；若只是个人草稿，保留本地排除是合理的。

## 18. 安全、鲁棒性与供应链

项目是本地科学计算库，当前没有网络服务、动态 eval、pickle 反序列化或 shell 拼接等高暴露面设计；YAML 使用 safe_load，外部进程调用采用参数列表，基础安全面较好。

SEC-01 改进建议：

- 对 XML/YAML/JSON/文本输入设置可配置大小、维度和样本数上限；当前全量读入可被超大文件耗尽内存；
- XML 解析采用明确的安全解析策略，并禁用不需要的外部实体能力；
- 依赖通过 constraints/lock 与定期扫描管理，发布制品记录 SBOM 或至少依赖清单；
- CI Action 若追求强供应链保证则固定 commit SHA；
- 输出目录处理应防止意外覆盖，批次任务使用唯一 run 目录；
- 日志和 reproducibility bundle 不默认泄露完整输入路径、采集数据或环境秘密。

这些属于工程加固，不是当前最主要风险；优先级应低于物理正确性、HARQ 语义和内存问题。

## 19. 建议的新架构

### 19.1 变动必要性

架构级调整的目标不是“更抽象”，而是解决五个已经出现的实际问题：

1. 配置错误被静默接受，原始输入与派生状态混合；
2. 频域信道无法满足统一接口，新模型需要改中心引擎；
3. RX 扩展点重叠且普遍依赖完整配置；
4. 全局 context 隐式保存 artifact，结果与复现信息不完整；
5. 多 TTI 和 HARQ 的状态、指标、内存模型不能扩展。

这些问题相互关联，单独打补丁会继续产生 model 分支、dict key 和复制逻辑。因此需要目标架构，但应通过兼容 adapter 渐进迁移，不能大爆炸重写。

### 19.2 目标结构

建议的数据流如下：

    Config file
        |
        v
    parse -> RawConfig -> validate -> resolve
                                      |
                                      v
                               ResolvedRunPlan
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
        TxPipeline              Channel.process           RxPipeline
        Stage Protocols         SignalFrame -> Output      Stage Protocols
             |                        |                        |
             +------------------------+------------------------+
                                      |
                                      v
                          MetricsAggregator + Sinks
                                      |
                                      v
                       RunResult + RunMetadata + Debug refs

关键对象：

- RawConfig：严格、可序列化、只含用户意图；
- ResolvedRunPlan：不可变，包含所有标准推导和资源索引；
- RunContext：显式、只读，包含 plan、分层 RNG、collector 和 logger adapter；
- SignalFrame：统一表达时域/频域信号及轴元数据；
- Stage Protocol：小接口，输入输出类型明确，不读取全局 config；
- ChannelOutput：信号、typed summary、可选 debug reference；
- MetricsAggregator：在线聚合，按 TB/attempt/TTI 区分口径；
- ResultSink：可选流式输出；
- CapturePolicy：控制大数组与 artifact 生命周期。

### 19.3 场景与协议差异

PUSCH/PDSCH 场景不应只是同一主链路的空壳名字，也不应复制完整引擎。建议 ScenarioPlanner 根据 RawConfig 生成不同的 Resource/ReferenceSignal/Transport plans；公共 Stage 复用，真正不同的 mapping、scrambling、DMRS/端口或译码逻辑由 plan 和专用 stage 表达。

这样能够清楚回答“PDSCH 当前支持到什么程度”，也便于建立 3GPP 条款追踪，而不是通过场景类是否存在推断能力。

### 19.4 信道边界

Channel.process 的输入和输出必须明确 domain。若一个信道只能接受 resource grid，它在 capability 中声明 frequency；引擎在 resolve 阶段验证整个 pipeline 是否可组合。时域/频域转换由标准 adapter stage 完成，信道内部不偷偷改变中心流程。

ChannelSummary 只保留稳定、轻量、可观测的信息，例如实际噪声功率、路径延迟/平均功率、Doppler、增益 shape 和 seed；大矩阵或逐样点系数放入按需 debug storage。

### 19.5 HARQ 与批次执行

BatchRunner 负责调度，不负责保存所有结果；HarqCoordinator 管理 process lifecycle 和 SoftBuffer；MetricsAggregator 基于事件更新：

    NewTB -> TxAttempt(rv) -> DecodeAttempt
                           |-> ACK -> CompleteTB(success)
                           |-> NACK -> Retx or CompleteTB(drop)

事件可同时生成 attempt BLER 和 post-HARQ TB BLER，不再混用分母。ResultSink 可在每个事件后落盘，内存只保留活跃 process 和在线统计。

### 19.6 过渡与兼容

- 为旧 ChannelModel、ReceiverProcessor 和结果 dataclass 提供 adapter；
- 新旧接口并存一个明确版本周期，运行时发 DeprecationWarning；
- 每次迁移一个垂直切片，并保持现有 AWGN baseline；
- 不在修 P0 的同一个提交中同时重构整个信道架构，以便验证行为变化来源；
- 迁移文档给出旧 API 到新 API 的一对一示例。

## 20. 分阶段修复路线图

### 阶段 0：正确性止血（立即）

- 修复 COR-01 分数延迟，建立独立解析测试；
- 修复 CFG-02 null 行为；
- 文档将 TDL/CDL 标记为待重新验证，将 HARQ 标记为 scheduler-only；
- 修复失效类名和 CHANGELOG/CI 表述；
- 将关键解析衰落测试放入 PR CI。

退出标准：整数/分数延迟和已知多径 oracle 全通过；所有文档示例可执行；不存在已知会反转物理语义的核心缺陷。

### 阶段 1：契约与配置（近期）

- 建立 strict RawConfig、跨字段 validator、ConfigError 和 resolved config 展示；
- 引入不可变 ResolvedRunPlan，停止修改调用者 config；
- 定义 public API 和 typed ChannelOutput/RunMetadata；
- 增加 wheel 安装 smoke 与正式 CLI；
- 提供配置 reference/schema。

退出标准：已知非法配置在运行前失败；未知字段不再静默；安装 wheel 后可在任意目录完成 validate 和最小仿真。

### 阶段 2：批处理、指标和 HARQ（近期至中期）

- 在线 MetricsAggregator、CapturePolicy、ResultSink；
- 修正 EVM 和 BLER 定义；
- 短期明确 scheduler-only，或实现完整软合并；
- 增加 10k TTI 内存稳定性测试和性能基线。

退出标准：默认内存近似 O(活跃状态)；attempt 与 TB 级指标可验证；HARQ 文档和实现完全一致。

### 阶段 3：接口收敛（中期）

- SignalFrame + Channel.process；
- 统一 RX Stage pipeline，提供旧接口 adapter；
- 显式 RunContext，移除对全局 active context 的主路径依赖；
- 按稳定职责拆分热点文件。

退出标准：新增一个时域或频域信道无需修改 SharedChannelSimulation；自定义 RX stage 只实现一个小协议；嵌套/并发运行上下文隔离。

### 阶段 4：质量与发布治理（持续）

- py3gpp 兼容层或内置受控 LDPC 实现；
- 依赖支持矩阵、constraints、定期上界测试；
- lint/type、覆盖率门槛、构建制品、Action 固定策略；
- API reference、ADR、标准一致性矩阵和 release guide；
- 扩展独立标准向量和跨实现比对。

## 21. 建议的验收清单

### 正确性

- [ ] 正延迟使脉冲向后移动，分数延迟相位符合解析公式；
- [ ] TDL/CDL 至少有一组独立 oracle 或交叉实现基线；
- [ ] 数组 shape 不匹配不再被 EVM 静默截断；
- [ ] HARQ 的软合并/非软合并语义与指标命名一致。

### 配置与接口

- [ ] 所有公开配置字段有类型、默认值、枚举、单位和约束；
- [ ] 未知字段默认报错，扩展字段有专用命名空间；
- [ ] SCS/FFT/sample rate/资源/天线组合在运行前验证；
- [ ] 时域和频域信道满足显式可替换契约；
- [ ] 结果包含可复现 metadata 和 typed summary。

### 性能与可观测性

- [ ] 默认多 TTI 常驻内存不随结果数线性增长；
- [ ] 可配置结果保留和流式输出；
- [ ] 输出 run ID、config hash、seed、版本、阶段耗时和 warning；
- [ ] 性能基线包含吞吐和峰值 RSS。

### 测试与交付

- [ ] wheel 干净安装 smoke 通过；
- [ ] PR CI 运行关键解析基线并有真实覆盖率阈值；
- [ ] 最低/最高依赖组合受测；
- [ ] 文档代码片段在 CI 执行或从测试生成；
- [ ] 发布说明、迁移指南和标准追踪矩阵可用。

## 22. 证据索引与参考资料

### 22.1 关键问题的代码证据

| 问题 | 主要证据位置 | 复核要点 |
|---|---|---|
| COR-01 | [fading_base.py](../src/nr_phy_simu/channels/fading_base.py) 的 _apply_time_varying_channel、_fractional_delay | 卷积结果截取起点叠加正 integer_delay，单位脉冲峰值反向移动 |
| CFG-01 | [config.py](../src/nr_phy_simu/config.py) 的 CarrierConfig、validate、_build_config_dataclass | numerology 使用 round 推导；跨字段与未知字段约束不完整 |
| CFG-02 | [external_frequency_response.py](../src/nr_phy_simu/channels/external_frequency_response.py) 的 tap_length 解析；[README](../README.md) 外部频响配置段 | 文档允许 null，实现直接 int(None) |
| API-01 | [interfaces.py](../src/nr_phy_simu/common/interfaces.py) 的 ChannelModel；[external_frequency_response.py](../src/nr_phy_simu/channels/external_frequency_response.py) 的 propagate_grid；[base.py](../src/nr_phy_simu/scenarios/base.py) 的 run | 公共接口只有 waveform，频域实现和引擎另走特殊分支 |
| HARQ-01 | [harq.py](../src/nr_phy_simu/common/harq.py)、[decoding.py](../src/nr_phy_simu/rx/decoding.py)、[multi_tti.py](../src/nr_phy_simu/scenarios/multi_tti.py) | process 状态无软缓冲；每次 CRC false 都累加 packet_errors |
| PERF-01 | [multi_tti.py](../src/nr_phy_simu/scenarios/multi_tti.py) 的 tti_results；[types.py](../src/nr_phy_simu/common/types.py) 的 SimulationResult | 每 TTI 保存包含大数组的完整结果 |
| MET-01 | [base.py](../src/nr_phy_simu/scenarios/base.py) 的 _compute_evm_metrics；[multi_tti.py](../src/nr_phy_simu/scenarios/multi_tti.py) 的聚合 | EVM 取最短长度；批次对每 TTI 指标做算术平均 |
| DEP-01 | [ulsch_ldpc.py](../src/nr_phy_simu/common/ulsch_ldpc.py) 顶部导入；[pyproject.toml](../pyproject.toml) 的依赖范围 | 导入 py3gpp 下划线私有符号，但允许整个 0.6 小版本范围 |
| CTX-01 | [runtime_context.py](../src/nr_phy_simu/common/runtime_context.py)、[base.py](../src/nr_phy_simu/scenarios/base.py)、[waveform_replay.py](../src/nr_phy_simu/scenarios/waveform_replay.py) | 设置 active context 后主路径不按 token 恢复，绘图隐式读取 |
| DOC-01 | [README](../README.md)、[custom_intermediate_plotting.md](custom_intermediate_plotting.md)、[CHANGELOG](../CHANGELOG.md)、[CI](../.github/workflows/ci.yml) | 类名、null 行为、覆盖率门禁和安装后示例存在不一致 |
| UX-01 | [pyproject.toml](../pyproject.toml)、[run_from_config.py](../examples/run_from_config.py) | 无 project.scripts，示例依赖源码树相对资源 |

### 22.2 复核时使用的针对性诊断

- 对 _fractional_delay 输入单天线单位脉冲并比较 delay=0、2、5 的峰值位置；
- 分别加载省略 time_domain_tap_length 与显式 null 的外部频响配置；
- 构造 SCS=20 kHz、start_symbol=-1、52 RB/FFT=128、TX 天线数 0 和拼错 num_tti 的配置；
- 对一次默认 PUSCH SimulationResult 按 ndarray 对象身份去重后累加 nbytes；
- 构建 wheel 并检查 RECORD/文件清单，确认没有 examples、configs、scripts、docs 和 console entry point；
- 分开运行普通全量测试与 slow 标记协议基线，避免默认标记选择掩盖结果。

### 22.3 外部工程参考

打包与 CLI 建议依据 Python Packaging User Guide 的现代 pyproject 和命令行工具指导：

- [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Creating and packaging command-line tools](https://packaging.python.org/en/latest/guides/creating-command-line-tools/)
- [Declaring project metadata](https://packaging.python.org/specifications/declaring-project-metadata/)

后续协议一致性工作还应在项目内建立 3GPP TS 38.211、38.212、38.901 的具体版本与条款追踪；本报告没有用非独立的生产实现自证标准一致性。

## 23. 最终评价

NR_L1_Simu 的价值不在于抽象数量，而在于它已经把配置、发射、信道、接收、指标、重放和可视化串成了一条可实验的链路。现阶段最重要的工作不是继续扩充表面功能，而是把“可运行”提升为“契约清楚、结果可解释、正确性有独立证据、批量运行可扩展”。

建议以 P0 分数延迟为第一优先级，紧接着收敛 HARQ 指标、配置校验和多 TTI 内存。架构调整采用 RawConfig → ResolvedRunPlan、SignalFrame/ChannelOutput、单一 Stage pipeline、显式 RunContext 和在线 MetricsAggregator 这五个支点，能够直接解决已经观察到的问题，同时避免不必要的大规模重写。

在这些问题修复并建立独立标准验证之前，最准确的项目定位是：**可扩展的 5G NR 链路级研发/教学参考框架，AWGN 主路径较成熟，衰落与 HARQ 能力仍处于需要严格验证和语义收敛的实验阶段。**
