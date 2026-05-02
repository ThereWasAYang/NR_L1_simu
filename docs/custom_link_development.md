# 新链路搭建开发指南

这份文档面向希望在本工程中接入新算法的开发者。这里的“新链路”不是指一定要重写完整的发射机、信道和接收机，而是指：

- 你保留大部分默认模块，只替换其中一个或几个处理块。
- 这些处理块共同组成一条新的 `TX -> Channel -> RX` 仿真路径。
- 例如：你新写了一个从 `ChannelEstimator` 派生的信道估计类，并希望接收机用它完成解调，这条“默认发射机 + 默认信道 + 新信道估计器接收机”的组合就是一条新链路。

当前推荐做法是：**新增算法类，再通过自定义 `SimulationComponentFactory` 注入到场景对象中**。这样不需要修改 `Receiver` 主流程，也不需要在 `run_from_config.py` 里为每个实验算法增加特殊分支。

## 1. 当前链路如何组装

默认组件装配位于：

- [component_factory.py](../src/nr_phy_simu/scenarios/component_factory.py)

场景对象位于：

- [pusch.py](../src/nr_phy_simu/scenarios/pusch.py)
- [pdsch.py](../src/nr_phy_simu/scenarios/pdsch.py)
- [base.py](../src/nr_phy_simu/scenarios/base.py)

默认流程可以简化理解为：

```text
PuschSimulation / PdschSimulation
  -> component_factory.create_components(config)
  -> build_transmitter(components)
  -> build_receiver(components)
  -> component_factory.create_channel_factory().create(config)
  -> simulation.run()
```

其中 `DefaultSimulationComponentFactory` 会创建：

- `TransmitterComponents`：编码、扰码、调制、资源映射、时域处理
- `ReceiverComponents`：时域处理、频域抽取、信道估计、均衡、解调、解扰、译码，以及可选的 `data_processor / receiver_processor`
- `SharedComponents`：发射机和接收机共享的公共对象，例如 `DMRS` 生成器

这意味着你要替换某个模块时，优先改“组件工厂返回什么”，而不是改 `Transmitter` 或 `Receiver` 的内部流程。

## 2. 可以替换哪些模块

核心抽象接口位于：

- [interfaces.py](../src/nr_phy_simu/common/interfaces.py)

当前主要可替换模块包括：

- `ChannelCoder`：信道编码，例如 `CRC/LDPC/rate matching`
- `BitScrambler`：数据扰码与 LLR 解扰
- `Modulator`：调制映射
- `ResourceMapper`：频域资源映射，包含数据和导频位置
- `TimeDomainProcessor`：OFDM / DFT-s-OFDM 时域处理
- `ChannelModel`：信道模型
- `FrequencyExtractor`：频域 RE 抽取
- `ChannelEstimator`：信道估计
- `MimoEqualizer`：MIMO 均衡
- `Demodulator`：软解调
- `ChannelDecoder`：译码
- `DmrsSequenceGenerator`：DMRS 信息与序列生成

如果你的新算法能归入上述接口，建议直接派生对应接口或继承现有默认实现。如果你的算法会改变上下游数据形状，则还需要同步替换相邻模块。

## 3. 最小示例：替换信道估计器

假设你设计了一个新的信道估计算法，只想替换接收机中的 `LeastSquaresEstimator`。

可以新增文件：

```text
src/nr_phy_simu/rx/my_channel_estimation.py
```

示例代码：

```python
from __future__ import annotations

import numpy as np

from nr_phy_simu.common.runtime_context import get_runtime_context
from nr_phy_simu.common.types import PlotArtifact
from nr_phy_simu.rx.channel_estimation import LeastSquaresEstimator


class MyChannelEstimator(LeastSquaresEstimator):
    """Example estimator that customizes frequency interpolation."""

    def _interpolate_frequency(
        self,
        pilot_subcarriers: np.ndarray,
        pilot_values: np.ndarray,
        num_subcarriers: int,
    ) -> np.ndarray:
        """Interpolate pilot estimates and record an intermediate metric."""
        interpolated = super()._interpolate_frequency(
            pilot_subcarriers=pilot_subcarriers,
            pilot_values=pilot_values,
            num_subcarriers=num_subcarriers,
        )

        metric = np.abs(interpolated)
        context = get_runtime_context()
        context.set("my_channel_estimator", "frequency_metric", metric)
        context.add_plot_artifact(
            PlotArtifact(
                name="my_frequency_metric",
                values=metric,
                title="My Channel Estimator Frequency Metric",
                plot_type="magnitude",
                xlabel="Subcarrier Index",
                ylabel="Magnitude",
            )
        )
        return interpolated
```

这个例子继承了 `LeastSquaresEstimator`，只改频域插值函数。这样可以复用默认的：

- 多接收天线循环
- pilot RE 抽取
- LS 估计
- 时域插值
- `ChannelEstimateResult` 返回结构

如果你的算法完全不同，也可以直接继承 `ChannelEstimator` 并完整实现 `estimate(...)`。

## 4. 用自定义工厂注入新组件

新增一个工厂类，例如：

```text
src/nr_phy_simu/scenarios/my_link_factory.py
```

示例代码：

```python
from __future__ import annotations

from dataclasses import replace

from nr_phy_simu.config import SimulationConfig
from nr_phy_simu.rx.my_channel_estimation import MyChannelEstimator
from nr_phy_simu.scenarios.component_factory import (
    DefaultSimulationComponentFactory,
    SimulationComponents,
)


class MyLinkComponentFactory(DefaultSimulationComponentFactory):
    """Build the default link, replacing only the channel estimator."""

    def create_components(self, config: SimulationConfig) -> SimulationComponents:
        components = super().create_components(config)
        return replace(
            components,
            receiver=replace(
                components.receiver,
                estimator=MyChannelEstimator(),
            ),
        )
```

这里使用 `dataclasses.replace(...)` 的原因是：`SimulationComponents`、`ReceiverComponents` 等结构是 frozen dataclass。用 `replace(...)` 可以清晰表达“保留默认链路，只替换某一个字段”。

如果你要同时替换均衡器和解调器，可以写成：

```python
return replace(
    components,
    receiver=replace(
        components.receiver,
        estimator=MyChannelEstimator(),
        equalizer=MyEqualizer(),
        demodulator=MyDemodulator(),
    ),
)
```

### 一个模块直接替换信道估计、均衡、解调

有些算法并不能自然拆成 `ChannelEstimator -> MimoEqualizer -> Demodulator` 三步。例如，一个神经网络接收机可能直接输入：

- 接收频域栅格 `rx_grid`
- 本地 DMRS 序列 `dmrs_symbols`
- `dmrs_mask / data_mask`
- 噪声方差、MCS、RB 位置、天线数等配置

然后直接输出解调 LLR。对于这种情况，不要强行把算法拆成三个接口。当前工程提供了更高层的 `ReceiverDataProcessor` 接口，专门用于一次性替换“信道估计 + 均衡 + 解调”这一段。

接口定义位于：

- [interfaces.py](../src/nr_phy_simu/common/interfaces.py)

返回结构位于：

- [types.py](../src/nr_phy_simu/common/types.py)

一个简化的神经网络接收机骨架如下：

```python
from __future__ import annotations

import numpy as np

from nr_phy_simu.common.interfaces import ReceiverDataProcessor
from nr_phy_simu.common.types import ReceiverDataProcessingResult
from nr_phy_simu.config import SimulationConfig


class MyNeuralReceiver(ReceiverDataProcessor):
    """Example processor that directly maps received grid to scrambled-domain LLRs."""

    def __init__(self, model) -> None:
        self.model = model

    def process(
        self,
        rx_grid: np.ndarray,
        dmrs_symbols: np.ndarray,
        dmrs_mask: np.ndarray,
        data_mask: np.ndarray,
        noise_variance: float,
        config: SimulationConfig,
    ) -> ReceiverDataProcessingResult:
        # rx_grid shape: (num_rx_ant, num_subcarriers, num_symbols)
        # dmrs_symbols shape: (num_dmrs_re,)
        # dmrs_mask/data_mask shape: (num_subcarriers, num_symbols)
        features = self._build_features(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )
        llrs = self.model(features)
        return ReceiverDataProcessingResult(
            llrs=np.asarray(llrs, dtype=np.float64).reshape(-1),
        )

    def _build_features(self, **kwargs):
        ...
```

这里返回的 `llrs` 仍然应处于“扰码域”，也就是和普通解调器输出一致：接收机会继续执行：

```text
data descrambling -> channel decoding
```

如果你的算法已经输出了解扰后的 LLR，建议同时替换 `BitScrambler`，让 `descramble_llrs(...)` 直接透传，避免重复解扰。

如果你的算法能够额外输出信道估计、均衡星座点或调试图，也可以填充更多字段：

```python
return ReceiverDataProcessingResult(
    llrs=llrs,
    channel_estimation=my_channel_estimation,   # 可选
    equalized_symbols=my_equalized_symbols,     # 可选，用于星座图和 EVM
    plot_artifacts=my_plot_artifacts,           # 可选
)
```

如果不提供 `channel_estimation`，系统不会绘制导频信道估计图；如果不提供 `equalized_symbols`，系统不会绘制星座图，也不会计算 EVM。这对“黑盒神经网络直接输出 LLR”的算法是合理的。

在组件工厂中注入这个模块：

```python
from dataclasses import replace

from nr_phy_simu.scenarios.component_factory import DefaultSimulationComponentFactory


class MyNeuralReceiverFactory(DefaultSimulationComponentFactory):
    def __init__(self, model) -> None:
        self.receiver_processor = MyNeuralReceiver(model)

    def create_components(self, config):
        components = super().create_components(config)
        return replace(
            components,
            receiver=replace(
                components.receiver,
                data_processor=self.receiver_processor,
            ),
        )
```

这里不会报错，因为当前 `ReceiverComponents` 已经显式定义了可选字段：

```python
data_processor: ReceiverDataProcessor | None = None
```

当 `data_processor` 为 `None` 时，`Receiver` 会自动创建默认的 `ThreeStageReceiverDataProcessor`，也就是继续按传统方式执行：

```text
ChannelEstimator -> MimoEqualizer -> Demodulator
```

当你通过 `replace(..., data_processor=self.receiver_processor)` 传入自定义对象后，接收机主流程会优先调用这个高层处理器，直接拿到解调 LLR。此时原来的 `estimator / equalizer / demodulator` 字段仍然存在，但不会被高层处理器路径调用；它们只是保留为默认链路和兼容旧扩展方式使用。

### 更灵活：任意组合多个接收处理 stage

如果你的算法不是一个完整黑盒，而是希望自由组合多个步骤，例如：

```text
自定义归一化 -> 特征构造 -> 神经网络推理 -> LLR 后处理
```

或者：

```text
传统 LS 信道估计 -> 自定义频域滤波 -> 神经网络均衡 -> 传统 QAM 解调
```

可以使用 `ReceiverDataProcessorPipeline`。它会创建一个 `ReceiverProcessingContext`，然后按顺序调用多个 `ReceiverProcessingStage`。每个 stage 都可以读写 context 中的变量。

相关实现位于：

- [data_processing.py](../src/nr_phy_simu/rx/data_processing.py)
- [interfaces.py](../src/nr_phy_simu/common/interfaces.py)
- [types.py](../src/nr_phy_simu/common/types.py)

一个最小示例：

```python
import numpy as np

from nr_phy_simu.common.interfaces import ReceiverProcessingStage
from nr_phy_simu.rx.data_processing import ReceiverDataProcessorPipeline


class FeatureStage(ReceiverProcessingStage):
    def process(self, context):
        # context.rx_grid shape: (num_rx_ant, num_subcarriers, num_symbols)
        context.metadata["features"] = np.stack(
            [context.rx_grid.real, context.rx_grid.imag],
            axis=0,
        )
        return context


class NeuralLlrStage(ReceiverProcessingStage):
    def __init__(self, model) -> None:
        self.model = model

    def process(self, context):
        features = context.metadata["features"]
        context.llrs = np.asarray(self.model(features), dtype=np.float64).reshape(-1)
        return context


processor = ReceiverDataProcessorPipeline(
    [
        FeatureStage(),
        NeuralLlrStage(model),
    ]
)
```

然后在组件工厂里注入：

```python
class MyPipelineReceiverFactory(DefaultSimulationComponentFactory):
    def __init__(self, processor) -> None:
        self.processor = processor

    def create_components(self, config):
        components = super().create_components(config)
        return replace(
            components,
            receiver=replace(
                components.receiver,
                data_processor=self.processor,
            ),
        )
```

`ReceiverProcessingContext` 中常用字段包括：

- `rx_grid`：接收频域栅格
- `dmrs_symbols`：本地 DMRS 参考序列
- `dmrs_mask / data_mask`：导频和数据 RE 位置
- `noise_variance`：噪声方差
- `config`：完整仿真配置
- `channel_estimation`：可选信道估计结果
- `rx_data_symbols`：可选抽取后的数据 RE
- `data_channel`：可选数据 RE 上的信道估计
- `equalized_symbols`：可选均衡星座点
- `llrs`：必须在最后由某个 stage 填充，且应为解扰前 LLR
- `metadata`：用户自定义中间变量字典

如果最后没有任何 stage 生成 `context.llrs`，pipeline 会报错。这样可以尽早暴露“链路组装不完整”的问题。

工程还提供了一些可复用的 stage：

- `ChannelEstimationStage`
- `DataExtractionStage`
- `EqualizationStage`
- `TransformPrecodingDespreadStage`
- `LayerDemappingStage`
- `DemodulationStage`

因此用户可以混合传统模块和新算法。例如：

```python
from nr_phy_simu.rx.data_processing import (
    ChannelEstimationStage,
    DataExtractionStage,
    DemodulationStage,
    ReceiverDataProcessorPipeline,
    TransformPrecodingDespreadStage,
)

processor = ReceiverDataProcessorPipeline(
    [
        ChannelEstimationStage(my_estimator),
        DataExtractionStage(default_extractor),
        MyNeuralEqualizationStage(model),
        TransformPrecodingDespreadStage(),
        DemodulationStage(default_demodulator),
    ]
)
```

这个模式适合“任意组合”的算法开发；如果你的模块本来就是完整黑盒，直接实现 `ReceiverDataProcessor` 会更简单。

### 更高一层：替换接收机中的任意几个步骤

`ReceiverDataProcessor` 和 `ReceiverDataProcessorPipeline` 的入口是 `rx_grid`，所以它们适合替换“时域处理之后、解扰译码之前”的接收机中段。如果你希望替换的范围更大，例如：

- 自定义时域处理 + 默认后续处理
- 默认时域处理 + 自定义解扰 + 默认译码
- 自定义频域抽取 + 神经网络接收机 + 自定义译码
- 完全绕过默认 `Receiver` 内部所有步骤

则可以使用更高层的 `ReceiverProcessor`。它直接接管 `Receiver.receive(...)` 和 `Receiver.receive_from_grid(...)`，因此可以自由决定复用哪些已有模块、替换哪些模块。

一个简化示例：

```python
import numpy as np

from nr_phy_simu.common.interfaces import ReceiverProcessor
from nr_phy_simu.common.types import ChannelEstimateResult, RxPayload


class MyReceiverProcessor(ReceiverProcessor):
    def receive(
        self,
        receiver,
        rx_waveform,
        dmrs_symbols,
        dmrs_mask,
        data_mask,
        noise_variance,
        config,
    ):
        # 这里可以替换时域处理，也可以复用默认 OFDM 解调。
        rx_grid = receiver.time_processor.demodulate(rx_waveform, config)
        return self.receive_from_grid(
            receiver,
            rx_grid,
            dmrs_symbols,
            dmrs_mask,
            data_mask,
            noise_variance,
            config,
            rx_waveform,
        )

    def receive_from_grid(
        self,
        receiver,
        rx_grid,
        dmrs_symbols,
        dmrs_mask,
        data_mask,
        noise_variance,
        config,
        rx_waveform=None,
    ):
        # 这里可以任意组合：自定义抽取、神经网络、解扰、译码等。
        llrs = self._my_receiver_algorithm(
            rx_grid=rx_grid,
            dmrs_symbols=dmrs_symbols,
            dmrs_mask=dmrs_mask,
            data_mask=data_mask,
            noise_variance=noise_variance,
            config=config,
        )

        # 如果输出仍是扰码域 LLR，可以复用默认解扰和译码。
        descrambled_llrs = receiver.scrambler.descramble_llrs(llrs, config)
        decoded_bits = receiver.decoder.decode(descrambled_llrs, config)

        if rx_grid.ndim == 2:
            rx_grid = rx_grid[np.newaxis, ...]
        return RxPayload(
            rx_waveform=np.asarray([], dtype=np.complex128) if rx_waveform is None else rx_waveform,
            rx_grid=rx_grid,
            channel_estimation=ChannelEstimateResult(
                channel_estimate=np.array([], dtype=np.complex128),
                pilot_estimates=np.array([], dtype=np.complex128),
                pilot_symbol_indices=np.array([], dtype=int),
            ),
            equalized_symbols=np.array([], dtype=np.complex128),
            llrs=descrambled_llrs,
            decoded_bits=decoded_bits,
            crc_ok=getattr(receiver.decoder, "last_crc_ok", None),
            dmrs_symbols=dmrs_symbols,
        )
```

然后通过工厂注入：

```python
class MyReceiverFactory(DefaultSimulationComponentFactory):
    def __init__(self) -> None:
        self.receiver_processor = MyReceiverProcessor()

    def create_components(self, config):
        components = super().create_components(config)
        return replace(
            components,
            receiver=replace(
                components.receiver,
                receiver_processor=self.receiver_processor,
            ),
        )
```

当前 `ReceiverComponents` 中有两个不同层级的可选组合入口：

```python
data_processor: ReceiverDataProcessor | None = None
receiver_processor: ReceiverProcessor | None = None
```

推荐选择方式：

- 如果只想替换 `rx_grid -> LLR` 这段，用 `data_processor`。
- 如果想替换时域处理、频域处理、解扰、译码等任意更大范围，用 `receiver_processor`。
- 如果只是替换单个传统模块，继续替换 `estimator / equalizer / demodulator / decoder` 等字段即可。

## 5. 写一个专用运行脚本

推荐为新链路写一个小脚本，而不是直接修改 [run_from_config.py](../examples/run_from_config.py)。例如：

```text
examples/run_pusch_my_link.py
```

示例代码：

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nr_phy_simu.common.runtime_context import SimulationRuntimeContext
from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.my_link_factory import MyLinkComponentFactory
from nr_phy_simu.scenarios.pusch import PuschSimulation
from nr_phy_simu.visualization import save_simulation_plots


def main(config_relpath: str = "configs/pusch_awgn.yaml") -> None:
    config_path = ROOT / config_relpath
    config = load_simulation_config(config_path)
    runtime_context = SimulationRuntimeContext()
    factory = MyLinkComponentFactory()
    simulation = PuschSimulation(
        config=config,
        component_factory=factory,
        runtime_context=runtime_context,
    )
    result = simulation.run()

    if config.plotting.enabled:
        save_simulation_plots(
            result=result,
            config=config,
            output_dir=ROOT / "outputs",
            prefix=config_path.stem,
            show=True,
            block=False,
        )

    print(f"SNR: {result.snr_db:.2f} dB")
    print(f"CRC OK: {result.crc_ok}")
    if result.evm_percent is not None:
        print(f"EVM: {result.evm_percent:.6f} %")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/pusch_awgn.yaml"
    main(arg)
```

这种写法有两个好处：

- 从脚本入口就能看出使用了哪个 `component_factory`，读代码时不容易误解。
- 默认示例脚本仍保持稳定，不会因为某个实验算法变得越来越复杂。

## 6. 替换信道模型

如果新链路的核心变化是信道模型，可以实现新的 `ChannelModel`，然后替换工厂的 `create_channel_factory()`。

简化示例：

```python
from nr_phy_simu.channels.channel_factory import ChannelFactory
from nr_phy_simu.common.interfaces import ChannelModel
from nr_phy_simu.config import SimulationConfig


class MyChannel(ChannelModel):
    def propagate(self, waveform, config: SimulationConfig):
        rx_waveform = ...
        channel_info = {"noise_variance": ..., "snr_db": config.snr_db}
        return rx_waveform, channel_info


class MyChannelFactory(ChannelFactory):
    def create(self, config: SimulationConfig) -> ChannelModel:
        return MyChannel()
```

然后在组件工厂里：

```python
class MyLinkComponentFactory(DefaultSimulationComponentFactory):
    def create_channel_factory(self) -> ChannelFactory:
        return MyChannelFactory()
```

注意：`channel_info` 至少应包含接收机后续会使用的 `noise_variance`。如果你希望打印或报告实际信噪比，也建议写入 `snr_db`。

## 7. 关键数据维度约定

新链路最容易出错的地方通常不是类继承，而是数据维度。当前接收机内部约定如下：

- `rx_grid`：`num_rx_ant x num_subcarrier x num_symbol`
- `channel_estimate`：`num_rx_ant x num_subcarrier x num_symbol`
- `pilot_estimates`：`num_rx_ant x num_dmrs_re`
- `dmrs_mask`：`num_subcarrier x num_symbol`
- `data_mask`：`num_subcarrier x num_symbol`
- `dmrs_symbols`：按资源映射顺序序列化的一维复数数组
- `equalized_symbols`：按数据 RE 抽取顺序排列的一维或多层符号数组

即使只有一根接收天线，`rx_grid` 和 `channel_estimate` 也应保留天线维，长度为 `1`。如果某个新模块临时产生二维数据，建议在模块边界处恢复成统一三维格式。

## 8. 共享对象不要随意拆开

有些对象需要在发射机和接收机之间保持一致，除非你明确知道自己在做协议变更：

- `DmrsGenerator`：资源映射与接收机参考 DMRS 应使用同一套规则。
- `NrDataScrambler`：发端扰码与收端 LLR 解扰应使用同一套初始化规则。
- `OfdmProcessor`：发端调制和收端解调应使用一致的 numerology 推导。

因此，替换这些共享规则时，建议通过 `SharedComponents` 或同一个实例同时接入 TX/RX，而不是只替换一侧。

## 9. 新算法里的绘图和中间变量

如果新链路中的算法需要保存中间变量并绘图，优先使用：

- `SimulationRuntimeContext`
- `PlotArtifact`

详细说明见：

- [custom_intermediate_plotting.md](custom_intermediate_plotting.md)

核心原则是：

- 算法函数负责把已经计算出的变量保存下来。
- 绘图函数只消费保存的变量，不重新运行算法。
- 不属于配置输入的临时量不要塞进 `config`。
- 不属于稳定仿真输出的调试量不要急着塞进 `SimulationResult`。

## 10. 给新链路增加测试

建议至少增加一个 smoke test，验证新链路能完整跑通。

示例：

```python
from pathlib import Path

from nr_phy_simu.io.config_loader import load_simulation_config
from nr_phy_simu.scenarios.my_link_factory import MyLinkComponentFactory
from nr_phy_simu.scenarios.pusch import PuschSimulation


def test_my_link_runs():
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "pusch_awgn.yaml")
    config.plotting.enabled = False

    result = PuschSimulation(
        config=config,
        component_factory=MyLinkComponentFactory(),
    ).run()

    assert result.bit_error_rate >= 0.0
    assert result.rx.channel_estimation.channel_estimate.ndim == 3
```

如果你的目标是在标准底线场景上保持性能，还应运行 baseline 用例：

```bash
conda run -n NRpy312 python -m unittest -v tests.test_smoke.BaselineRegressionTest
```

当前约定是：任何上库前都应完成这些底线用例，避免新链路破坏已有 PUSCH AWGN 基线。

## 11. 推荐开发顺序

建议按下面顺序开发：

1. 先继承现有默认实现，只替换最小函数，例如只替换 `_interpolate_frequency(...)`。
2. 写自定义 `SimulationComponentFactory`，用 `replace(...)` 注入新模块。
3. 写专用 `examples/run_xxx.py`，显式传入 `component_factory`。
4. 先在单 TTI、AWGN、高 SNR 下跑通。
5. 再打开绘图，检查星座图、信道估计图和自定义中间变量图。
6. 再跑多 TTI 和 baseline。
7. 最后才扩展到 TDL/CDL、多天线、干扰或外部信道系数。

这种顺序比较保守，但能快速定位问题到底来自算法本身、链路装配、资源映射、信道模型还是译码。

## 12. 常见问题

### 为什么不直接修改 Receiver？

`Receiver` 表示接收机主处理流程。直接在里面加实验分支会让主流程越来越难读，也容易影响其他链路。通过工厂注入组件，可以让“默认链路”和“实验链路”并存。

### 为什么不把自定义工厂写进 YAML？

当前配置文件主要描述仿真参数，不负责动态导入 Python 类。这样更适合普通用户运行，也能减少 Windows 环境下路径和模块导入问题。对于开发者实验链路，推荐用专用 Python 脚本显式传入工厂。

如果后续需要“配置文件选择算法类”，可以在工厂层增加一个受控 registry，而不是直接让 YAML 任意导入模块。

### 什么时候需要替换多个模块？

如果新模块改变了输出语义或形状，就可能需要替换下游模块。例如：

- 新 `ResourceMapper` 改变了 `data_mask` 排列方式，可能需要同步检查 `FrequencyExtractor`。
- 新 `ChannelEstimator` 输出了多层或多端口估计，可能需要同步替换 `MimoEqualizer`。
- 新 `TimeDomainProcessor` 改变了波形结构，可能需要同步替换发端和收端同一个 processor。

### 新链路结果不对时先看哪里？

优先检查：

- `rx_grid.shape`
- `dmrs_mask.shape`
- `data_mask.shape`
- `dmrs_symbols.size`
- `channel_estimate.shape`
- `noise_variance`
- 星座图是否有大量点堆在零附近
- 导频信道估计幅度和相位是否按接收天线分别合理

这些信息通常能比最终 BER/BLER 更早暴露链路装配问题。
