# 自定义中间变量绘图指南

这份文档面向算法开发者，说明如何在任意函数实现中保存中间变量、中间参数，并把它们交给绘图系统。典型场景是：

- 你在某个算法函数里算出了一个中间变量，例如新的信道估计权重、滤波器系数、置信度矩阵、迭代误差曲线
- 这个变量可能是 `numpy.ndarray`、`torch.Tensor`、Python `list`，维度也可能是一维、二维或更高维
- 绘图还需要一些临时参数，例如横坐标、天线编号、迭代次数、阈值、算法名称、归一化因子
- 这些参数不属于配置 `config`，也不适合作为最终仿真结果 `result`
- 具体画什么、怎么画，由你自己实现

当前推荐方案是：

1. 在算法函数里使用 `SimulationRuntimeContext` 保存中间变量和中间参数。
2. 使用 `PlotArtifact` 把要画的数据注册给绘图系统。
3. 如果通用绘图类型不够用，在 `visualization.py` 里新增一个专用 `plot_type` 和对应绘图函数。

## 1. 相关文件

主要涉及这些文件：

- [runtime_context.py](../src/nr_phy_simu/common/runtime_context.py)
- [types.py](../src/nr_phy_simu/common/types.py)
- [visualization.py](../src/nr_phy_simu/visualization.py)
- [run_from_config.py](../examples/run_from_config.py)

调用关系是：

```text
算法函数
  -> get_runtime_context().set(...)
  -> get_runtime_context().add_plot_artifact(...)
  -> simulation.run()
  -> run_from_config.py
  -> save_simulation_plots(...)
  -> visualization.py 根据 PlotArtifact 生成图片
```

## 2. 什么时候用 RuntimeContext

`config` 表示仿真输入参数，例如带宽、MCS、信道类型。

`result` 表示稳定的仿真输出，例如 BER、BLER、星座点、信道估计结果。

如果一个变量只是某次算法运行时的临时数据，建议放进 `SimulationRuntimeContext`。例如：

- 某个信道估计算法的内部权重矩阵
- 插值前后的误差
- 每次迭代的残差
- 某个 debug 阈值
- 为绘图准备的横坐标
- 自定义绘图函数需要的标量参数

## 3. 最小示例：画一个中间数组

假设你在 `channel_estimation.py` 的某个函数中得到了 `my_metric`：

```python
from nr_phy_simu.common.runtime_context import get_runtime_context
from nr_phy_simu.common.types import PlotArtifact


def my_algorithm_step(...):
    my_metric = ...

    context = get_runtime_context()
    context.add_plot_artifact(
        PlotArtifact(
            name="my_metric",
            values=my_metric,
            title="My Metric",
            plot_type="magnitude",
            xlabel="Index",
            ylabel="Magnitude",
        )
    )
```

运行仿真且 `plotting.enabled: true` 时，系统会自动输出：

```text
outputs/<prefix>_context_my_metric.png
```

`values` 可以是：

- `numpy.ndarray`
- `torch.Tensor`
- Python `list`
- 标量列表，例如 `[1.0, 0.8, 0.5]`
- 复数数组，例如 `np.array([1+1j, 0.5-0.2j])`

绘图层会在进入 `matplotlib` 前转换为 `numpy`。如果是 `torch.Tensor`，会先执行类似 `detach().cpu().numpy()` 的转换，所以 CPU/GPU tensor 都可以按这个路径使用。

## 4. 通用 plot_type

如果你的绘图需求比较简单，可以直接使用内置 `plot_type`：

```text
magnitude  画幅度，默认类型
phase      画相位
angle      phase 的别名
real       画实部
i          real 的别名
imag       画虚部
q          imag 的别名
image      二维热力图
auto       二维数据走热力图，一维数据走曲线
```

一维数据会画成一条曲线。

二维数据如果 `plot_type="image"` 或 `plot_type="auto"`，会画成热力图。

二维数据如果 `plot_type="magnitude"`、`real`、`imag` 或 `phase`，会把第一维当成多条序列，每一行画一条曲线。

## 5. 保存绘图需要的临时参数

如果绘图除了 `values` 还需要额外参数，推荐放在 `PlotArtifact.metadata` 里。

例如，你的算法生成了一个二维权重矩阵，同时还需要标出参考子载波和门限：

```python
context.add_plot_artifact(
    PlotArtifact(
        name="ce_weight_matrix",
        values=weight_matrix,
        title="Channel Estimation Weight Matrix",
        plot_type="ce_weight_matrix",
        xlabel="Subcarrier",
        ylabel="Symbol",
        metadata={
            "reference_subcarriers": reference_subcarriers,
            "threshold": threshold,
            "algorithm": "my_lmmse_v2",
        },
    )
)
```

`metadata` 可以保存任意 Python 对象。为了绘图代码更容易维护，建议优先保存这些类型：

- `int`
- `float`
- `str`
- `list`
- `tuple`
- `numpy.ndarray`
- `torch.Tensor`

如果某个参数其他模块也需要读取，可以同时用 `context.set(...)` 保存：

```python
context.set("channel_estimation", "reference_subcarriers", reference_subcarriers)
context.set("channel_estimation", "threshold", threshold)
```

其他模块可以这样读取：

```python
context = get_runtime_context()
threshold = context.get("channel_estimation", "threshold", default=None)
```

## 6. 自定义绘图函数

如果通用 `plot_type` 不够用，就在 [visualization.py](../src/nr_phy_simu/visualization.py) 里新增一个专用绘图函数。

假设你注册的 artifact 是：

```python
context.add_plot_artifact(
    PlotArtifact(
        name="ce_weight_matrix",
        values=weight_matrix,
        plot_type="ce_weight_matrix",
        metadata={
            "reference_subcarriers": reference_subcarriers,
            "threshold": threshold,
        },
    )
)
```

在 `visualization.py` 的 `_build_artifact_figure(...)` 中增加分支：

```python
def _build_artifact_figure(artifact: PlotArtifact) -> object:
    if artifact.plot_type == "constellation":
        return _build_constellation_figure(artifact)
    if artifact.plot_type == "pilot_estimates":
        return _build_pilot_estimate_figure(artifact)
    if artifact.plot_type == "rx_time":
        return _build_rx_time_domain_figure(artifact)
    if artifact.plot_type == "rx_freq":
        return _build_rx_frequency_domain_figure(artifact)
    if artifact.plot_type == "ce_weight_matrix":
        return _build_ce_weight_matrix_figure(artifact)

    ...
```

然后实现你的绘图函数：

```python
def _build_ce_weight_matrix_figure(artifact: PlotArtifact) -> object:
    values = _as_plot_array(artifact.values)
    metadata = artifact.metadata or {}
    reference_subcarriers = _as_plot_array(metadata.get("reference_subcarriers", []))
    threshold = metadata.get("threshold")

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(np.abs(values), aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="Weight Magnitude")

    for sc in reference_subcarriers:
        ax.axvline(int(sc), color="tab:red", linestyle="--", linewidth=0.8)

    if threshold is not None:
        ax.set_title(f"{artifact.title or artifact.name} (threshold={float(threshold):.3g})")
    else:
        ax.set_title(artifact.title or artifact.name)

    ax.set_xlabel(artifact.xlabel)
    ax.set_ylabel(artifact.ylabel or "Symbol Index")
    fig.tight_layout()
    return fig
```

这里的关键点是：

- `artifact.values` 是你要画的主数据
- `artifact.metadata` 是绘图所需的临时参数
- `_as_plot_array(...)` 会把 `list / numpy / torch` 统一转成 `numpy.ndarray`
- 绘图函数只负责画图，不重新计算算法结果

## 7. 在类成员函数中使用

如果你是在某个类里实现新算法，例如新的信道估计器，可以这样写：

```python
class MyChannelEstimator(LeastSquaresChannelEstimator):
    def _interpolate_frequency(self, pilot_subcarriers, pilot_values, num_subcarriers):
        interpolated = super()._interpolate_frequency(
            pilot_subcarriers,
            pilot_values,
            num_subcarriers,
        )

        interpolation_error = self._estimate_interpolation_error(...)
        context = get_runtime_context()
        context.set("my_channel_estimator", "interpolation_error", interpolation_error)
        context.add_plot_artifact(
            PlotArtifact(
                name="interpolation_error",
                values=interpolation_error,
                title="Interpolation Error",
                plot_type="magnitude",
                xlabel="Subcarrier Index",
            )
        )
        return interpolated
```

这种写法的好处是：

- 不需要改 `SimulationResult`
- 不需要改接收机主流程
- 不需要把 debug 变量塞进 `config`
- 后续如果这个变量稳定下来，再提升到正式结果结构也不晚

## 8. 多 TTI 场景

当前 runtime context 在每个单 TTI 开始时会清空。

多 TTI 仿真结束后，context 中保留的是最后一个 TTI 的运行期变量和绘图 artifact。因此：

- 如果你只关心最后一个 TTI 的中间变量，直接使用 `context.add_plot_artifact(...)` 即可
- 如果你要统计全部 TTI 的中间变量，需要自己在多 TTI runner 或外部脚本中聚合
- 如果某个图必须展示全部 TTI 的历史，建议把聚合结果作为稳定输出字段加入 `MultiTtiSimulationResult`，或者新增专门的 sweep/report 逻辑

## 9. 推荐命名

建议 artifact 名称使用小写加下划线：

```text
ce_weight_matrix
interpolation_error
mmse_noise_power
pilot_residual
```

系统会给不同来源自动加前缀：

- `result.rx.plot_artifacts` 中的图会保存为 `artifact_<name>`
- runtime context 中的图会保存为 `context_<name>`

最终文件名类似：

```text
outputs/pusch_awgn_context_ce_weight_matrix.png
outputs/pusch_awgn_artifact_interpolation_error.png
```

## 10. 常见问题

### 图没有生成

先检查配置文件：

```yaml
plotting:
  enabled: true
```

再确认代码确实调用了：

```python
context.add_plot_artifact(...)
```

如果你只调用了 `context.set(...)`，变量会被保存，但不会自动绘图。

### 图生成了但数据形状不对

先打印或断点检查：

```python
np.asarray(values).shape
```

对于 `torch.Tensor`，可以检查：

```python
values.detach().cpu().numpy().shape
```

如果是三维或更高维数据，建议你自己实现专用 `plot_type`，明确每个维度的含义。

### 自定义绘图函数里不要重新跑算法

绘图函数应该只消费已经保存的中间变量。

不推荐在绘图函数里重新做信道估计、均衡、译码等算法步骤。这样会让图中结果和接收机实际使用结果产生偏差，也会浪费仿真时间。

### 什么时候应该扩展 result

如果一个变量已经不只是调试量，而是稳定的算法输出，例如新的信道估计质量指标、译码迭代次数、HARQ 合并统计，建议把它加到正式结果结构中：

- `ChannelEstimateResult`
- `RxPayload`
- `SimulationResult`
- `MultiTtiSimulationResult`

`SimulationRuntimeContext` 更适合运行期 scratch 数据和快速绘图。
