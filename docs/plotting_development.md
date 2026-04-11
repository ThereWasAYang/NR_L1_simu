# 绘图开发文档

这份文档说明当前仿真系统里的绘图是如何被调用的，以及如果后续要新增绘图节点，应该修改哪些位置。

## 1. 当前绘图调用链

当前主入口是：

- `examples/run_from_config.py`

调用顺序如下：

1. `run_from_config.py` 读取配置文件，得到 `SimulationConfig`
2. 根据 `channel_type` 或 `waveform_input` 选择场景对象
3. 调用 `simulation.run()` 完成一次仿真，拿到 `SimulationResult`
4. 如果 `config.plotting.enabled == true`
   则调用 `save_simulation_plots(result, config, output_dir, prefix, ...)`
5. `save_simulation_plots(...)` 在 `src/nr_phy_simu/visualization.py` 中构建所有图，并保存到 `outputs/`

也就是说：

- 仿真本体负责产生数据
- `visualization.py` 负责把数据画出来
- `run_from_config.py` 负责决定“这次要不要画图”

## 2. 绘图开关在哪里

统一开关在配置文件里：

```yaml
plotting:
  enabled: true
```

对应配置类在：

- `src/nr_phy_simu/config.py`

当前 `run_from_config.py` 的逻辑是：

- `enabled = true`：生成并显示/保存所有图
- `enabled = false`：完全跳过绘图

## 3. 现在有哪些绘图节点

当前所有绘图函数都集中在：

- `src/nr_phy_simu/visualization.py`

统一入口函数：

- `save_simulation_plots(...)`

它当前会调用这些内部函数：

- `_build_constellation_figure(result)`
  画均衡后的星座图
- `_build_pilot_estimate_figure(result)`
  画导频位置上的信道估计幅度/相位
- `_build_rx_time_domain_figures(result, config)`
  画接收机入口时域幅值图
- `_build_rx_frequency_domain_figures(result, config)`
  画接收机时域处理后的频域幅值图

## 4. 图里用到的数据从哪里来

绘图函数只使用 `SimulationResult` 和 `SimulationConfig`，不直接参与算法处理。

主要数据来源：

- `result.rx.rx_waveform`
  接收机入口时域波形
- `result.rx.rx_grid`
  OFDM 解调后的频域栅格
- `result.rx.channel_estimate`
  信道估计结果
- `result.rx.equalized_symbols`
  均衡后的调制符号
- `result.tx.dmrs_mask`
  DMRS 资源位置
- `config.carrier`
  FFT、大带宽、CP 等绘图坐标所需信息

这些字段的定义在：

- `src/nr_phy_simu/common/types.py`

## 5. 如果要新增一个绘图节点，应该改哪里

推荐做法是只改两个地方：

1. 在 `src/nr_phy_simu/visualization.py` 新增一个 `_build_xxx_figure(...)` 或 `_build_xxx_figures(...)`
2. 在 `save_simulation_plots(...)` 里把它加入 `figures` 字典

例如：

```python
def _build_my_new_figure(result: SimulationResult, config: SimulationConfig):
    fig, ax = plt.subplots()
    ax.plot(...)
    return fig
```

然后在 `save_simulation_plots(...)` 中加入：

```python
figures["my_new_plot"] = _build_my_new_figure(result, config)
```

这样系统会自动：

- 保存成 `outputs/<prefix>_my_new_plot.png`
- 在前台显示
- 在 `run_from_config.py` 里打印对应路径

## 6. 新增绘图节点时的推荐原则

### 原则 1：不要在绘图函数里重复实现算法

绘图应该只消费已经算好的结果。

例如：

- 好做法：直接画 `result.rx.channel_estimate`
- 不好做法：在绘图函数里重新从 `rx_grid / dmrs_symbols` 再做一次信道估计

原因：

- 避免浪费算力
- 避免“图里画的结果”和“接收机真实使用的结果”不一致

### 原则 2：如果图需要新数据，优先把数据挂到 `SimulationResult`

如果你要画一个新的中间节点，例如：

- 去 CP 后的时域波形
- 频域抽取后的数据 RE
- 均衡前后的对比

推荐做法是：

1. 在 `src/nr_phy_simu/common/types.py` 的 `RxPayload` 或 `SimulationResult` 中增加字段
2. 在接收机链路里把该中间量存进去
3. 在 `visualization.py` 中读取这个字段绘图

不要在绘图函数里重新跑一遍主链路。

### 原则 3：一类图建议对应一个稳定的数据语义

例如：

- 时域图：横轴是样点索引
- 频域图：横轴是拼接后的频域索引，或者明确写成 `(symbol, subcarrier)`
- 导频图：横轴是导频子载波索引

如果横轴不是直观物理量，建议在标题或 `xlabel` 中明确写出来。

## 7. 如果我想画“更前面的接收机中间节点”，该改哪里

当前接收机主流程在：

- `src/nr_phy_simu/rx/chain.py`

其中关键步骤是：

1. `rx_waveform -> rx_grid`
2. `rx_grid -> channel_estimate`
3. `rx_grid -> rx_data_symbols`
4. `rx_data_symbols + channel_estimate -> equalized_symbols`
5. `equalized_symbols -> llrs`
6. `llrs -> decoded_bits`

如果你想新增例如：

- “去 CP 后的单符号时域波形图”
- “频域抽取后的数据 RE 图”
- “均衡前/后星座图”

通常需要：

1. 在 `rx/chain.py` 把对应中间量保存下来
2. 扩展 `RxPayload`
3. 在 `visualization.py` 里画出来

## 8. 最小修改模板

如果你只是想增加一张图，最小修改通常是：

1. 修改 `src/nr_phy_simu/common/types.py`
   如果现有 `result` 里没有你要的数据，就加字段
2. 修改 `src/nr_phy_simu/rx/chain.py` 或 `src/nr_phy_simu/scenarios/base.py`
   把中间结果写进 `result`
3. 修改 `src/nr_phy_simu/visualization.py`
   新增 `_build_xxx_figure(...)`
4. 修改 `save_simulation_plots(...)`
   把新图挂进去

## 9. 当前最关键的文件

如果后续你要自己扩图，最常用的是这几个文件：

- `examples/run_from_config.py`
  控制本次仿真是否调用绘图
- `src/nr_phy_simu/visualization.py`
  所有图的统一实现入口
- `src/nr_phy_simu/common/types.py`
  图所消费的数据结构定义
- `src/nr_phy_simu/rx/chain.py`
  接收机主链路，中间结果从这里拿
- `src/nr_phy_simu/config.py`
  全局绘图开关和其他绘图相关配置

## 10. 一句话建议

如果你后续要加新图，最稳的方式是：

- 先在主链路里把中间结果存下来
- 再让 `visualization.py` 只负责“把这个结果画出来”

这样最不容易把仿真逻辑和可视化逻辑缠在一起。
