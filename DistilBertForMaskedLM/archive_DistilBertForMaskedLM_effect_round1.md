# DistilBertForMaskedLM 融合效果采集归档

按 `npu-graph-benchmark-collect` 技能规范整理（2026-10-28）。

## 1. 环境信息

| 项 | 值 |
|---|---|
| 容器 | npu-z00896972-benchmark-cann920 |
| CANN | 9.2.0 正式版（/opt/ascend/cann-9.2.0） |
| benchmark 源码 | /opt/ascend_pytorch/pytorch/benchmarks/torchbench |
| runner | huggingface.py |
| 后端组合 | inductor + triton_experimental + disable-aclgraph + 静态 shape（不传 dynamic-shapes/mfusion） |
| 卡 | 7（ASCEND_RT_VISIBLE_DEVICES=7） |
| 环境规避 | TORCHINDUCTOR_COMPILE_THREADS=1（v2.13 fork 编译坑） |

## 2. 软件栈版本

| env | torch | torch_npu | triton | triton-ascend |
|---|---|---|---|---|
| tnpu-2.13.0 | 2.13.0+cpu | 2.13.0+git14dc54e | 3.6.0 | 3.6.0 |
| tnpu-2.14.0 | 2.14.0+cpu | 2.14.0+git14dc54e | 3.6.0 | 3.6.0 |
| tnpu-2.15.0 | 2.15.0.dev20261005+cpu | 2.15.0.dev+git14dc54e | 3.6.0 | 3.6.0 |

## 3. 采集结果汇总（融合效果：debug+profiler 双开，观测值）

| 版本 | 模式 | 精度 | eager 观测(ms) | compile 观测(ms) | 观测比值 |
|---|---|---|---|---|---|
| 2.13 | train | pass_accuracy | 35.64 | 23.21 | 1.54x |
| 2.13 | inference | pass_accuracy | 10.83 | 3.74 | 2.90x |
| 2.14 | train | pass_accuracy | 32.07 | 24.96 | 1.28x |
| 2.14 | inference | pass_accuracy | 10.27 | 3.65 | 2.81x |
| 2.15 | train | pass_accuracy | 34.71 | 60.86* | 0.57x* |
| 2.15 | inference | pass_accuracy | 10.61 | 3.96 | 2.68x |

> *2.15 train compile 含 step38 一处 1037ms 慢步（其余 step ~20ms），拉高均值。按技能口径不删慢步，如实记录为观测异常。
>
> 以上均为 debug+profiler 双开观测值，**非正式加速比**。

## 3.5 NPU 侧总结对比表（summarize 口径）

| 配置 | eager 均值(ms) | compile 均值(ms) | 加速比 | 耗时降低 | 统计区间 | 精度通过 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 2.13 train（基线·双关） | 28.17 | 20.04 | 1.404x | 28.86% | 16~40 | 是 |
| 2.13 train（效果·观测） | 35.64 | 23.21 | 1.536x | 34.87% | 16~40 | 是 |
| 2.13 inference（效果·观测） | 10.83 | 3.74 | 2.896x | 65.47% | 16~40 | 是 |
| 2.14 train（效果·观测） | 32.07 | 24.96 | 1.285x | 22.18% | 16~40 | 是 |
| 2.14 inference（效果·观测） | 10.27 | 3.65 | 2.814x | 64.46% | 16~40 | 是 |
| 2.15 train（效果·观测） | 34.71 | 60.86 | 0.570x | -75.34% | 16~40 | 是 |
| 2.15 inference（效果·观测） | 10.61 | 3.96 | 2.679x | 62.68% | 16~40 | 是 |

> 效果轮（debug+profiler 双开）均为观测值，**正式加速比仅基线一行**。2.15 train 的负加速比由 step38 的 1037ms 慢步造成（未剔除，按技能口径保留）。

## 4. 正式基线（2.13 train，双关）

| 指标 | eager | compile | 加速比 |
|---|---|---|---|
| avg [16-40] | 28.17 ms | 20.04 ms | 1.41x |

## 5. 产物路径（容器内绝对路径）

| 轮次 | debug 目录 | profile 目录 |
|---|---|---|
| 2.13 train | collect/dmb/2.13/effect/debug | collect/dmb/2.13/effect/profile |
| 2.13 inference | collect/dmb/2.13/effect_inference/debug | collect/dmb/2.13/effect_inference/profile |
| 2.14 train | collect/dmb/2.14/effect_train/debug | collect/dmb/2.14/effect_train/profile |
| 2.14 inference | collect/dmb/2.14/effect_inference/debug | collect/dmb/2.14/effect_inference/profile |
| 2.15 train | collect/dmb/2.15/effect_train/debug | collect/dmb/2.15/effect_train/profile |
| 2.15 inference | collect/dmb/2.15/effect_inference/debug | collect/dmb/2.15/effect_inference/profile |

> 完整前缀：`/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb/...`

## 6. 产物核验状态（全部通过）

| 类别 | 结果 |
|---|---|
| debug：前向/反向（train）或 inference 单图四件套 | ✅ fx_graph_readable.py + output_code.py + ir_pre_fusion.txt + ir_post_fusion.txt |
| profiler：eager/compile 各 ASCEND_PROFILER_OUTPUT | ✅ step_trace_time / kernel_details / operator_details / task_time CSV + db |
| 同一轮配对 | ✅ 各轮 debug 与 profile 同 pid |
| 精度 | ✅ 6 轮全 pass_accuracy，loss 无 nan |

## 7. 采集命令模板（2.13 train 实际使用）

```bash
cd /opt/ascend_pytorch/pytorch/benchmarks/torchbench
source /opt/ascend/cann-9.2.0/set_env.sh
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb/2.13/effect/debug
export TORCHINDUCTOR_CACHE_DIR=/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb/2.13/effect/cache
export TORCH_LOGS=graph_breaks,recompiles
TORCHINDUCTOR_COMPILE_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=7 /opt/npu-envs/tnpu-2.13.0/bin/python3 -u huggingface.py \
  --devices npu --backend inductor --npu-backend triton_experimental \
  --accuracy --train --float32 \
  --only DistilBertForMaskedLM --iterations 50 --timeout 1500 --disable-aclgraph \
  --enable-profiler=0 --prof-output-path /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb/2.13/effect/profile
```

（inference 换 `--inference` + `effect_inference`；2.14/2.15 换 python 路径与 `2.14/2.15` 目录。）

## 8. SHA256 清单（已生成）

容器内产物 SHA256 清单已生成，覆盖 6 轮采集的 debug 四件套（fx_graph_readable.py / output_code.py / ir_pre_fusion.txt / ir_post_fusion.txt）与 profiler CSV/db（step_trace_time / kernel_details / operator_details / task_time + ascend_pytorch_profiler.db / analysis.db）。

清单要点：
- cache 与 debug 同名文件 SHA 一致（编译缓存与 debug dump 同源，符合预期）
- eager/compile 两模式产物齐全
- 额外存在 2 个 inference 重复 run（模型名 `model__0_inference_0.0`，pid_213830 / pid_284696），与主 6 轮（`DistilBertForMaskedLM__0_inference_0.0`）为不同次运行，归档时区分标注、不混用

容器内落盘建议：
```bash
cd /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb
find . -type f \( -name "fx_graph_readable.py" -o -name "output_code.py" -o -name "ir_pre_fusion.txt" -o -name "ir_post_fusion.txt" -o -name "step_trace_time.csv" -o -name "kernel_details.csv" -o -name "operator_details.csv" -o -name "task_time.csv" -o -name "*.db" \) -exec sha256sum {} \; | sort > sha256_manifest.txt
```

- [ ] 本地回收（可选，当前全部留在远端容器）
