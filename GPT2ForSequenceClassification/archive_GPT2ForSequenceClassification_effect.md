# GPT2ForSequenceClassification 融合效果采集归档

模型：**GPT2ForSequenceClassification**（HuggingFace，`huggingface.py`）
后端：**inductor + triton_experimental + 关闭 ACLGraph + 静态 shape**
采集：**融合效果（debug + profiler 双开）**，batch_size = 4

## 1. 环境

| 项 | 值 |
|---|---|
| 宿主机 | 179.173.2.2 |
| 容器 | npu-z00896972-benckmark-cann920 |
| CANN | 9.2.0 正式版 |
| 卡 | 7 |
| 规避 | TORCHINDUCTOR_COMPILE_THREADS=1 |

## 2. 软件栈

| env | torch | torch_npu |
|---|---|---|
| tnpu-2.13.0 | 2.13.0+cpu | 2.13.0+git14dc54e |
| tnpu-2.14.0 | 2.14.0+cpu | 2.14.0+git14dc54e |
| tnpu-2.15.0 | 2.15.0.dev20261005+cpu | 2.15.0.dev+git14dc54e |

## 3. 采集结果（融合效果，观测值，batch=4）

| 版本 | 模式 | eager(ms) | compile(ms) | 加速比 | 精度 |
| --- | --- | ---: | ---: | ---: | --- |
| 2.13 | train | 201.41 | 151.27 | 1.33x | pass |
| 2.13 | inference | 67.98 | 43.91 | 1.55x | pass |
| 2.14 | train | 194.49 | 152.27 | 1.28x | pass |
| 2.14 | inference | 65.09 | 54.14 | 1.20x | pass |
| 2.15 | train | 212.89 | 152.28 | 1.40x | pass |
| 2.15 | inference | 65.20 | 43.15 | 1.51x | pass |

## 4. 产物清单（按「Gap 报告采集清单」范围）

本地文件夹：`F:\AI Agent\CodeAgent\Benchmark\GPT2ForSequenceClassification\`

| 清单项 | 内容 |
|---|---|
| effect_*.log | 6 个（每版本 train/inference） |
| kernel_details.csv | 12 份（每模式 eager/compile） |
| step_trace_time.csv | 12 份 |
| output_code.py / fx_graph_readable.py | debug 目录下 |
| torchdynamo/debug.log | 各 run |
| report/ | summary.json + timing_samples.csv + comparison.md |

## 5. 清单外未下载（留在容器，未删除）

GPT2 的 profiler 产物异常大（总 45G），**清单范围外**的大文件未下载、未删除，仍在容器 `/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/`：

- trace_view.json（inference 单文件 2.7G，train 92~144M）
- ascend_pytorch_profiler.db（inference 1.2G，train 66~75M）
- operator_details.csv（inference 470M，train <50M）
- FRAMEWORK/torch.op_range（1.1G）
- msprof_*.json（原始数据）
- PROF_* 二进制原始数据、cache 编译缓存

如需其中某类，再单独取。

## 6. 备注

- 本清单按「新模型 Gap 报告采集清单（基于 DistilBertForMaskedLM 分析经验）」的 NPU 侧必需文件范围整理，只取 #1~#6，其余未动。
- 加速比为 debug+profiler 双开观测值，非正式双关基线。
- inference 的 profiler 数据（trace_view/db/operator/op_range）异常大，是 GPT2 推理时算子事件数量巨大所致，与 kernel_details.csv（正常 19~20M）无关。
