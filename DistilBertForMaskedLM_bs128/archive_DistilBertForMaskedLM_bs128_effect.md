# DistilBertForMaskedLM 复测归档（batch=128）

模型：**DistilBertForMaskedLM**（`huggingface.py`）
采集：融合效果（debug + profiler 双开）
batch_size：**128**
范围：仅 2.13.0 / 2.14.0（2.15 未跑，按用户要求只取这两个版本）

## 1. 环境

| 项 | 值 |
|---|---|
| 宿主机 | 179.173.2.2 |
| 容器 | npu-z00896972-benckmark-cann920 |
| CANN | 9.2.0 正式版 |
| 卡 | 7 |

## 2. 结果（batch=128，观测值）

| 版本 | 模式 | eager(ms) | compile(ms) | 加速比 | 精度 |
| --- | --- | ---: | ---: | ---: | --- |
| 2.13 | train | 283.04 | 973.71 | 0.291x | pass |
| 2.13 | inference | 81.05 | 61.37 | 1.321x | pass |
| 2.14 | train | 259.78 | 267.94 | 0.970x | pass |
| 2.14 | inference | 86.95 | 61.00 | 1.425x | pass |

> batch=128 下 train 的 compile 反而慢于 eager（0.29x/0.97x）；inference 有加速（1.32x/1.43x）。

## 3. 产物清单（按「Gap 报告采集清单」范围）

本地：`F:\AI Agent\CodeAgent\Benchmark\DistilBertForMaskedLM_bs128\`

- effect_*.log（4 个）
- kernel_details.csv（8 份）
- step_trace_time.csv（8 份）
- output_code.py / fx_graph_readable.py（debug）
- torchdynamo/debug.log
- report/（comparison.md + summary.json + timing_samples.csv）

## 4. 清单外未保留

容器内 `collect/DistilBertForMaskedLM_bs128/` 清单外产物已删（1.5G → 32M）：cache、PROF 二进制、trace_view.json、msprof、operator_details.csv、ascend_pytorch_profiler.db 等。