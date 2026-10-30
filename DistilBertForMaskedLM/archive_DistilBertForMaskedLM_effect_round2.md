# DistilBertForMaskedLM 图模式 benchmark 采集报告

模型：**DistilBertForMaskedLM**（HuggingFace，`huggingface.py` runner）
后端：**inductor + triton_experimental + 关闭 ACLGraph + 静态 shape**
采集类型：**融合效果**（debug + profiler 双开，观测值）

---

## 1. 环境信息

| 项 | 值 |
|---|---|
| 宿主机 | 179.173.2.2（root，SSH 免密） |
| 容器 | npu-z00896972-benckmark-cann920 |
| CANN | 9.2.0 正式版（/opt/ascend/cann-9.2.0） |
| benchmark 源码 | /opt/ascend_pytorch/pytorch/benchmarks/torchbench（容器内） |
| 卡 | 7（ASCEND_RT_VISIBLE_DEVICES=7） |
| 环境规避 | TORCHINDUCTOR_COMPILE_THREADS=1 |

## 2. 软件栈版本

| env | torch | torch_npu | triton |
|---|---|---|---|
| tnpu-2.13.0 | 2.13.0+cpu | 2.13.0+git14dc54e | 3.6.0 |
| tnpu-2.14.0 | 2.14.0+cpu | 2.14.0+git14dc54e | 3.6.0 |
| tnpu-2.15.0 | 2.15.0.dev20261005+cpu | 2.15.0.dev+git14dc54e | 3.6.0 |

## 3. 采集结果

### 3.1 融合效果（debug+profiler 双开，观测值，本轮 v2）

| 配置 | eager 均值(ms) | compile 均值(ms) | 加速比 | 耗时降低 | 统计区间 | 精度 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 2.13.0 train | 39.068 | 21.069 | 1.854x | 46.07% | 16~40 | pass |
| 2.13.0 inference | 10.440 | 3.941 | 2.649x | 62.25% | 16~40 | pass |
| 2.14.0 train | 36.246 | 18.950 | 1.913x | 47.72% | 16~40 | pass |
| 2.14.0 inference | 10.312 | 3.381 | 3.050x | 67.21% | 16~40 | pass |
| 2.15.0 train | 35.547 | 20.064 | 1.772x | 43.56% | 16~40 | pass |
| 2.15.0 inference | 9.226 | 3.026 | 3.049x | 67.20% | 16~40 | pass |

### 3.2 性能基线（双关，第一轮 collect/dmb 测得）

| 配置 | eager(ms) | compile(ms) | 加速比 |
| --- | ---: | ---: | ---: |
| 2.13 train | 28.17 | 20.04 | 1.41x |

> 融合效果轮（debug+profiler 双开）均为**观测值**，正式加速比仅基线一行。训练模式 compile 加速约 1.8~1.9x，推理模式 compile 加速约 2.6~3.0x。

## 4. 后端组合

- `--backend inductor --npu-backend triton_experimental --disable-aclgraph`
- 不传 `--dynamic-shapes` / `--dynamic-batch-only` / `--aclgraph-mode` / `--mfusion`（静态 shape）

## 5. 产物清单

### 本地（本机 PC，供与 GPU 对比）
```
F:\AI Agent\CodeAgent\Benchmark\DistilBertForMaskedLM\
├── REPORT.md                 # 本报告
├── ARCHIVE.md                # 第二轮采集归档
├── report\
│   ├── comparison.md         # 对比表
│   ├── summary.json          # 结构化汇总
│   └── timing_samples.csv    # 逐步耗时（6 轮 × 100 步）
├── 2.13.0\ / 2.14.0\ / 2.15.0\
│   ├── effect_train\{debug, profile} + effect_train.log
│   └── effect_inference\{debug, profile} + effect_inference.log
└── dmb_v2_artifacts.tar.gz   # 容器产物原始打包（137M，解压即上述内容）
```

各轮 debug 目录含：`fx_graph_readable.py`（前向/反向 FX）、`output_code.py`（生成代码）、`ir_pre_fusion.txt`、`ir_post_fusion.txt`（融合前后 IR）。
各轮 profile 目录含 eager/compile 各自的 `ASCEND_PROFILER_OUTPUT/`：`step_trace_time.csv`、`kernel_details.csv`、`operator_details.csv`、`task_time.csv`、`ascend_pytorch_profiler.db`、`analysis.db`。

### 容器内（原始，未删）
`/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/`（含 cache 编译缓存，本机已排除缓存但部分 triton 缓存文件因超长文件名未能清理，可忽略）

## 6. 采集命令模板（2.13.0 train）

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
cd /opt/ascend_pytorch/pytorch/benchmarks/torchbench
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=.../collect/dmb_v2/2.13.0/effect_train/debug
export TORCHINDUCTOR_CACHE_DIR=.../collect/dmb_v2/2.13.0/effect_train/cache
export TORCH_LOGS=graph_breaks,recompiles
TORCHINDUCTOR_COMPILE_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=7 /opt/npu-envs/tnpu-2.13.0/bin/python3 -u huggingface.py \
  --devices npu --backend inductor --npu-backend triton_experimental \
  --accuracy --train --float32 \
  --only DistilBertForMaskedLM --iterations 50 --timeout 1500 --disable-aclgraph \
  --enable-profiler=0 --prof-output-path .../collect/dmb_v2/2.13.0/effect_train/profile \
  2>&1 | tee .../collect/dmb_v2/2.13.0/effect_train.log
```

（inference 换 `--inference` + `effect_inference`；2.14.0/2.15.0 换 python 路径与目录。）

## 7. 说明

- 本轮 6 轮全部 pass、无 nan、无慢步；对比第一轮 2.15 train 曾出现 1037ms 慢步（compile 均值 60.86ms），本轮 20.06ms，说明首轮慢步为偶发。
- 逐步耗时已落盘（第一轮缺失），`timing_samples.csv` 完整 600 条（6 轮 × eager/compile 各 50 步）。
- 本报告 NPU 侧数据可直接与 GPU 同配置结果做跨设备比对。
