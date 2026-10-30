# DistilBertForMaskedLM 融合效果采集归档（第二轮 v2）

按 `npu-graph-benchmark-collect` 技能规范整理（2026-10-29），本轮**每条命令 tee 日志落盘**，产出了逐步 CSV。

## 1. 环境信息

| 项 | 值 |
|---|---|
| 宿主机 | 179.173.2.2（root，SSH 免密） |
| 容器 | npu-z00896972-benckmark-cann920 |
| CANN | 9.2.0 正式版（/opt/ascend/cann-9.2.0） |
| benchmark 源码 | /opt/ascend_pytorch/pytorch/benchmarks/torchbench（容器内） |
| runner | huggingface.py |
| 后端组合 | inductor + triton_experimental + disable-aclgraph + 静态 shape |
| 卡 | 7（ASCEND_RT_VISIBLE_DEVICES=7） |
| 环境规避 | TORCHINDUCTOR_COMPILE_THREADS=1 |
| 采集阶段 | 融合效果（debug+profiler 双开） |

## 2. 软件栈版本

| env | torch | torch_npu |
|---|---|---|
| tnpu-2.13.0 | 2.13.0+cpu | 2.13.0+git14dc54e |
| tnpu-2.14.0 | 2.14.0+cpu | 2.14.0+git14dc54e |
| tnpu-2.15.0 | 2.15.0.dev20261005+cpu | 2.15.0.dev+git14dc54e |

（triton 3.6.0 / triton-ascend 3.6.0）

## 3. 采集结果（融合效果，观测值）

| 配置 | eager 均值(ms) | compile 均值(ms) | 加速比 | 耗时降低 | 统计区间 | 精度 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 2.13.0 train | 39.068 | 21.069 | 1.854x | 46.07% | 16~40 | pass |
| 2.13.0 inference | 10.440 | 3.941 | 2.649x | 62.25% | 16~40 | pass |
| 2.14.0 train | 36.246 | 18.950 | 1.913x | 47.72% | 16~40 | pass |
| 2.14.0 inference | 10.312 | 3.381 | 3.050x | 67.21% | 16~40 | pass |
| 2.15.0 train | 35.547 | 20.064 | 1.772x | 43.56% | 16~40 | pass |
| 2.15.0 inference | 9.226 | 3.026 | 3.049x | 67.20% | 16~40 | pass |

> 本轮 6 轮全 pass、无 nan、无慢步。对比上一轮（collect/dmb，2.15 train 出现 1037ms 慢步导致 compile 均值 60.86ms），本轮 2.15 train compile 均值 20.06ms，说明上次慢步为偶发，本轮数据更干净。
>
> 以上均为 debug+profiler 双开观测值，非正式加速比。

## 4. 产物路径

### 容器内（原始日志 + debug/profile）
```
/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/
├── 2.13.0/effect_train/{debug,cache,profile} + effect_train.log
├── 2.13.0/effect_inference/{debug,cache,profile} + effect_inference.log
├── 2.14.0/effect_train/{...} + effect_train.log
├── 2.14.0/effect_inference/{...} + effect_inference.log
├── 2.15.0/effect_train/{...} + effect_train.log
├── 2.15.0/effect_inference/{...} + effect_inference.log
└── report/{summary.json, timing_samples.csv, comparison.md}
```

### PC 本机（汇总产物，已下载）
```
F:\AI Agent\CodeAgent\Benchmark\collect_dmb_v2\
├── comparison.md        # 对比表
├── summary.json         # 结构化汇总
└── timing_samples.csv   # 逐步耗时（601 行 = 6 轮 × 100 步）
```

## 5. 采集命令模板（2.13.0 train 实际使用）

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

## 6. 与上一轮（collect/dmb）的差异

- 上轮：日志未 tee 落盘，逐步 CSV 丢失；2.15 train 含 1 秒慢步
- 本轮：全部 tee 落盘，逐步 CSV 完整；2.15 train 无慢步，数据干净