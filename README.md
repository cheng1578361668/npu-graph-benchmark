# NPU 图模式 Benchmark 采集归档

本仓库归档 Ascend NPU 图模式（triton_experimental 后端）各模型的 benchmark 采集结果，用于 NPU / CPU / GPU 跨设备性能对比分析。

## 目录结构

每个模型一个文件夹，文件夹名即模型名：

```
benchmark/
├── README.md                              # 本说明
├── DistilBertForMaskedLM/                 # 模型 1
│   ├── archive_<模型名>_effect_roundN.md   # 各轮采集报告
│   ├── report/                            # 汇总产物
│   │   ├── comparison.md                  # 对比表
│   │   ├── summary.json                   # 结构化汇总
│   │   └── timing_samples.csv             # 逐步耗时
│   └── 2.13.0/ 2.14.0/ 2.15.0/           # 各 PyTorch 版本
│       ├── effect_train/                  # 训练融合效果（debug + profile）
│       ├── effect_inference/              # 推理融合效果（debug + profile）
│       └── *.log                          # 采集日志
├── <模型名 2>/
└── ...
```

## 已采集模型

### DistilBertForMaskedLM

- 后端：inductor + triton_experimental + 关闭 ACLGraph + 静态 shape
- 采集类型：融合效果（debug + profiler 双开）
- CANN：9.2.0 正式版

| PyTorch 版本 | train 加速比 | inference 加速比 | 精度 |
| --- | ---: | ---: | --- |
| 2.13 | 1.854x | 2.649x | pass |
| 2.14 | 1.913x | 3.050x | pass |
| 2.15 | 1.772x | 3.049x | pass |

> 上表为融合效果轮观测值（debug+profiler 双开），正式 E2E 加速比需双关基线；详见各模型文件夹内的归档报告。

## 采集环境

| 项 | 值 |
| --- | --- |
| 设备 | Ascend 910B4（32GB） |
| 后端 | inductor + triton_experimental |
| CANN | 9.2.0 正式版 |
| 统一配置 | 关闭 ACLGraph、静态 shape、不开 MFusion、`TORCHINDUCTOR_COMPILE_THREADS=1` |
| 统计口径 | 50 轮取原生第 16~40 步算术均值；不删慢步 |

## 采集命令模板

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
cd /opt/ascend_pytorch/pytorch/benchmarks/torchbench
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=<模型目录>/<版本>/effect_<模式>/debug
export TORCHINDUCTOR_CACHE_DIR=<模型目录>/<版本>/effect_<模式>/cache
export TORCH_LOGS=graph_breaks,recompiles
TORCHINDUCTOR_COMPILE_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=<卡号> <python> -u huggingface.py \
  --devices npu --backend inductor --npu-backend triton_experimental \
  --accuracy --<train|inference> --float32 \
  --only <模型名> --iterations 50 --timeout 1500 --disable-aclgraph \
  --enable-profiler=0 --prof-output-path <模型目录>/<版本>/effect_<模式>/profile \
  2>&1 | tee <模型目录>/<版本>/effect_<模式>.log
```

## 如何新增模型

1. 在仓库根目录新建以**模型名**命名的文件夹；
2. 按上表结构放置 `report/` 汇总产物、`<版本>/effect_<模式>/` 的 debug/profile 目录及日志；
3. 在文件夹内放一份 `archive_<模型名>_effect.md` 报告，记录环境、版本、结果与命令；
4. 在本 README 的「已采集模型」表格中追加该模型一节。

每个模型独立一个文件夹，互不影响，可随采随归档。

## 汇总脚本

使用 `npu-graph-benchmark-collect` 技能的 `scripts/summarize_timings.py` 解析日志，生成 `comparison.md` / `summary.json` / `timing_samples.csv`：

```bash
python summarize_timings.py \
  --run 2.13_train=<日志> --run 2.13_infer=<日志> \
  --run 2.14_train=<日志> --run 2.14_infer=<日志> \
  --run 2.15_train=<日志> --run 2.15_infer=<日志> \
  --output-dir report
```