# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/ec/cecv4itvj2zzwz73cudpacwdhdw3uwzqopz5t35mmmfnj7s7b5h5.py
# Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   position_ids => iota
# Graph fragment:
#   %iota : Tensor "i64[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   return %iota
triton_unk_fused_arange_0 = async_compile.triton('triton_unk_fused_arange_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 1024
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = (x0).to(tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/f2/cf2pndp7jkl5xt3yzg6ztx4ee72afdbcrxuzpm4ippuqrzkvsk5g.py
# Topologically Sorted Source Nodes: [eq, int_1], Original ATen: [aten.eq, aten._to_copy]
# Source node to ATen node mapping:
#   eq => eq
#   int_1 => convert_element_type_12
# Graph fragment:
#   %primals_1 : Tensor "i64[4, 1024][1024, 1]npu:0" = PlaceHolder[target=primals_1]
#   %eq : Tensor "b8[4, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, 0), kwargs = {})
#   %convert_element_type_12 : Tensor "i32[4, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.int32), kwargs = {})
#   return %convert_element_type_12
triton_unk_fused__to_copy_eq_1 = async_compile.triton('triton_unk_fused__to_copy_eq_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_eq_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_eq_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2.to(tl.int32)
        tl.store(out_ptr0 + (x0), tmp3, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/tf/ctfypjwbvgfg72gtl5s373u2lcypniiuka4msc5ohxy3ay4h255o.py
# Topologically Sorted Source Nodes: [arange_1], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   arange_1 => iota_1
# Graph fragment:
#   %iota_1 : Tensor "i64[4][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   return %iota_1
triton_unk_fused_arange_2 = async_compile.triton('triton_unk_fused_arange_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = (x0).to(tl.int32)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/dm/cdmgaeiwpjmivqijowerjshpcjyztsy27il6enbrvsm64cuz4pmh.py
# Topologically Sorted Source Nodes: [loss, unsqueeze_2, ne_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   loss => full_default_24, ne, where_12
#   ne_3 => ne_3
#   unsqueeze_2 => unsqueeze_2
# Graph fragment:
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %ne : Tensor "b8[4][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_163, -100), kwargs = {})
#   %full_default_24 : Tensor "i64[][]npu:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_12 : Tensor "i64[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %primals_163, %full_default_24), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[4, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_163, 1), kwargs = {})
#   %ne_3 : Tensor "b8[4, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_2, -100), kwargs = {})
#   return %where_12,%ne_3
triton_unk_fused_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_unk_fused_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_backward_nll_loss_forward_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], -100, tl.int32)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tl.store(out_ptr0 + (x0), tmp4, x0mask)
        tl.store(out_ptr1 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/t5/ct5hktvgot7ebgx3djhq3pyfjl5ar6rl4mdxfczgmjj63zof4r5h.py
# Topologically Sorted Source Nodes: [loss, unsqueeze_2, where_14, ], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward, aten.expand]
# Source node to ATen node mapping:
#    => expand_default
#   loss => full_default_24
#   unsqueeze_2 => unsqueeze_2
#   where_14 => where_14
# Graph fragment:
#   %ne_3 : Tensor "b8[4, 1][1, 1]npu:0" = PlaceHolder[target=ne_3]
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %full_default_24 : Tensor "i64[][]npu:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %unsqueeze_2 : Tensor "i64[4, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_163, 1), kwargs = {})
#   %where_14 : Tensor "i64[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_2, %full_default_24), kwargs = {})
#   %expand_default : Tensor "i64[4, 2][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_14, [4, 2]), kwargs = {})
#   return %expand_default
triton_unk_fused_expand_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_unk_fused_expand_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr1': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_expand_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_expand_nll_loss_backward_nll_loss_forward_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 8
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_1 = x0_blocks
    total_blocks = x0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        tmp0 = tl.load(in_ptr0 + x1, x1mask, eviction_policy='evict_last') != 0
        tmp1 = tl.load(in_ptr1 + (x1), x1mask, eviction_policy='evict_last')
        tmp2 = tl.full([1, 1], 0, tl.int32)
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tl.store(out_ptr0 + (x0 + 2*x1), tmp3, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/m2/cm2kncind43xm3vbk5jm7okh4vewrmgzl2a2kgn4qapahi4lcoug.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %embedding : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %add : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   return %add
triton_unk_fused_add_5 = async_compile.triton('triton_unk_fused_add_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 3145728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [786432, 4, {'divisors': [1, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 786432, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3145728
    x_g_tile0 : tl.constexpr = (786432) if (786432) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 786432
    x1numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_1 = x0_blocks
    total_blocks = x0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        tmp0 = tl.load(in_out_ptr0 + (x0 + 786432*x1), x0mask & x1mask)
        tmp1 = tl.load(in_ptr0 + (x0), x0mask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tl.store(in_out_ptr0 + (x0 + 786432*x1), tmp2, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/sf/csfxqis3ox7zgvfmgmcqozgimnfok6imzdywcgsd7f6qktvo6lvk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.arange, aten.view, aten.eq]
# Source node to ATen node mapping:
#    => eq_tensor, iota_default, view_default
# Graph fragment:
#   %expand_default : Tensor "i64[4, 2][2, 1]npu:0" = PlaceHolder[target=expand_default]
#   %iota_default : Tensor "i64[2][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 2]), kwargs = {})
#   %eq_tensor : Tensor "b8[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   return %eq_tensor
triton_unk_fused_arange_eq_view_6 = async_compile.triton('triton_unk_fused_arange_eq_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_eq_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_eq_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 8
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_1 = x0_blocks
    total_blocks = x0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        tmp0 = tl.load(in_ptr0 + (x0 + 2*x1), x0mask & x1mask)
        tmp1 = (x0).to(tl.int32)
        tmp2 = tmp0 == tmp1
        tl.store(out_ptr0 + (x0 + 2*x1), tmp2, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/vo/cvobymfu2jpaihj77xffpze5geuuu3oc56syqbfd7f2jjulhekoz.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub => sub_25
# Graph fragment:
#   %argmax : Tensor "i64[4][1]npu:0" = PlaceHolder[target=argmax]
#   %sub_25 : Tensor "i64[4][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%argmax, 1), kwargs = {})
#   return %sub_25
triton_unk_fused_sub_7 = async_compile.triton('triton_unk_fused_sub_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_sub_7(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 1, tl.int32)
        tmp2 = tmp0 - tmp1
        tl.store(in_out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/td/ctd3qzptstpa7lk2ei7nl6yxistlwjmtuc3n5gfdb42fksmqzvs6.py
# Topologically Sorted Source Nodes: [hidden_states, div_50], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_50 => div_50
#   hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %buf9 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf9]
#   %getitem_3 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_3]
#   %buf11 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf11]
#   %mul : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul]
#   %primals_4 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_5]
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%getitem, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %getitem_3), kwargs = {})
#   %mul : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_4), kwargs = {})
#   %add_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_5), kwargs = {})
#   %div_50 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 768), kwargs = {})
#   return %buf9,%getitem_3,%buf11,%mul,%add_2,%div_50
triton_unk_fused_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_unk_fused_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [real_block_x0, R0_BLOCK])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 1)[:, None]
        tmp4 = tl.full([1, 1], 768.0, tl.float32)
        tmp5 = (tmp2 / tmp4)
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp6 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp7 = tmp6 - tmp5
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp20 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp13 = tmp12 - tmp5
            tmp14 = tl.full([1, 1], 768.0, tl.float32)
            tmp15 = (tmp10 / tmp14)
            tmp16 = tl.full([1, 1], 1e-05, tl.float32)
            tmp17 = tmp15 + tmp16
            tmp18 = tl.rsqrt(tmp17)
            tmp19 = tmp13 * tmp18
            tmp21 = tmp19 * tmp20
            tmp23 = tmp21 + tmp22
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp19, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp23, r0_mask & x0mask)
        tmp24 = tl.full([1, 1], 768.0, tl.float32)
        tmp25 = (tmp10 / tmp24)
        tmp26 = tl.full([1, 1], 1e-05, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = tl.rsqrt(tmp27)
        tmp29 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp30 = tmp28 * tmp29
        tl.store(out_ptr3 + (x0), tmp30, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/o3/co3sxrcrhqjr3esvdez667uke5zhykpe3lxwrwht75rb5awtnjnr.py
# Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   attn_weights => clone
#   query_1 => permute
#   split => split
#   tensor => view_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_3 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 1024, 12, 64]), kwargs = {})
#   %permute : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_unk_fused_matmul_permute_split_view_9 = async_compile.triton('triton_unk_fused_matmul_permute_split_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 3145728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3145728
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (1024) if (1024) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (4) if (4) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 1024
    x2numel : tl.constexpr = 12
    x3numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 2304*x1 + 2359296*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/qt/cqttnvxdlwtuaecutx7o4z4tr5dvetgfnrtqbk2o5k2oq7xrv5ww.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#   attn_weights => clone_1
#   key_1 => permute_1
#   split => split
#   tensor_1 => view_4
#   transpose => permute_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_4 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_5, [4, 1024, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_matmul_permute_split_transpose_view_10 = async_compile.triton('triton_unk_fused_matmul_permute_split_transpose_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 3072, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_transpose_view_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_transpose_view_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 3072
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 768
    y1numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 1024
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        xoffset = x2offset
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (768 + y0 + 2304*x2 + 2359296*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 1024*y0 + 786432*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/jy/cjykoulfaapate2u4qy6o27f2svinuzuwyeqlzdh4ry5wdken64d.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   attn_output => clone_2
#   split => split
#   tensor_2 => view_5
#   value_1 => permute_2
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_5 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_6, [4, 1024, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_unk_fused_matmul_permute_split_view_11 = async_compile.triton('triton_unk_fused_matmul_permute_split_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 3145728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_view_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3145728
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (1024) if (1024) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (4) if (4) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 1024
    x2numel : tl.constexpr = 12
    x3numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (1536 + x0 + 64*x2 + 2304*x1 + 2359296*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/tz/ctztf57m4hobohmbaomt64tzs54oufw5jt7atzf7o2yeqphinjic.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, permute_258, clone_115], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
# Source node to ATen node mapping:
#   clone_115 => clone_115
#   permute_258 => permute_258
#   split => split
#   tensor_2 => view_5
#   value_1 => permute_2
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_5 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_6, [4, 1024, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %permute_258 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 1, 3, 2]), kwargs = {})
#   %clone_115 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_137,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_115
triton_unk_fused_matmul_backward_permute_split_view_12 = async_compile.triton('triton_unk_fused_matmul_backward_permute_split_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 3072, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_permute_split_view_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_permute_split_view_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 3072
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 768
    y1numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 1024
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        xoffset = x2offset
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (1536 + y0 + 2304*x2 + 2359296*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 1024*y0 + 786432*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/n7/cn7i6mjfexub6vxhpm2oln4g5qg422buzisutxeclmqpd7xvrd2r.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, permute_260, clone_117], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
# Source node to ATen node mapping:
#   clone_117 => clone_117
#   key_1 => permute_1
#   permute_260 => permute_260
#   split => split
#   tensor_1 => view_4
#   transpose => permute_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_4 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_5, [4, 1024, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %permute_260 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_3, [0, 1, 3, 2]), kwargs = {})
#   %clone_117 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_141,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_117
triton_unk_fused_matmul_backward_permute_split_transpose_view_13 = async_compile.triton('triton_unk_fused_matmul_backward_permute_split_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 3145728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_permute_split_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_permute_split_transpose_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3145728
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (1024) if (1024) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (4) if (4) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 1024
    x2numel : tl.constexpr = 12
    x3numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (768 + x0 + 64*x2 + 2304*x1 + 2359296*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/3i/c3ih37qovs3omspbhbv2mbmrnlooymdrosv3uqku3u7xwfb3t2ui.py
# Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, permute_261, clone_118], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
# Source node to ATen node mapping:
#   clone_118 => clone_118
#   permute_261 => permute_261
#   query_1 => permute
#   split => split
#   tensor => view_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_3 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 1024, 12, 64]), kwargs = {})
#   %permute : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %permute_261 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute, [0, 1, 3, 2]), kwargs = {})
#   %clone_118 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_142,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_118
triton_unk_fused_matmul_backward_permute_split_view_14 = async_compile.triton('triton_unk_fused_matmul_backward_permute_split_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 3072, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_permute_split_view_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_permute_split_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 3072
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 768
    y1numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 1024
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        xoffset = x2offset
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (y0 + 2304*x2 + 2359296*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 1024*y0 + 786432*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/76/c7664vzzy2g64jpet5oteaiksotolp4sbaq2g4mrk7sddauwctcm.py
# Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
# Source node to ATen node mapping:
#   attn_weights => view_8
#   attn_weights_1 => div
#   attn_weights_2 => where
#   full => full_default
#   mask_value => full_default_1
# Graph fragment:
#   %primals_8 : Tensor "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]npu:0" = PlaceHolder[target=primals_8]
#   %bmm : Tensor "f32[48, 1024, 1024][1048576, 1024, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_8 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 12, 1024, 1024]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 8.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %div : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_8, %full_default), kwargs = {})
#   %full_default_1 : Tensor "f32[][]npu:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%primals_8, %div, %full_default_1), kwargs = {})
#   return %where
triton_unk_fused__to_copy_div_full_matmul_where_15 = async_compile.triton('triton_unk_fused__to_copy_div_full_matmul_where_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 50331648}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1048576, 48, {'divisors': [1, 1048576]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1048576, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 48, 'divisor': 1048576, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_div_full_matmul_where_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_div_full_matmul_where_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 50331648
    x_g_tile0 : tl.constexpr = (1048576) if (1048576) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (48) if (48) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 1048576
    x1numel : tl.constexpr = 48
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_1 = x0_blocks
    total_blocks = x0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        tmp0 = tl.load(in_ptr0 + x0, x0mask, eviction_policy='evict_last') != 0
        tmp1 = tl.load(in_out_ptr0 + (x0 + 1048576*x1), x0mask & x1mask)
        tmp2 = tl.full([1, 1], 0.125, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
        tmp5 = tl.where(tmp0, tmp3, tmp4)
        tl.store(in_out_ptr0 + (x0 + 1048576*x1), tmp5, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/6a/c6a7tvu6xr7hvmnfzv457o7ywjtahcogohyblahhbqa7qyqnxct5.py
# Topologically Sorted Source Nodes: [permute_259], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   permute_259 => permute_259
# Graph fragment:
#   %expand_2 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0" = PlaceHolder[target=expand_2]
#   %permute_259 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1, 1024]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_7, [0, 1, 3, 2]), kwargs = {})
#   return %expand_138
triton_unk_fused_matmul_backward_16 = async_compile.triton('triton_unk_fused_matmul_backward_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 49152, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 48, {'divisors': [1, 1024]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 48, 'divisor': 1024, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 49152
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (1024) if (1024) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (48) if (48) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 1024
    y1numel : tl.constexpr = 48
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 1024
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        xoffset = x2offset
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (y0 + 1024*x2 + 1048576*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 1024*y0 + 1048576*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/2j/c2jjecwlwabxs5lr75cro7xtwtp77kao4a6fxpaib5snc35qonaz.py
# Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3], Original ATen: [aten.matmul, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   attn_output => view_11
#   permute_3 => permute_4
#   tensor_3 => clone_3
# Graph fragment:
#   %bmm_1 : Tensor "f32[48, 1024, 64][65536, 64, 1]npu:0" = PlaceHolder[target=bmm_1]
#   %view_11 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [4, 12, 1024, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[4, 1024, 12, 64][786432, 64, 65536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_unk_fused_clone_matmul_permute_17 = async_compile.triton('triton_unk_fused_clone_matmul_permute_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 3145728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, 1024, 4, {'divisors': [1, 64, 768, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_permute_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_permute_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3145728
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (12) if (12) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (1024) if (1024) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (4) if (4) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 12
    x2numel : tl.constexpr = 1024
    x3numel : tl.constexpr = 4
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 65536*x1 + 786432*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 768*x2 + 786432*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/3m/c3mfluj4g6eeehnl3duok26tozibnjdltzbyh3gxuex7rwip6rfv.py
# Topologically Sorted Source Nodes: [view_549], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   view_549 => view_549
# Graph fragment:
#   %expand_138 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0" = PlaceHolder[target=expand_138]
#   %view_549 : Tensor "f32[48, 1024, 1024][1048576, 1, 1024]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_138, [48, 1024, 1024]), kwargs = {})
#   return %view_549
triton_unk_fused_matmul_backward_18 = async_compile.triton('triton_unk_fused_matmul_backward_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 49152, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 48, {'divisors': [1, 1024]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 48, 'divisor': 1024, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 49152
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (1024) if (1024) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (48) if (48) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 1024
    y1numel : tl.constexpr = 48
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 1024
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        xoffset = x2offset
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (x2 + 1024*y0 + 1048576*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (y0 + 1024*x2 + 1048576*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/vs/cvsu5jsq2om6bzjv6fiuwt6xgvay7rylxu2wwu66hogaek4c5ao4.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, div_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_48 => div_48
#   hidden_states_1 => add_3
#   hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %getitem_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_9]
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %buf31 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf31]
#   %getitem_12 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_12]
#   %buf33 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf33]
#   %mul_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_2]
#   %primals_11 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_11]
#   %primals_12 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_12]
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, %getitem), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_1 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_12), kwargs = {})
#   %mul_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %add_5 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_12), kwargs = {})
#   %div_48 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 768), kwargs = {})
#   return %buf31,%getitem_12,%buf33,%mul_2,%add_5,%div_48
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp4 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp6 = tl.full([1, 1], 768.0, tl.float32)
        tmp7 = (tmp4 / tmp6)
        _tmp14 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp8 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10 - tmp7
            tmp12 = tmp11 * tmp11
            tmp13 = tl.broadcast_to(tmp12, [real_block_x0, R0_BLOCK])
            tmp15 = _tmp14 + tmp13
            _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp14 = tl.sum(_tmp14, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp16 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp28 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp7
            tmp20 = tl.full([1, 1], 768.0, tl.float32)
            tmp21 = (tmp14 / tmp20)
            tmp22 = tl.full([1, 1], 1e-05, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp25, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp29, r0_mask & x0mask)
        tmp30 = tl.full([1, 1], 768.0, tl.float32)
        tmp31 = (tmp14 / tmp30)
        tmp32 = tl.full([1, 1], 1e-05, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tl.rsqrt(tmp33)
        tmp35 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp36 = tmp34 * tmp35
        tl.store(out_ptr3 + (x0), tmp36, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/jx/cjxy46hhtckh3ngjyjmepdj67vg7f34kcdiemijrh4j7vhkp6p7q.py
# Topologically Sorted Source Nodes: [x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_6
#   add_3 => add_7
#   hidden_states_3 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   x_5 => view_16
# Graph fragment:
#   %addmm_2 : Tensor "f32[4096, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_2]
#   %view_16 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [4, 1024, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_16, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_7 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_7
triton_unk_fused_add_mul_pow_tanh_view_20 = async_compile.triton('triton_unk_fused_add_mul_pow_tanh_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 12582912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [12582912, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 12582912, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_pow_tanh_view_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_mul_pow_tanh_view_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 12582912
    x_g_tile0 : tl.constexpr = (12582912) if (12582912) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 12582912
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0.5, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp0 * tmp0
        tmp4 = tmp3 * tmp0
        tmp5 = tl.full([1], 0.044715, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 + tmp6
        tmp8 = tl.full([1], 0.7978845608028654, tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = libdevice.tanh(tmp9)
        tmp11 = tl.full([1], 1.0, tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp2 * tmp12
        tl.store(out_ptr0 + (x0), tmp13, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/g2/cg2emhvi7qnsrcuijbxkjc6t5sfa7vwivmvpenzg7m6g7gyrjtsk.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_6, div_47], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_47 => div_47
#   hidden_states_1 => add_3
#   hidden_states_5 => add_8
#   hidden_states_6 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %getitem_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_9]
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %getitem_13 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_13]
#   %buf42 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf42]
#   %getitem_16 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_16]
#   %buf44 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf44]
#   %mul_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_8]
#   %primals_17 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_17]
#   %primals_18 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_18]
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, %getitem), kwargs = {})
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %getitem_13), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_15, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_16), kwargs = {})
#   %mul_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_17), kwargs = {})
#   %add_10 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_18), kwargs = {})
#   %div_47 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
#   return %buf42,%getitem_16,%buf44,%mul_8,%add_10,%div_47
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp6 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            tmp5 = tl.broadcast_to(tmp4, [real_block_x0, R0_BLOCK])
            tmp7 = _tmp6 + tmp5
            _tmp6 = tmp7
        tmp6 = tl.sum(_tmp6, 1)[:, None]
        tmp8 = tl.full([1, 1], 768.0, tl.float32)
        tmp9 = (tmp6 / tmp8)
        _tmp18 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp10 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp11 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp13 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp12 = tmp10 + tmp11
            tmp14 = tmp12 + tmp13
            tmp15 = tmp14 - tmp9
            tmp16 = tmp15 * tmp15
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp21 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp23 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp34 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp22 = tmp20 + tmp21
            tmp24 = tmp22 + tmp23
            tmp25 = tmp24 - tmp9
            tmp26 = tl.full([1, 1], 768.0, tl.float32)
            tmp27 = (tmp18 / tmp26)
            tmp28 = tl.full([1, 1], 1e-05, tl.float32)
            tmp29 = tmp27 + tmp28
            tmp30 = tl.rsqrt(tmp29)
            tmp31 = tmp25 * tmp30
            tmp33 = tmp31 * tmp32
            tmp35 = tmp33 + tmp34
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & x0mask)
        tmp36 = tl.full([1, 1], 768.0, tl.float32)
        tmp37 = (tmp18 / tmp36)
        tmp38 = tl.full([1, 1], 1e-05, tl.float32)
        tmp39 = tmp37 + tmp38
        tmp40 = tl.rsqrt(tmp39)
        tmp41 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr3 + (x0), tmp42, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/gj/cgj6n7sjb2cuw6c7xakzjniz3zxrjhyy3sp4dw4gatee7xr5v6hq.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_8, div_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_45 => div_45
#   hidden_states_1 => add_3
#   hidden_states_5 => add_8
#   hidden_states_7 => add_11
#   hidden_states_8 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %getitem_22 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_22]
#   %getitem_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_9]
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %getitem_13 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_13]
#   %buf64 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf64]
#   %getitem_25 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_25]
#   %buf66 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf66]
#   %mul_10 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_10]
#   %primals_24 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_24]
#   %primals_25 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_25]
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, %getitem), kwargs = {})
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %getitem_13), kwargs = {})
#   %add_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, %add_8), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_12 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %sub_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_25), kwargs = {})
#   %mul_10 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %primals_24), kwargs = {})
#   %add_13 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %primals_25), kwargs = {})
#   %div_45 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 768), kwargs = {})
#   return %buf64,%getitem_25,%buf66,%mul_10,%add_13,%div_45
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp8 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp4 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tmp1 + tmp2
            tmp5 = tmp3 + tmp4
            tmp6 = tmp0 + tmp5
            tmp7 = tl.broadcast_to(tmp6, [real_block_x0, R0_BLOCK])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tmp9
        tmp8 = tl.sum(_tmp8, 1)[:, None]
        tmp10 = tl.full([1, 1], 768.0, tl.float32)
        tmp11 = (tmp8 / tmp10)
        _tmp22 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp13 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp14 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp16 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tmp13 + tmp14
            tmp17 = tmp15 + tmp16
            tmp18 = tmp12 + tmp17
            tmp19 = tmp18 - tmp11
            tmp20 = tmp19 * tmp19
            tmp21 = tl.broadcast_to(tmp20, [real_block_x0, R0_BLOCK])
            tmp23 = _tmp22 + tmp21
            _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
        tmp22 = tl.sum(_tmp22, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp24 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp25 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp26 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp28 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp38 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp40 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp27 = tmp25 + tmp26
            tmp29 = tmp27 + tmp28
            tmp30 = tmp24 + tmp29
            tmp31 = tmp30 - tmp11
            tmp32 = tl.full([1, 1], 768.0, tl.float32)
            tmp33 = (tmp22 / tmp32)
            tmp34 = tl.full([1, 1], 1e-05, tl.float32)
            tmp35 = tmp33 + tmp34
            tmp36 = tl.rsqrt(tmp35)
            tmp37 = tmp31 * tmp36
            tmp39 = tmp37 * tmp38
            tmp41 = tmp39 + tmp40
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp37, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp41, r0_mask & x0mask)
        tmp42 = tl.full([1, 1], 768.0, tl.float32)
        tmp43 = (tmp22 / tmp42)
        tmp44 = tl.full([1, 1], 1e-05, tl.float32)
        tmp45 = tmp43 + tmp44
        tmp46 = tl.rsqrt(tmp45)
        tmp47 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp48 = tmp46 * tmp47
        tl.store(out_ptr3 + (x0), tmp48, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/cm/ccmvqegaf2si5gcxy57utkyag2yresoumr6yaycwj6xqucpqiwhr.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_11, hidden_states_12, div_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_44 => div_44
#   hidden_states_1 => add_3
#   hidden_states_11 => add_16
#   hidden_states_12 => add_17, add_18, mul_16, mul_17, rsqrt_4, sub_4, var_mean_4
#   hidden_states_5 => add_8
#   hidden_states_7 => add_11
# Graph fragment:
#   %getitem_22 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_22]
#   %getitem_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_9]
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %getitem_13 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_13]
#   %getitem_26 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_26]
#   %add_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_16]
#   %buf76 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf76]
#   %getitem_29 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_29]
#   %buf78 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf78]
#   %mul_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_16]
#   %primals_30 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_30]
#   %primals_31 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_31]
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, %getitem), kwargs = {})
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %getitem_13), kwargs = {})
#   %add_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, %add_8), kwargs = {})
#   %add_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %getitem_26), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_4 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_29), kwargs = {})
#   %mul_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_17 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %primals_30), kwargs = {})
#   %add_18 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %primals_31), kwargs = {})
#   %div_44 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 768), kwargs = {})
#   return %add_16,%buf76,%getitem_29,%buf78,%mul_16,%add_18,%div_44
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp7 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tmp1 + tmp2
            tmp5 = tmp3 + tmp4
            tmp6 = tmp0 + tmp5
            tmp8 = tmp6 + tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tmp11
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & x0mask)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tmp12 = tl.full([1, 1], 768.0, tl.float32)
        tmp13 = (tmp10 / tmp12)
        _tmp18 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp14 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tmp14 - tmp13
            tmp16 = tmp15 * tmp15
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp28 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp30 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp21 = tmp20 - tmp13
            tmp22 = tl.full([1, 1], 768.0, tl.float32)
            tmp23 = (tmp18 / tmp22)
            tmp24 = tl.full([1, 1], 1e-05, tl.float32)
            tmp25 = tmp23 + tmp24
            tmp26 = tl.rsqrt(tmp25)
            tmp27 = tmp21 * tmp26
            tmp29 = tmp27 * tmp28
            tmp31 = tmp29 + tmp30
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp27, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
        tmp32 = tl.full([1, 1], 768.0, tl.float32)
        tmp33 = (tmp18 / tmp32)
        tmp34 = tl.full([1, 1], 1e-05, tl.float32)
        tmp35 = tmp33 + tmp34
        tmp36 = tl.rsqrt(tmp35)
        tmp37 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp38 = tmp36 * tmp37
        tl.store(out_ptr3 + (x0), tmp38, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/kq/ckquyuzqhv4n66z7om43ibh6mia7b3nadygxejvaijf67dp6wcl5.py
# Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, attn_weights_66, permute_74, clone_52], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#   attn_weights_66 => clone_44
#   clone_52 => clone_52
#   permute_74 => permute_74
#   query_23 => permute_55
#   split_11 => split_11
#   tensor_44 => view_201
#   x_89 => view_200
# Graph fragment:
#   %addmm_44 : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm_44]
#   %view_200 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [4, 1024, 2304]), kwargs = {})
#   %split_11 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_200, 768, 2), kwargs = {})
#   %view_201 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_147, [4, 1024, 12, 64]), kwargs = {})
#   %permute_55 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_201, [0, 2, 1, 3]), kwargs = {})
#   %clone_44 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_44,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_74 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_55, [0, 1, 3, 2]), kwargs = {})
#   %clone_52 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_54,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_44,%clone_52
triton_unk_fused_matmul_matmul_backward_permute_split_view_24 = async_compile.triton('triton_unk_fused_matmul_matmul_backward_permute_split_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 3072, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 4, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_matmul_backward_permute_split_view_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_matmul_backward_permute_split_view_24(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 3072
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (64) if (64) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (4) if (4) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 64
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 1024
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_0 = y0_blocks * y1_blocks * y2_blocks
    total_blocks = y0_blocks * y1_blocks * y2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x3offset = (group_base + i) // x_cumblk_0 % x3_blocks * real_block_x3
        xoffset = x3offset
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x3mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y2offset = (group_base + i) // y_cumblk_2 % y2_blocks * real_block_y2
        y0index = y0offset + tl.arange(0, real_block_y0)[None, None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[None, :, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        y2index = y2offset + tl.arange(0, real_block_y2)[:, None, None, None]
        y2 = y2index
        y2mask = y2index < y2numel
        ymask = y0mask & y1mask & y2mask
        y4 = y0 + 64*y1
        y4mask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (y0 + 64*y1 + 2304*x3 + 2359296*y2), x3mask & y0mask & y1mask & y2mask)
        tmp1 = tl.load(in_ptr0 + (y4 + 2304*x3 + 2359296*y2), x3mask & y2mask & y4mask)
        tl.store(out_ptr0 + (y0 + 64*x3 + 65536*y1 + 786432*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 1024*y4 + 786432*y2), tmp1, x3mask & y2mask & y4mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/4l/c4lnaywg73vvu6gfjf64syfdjlqs7j4mxgfbgxkx72fsf3gpymlf.py
# Topologically Sorted Source Nodes: [x_89, split_11, tensor_45, key_23, transpose_11, attn_weights_66, permute_73, clone_51], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#   attn_weights_66 => clone_45
#   clone_51 => clone_51
#   key_23 => permute_56
#   permute_73 => permute_73
#   split_11 => split_11
#   tensor_45 => view_202
#   transpose_11 => permute_58
#   x_89 => view_200
# Graph fragment:
#   %addmm_44 : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm_44]
#   %view_200 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [4, 1024, 2304]), kwargs = {})
#   %split_11 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_200, 768, 2), kwargs = {})
#   %view_202 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_148, [4, 1024, 12, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_202, [0, 2, 1, 3]), kwargs = {})
#   %permute_58 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_56, [0, 1, 3, 2]), kwargs = {})
#   %clone_45 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_45,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_73 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_58, [0, 1, 3, 2]), kwargs = {})
#   %clone_51 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_53,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_45,%clone_51
triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25 = async_compile.triton('triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 49152, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 12, 4, {'divisors': [1, 1024, 12288]}], 'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 1024, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 12288, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 49152
    xnumel = 64
    y_g_tile0 : tl.constexpr = (1024) if (1024) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (4) if (4) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 1024
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 64
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_0 = y0_blocks * y1_blocks * y2_blocks
    total_blocks = y0_blocks * y1_blocks * y2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x3offset = (group_base + i) // x_cumblk_0 % x3_blocks * real_block_x3
        xoffset = x3offset
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x3mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y2offset = (group_base + i) // y_cumblk_2 % y2_blocks * real_block_y2
        y0index = y0offset + tl.arange(0, real_block_y0)[None, None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[None, :, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        y2index = y2offset + tl.arange(0, real_block_y2)[:, None, None, None]
        y2 = y2index
        y2mask = y2index < y2numel
        ymask = y0mask & y1mask & y2mask
        tmp0 = tl.load(in_ptr0 + (768 + x3 + 64*y1 + 2304*y0 + 2359296*y2), x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr0 + (y0 + 1024*x3 + 65536*y1 + 786432*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 64*y0 + 65536*y1 + 786432*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/al/calkll3hhboosirjy34btgs62ldt2ro7vbzjazzltce7iamirdmu.py
# Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33, permute_71, clone_49], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#   attn_output_33 => clone_46
#   clone_49 => clone_49
#   permute_71 => permute_71
#   split_11 => split_11
#   tensor_46 => view_203
#   value_23 => permute_57
#   x_89 => view_200
# Graph fragment:
#   %addmm_44 : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm_44]
#   %view_200 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [4, 1024, 2304]), kwargs = {})
#   %split_11 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_200, 768, 2), kwargs = {})
#   %view_203 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_149, [4, 1024, 12, 64]), kwargs = {})
#   %permute_57 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1, 3]), kwargs = {})
#   %clone_46 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_47,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_71 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_57, [0, 1, 3, 2]), kwargs = {})
#   %clone_49 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_49,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_46,%clone_49
triton_unk_fused_matmul_matmul_backward_permute_split_view_26 = async_compile.triton('triton_unk_fused_matmul_matmul_backward_permute_split_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'y': 3072, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 4, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_matmul_backward_permute_split_view_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_matmul_backward_permute_split_view_26(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 3072
    xnumel = 1024
    y_g_tile0 : tl.constexpr = (64) if (64) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (4) if (4) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 64
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 1024
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_0 = y0_blocks * y1_blocks * y2_blocks
    total_blocks = y0_blocks * y1_blocks * y2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x3offset = (group_base + i) // x_cumblk_0 % x3_blocks * real_block_x3
        xoffset = x3offset
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x3mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y2offset = (group_base + i) // y_cumblk_2 % y2_blocks * real_block_y2
        y0index = y0offset + tl.arange(0, real_block_y0)[None, None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[None, :, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        y2index = y2offset + tl.arange(0, real_block_y2)[:, None, None, None]
        y2 = y2index
        y2mask = y2index < y2numel
        ymask = y0mask & y1mask & y2mask
        y4 = y0 + 64*y1
        y4mask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (1536 + y0 + 64*y1 + 2304*x3 + 2359296*y2), x3mask & y0mask & y1mask & y2mask)
        tmp1 = tl.load(in_ptr0 + (1536 + y4 + 2304*x3 + 2359296*y2), x3mask & y2mask & y4mask)
        tl.store(out_ptr0 + (y0 + 64*x3 + 65536*y1 + 786432*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 1024*y4 + 786432*y2), tmp1, x3mask & y2mask & y4mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/x3/cx3sfhqupavhq6km2ml5mfpw5ft7piejjajkqdpsdxdkhixvfpxl.py
# Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_67, hidden_states_71, layer_norm_24, div_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_14 => div_14
#   hidden_states_61 => add_83
#   hidden_states_65 => add_88
#   hidden_states_67 => add_91
#   hidden_states_71 => add_96
#   layer_norm_24 => add_97, add_98, mul_96, mul_97, rsqrt_24, sub_24, var_mean_24
# Graph fragment:
#   %getitem_152 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_152]
#   %getitem_139 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_139]
#   %add_80 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_80]
#   %getitem_143 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_143]
#   %getitem_156 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_156]
#   %add_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_96]
#   %buf411 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf411]
#   %getitem_159 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_159]
#   %buf413 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf413]
#   %mul_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_96]
#   %primals_160 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_160]
#   %primals_161 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_161]
#   %add_83 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_139, %add_80), kwargs = {})
#   %add_88 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %getitem_143), kwargs = {})
#   %add_91 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_152, %add_88), kwargs = {})
#   %add_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %getitem_156), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_97 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_158, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_97,), kwargs = {})
#   %sub_24 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_96, %getitem_159), kwargs = {})
#   %mul_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %primals_160), kwargs = {})
#   %add_98 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %primals_161), kwargs = {})
#   %div_14 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_24, 768), kwargs = {})
#   return %add_96,%buf411,%getitem_159,%buf413,%mul_96,%add_98,%div_14
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp7 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tmp1 + tmp2
            tmp5 = tmp3 + tmp4
            tmp6 = tmp0 + tmp5
            tmp8 = tmp6 + tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tmp11
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & x0mask)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tmp12 = tl.full([1, 1], 768.0, tl.float32)
        tmp13 = (tmp10 / tmp12)
        _tmp18 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp14 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tmp14 - tmp13
            tmp16 = tmp15 * tmp15
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp28 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp30 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp21 = tmp20 - tmp13
            tmp22 = tl.full([1, 1], 768.0, tl.float32)
            tmp23 = (tmp18 / tmp22)
            tmp24 = tl.full([1, 1], 1e-05, tl.float32)
            tmp25 = tmp23 + tmp24
            tmp26 = tl.rsqrt(tmp25)
            tmp27 = tmp21 * tmp26
            tmp29 = tmp27 * tmp28
            tmp31 = tmp29 + tmp30
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp27, r0_mask & x0mask)
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
        tmp32 = tl.full([1, 1], 768.0, tl.float32)
        tmp33 = (tmp18 / tmp32)
        tmp34 = tl.full([1, 1], 1e-05, tl.float32)
        tmp35 = tmp33 + tmp34
        tmp36 = tl.rsqrt(tmp35)
        tmp37 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp38 = tmp36 * tmp37
        tl.store(out_ptr2 + (x0), tmp38, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/6b/c6bz5dphpwv4obbanf6rgyfthm3kos47kogqgwzkttiayyyxlxhu.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_13, div_12, full_default_25, ne, neg, squeeze, sum_1, sum_2, where_13
# Graph fragment:
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %gather : Tensor "f32[4, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %convert_element_type_13 : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type_13]
#   %ne : Tensor "b8[4][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_163, -100), kwargs = {})
#   %squeeze : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_13 : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_25), kwargs = {})
#   %sum_1 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %sum_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_13,), kwargs = {})
#   %div_12 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %convert_element_type_13), kwargs = {})
#   return %convert_element_type_13,%div_12
triton_unk_fused_nll_loss_forward_28 = async_compile.triton('triton_unk_fused_nll_loss_forward_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import npu_triton_helpers
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': []},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1
    group_size = 1 if group_id < 1 else 0
    group_base = group_id
    for i in range(group_size):
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tmp0
        tmp5 = tl.load(in_ptr0 + (1))
        tmp6 = tmp5
        tmp10 = tl.load(in_ptr0 + (2))
        tmp11 = tmp10
        tmp15 = tl.load(in_ptr0 + (3))
        tmp16 = tmp15
        tmp21 = tl.load(in_ptr1 + (0))
        tmp22 = tmp21
        tmp26 = tl.load(in_ptr1 + (1))
        tmp27 = tmp26
        tmp31 = tl.load(in_ptr1 + (2))
        tmp32 = tmp31
        tmp36 = tl.load(in_ptr1 + (3))
        tmp37 = tmp36
        tmp2 = tl.full([], -100, tl.int32)
        tmp3 = tmp1 != tmp2
        tmp4 = tmp3.to(tl.int32)
        tmp7 = tmp6 != tmp2
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tmp4 + tmp8
        tmp12 = tmp11 != tmp2
        tmp13 = tmp12.to(tl.int32)
        tmp14 = tmp9 + tmp13
        tmp17 = tmp16 != tmp2
        tmp18 = tmp17.to(tl.int32)
        tmp19 = tmp14 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp23 = -tmp22
        tmp24 = tl.full([], 0.0, tl.float32)
        tmp25 = tl.where(tmp3, tmp23, tmp24)
        tmp28 = -tmp27
        tmp29 = tl.where(tmp7, tmp28, tmp24)
        tmp30 = tmp25 + tmp29
        tmp33 = -tmp32
        tmp34 = tl.where(tmp12, tmp33, tmp24)
        tmp35 = tmp30 + tmp34
        tmp38 = -tmp37
        tmp39 = tl.where(tmp17, tmp38, tmp24)
        tmp40 = tmp35 + tmp39
        tmp41 = (tmp40 / tmp20)
        tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp20, None)
        tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp41, None)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((1024, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_0.run(buf0, 1024, stream=raw_stream0)
            primals_1 = copy_if_misaligned(primals_1)
            # Topologically Sorted Source Nodes: [inputs_embeds], Original ATen: [aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:1
            buf1 = torch.ops.aten.embedding.default(primals_2, primals_1)
            del primals_2
            buf417 = empty_strided_npu((4, 1024), (1024, 1), torch.int32)
            # Topologically Sorted Source Nodes: [eq, int_1], Original ATen: [aten.eq, aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_eq_1.run(primals_1, buf417, 4096, stream=raw_stream0)
            buf421 = empty_strided_npu((4, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [arange_1], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_2.run(buf421, 4, stream=raw_stream0)
            primals_163 = copy_if_misaligned(primals_163)
            buf426 = empty_strided_npu((4, ), (1, ), torch.int64)
            buf430 = empty_strided_npu((4, 1), (1, 1), torch.bool)
            # Topologically Sorted Source Nodes: [loss, unsqueeze_2, ne_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_backward_nll_loss_forward_3.run(primals_163, buf426, buf430, 4, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [position_ids, position_ids_1, position_embeds], Original ATen: [aten.arange, aten.unsqueeze, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf3 = torch.ops.aten.embedding.default(primals_3, reinterpret_tensor(buf0, (1, 1024), (0, 1), 0))
            del primals_3
            buf2 = buf1
            assert_alignment(buf2, 16, 'torch.ops.aten.embedding.default')
            del buf1
            # Topologically Sorted Source Nodes: [eq, int_1, argmax], Original ATen: [aten.eq, aten._to_copy, aten.argmax]
            # [Provenance debug handles] torch.ops.aten.argmax.default:3
            buf418 = torch.ops.aten.argmax.default(buf417, -1)
            del buf417
            buf431 = empty_strided_npu((4, 2), (2, 1), torch.int64)
            # Topologically Sorted Source Nodes: [loss, unsqueeze_2, where_14, ], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward, aten.expand]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_expand_nll_loss_backward_nll_loss_forward_4.run(buf430, primals_163, buf431, 8, stream=raw_stream0)
            buf4 = buf3
            assert_alignment(buf4, 16, 'torch.ops.aten.embedding.default')
            del buf3
            buf5 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_5.run(buf5, buf4, 3145728, stream=raw_stream0)
            del buf4
            buf419 = buf418
            assert_alignment(buf419, 16, 'torch.ops.aten.argmax.default')
            del buf418
            buf432 = empty_strided_npu((4, 2), (2, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.arange, aten.view, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_eq_view_6.run(buf431, buf432, 8, stream=raw_stream0)
            del buf431
            # Topologically Sorted Source Nodes: [add, residual], Original ATen: [aten.add, aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:4
            buf6 = torch.ops.npu._npu_dropout.default(buf5, 0.1)
            buf420 = buf419; del buf419  # reuse
            # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sub_7.run(buf420, 4, stream=raw_stream0)
            buf7 = buf6[0]
            assert_alignment(buf7, 16, 'torch.ops.npu._npu_dropout.default')
            buf8 = buf6[1]
            assert_alignment(buf8, 16, 'torch.ops.npu._npu_dropout.default')
            del buf6
            buf12 = buf5; del buf5  # reuse
            buf13 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf517 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states, div_50], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_native_layer_norm_backward_8.run(buf7, primals_4, primals_5, buf12, buf13, buf517, 4096, 768, stream=raw_stream0)
            del primals_5
            buf14 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states, view_1, x], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(primals_6, reinterpret_tensor(buf13, (4096, 768), (768, 1), 0), primals_7, alpha=1, beta=1, out=buf14)
            del primals_6
            buf15 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf14, buf15, 3145728, stream=raw_stream0)
            buf16 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf14, buf16, 3072, 1024, stream=raw_stream0)
            buf24 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf14, buf24, 3145728, stream=raw_stream0)
            buf512 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, permute_258, clone_115], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf14, buf512, 3072, 1024, stream=raw_stream0)
            buf515 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, permute_260, clone_117], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf14, buf515, 3145728, stream=raw_stream0)
            buf516 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, permute_261, clone_118], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf14, buf516, 3072, 1024, stream=raw_stream0)
            buf17 = empty_strided_npu((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:6
            extern_kernels.bmm(reinterpret_tensor(buf15, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf16, (48, 64, 1024), (65536, 1024, 1), 0), out=buf17)
            buf18 = reinterpret_tensor(buf17, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf18, primals_8, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2, attn_weights_3], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:7
            buf19 = torch.ops.aten._softmax.default(buf18, -1, False)
            buf20 = buf19
            assert_alignment(buf20, 16, 'torch.ops.aten._softmax.default')
            del buf19
            # Unsorted Source Nodes: [attn_weights_5], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:8
            buf21 = torch.ops.npu._npu_dropout.default(buf20, 0.1)
            buf22 = buf21[0]
            assert_alignment(buf22, 16, 'torch.ops.npu._npu_dropout.default')
            buf23 = buf21[1]
            assert_alignment(buf23, 16, 'torch.ops.npu._npu_dropout.default')
            del buf21
            buf25 = reinterpret_tensor(buf16, (48, 1024, 64), (65536, 64, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:9
            extern_kernels.bmm(reinterpret_tensor(buf22, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf24, (48, 1024, 64), (65536, 64, 1), 0), out=buf25)
            buf513 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [permute_259], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf22, buf513, 49152, 1024, stream=raw_stream0)
            buf26 = reinterpret_tensor(buf24, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf25, buf26, 3145728, stream=raw_stream0)
            buf514 = reinterpret_tensor(buf22, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [view_549], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf513, buf514, 49152, 1024, stream=raw_stream0)
            buf27 = reinterpret_tensor(buf25, (4096, 768), (768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3, attn_output_1, view_7, x_2], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:10
            extern_kernels.addmm(primals_9, reinterpret_tensor(buf26, (4096, 768), (768, 1), 0), primals_10, alpha=1, beta=1, out=buf27)
            del primals_9
            # Unsorted Source Nodes: [attn_output_2, x_3], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:11
            buf28 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf27, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf29 = buf28[0]
            assert_alignment(buf29, 16, 'torch.ops.npu._npu_dropout.default')
            buf34 = reinterpret_tensor(buf27, (4, 1024, 768), (786432, 768, 1), 0); del buf27  # reuse
            buf35 = reinterpret_tensor(buf15, (4, 1024, 768), (786432, 768, 1), 0); del buf15  # reuse
            buf511 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, div_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf29, buf7, primals_11, primals_12, buf34, buf35, buf511, 4096, 768, stream=raw_stream0)
            del primals_12
            buf30 = buf28[1]
            assert_alignment(buf30, 16, 'torch.ops.npu._npu_dropout.default')
            del buf28
            buf36 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_2, view_9, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(primals_13, reinterpret_tensor(buf35, (4096, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf36)
            del primals_13
            buf37 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf36, buf37, 12582912, stream=raw_stream0)
            buf38 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3, view_11, x_6], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:13
            extern_kernels.addmm(primals_15, reinterpret_tensor(buf37, (4096, 3072), (3072, 1), 0), primals_16, alpha=1, beta=1, out=buf38)
            del primals_15
            # Unsorted Source Nodes: [hidden_states_4, x_7], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:14
            buf39 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf38, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf40 = buf39[0]
            assert_alignment(buf40, 16, 'torch.ops.npu._npu_dropout.default')
            buf45 = reinterpret_tensor(buf38, (4, 1024, 768), (786432, 768, 1), 0); del buf38  # reuse
            buf46 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf510 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_6, div_47], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf29, buf7, buf40, primals_17, primals_18, buf45, buf46, buf510, 4096, 768, stream=raw_stream0)
            del primals_18
            buf41 = buf39[1]
            assert_alignment(buf41, 16, 'torch.ops.npu._npu_dropout.default')
            del buf39
            buf47 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_6, view_13, x_8], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:15
            extern_kernels.addmm(primals_19, reinterpret_tensor(buf46, (4096, 768), (768, 1), 0), primals_20, alpha=1, beta=1, out=buf47)
            del primals_19
            buf48 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf47, buf48, 3145728, stream=raw_stream0)
            buf49 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_5, key_3, transpose_1, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf47, buf49, 3072, 1024, stream=raw_stream0)
            buf57 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, attn_output_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf47, buf57, 3145728, stream=raw_stream0)
            buf505 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, permute_241, clone_109], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf47, buf505, 3072, 1024, stream=raw_stream0)
            buf508 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_5, key_3, transpose_1, permute_243, clone_111], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf47, buf508, 3145728, stream=raw_stream0)
            buf509 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, permute_244, clone_112], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf47, buf509, 3072, 1024, stream=raw_stream0)
            buf50 = reinterpret_tensor(buf513, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, tensor_5, key_3, transpose_1, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:16
            extern_kernels.bmm(reinterpret_tensor(buf48, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf49, (48, 64, 1024), (65536, 1024, 1), 0), out=buf50)
            buf51 = reinterpret_tensor(buf50, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_6, attn_weights_7, attn_weights_8], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf51, primals_21, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_6, attn_weights_7, attn_weights_8, attn_weights_9], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:17
            buf52 = torch.ops.aten._softmax.default(buf51, -1, False)
            buf53 = buf52
            assert_alignment(buf53, 16, 'torch.ops.aten._softmax.default')
            del buf52
            # Unsorted Source Nodes: [attn_weights_11], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:18
            buf54 = torch.ops.npu._npu_dropout.default(buf53, 0.1)
            buf55 = buf54[0]
            assert_alignment(buf55, 16, 'torch.ops.npu._npu_dropout.default')
            buf56 = buf54[1]
            assert_alignment(buf56, 16, 'torch.ops.npu._npu_dropout.default')
            del buf54
            buf58 = reinterpret_tensor(buf49, (48, 1024, 64), (65536, 64, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, attn_output_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:19
            extern_kernels.bmm(reinterpret_tensor(buf55, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf57, (48, 1024, 64), (65536, 64, 1), 0), out=buf58)
            buf506 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [permute_242], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf55, buf506, 49152, 1024, stream=raw_stream0)
            buf59 = reinterpret_tensor(buf57, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf58, buf59, 3145728, stream=raw_stream0)
            buf507 = reinterpret_tensor(buf55, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [view_521], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf506, buf507, 49152, 1024, stream=raw_stream0)
            buf60 = reinterpret_tensor(buf58, (4096, 768), (768, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7, attn_output_4, view_19, x_10], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:20
            extern_kernels.addmm(primals_22, reinterpret_tensor(buf59, (4096, 768), (768, 1), 0), primals_23, alpha=1, beta=1, out=buf60)
            del primals_22
            # Unsorted Source Nodes: [attn_output_5, x_11], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:21
            buf61 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf60, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf62 = buf61[0]
            assert_alignment(buf62, 16, 'torch.ops.npu._npu_dropout.default')
            buf67 = reinterpret_tensor(buf60, (4, 1024, 768), (786432, 768, 1), 0); del buf60  # reuse
            buf68 = reinterpret_tensor(buf48, (4, 1024, 768), (786432, 768, 1), 0); del buf48  # reuse
            buf504 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_8, div_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf62, buf29, buf7, buf40, primals_24, primals_25, buf67, buf68, buf504, 4096, 768, stream=raw_stream0)
            del primals_25
            buf63 = buf61[1]
            assert_alignment(buf63, 16, 'torch.ops.npu._npu_dropout.default')
            del buf61
            buf69 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_8, view_21, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(primals_26, reinterpret_tensor(buf68, (4096, 768), (768, 1), 0), primals_27, alpha=1, beta=1, out=buf69)
            del primals_26
            buf70 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf69, buf70, 12582912, stream=raw_stream0)
            buf71 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9, view_23, x_14], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:23
            extern_kernels.addmm(primals_28, reinterpret_tensor(buf70, (4096, 3072), (3072, 1), 0), primals_29, alpha=1, beta=1, out=buf71)
            del primals_28
            # Unsorted Source Nodes: [hidden_states_10, x_15], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:24
            buf72 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf71, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf73 = buf72[0]
            assert_alignment(buf73, 16, 'torch.ops.npu._npu_dropout.default')
            buf75 = buf29; del buf29  # reuse
            buf79 = reinterpret_tensor(buf71, (4, 1024, 768), (786432, 768, 1), 0); del buf71  # reuse
            buf80 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf503 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_11, hidden_states_12, div_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf75, buf62, buf7, buf40, buf73, primals_30, primals_31, buf79, buf80, buf503, 4096, 768, stream=raw_stream0)
            del primals_31
            buf74 = buf72[1]
            assert_alignment(buf74, 16, 'torch.ops.npu._npu_dropout.default')
            del buf72
            buf81 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_12, view_25, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(primals_32, reinterpret_tensor(buf80, (4096, 768), (768, 1), 0), primals_33, alpha=1, beta=1, out=buf81)
            del primals_32
            buf82 = reinterpret_tensor(buf73, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf81, buf82, 3145728, stream=raw_stream0)
            buf83 = reinterpret_tensor(buf7, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_9, key_5, transpose_2, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf81, buf83, 3072, 1024, stream=raw_stream0)
            buf91 = reinterpret_tensor(buf62, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, attn_output_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf81, buf91, 3145728, stream=raw_stream0)
            buf498 = reinterpret_tensor(buf40, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, permute_224, clone_103], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf81, buf498, 3072, 1024, stream=raw_stream0)
            buf501 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_9, key_5, transpose_2, permute_226, clone_105], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf81, buf501, 3145728, stream=raw_stream0)
            buf502 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, permute_227, clone_106], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf81, buf502, 3072, 1024, stream=raw_stream0)
            buf84 = reinterpret_tensor(buf506, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf506  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, tensor_9, key_5, transpose_2, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:26
            extern_kernels.bmm(reinterpret_tensor(buf82, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf83, (48, 64, 1024), (65536, 1024, 1), 0), out=buf84)
            buf85 = reinterpret_tensor(buf84, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_12, attn_weights_13, attn_weights_14], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf85, primals_34, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_12, attn_weights_13, attn_weights_14, attn_weights_15], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:27
            buf86 = torch.ops.aten._softmax.default(buf85, -1, False)
            buf87 = buf86
            assert_alignment(buf87, 16, 'torch.ops.aten._softmax.default')
            del buf86
            # Unsorted Source Nodes: [attn_weights_17], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:28
            buf88 = torch.ops.npu._npu_dropout.default(buf87, 0.1)
            buf89 = buf88[0]
            assert_alignment(buf89, 16, 'torch.ops.npu._npu_dropout.default')
            buf90 = buf88[1]
            assert_alignment(buf90, 16, 'torch.ops.npu._npu_dropout.default')
            del buf88
            buf92 = reinterpret_tensor(buf83, (48, 1024, 64), (65536, 64, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, attn_output_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:29
            extern_kernels.bmm(reinterpret_tensor(buf89, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf91, (48, 1024, 64), (65536, 64, 1), 0), out=buf92)
            buf499 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [permute_225], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf89, buf499, 49152, 1024, stream=raw_stream0)
            buf93 = reinterpret_tensor(buf91, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf92, buf93, 3145728, stream=raw_stream0)
            buf500 = reinterpret_tensor(buf89, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [view_493], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf499, buf500, 49152, 1024, stream=raw_stream0)
            buf94 = reinterpret_tensor(buf92, (4096, 768), (768, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11, attn_output_7, view_31, x_18], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:30
            extern_kernels.addmm(primals_35, reinterpret_tensor(buf93, (4096, 768), (768, 1), 0), primals_36, alpha=1, beta=1, out=buf94)
            del primals_35
            # Unsorted Source Nodes: [attn_output_8, x_19], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:31
            buf95 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf94, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf96 = buf95[0]
            assert_alignment(buf96, 16, 'torch.ops.npu._npu_dropout.default')
            buf101 = reinterpret_tensor(buf94, (4, 1024, 768), (786432, 768, 1), 0); del buf94  # reuse
            buf102 = reinterpret_tensor(buf82, (4, 1024, 768), (786432, 768, 1), 0); del buf82  # reuse
            buf497 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14, div_42], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf96, buf75, primals_37, primals_38, buf101, buf102, buf497, 4096, 768, stream=raw_stream0)
            del primals_38
            buf97 = buf95[1]
            assert_alignment(buf97, 16, 'torch.ops.npu._npu_dropout.default')
            del buf95
            buf103 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_14, view_33, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:32
            extern_kernels.addmm(primals_39, reinterpret_tensor(buf102, (4096, 768), (768, 1), 0), primals_40, alpha=1, beta=1, out=buf103)
            del primals_39
            buf104 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf103, buf104, 12582912, stream=raw_stream0)
            buf105 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15, view_35, x_22], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:33
            extern_kernels.addmm(primals_41, reinterpret_tensor(buf104, (4096, 3072), (3072, 1), 0), primals_42, alpha=1, beta=1, out=buf105)
            del primals_41
            # Unsorted Source Nodes: [hidden_states_16, x_23], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:34
            buf106 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf105, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf107 = buf106[0]
            assert_alignment(buf107, 16, 'torch.ops.npu._npu_dropout.default')
            buf112 = reinterpret_tensor(buf105, (4, 1024, 768), (786432, 768, 1), 0); del buf105  # reuse
            buf113 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf496 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_18, div_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf96, buf75, buf107, primals_43, primals_44, buf112, buf113, buf496, 4096, 768, stream=raw_stream0)
            del primals_44
            buf108 = buf106[1]
            assert_alignment(buf108, 16, 'torch.ops.npu._npu_dropout.default')
            del buf106
            buf114 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_18, view_37, x_24], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:35
            extern_kernels.addmm(primals_45, reinterpret_tensor(buf113, (4096, 768), (768, 1), 0), primals_46, alpha=1, beta=1, out=buf114)
            del primals_45
            buf115 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf114, buf115, 3145728, stream=raw_stream0)
            buf116 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_13, key_7, transpose_3, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf114, buf116, 3072, 1024, stream=raw_stream0)
            buf124 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, attn_output_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf114, buf124, 3145728, stream=raw_stream0)
            buf491 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, permute_207, clone_97], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf114, buf491, 3072, 1024, stream=raw_stream0)
            buf494 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_13, key_7, transpose_3, permute_209, clone_99], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf114, buf494, 3145728, stream=raw_stream0)
            buf495 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, permute_210, clone_100], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf114, buf495, 3072, 1024, stream=raw_stream0)
            buf117 = reinterpret_tensor(buf499, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf499  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, tensor_13, key_7, transpose_3, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:36
            extern_kernels.bmm(reinterpret_tensor(buf115, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf116, (48, 64, 1024), (65536, 1024, 1), 0), out=buf117)
            buf118 = reinterpret_tensor(buf117, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_18, attn_weights_19, attn_weights_20], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf118, primals_47, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_18, attn_weights_19, attn_weights_20, attn_weights_21], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:37
            buf119 = torch.ops.aten._softmax.default(buf118, -1, False)
            buf120 = buf119
            assert_alignment(buf120, 16, 'torch.ops.aten._softmax.default')
            del buf119
            # Unsorted Source Nodes: [attn_weights_23], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:38
            buf121 = torch.ops.npu._npu_dropout.default(buf120, 0.1)
            buf122 = buf121[0]
            assert_alignment(buf122, 16, 'torch.ops.npu._npu_dropout.default')
            buf123 = buf121[1]
            assert_alignment(buf123, 16, 'torch.ops.npu._npu_dropout.default')
            del buf121
            buf125 = reinterpret_tensor(buf116, (48, 1024, 64), (65536, 64, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, attn_output_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf122, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf124, (48, 1024, 64), (65536, 64, 1), 0), out=buf125)
            buf492 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [permute_208], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf122, buf492, 49152, 1024, stream=raw_stream0)
            buf126 = reinterpret_tensor(buf124, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf125, buf126, 3145728, stream=raw_stream0)
            buf493 = reinterpret_tensor(buf122, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [view_465], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf492, buf493, 49152, 1024, stream=raw_stream0)
            buf127 = reinterpret_tensor(buf125, (4096, 768), (768, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15, attn_output_10, view_43, x_26], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:40
            extern_kernels.addmm(primals_48, reinterpret_tensor(buf126, (4096, 768), (768, 1), 0), primals_49, alpha=1, beta=1, out=buf127)
            del primals_48
            # Unsorted Source Nodes: [attn_output_11, x_27], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:41
            buf128 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf127, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf129 = buf128[0]
            assert_alignment(buf129, 16, 'torch.ops.npu._npu_dropout.default')
            buf134 = reinterpret_tensor(buf127, (4, 1024, 768), (786432, 768, 1), 0); del buf127  # reuse
            buf135 = reinterpret_tensor(buf115, (4, 1024, 768), (786432, 768, 1), 0); del buf115  # reuse
            buf490 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_19, hidden_states_20, div_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf129, buf96, buf75, buf107, primals_50, primals_51, buf134, buf135, buf490, 4096, 768, stream=raw_stream0)
            del primals_51
            buf130 = buf128[1]
            assert_alignment(buf130, 16, 'torch.ops.npu._npu_dropout.default')
            del buf128
            buf136 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_20, view_45, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:42
            extern_kernels.addmm(primals_52, reinterpret_tensor(buf135, (4096, 768), (768, 1), 0), primals_53, alpha=1, beta=1, out=buf136)
            del primals_52
            buf137 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf136, buf137, 12582912, stream=raw_stream0)
            buf138 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21, view_47, x_30], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:43
            extern_kernels.addmm(primals_54, reinterpret_tensor(buf137, (4096, 3072), (3072, 1), 0), primals_55, alpha=1, beta=1, out=buf138)
            del primals_54
            # Unsorted Source Nodes: [hidden_states_22, x_31], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:44
            buf139 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf138, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf140 = buf139[0]
            assert_alignment(buf140, 16, 'torch.ops.npu._npu_dropout.default')
            buf142 = buf96; del buf96  # reuse
            buf146 = reinterpret_tensor(buf138, (4, 1024, 768), (786432, 768, 1), 0); del buf138  # reuse
            buf147 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf489 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_19, hidden_states_23, hidden_states_24, div_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf142, buf129, buf75, buf107, buf140, primals_56, primals_57, buf146, buf147, buf489, 4096, 768, stream=raw_stream0)
            del primals_57
            buf141 = buf139[1]
            assert_alignment(buf141, 16, 'torch.ops.npu._npu_dropout.default')
            del buf139
            buf148 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_24, view_49, x_32], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:45
            extern_kernels.addmm(primals_58, reinterpret_tensor(buf147, (4096, 768), (768, 1), 0), primals_59, alpha=1, beta=1, out=buf148)
            del primals_58
            buf149 = reinterpret_tensor(buf75, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf148, buf149, 3145728, stream=raw_stream0)
            buf150 = reinterpret_tensor(buf140, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_17, key_9, transpose_4, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf148, buf150, 3072, 1024, stream=raw_stream0)
            buf158 = reinterpret_tensor(buf129, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, attn_output_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf148, buf158, 3145728, stream=raw_stream0)
            buf484 = reinterpret_tensor(buf107, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, permute_190, clone_91], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf148, buf484, 3072, 1024, stream=raw_stream0)
            buf487 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_17, key_9, transpose_4, permute_192, clone_93], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf148, buf487, 3145728, stream=raw_stream0)
            buf488 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, permute_193, clone_94], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf148, buf488, 3072, 1024, stream=raw_stream0)
            buf151 = reinterpret_tensor(buf492, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf492  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, tensor_17, key_9, transpose_4, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:46
            extern_kernels.bmm(reinterpret_tensor(buf149, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf150, (48, 64, 1024), (65536, 1024, 1), 0), out=buf151)
            buf152 = reinterpret_tensor(buf151, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_24, attn_weights_25, attn_weights_26], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf152, primals_60, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_24, attn_weights_25, attn_weights_26, attn_weights_27], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:47
            buf153 = torch.ops.aten._softmax.default(buf152, -1, False)
            buf154 = buf153
            assert_alignment(buf154, 16, 'torch.ops.aten._softmax.default')
            del buf153
            # Unsorted Source Nodes: [attn_weights_29], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:48
            buf155 = torch.ops.npu._npu_dropout.default(buf154, 0.1)
            buf156 = buf155[0]
            assert_alignment(buf156, 16, 'torch.ops.npu._npu_dropout.default')
            buf157 = buf155[1]
            assert_alignment(buf157, 16, 'torch.ops.npu._npu_dropout.default')
            del buf155
            buf159 = reinterpret_tensor(buf150, (48, 1024, 64), (65536, 64, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, attn_output_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:49
            extern_kernels.bmm(reinterpret_tensor(buf156, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf158, (48, 1024, 64), (65536, 64, 1), 0), out=buf159)
            buf485 = buf152; del buf152  # reuse
            # Topologically Sorted Source Nodes: [permute_191], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf156, buf485, 49152, 1024, stream=raw_stream0)
            buf160 = reinterpret_tensor(buf158, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf159, buf160, 3145728, stream=raw_stream0)
            buf486 = reinterpret_tensor(buf156, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [view_437], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf485, buf486, 49152, 1024, stream=raw_stream0)
            buf161 = reinterpret_tensor(buf159, (4096, 768), (768, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19, attn_output_13, view_55, x_34], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:50
            extern_kernels.addmm(primals_61, reinterpret_tensor(buf160, (4096, 768), (768, 1), 0), primals_62, alpha=1, beta=1, out=buf161)
            del primals_61
            # Unsorted Source Nodes: [attn_output_14, x_35], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:51
            buf162 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf161, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf163 = buf162[0]
            assert_alignment(buf163, 16, 'torch.ops.npu._npu_dropout.default')
            buf168 = reinterpret_tensor(buf161, (4, 1024, 768), (786432, 768, 1), 0); del buf161  # reuse
            buf169 = reinterpret_tensor(buf149, (4, 1024, 768), (786432, 768, 1), 0); del buf149  # reuse
            buf483 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, div_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf163, buf142, primals_63, primals_64, buf168, buf169, buf483, 4096, 768, stream=raw_stream0)
            del primals_64
            buf164 = buf162[1]
            assert_alignment(buf164, 16, 'torch.ops.npu._npu_dropout.default')
            del buf162
            buf170 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_26, view_57, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(primals_65, reinterpret_tensor(buf169, (4096, 768), (768, 1), 0), primals_66, alpha=1, beta=1, out=buf170)
            del primals_65
            buf171 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf170, buf171, 12582912, stream=raw_stream0)
            buf172 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27, view_59, x_38], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:53
            extern_kernels.addmm(primals_67, reinterpret_tensor(buf171, (4096, 3072), (3072, 1), 0), primals_68, alpha=1, beta=1, out=buf172)
            del primals_67
            # Unsorted Source Nodes: [hidden_states_28, x_39], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:54
            buf173 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf172, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf174 = buf173[0]
            assert_alignment(buf174, 16, 'torch.ops.npu._npu_dropout.default')
            buf179 = reinterpret_tensor(buf172, (4, 1024, 768), (786432, 768, 1), 0); del buf172  # reuse
            buf180 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf482 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_30, div_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf163, buf142, buf174, primals_69, primals_70, buf179, buf180, buf482, 4096, 768, stream=raw_stream0)
            del primals_70
            buf175 = buf173[1]
            assert_alignment(buf175, 16, 'torch.ops.npu._npu_dropout.default')
            del buf173
            buf181 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_30, view_61, x_40], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:55
            extern_kernels.addmm(primals_71, reinterpret_tensor(buf180, (4096, 768), (768, 1), 0), primals_72, alpha=1, beta=1, out=buf181)
            del primals_71
            buf182 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf181, buf182, 3145728, stream=raw_stream0)
            buf183 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_21, key_11, transpose_5, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf181, buf183, 3072, 1024, stream=raw_stream0)
            buf191 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, attn_output_15], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf181, buf191, 3145728, stream=raw_stream0)
            buf477 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, permute_173, clone_85], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf181, buf477, 3072, 1024, stream=raw_stream0)
            buf480 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_21, key_11, transpose_5, permute_175, clone_87], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf181, buf480, 3145728, stream=raw_stream0)
            buf481 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, permute_176, clone_88], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf181, buf481, 3072, 1024, stream=raw_stream0)
            buf184 = reinterpret_tensor(buf485, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf485  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, tensor_21, key_11, transpose_5, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:56
            extern_kernels.bmm(reinterpret_tensor(buf182, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf183, (48, 64, 1024), (65536, 1024, 1), 0), out=buf184)
            buf185 = reinterpret_tensor(buf184, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_30, attn_weights_31, attn_weights_32], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf185, primals_73, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_30, attn_weights_31, attn_weights_32, attn_weights_33], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:57
            buf186 = torch.ops.aten._softmax.default(buf185, -1, False)
            buf187 = buf186
            assert_alignment(buf187, 16, 'torch.ops.aten._softmax.default')
            del buf186
            # Unsorted Source Nodes: [attn_weights_35], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:58
            buf188 = torch.ops.npu._npu_dropout.default(buf187, 0.1)
            buf189 = buf188[0]
            assert_alignment(buf189, 16, 'torch.ops.npu._npu_dropout.default')
            buf190 = buf188[1]
            assert_alignment(buf190, 16, 'torch.ops.npu._npu_dropout.default')
            del buf188
            buf192 = reinterpret_tensor(buf183, (48, 1024, 64), (65536, 64, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, attn_output_15], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:59
            extern_kernels.bmm(reinterpret_tensor(buf189, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf191, (48, 1024, 64), (65536, 64, 1), 0), out=buf192)
            buf478 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [permute_174], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf189, buf478, 49152, 1024, stream=raw_stream0)
            buf193 = reinterpret_tensor(buf191, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf192, buf193, 3145728, stream=raw_stream0)
            buf479 = reinterpret_tensor(buf189, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [view_409], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf478, buf479, 49152, 1024, stream=raw_stream0)
            buf194 = reinterpret_tensor(buf192, (4096, 768), (768, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23, attn_output_16, view_67, x_42], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:60
            extern_kernels.addmm(primals_74, reinterpret_tensor(buf193, (4096, 768), (768, 1), 0), primals_75, alpha=1, beta=1, out=buf194)
            del primals_74
            # Unsorted Source Nodes: [attn_output_17, x_43], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:61
            buf195 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf194, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf196 = buf195[0]
            assert_alignment(buf196, 16, 'torch.ops.npu._npu_dropout.default')
            buf201 = reinterpret_tensor(buf194, (4, 1024, 768), (786432, 768, 1), 0); del buf194  # reuse
            buf202 = reinterpret_tensor(buf182, (4, 1024, 768), (786432, 768, 1), 0); del buf182  # reuse
            buf476 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_31, hidden_states_32, div_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf196, buf163, buf142, buf174, primals_76, primals_77, buf201, buf202, buf476, 4096, 768, stream=raw_stream0)
            del primals_77
            buf197 = buf195[1]
            assert_alignment(buf197, 16, 'torch.ops.npu._npu_dropout.default')
            del buf195
            buf203 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_32, view_69, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:62
            extern_kernels.addmm(primals_78, reinterpret_tensor(buf202, (4096, 768), (768, 1), 0), primals_79, alpha=1, beta=1, out=buf203)
            del primals_78
            buf204 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf203, buf204, 12582912, stream=raw_stream0)
            buf205 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33, view_71, x_46], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:63
            extern_kernels.addmm(primals_80, reinterpret_tensor(buf204, (4096, 3072), (3072, 1), 0), primals_81, alpha=1, beta=1, out=buf205)
            del primals_80
            # Unsorted Source Nodes: [hidden_states_34, x_47], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:64
            buf206 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf205, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf207 = buf206[0]
            assert_alignment(buf207, 16, 'torch.ops.npu._npu_dropout.default')
            buf209 = buf163; del buf163  # reuse
            buf213 = reinterpret_tensor(buf205, (4, 1024, 768), (786432, 768, 1), 0); del buf205  # reuse
            buf214 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf475 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_31, hidden_states_35, hidden_states_36, div_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf209, buf196, buf142, buf174, buf207, primals_82, primals_83, buf213, buf214, buf475, 4096, 768, stream=raw_stream0)
            del primals_83
            buf208 = buf206[1]
            assert_alignment(buf208, 16, 'torch.ops.npu._npu_dropout.default')
            del buf206
            buf215 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_36, view_73, x_48], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(primals_84, reinterpret_tensor(buf214, (4096, 768), (768, 1), 0), primals_85, alpha=1, beta=1, out=buf215)
            del primals_84
            buf216 = reinterpret_tensor(buf207, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf215, buf216, 3145728, stream=raw_stream0)
            buf217 = reinterpret_tensor(buf196, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_25, key_13, transpose_6, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf215, buf217, 3072, 1024, stream=raw_stream0)
            buf225 = reinterpret_tensor(buf174, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, attn_output_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf215, buf225, 3145728, stream=raw_stream0)
            buf470 = reinterpret_tensor(buf142, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, permute_156, clone_79], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf215, buf470, 3072, 1024, stream=raw_stream0)
            buf473 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_25, key_13, transpose_6, permute_158, clone_81], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf215, buf473, 3145728, stream=raw_stream0)
            buf474 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, permute_159, clone_82], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf215, buf474, 3072, 1024, stream=raw_stream0)
            buf218 = reinterpret_tensor(buf478, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf478  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, tensor_25, key_13, transpose_6, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:66
            extern_kernels.bmm(reinterpret_tensor(buf216, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf217, (48, 64, 1024), (65536, 1024, 1), 0), out=buf218)
            buf219 = reinterpret_tensor(buf218, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_36, attn_weights_37, attn_weights_38], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf219, primals_86, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_36, attn_weights_37, attn_weights_38, attn_weights_39], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:67
            buf220 = torch.ops.aten._softmax.default(buf219, -1, False)
            buf221 = buf220
            assert_alignment(buf221, 16, 'torch.ops.aten._softmax.default')
            del buf220
            # Unsorted Source Nodes: [attn_weights_41], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:68
            buf222 = torch.ops.npu._npu_dropout.default(buf221, 0.1)
            buf223 = buf222[0]
            assert_alignment(buf223, 16, 'torch.ops.npu._npu_dropout.default')
            buf224 = buf222[1]
            assert_alignment(buf224, 16, 'torch.ops.npu._npu_dropout.default')
            del buf222
            buf226 = reinterpret_tensor(buf217, (48, 1024, 64), (65536, 64, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, attn_output_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:69
            extern_kernels.bmm(reinterpret_tensor(buf223, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf225, (48, 1024, 64), (65536, 64, 1), 0), out=buf226)
            buf471 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [permute_157], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf223, buf471, 49152, 1024, stream=raw_stream0)
            buf227 = reinterpret_tensor(buf225, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf226, buf227, 3145728, stream=raw_stream0)
            buf472 = reinterpret_tensor(buf223, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [view_381], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf471, buf472, 49152, 1024, stream=raw_stream0)
            buf228 = reinterpret_tensor(buf226, (4096, 768), (768, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27, attn_output_19, view_79, x_50], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:70
            extern_kernels.addmm(primals_87, reinterpret_tensor(buf227, (4096, 768), (768, 1), 0), primals_88, alpha=1, beta=1, out=buf228)
            del primals_87
            # Unsorted Source Nodes: [attn_output_20, x_51], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:71
            buf229 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf228, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf230 = buf229[0]
            assert_alignment(buf230, 16, 'torch.ops.npu._npu_dropout.default')
            buf235 = reinterpret_tensor(buf228, (4, 1024, 768), (786432, 768, 1), 0); del buf228  # reuse
            buf236 = reinterpret_tensor(buf216, (4, 1024, 768), (786432, 768, 1), 0); del buf216  # reuse
            buf469 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38, div_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf230, buf209, primals_89, primals_90, buf235, buf236, buf469, 4096, 768, stream=raw_stream0)
            del primals_90
            buf231 = buf229[1]
            assert_alignment(buf231, 16, 'torch.ops.npu._npu_dropout.default')
            del buf229
            buf237 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_38, view_81, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:72
            extern_kernels.addmm(primals_91, reinterpret_tensor(buf236, (4096, 768), (768, 1), 0), primals_92, alpha=1, beta=1, out=buf237)
            del primals_91
            buf238 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf237, buf238, 12582912, stream=raw_stream0)
            buf239 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39, view_83, x_54], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:73
            extern_kernels.addmm(primals_93, reinterpret_tensor(buf238, (4096, 3072), (3072, 1), 0), primals_94, alpha=1, beta=1, out=buf239)
            del primals_93
            # Unsorted Source Nodes: [hidden_states_40, x_55], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:74
            buf240 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf239, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf241 = buf240[0]
            assert_alignment(buf241, 16, 'torch.ops.npu._npu_dropout.default')
            buf246 = reinterpret_tensor(buf239, (4, 1024, 768), (786432, 768, 1), 0); del buf239  # reuse
            buf247 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf468 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_42, div_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf230, buf209, buf241, primals_95, primals_96, buf246, buf247, buf468, 4096, 768, stream=raw_stream0)
            del primals_96
            buf242 = buf240[1]
            assert_alignment(buf242, 16, 'torch.ops.npu._npu_dropout.default')
            del buf240
            buf248 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_42, view_85, x_56], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:75
            extern_kernels.addmm(primals_97, reinterpret_tensor(buf247, (4096, 768), (768, 1), 0), primals_98, alpha=1, beta=1, out=buf248)
            del primals_97
            buf249 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf248, buf249, 3145728, stream=raw_stream0)
            buf250 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_29, key_15, transpose_7, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf248, buf250, 3072, 1024, stream=raw_stream0)
            buf258 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, attn_output_21], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf248, buf258, 3145728, stream=raw_stream0)
            buf463 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, permute_139, clone_73], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf248, buf463, 3072, 1024, stream=raw_stream0)
            buf466 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_29, key_15, transpose_7, permute_141, clone_75], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf248, buf466, 3145728, stream=raw_stream0)
            buf467 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, permute_142, clone_76], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf248, buf467, 3072, 1024, stream=raw_stream0)
            buf251 = reinterpret_tensor(buf471, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf471  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, tensor_29, key_15, transpose_7, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:76
            extern_kernels.bmm(reinterpret_tensor(buf249, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf250, (48, 64, 1024), (65536, 1024, 1), 0), out=buf251)
            buf252 = reinterpret_tensor(buf251, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_42, attn_weights_43, attn_weights_44], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf252, primals_99, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_42, attn_weights_43, attn_weights_44, attn_weights_45], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:77
            buf253 = torch.ops.aten._softmax.default(buf252, -1, False)
            buf254 = buf253
            assert_alignment(buf254, 16, 'torch.ops.aten._softmax.default')
            del buf253
            # Unsorted Source Nodes: [attn_weights_47], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:78
            buf255 = torch.ops.npu._npu_dropout.default(buf254, 0.1)
            buf256 = buf255[0]
            assert_alignment(buf256, 16, 'torch.ops.npu._npu_dropout.default')
            buf257 = buf255[1]
            assert_alignment(buf257, 16, 'torch.ops.npu._npu_dropout.default')
            del buf255
            buf259 = reinterpret_tensor(buf250, (48, 1024, 64), (65536, 64, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, attn_output_21], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:79
            extern_kernels.bmm(reinterpret_tensor(buf256, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf258, (48, 1024, 64), (65536, 64, 1), 0), out=buf259)
            buf464 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [permute_140], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf256, buf464, 49152, 1024, stream=raw_stream0)
            buf260 = reinterpret_tensor(buf258, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf259, buf260, 3145728, stream=raw_stream0)
            buf465 = reinterpret_tensor(buf256, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf256  # reuse
            # Topologically Sorted Source Nodes: [view_353], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf464, buf465, 49152, 1024, stream=raw_stream0)
            buf261 = reinterpret_tensor(buf259, (4096, 768), (768, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31, attn_output_22, view_91, x_58], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:80
            extern_kernels.addmm(primals_100, reinterpret_tensor(buf260, (4096, 768), (768, 1), 0), primals_101, alpha=1, beta=1, out=buf261)
            del primals_100
            # Unsorted Source Nodes: [attn_output_23, x_59], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:81
            buf262 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf261, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf263 = buf262[0]
            assert_alignment(buf263, 16, 'torch.ops.npu._npu_dropout.default')
            buf268 = reinterpret_tensor(buf261, (4, 1024, 768), (786432, 768, 1), 0); del buf261  # reuse
            buf269 = reinterpret_tensor(buf249, (4, 1024, 768), (786432, 768, 1), 0); del buf249  # reuse
            buf462 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_43, hidden_states_44, div_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf263, buf230, buf209, buf241, primals_102, primals_103, buf268, buf269, buf462, 4096, 768, stream=raw_stream0)
            del primals_103
            buf264 = buf262[1]
            assert_alignment(buf264, 16, 'torch.ops.npu._npu_dropout.default')
            del buf262
            buf270 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_44, view_93, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:82
            extern_kernels.addmm(primals_104, reinterpret_tensor(buf269, (4096, 768), (768, 1), 0), primals_105, alpha=1, beta=1, out=buf270)
            del primals_104
            buf271 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf270, buf271, 12582912, stream=raw_stream0)
            buf272 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45, view_95, x_62], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:83
            extern_kernels.addmm(primals_106, reinterpret_tensor(buf271, (4096, 3072), (3072, 1), 0), primals_107, alpha=1, beta=1, out=buf272)
            del primals_106
            # Unsorted Source Nodes: [hidden_states_46, x_63], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:84
            buf273 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf272, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf274 = buf273[0]
            assert_alignment(buf274, 16, 'torch.ops.npu._npu_dropout.default')
            buf276 = buf230; del buf230  # reuse
            buf280 = reinterpret_tensor(buf272, (4, 1024, 768), (786432, 768, 1), 0); del buf272  # reuse
            buf281 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf461 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_43, hidden_states_47, hidden_states_48, div_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf276, buf263, buf209, buf241, buf274, primals_108, primals_109, buf280, buf281, buf461, 4096, 768, stream=raw_stream0)
            del primals_109
            buf275 = buf273[1]
            assert_alignment(buf275, 16, 'torch.ops.npu._npu_dropout.default')
            del buf273
            buf282 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_48, view_97, x_64], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:85
            extern_kernels.addmm(primals_110, reinterpret_tensor(buf281, (4096, 768), (768, 1), 0), primals_111, alpha=1, beta=1, out=buf282)
            del primals_110
            buf283 = reinterpret_tensor(buf274, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf282, buf283, 3145728, stream=raw_stream0)
            buf284 = reinterpret_tensor(buf263, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_33, key_17, transpose_8, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf282, buf284, 3072, 1024, stream=raw_stream0)
            buf292 = reinterpret_tensor(buf241, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, attn_output_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf282, buf292, 3145728, stream=raw_stream0)
            buf456 = reinterpret_tensor(buf209, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, permute_122, clone_67], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf282, buf456, 3072, 1024, stream=raw_stream0)
            buf459 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_33, key_17, transpose_8, permute_124, clone_69], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf282, buf459, 3145728, stream=raw_stream0)
            buf460 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, permute_125, clone_70], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf282, buf460, 3072, 1024, stream=raw_stream0)
            buf285 = reinterpret_tensor(buf464, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf464  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, tensor_33, key_17, transpose_8, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:86
            extern_kernels.bmm(reinterpret_tensor(buf283, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf284, (48, 64, 1024), (65536, 1024, 1), 0), out=buf285)
            buf286 = reinterpret_tensor(buf285, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_48, attn_weights_49, attn_weights_50], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf286, primals_112, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_48, attn_weights_49, attn_weights_50, attn_weights_51], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:87
            buf287 = torch.ops.aten._softmax.default(buf286, -1, False)
            buf288 = buf287
            assert_alignment(buf288, 16, 'torch.ops.aten._softmax.default')
            del buf287
            # Unsorted Source Nodes: [attn_weights_53], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:88
            buf289 = torch.ops.npu._npu_dropout.default(buf288, 0.1)
            buf290 = buf289[0]
            assert_alignment(buf290, 16, 'torch.ops.npu._npu_dropout.default')
            buf291 = buf289[1]
            assert_alignment(buf291, 16, 'torch.ops.npu._npu_dropout.default')
            del buf289
            buf293 = reinterpret_tensor(buf284, (48, 1024, 64), (65536, 64, 1), 0); del buf284  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, attn_output_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:89
            extern_kernels.bmm(reinterpret_tensor(buf290, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf292, (48, 1024, 64), (65536, 64, 1), 0), out=buf293)
            buf457 = buf286; del buf286  # reuse
            # Topologically Sorted Source Nodes: [permute_123], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf290, buf457, 49152, 1024, stream=raw_stream0)
            buf294 = reinterpret_tensor(buf292, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf293, buf294, 3145728, stream=raw_stream0)
            buf458 = reinterpret_tensor(buf290, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [view_325], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf457, buf458, 49152, 1024, stream=raw_stream0)
            buf295 = reinterpret_tensor(buf293, (4096, 768), (768, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35, attn_output_25, view_103, x_66], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:90
            extern_kernels.addmm(primals_113, reinterpret_tensor(buf294, (4096, 768), (768, 1), 0), primals_114, alpha=1, beta=1, out=buf295)
            del primals_113
            # Unsorted Source Nodes: [attn_output_26, x_67], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:91
            buf296 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf295, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf297 = buf296[0]
            assert_alignment(buf297, 16, 'torch.ops.npu._npu_dropout.default')
            buf302 = reinterpret_tensor(buf295, (4, 1024, 768), (786432, 768, 1), 0); del buf295  # reuse
            buf303 = reinterpret_tensor(buf283, (4, 1024, 768), (786432, 768, 1), 0); del buf283  # reuse
            buf455 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50, div_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf297, buf276, primals_115, primals_116, buf302, buf303, buf455, 4096, 768, stream=raw_stream0)
            del primals_116
            buf298 = buf296[1]
            assert_alignment(buf298, 16, 'torch.ops.npu._npu_dropout.default')
            del buf296
            buf304 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_50, view_105, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:92
            extern_kernels.addmm(primals_117, reinterpret_tensor(buf303, (4096, 768), (768, 1), 0), primals_118, alpha=1, beta=1, out=buf304)
            del primals_117
            buf305 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf304, buf305, 12582912, stream=raw_stream0)
            buf306 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51, view_107, x_70], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:93
            extern_kernels.addmm(primals_119, reinterpret_tensor(buf305, (4096, 3072), (3072, 1), 0), primals_120, alpha=1, beta=1, out=buf306)
            del primals_119
            # Unsorted Source Nodes: [hidden_states_52, x_71], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:94
            buf307 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf306, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf308 = buf307[0]
            assert_alignment(buf308, 16, 'torch.ops.npu._npu_dropout.default')
            buf313 = reinterpret_tensor(buf306, (4, 1024, 768), (786432, 768, 1), 0); del buf306  # reuse
            buf314 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf454 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_54, div_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf297, buf276, buf308, primals_121, primals_122, buf313, buf314, buf454, 4096, 768, stream=raw_stream0)
            del primals_122
            buf309 = buf307[1]
            assert_alignment(buf309, 16, 'torch.ops.npu._npu_dropout.default')
            del buf307
            buf315 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_54, view_109, x_72], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:95
            extern_kernels.addmm(primals_123, reinterpret_tensor(buf314, (4096, 768), (768, 1), 0), primals_124, alpha=1, beta=1, out=buf315)
            del primals_123
            buf316 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf315, buf316, 3145728, stream=raw_stream0)
            buf317 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_37, key_19, transpose_9, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf315, buf317, 3072, 1024, stream=raw_stream0)
            buf325 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, attn_output_27], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf315, buf325, 3145728, stream=raw_stream0)
            buf449 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, permute_105, clone_61], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf315, buf449, 3072, 1024, stream=raw_stream0)
            buf452 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_37, key_19, transpose_9, permute_107, clone_63], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf315, buf452, 3145728, stream=raw_stream0)
            buf453 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, permute_108, clone_64], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf315, buf453, 3072, 1024, stream=raw_stream0)
            buf318 = reinterpret_tensor(buf457, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf457  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, tensor_37, key_19, transpose_9, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:96
            extern_kernels.bmm(reinterpret_tensor(buf316, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf317, (48, 64, 1024), (65536, 1024, 1), 0), out=buf318)
            buf319 = reinterpret_tensor(buf318, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_54, attn_weights_55, attn_weights_56], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf319, primals_125, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_54, attn_weights_55, attn_weights_56, attn_weights_57], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:97
            buf320 = torch.ops.aten._softmax.default(buf319, -1, False)
            buf321 = buf320
            assert_alignment(buf321, 16, 'torch.ops.aten._softmax.default')
            del buf320
            # Unsorted Source Nodes: [attn_weights_59], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:98
            buf322 = torch.ops.npu._npu_dropout.default(buf321, 0.1)
            buf323 = buf322[0]
            assert_alignment(buf323, 16, 'torch.ops.npu._npu_dropout.default')
            buf324 = buf322[1]
            assert_alignment(buf324, 16, 'torch.ops.npu._npu_dropout.default')
            del buf322
            buf326 = reinterpret_tensor(buf317, (48, 1024, 64), (65536, 64, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, attn_output_27], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:99
            extern_kernels.bmm(reinterpret_tensor(buf323, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf325, (48, 1024, 64), (65536, 64, 1), 0), out=buf326)
            buf450 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [permute_106], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf323, buf450, 49152, 1024, stream=raw_stream0)
            buf327 = reinterpret_tensor(buf325, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf325  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf326, buf327, 3145728, stream=raw_stream0)
            buf451 = reinterpret_tensor(buf323, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [view_297], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf450, buf451, 49152, 1024, stream=raw_stream0)
            buf328 = reinterpret_tensor(buf326, (4096, 768), (768, 1), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39, attn_output_28, view_115, x_74], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:100
            extern_kernels.addmm(primals_126, reinterpret_tensor(buf327, (4096, 768), (768, 1), 0), primals_127, alpha=1, beta=1, out=buf328)
            del primals_126
            # Unsorted Source Nodes: [attn_output_29, x_75], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:101
            buf329 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf328, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf330 = buf329[0]
            assert_alignment(buf330, 16, 'torch.ops.npu._npu_dropout.default')
            buf335 = reinterpret_tensor(buf328, (4, 1024, 768), (786432, 768, 1), 0); del buf328  # reuse
            buf336 = reinterpret_tensor(buf316, (4, 1024, 768), (786432, 768, 1), 0); del buf316  # reuse
            buf448 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_55, hidden_states_56, div_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf330, buf297, buf276, buf308, primals_128, primals_129, buf335, buf336, buf448, 4096, 768, stream=raw_stream0)
            del primals_129
            buf331 = buf329[1]
            assert_alignment(buf331, 16, 'torch.ops.npu._npu_dropout.default')
            del buf329
            buf337 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_56, view_117, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:102
            extern_kernels.addmm(primals_130, reinterpret_tensor(buf336, (4096, 768), (768, 1), 0), primals_131, alpha=1, beta=1, out=buf337)
            del primals_130
            buf338 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf337, buf338, 12582912, stream=raw_stream0)
            buf339 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57, view_119, x_78], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:103
            extern_kernels.addmm(primals_132, reinterpret_tensor(buf338, (4096, 3072), (3072, 1), 0), primals_133, alpha=1, beta=1, out=buf339)
            del primals_132
            # Unsorted Source Nodes: [hidden_states_58, x_79], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:104
            buf340 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf339, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf341 = buf340[0]
            assert_alignment(buf341, 16, 'torch.ops.npu._npu_dropout.default')
            buf343 = buf297; del buf297  # reuse
            buf347 = reinterpret_tensor(buf339, (4, 1024, 768), (786432, 768, 1), 0); del buf339  # reuse
            buf348 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf447 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_55, hidden_states_59, hidden_states_60, div_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf343, buf330, buf276, buf308, buf341, primals_134, primals_135, buf347, buf348, buf447, 4096, 768, stream=raw_stream0)
            del primals_135
            buf342 = buf340[1]
            assert_alignment(buf342, 16, 'torch.ops.npu._npu_dropout.default')
            del buf340
            buf349 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_60, view_121, x_80], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:105
            extern_kernels.addmm(primals_136, reinterpret_tensor(buf348, (4096, 768), (768, 1), 0), primals_137, alpha=1, beta=1, out=buf349)
            del primals_136
            buf350 = reinterpret_tensor(buf341, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf349, buf350, 3145728, stream=raw_stream0)
            buf351 = reinterpret_tensor(buf330, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_41, key_21, transpose_10, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf349, buf351, 3072, 1024, stream=raw_stream0)
            buf359 = reinterpret_tensor(buf308, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf308  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, attn_output_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf349, buf359, 3145728, stream=raw_stream0)
            buf442 = reinterpret_tensor(buf276, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, permute_88, clone_55], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf349, buf442, 3072, 1024, stream=raw_stream0)
            buf445 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_41, key_21, transpose_10, permute_90, clone_57], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf349, buf445, 3145728, stream=raw_stream0)
            buf446 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, permute_91, clone_58], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf349, buf446, 3072, 1024, stream=raw_stream0)
            buf352 = reinterpret_tensor(buf450, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf450  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, tensor_41, key_21, transpose_10, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:106
            extern_kernels.bmm(reinterpret_tensor(buf350, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf351, (48, 64, 1024), (65536, 1024, 1), 0), out=buf352)
            buf353 = reinterpret_tensor(buf352, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_60, attn_weights_61, attn_weights_62], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf353, primals_138, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_60, attn_weights_61, attn_weights_62, attn_weights_63], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:107
            buf354 = torch.ops.aten._softmax.default(buf353, -1, False)
            buf355 = buf354
            assert_alignment(buf355, 16, 'torch.ops.aten._softmax.default')
            del buf354
            # Unsorted Source Nodes: [attn_weights_65], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:108
            buf356 = torch.ops.npu._npu_dropout.default(buf355, 0.1)
            buf357 = buf356[0]
            assert_alignment(buf357, 16, 'torch.ops.npu._npu_dropout.default')
            buf358 = buf356[1]
            assert_alignment(buf358, 16, 'torch.ops.npu._npu_dropout.default')
            del buf356
            buf360 = reinterpret_tensor(buf351, (48, 1024, 64), (65536, 64, 1), 0); del buf351  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, attn_output_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:109
            extern_kernels.bmm(reinterpret_tensor(buf357, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf359, (48, 1024, 64), (65536, 64, 1), 0), out=buf360)
            buf443 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [permute_89], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf357, buf443, 49152, 1024, stream=raw_stream0)
            buf361 = reinterpret_tensor(buf359, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf360, buf361, 3145728, stream=raw_stream0)
            buf444 = reinterpret_tensor(buf357, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf357  # reuse
            # Topologically Sorted Source Nodes: [view_269], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf443, buf444, 49152, 1024, stream=raw_stream0)
            buf362 = reinterpret_tensor(buf360, (4096, 768), (768, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43, attn_output_31, view_127, x_82], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:110
            extern_kernels.addmm(primals_139, reinterpret_tensor(buf361, (4096, 768), (768, 1), 0), primals_140, alpha=1, beta=1, out=buf362)
            del primals_139
            # Unsorted Source Nodes: [attn_output_32, x_83], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:111
            buf363 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf362, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf364 = buf363[0]
            assert_alignment(buf364, 16, 'torch.ops.npu._npu_dropout.default')
            buf369 = reinterpret_tensor(buf362, (4, 1024, 768), (786432, 768, 1), 0); del buf362  # reuse
            buf370 = reinterpret_tensor(buf350, (4, 1024, 768), (786432, 768, 1), 0); del buf350  # reuse
            buf441 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, div_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf364, buf343, primals_141, primals_142, buf369, buf370, buf441, 4096, 768, stream=raw_stream0)
            del primals_142
            buf365 = buf363[1]
            assert_alignment(buf365, 16, 'torch.ops.npu._npu_dropout.default')
            del buf363
            buf371 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_62, view_129, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:112
            extern_kernels.addmm(primals_143, reinterpret_tensor(buf370, (4096, 768), (768, 1), 0), primals_144, alpha=1, beta=1, out=buf371)
            del primals_143
            buf372 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf371, buf372, 12582912, stream=raw_stream0)
            buf373 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63, view_131, x_86], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:113
            extern_kernels.addmm(primals_145, reinterpret_tensor(buf372, (4096, 3072), (3072, 1), 0), primals_146, alpha=1, beta=1, out=buf373)
            del primals_145
            # Unsorted Source Nodes: [hidden_states_64, x_87], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:114
            buf374 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf373, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf375 = buf374[0]
            assert_alignment(buf375, 16, 'torch.ops.npu._npu_dropout.default')
            buf380 = reinterpret_tensor(buf373, (4, 1024, 768), (786432, 768, 1), 0); del buf373  # reuse
            buf381 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf440 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_66, div_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf364, buf343, buf375, primals_147, primals_148, buf380, buf381, buf440, 4096, 768, stream=raw_stream0)
            del primals_148
            buf376 = buf374[1]
            assert_alignment(buf376, 16, 'torch.ops.npu._npu_dropout.default')
            del buf374
            buf382 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_66, view_133, x_88], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:115
            extern_kernels.addmm(primals_149, reinterpret_tensor(buf381, (4096, 768), (768, 1), 0), primals_150, alpha=1, beta=1, out=buf382)
            del primals_149
            buf383 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            buf439 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, attn_weights_66, permute_74, clone_52], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_view_24.run(buf382, buf383, buf439, 3072, 1024, stream=raw_stream0)
            buf384 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            buf438 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_45, key_23, transpose_11, attn_weights_66, permute_73, clone_51], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25.run(buf382, buf384, buf438, 49152, 64, stream=raw_stream0)
            buf392 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            buf435 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33, permute_71, clone_49], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_view_26.run(buf382, buf392, buf435, 3072, 1024, stream=raw_stream0)
            buf385 = reinterpret_tensor(buf443, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf443  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, tensor_45, key_23, transpose_11, attn_weights_66], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:116
            extern_kernels.bmm(reinterpret_tensor(buf383, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf384, (48, 64, 1024), (65536, 1024, 1), 0), out=buf385)
            del buf383
            buf386 = reinterpret_tensor(buf385, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf385  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_66, attn_weights_67, attn_weights_68], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf386, primals_151, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_66, attn_weights_67, attn_weights_68, attn_weights_69], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:117
            buf387 = torch.ops.aten._softmax.default(buf386, -1, False)
            del buf386
            buf388 = buf387
            assert_alignment(buf388, 16, 'torch.ops.aten._softmax.default')
            del buf387
            # Unsorted Source Nodes: [attn_weights_71], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:118
            buf389 = torch.ops.npu._npu_dropout.default(buf388, 0.1)
            buf390 = buf389[0]
            assert_alignment(buf390, 16, 'torch.ops.npu._npu_dropout.default')
            buf391 = buf389[1]
            assert_alignment(buf391, 16, 'torch.ops.npu._npu_dropout.default')
            del buf389
            buf393 = reinterpret_tensor(buf384, (48, 1024, 64), (65536, 64, 1), 0); del buf384  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:119
            extern_kernels.bmm(reinterpret_tensor(buf390, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf392, (48, 1024, 64), (65536, 64, 1), 0), out=buf393)
            del buf392
            buf436 = empty_strided_npu((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_72], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf390, buf436, 49152, 1024, stream=raw_stream0)
            del buf390
            buf394 = empty_strided_npu((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf393, buf394, 3145728, stream=raw_stream0)
            del buf393
            buf437 = empty_strided_npu((48, 1024, 1024), (1048576, 1, 1024), torch.float32)
            # Topologically Sorted Source Nodes: [view_241], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf436, buf437, 49152, 1024, stream=raw_stream0)
            del buf436
            buf395 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47, attn_output_34, view_139, x_90], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:120
            extern_kernels.addmm(primals_152, reinterpret_tensor(buf394, (4096, 768), (768, 1), 0), primals_153, alpha=1, beta=1, out=buf395)
            del primals_152
            # Unsorted Source Nodes: [attn_output_35, x_91], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:121
            buf396 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf395, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf397 = buf396[0]
            assert_alignment(buf397, 16, 'torch.ops.npu._npu_dropout.default')
            buf402 = reinterpret_tensor(buf395, (4, 1024, 768), (786432, 768, 1), 0); del buf395  # reuse
            buf403 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf434 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_67, hidden_states_68, div_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf397, buf364, buf343, buf375, primals_154, primals_155, buf402, buf403, buf434, 4096, 768, stream=raw_stream0)
            del primals_155
            buf398 = buf396[1]
            assert_alignment(buf398, 16, 'torch.ops.npu._npu_dropout.default')
            del buf396
            buf404 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_68, view_141, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:122
            extern_kernels.addmm(primals_156, reinterpret_tensor(buf403, (4096, 768), (768, 1), 0), primals_157, alpha=1, beta=1, out=buf404)
            del primals_156
            buf405 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf404, buf405, 12582912, stream=raw_stream0)
            buf406 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69, view_143, x_94], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:123
            extern_kernels.addmm(primals_158, reinterpret_tensor(buf405, (4096, 3072), (3072, 1), 0), primals_159, alpha=1, beta=1, out=buf406)
            del primals_158
            # Unsorted Source Nodes: [hidden_states_70, x_95], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:124
            buf407 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf406, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf408 = buf407[0]
            assert_alignment(buf408, 16, 'torch.ops.npu._npu_dropout.default')
            buf410 = buf364; del buf364  # reuse
            buf414 = buf410; del buf410  # reuse
            buf415 = reinterpret_tensor(buf406, (4, 1024, 768), (786432, 768, 1), 0); del buf406  # reuse
            buf433 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_67, hidden_states_71, layer_norm_24, div_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf414, buf397, buf343, buf375, buf408, primals_160, primals_161, buf415, buf433, 4096, 768, stream=raw_stream0)
            del buf343
            del buf375
            del buf397
            del buf408
            del primals_161
            buf409 = buf407[1]
            assert_alignment(buf409, 16, 'torch.ops.npu._npu_dropout.default')
            del buf407
            buf416 = empty_strided_npu((4096, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_24, logits], Original ATen: [aten.native_layer_norm, aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:125
            extern_kernels.mm(reinterpret_tensor(buf415, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_162, (768, 2), (1, 768), 0), out=buf416)
            # Topologically Sorted Source Nodes: [logits, pooled_logits], Original ATen: [aten.matmul, aten.index]
            # [Provenance debug handles] torch.ops.aten.index.Tensor:126
            buf422 = torch.ops.aten.index.Tensor(reinterpret_tensor(buf416, (4, 1024, 2), (2048, 2, 1), 0), [buf421, buf420])
            del buf416
            buf423 = buf422
            assert_alignment(buf423, 16, 'torch.ops.aten.index.Tensor')
            del buf422
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:127
            buf424 = torch.ops.aten._log_softmax.default(buf423, 1, False)
            buf425 = buf424
            assert_alignment(buf425, 16, 'torch.ops.aten._log_softmax.default')
            del buf424
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:128
            buf427 = torch.ops.aten.gather.default(buf425, 1, reinterpret_tensor(buf426, (4, 1), (1, 0), 0))
            del buf426
            buf428 = buf427
            assert_alignment(buf428, 16, 'torch.ops.aten.gather.default')
            del buf427
            buf429 = empty_strided_npu((), (), torch.float32)
            buf518 = empty_strided_npu((), (), torch.float32)
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_28.run(primals_163, buf428, buf429, buf518, 1, stream=raw_stream0)
            del buf428
            del primals_163
        return (buf518, buf423, reinterpret_tensor(buf14, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf14, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf47, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf47, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf81, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf81, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf114, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf114, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf148, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf148, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf181, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf181, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf215, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf215, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf248, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf248, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf282, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf282, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf315, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf315, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf349, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf349, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf382, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf382, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), primals_1, primals_4, primals_7, primals_8, primals_10, primals_11, primals_14, primals_16, primals_17, primals_20, primals_21, primals_23, primals_24, primals_27, primals_29, primals_30, primals_33, primals_34, primals_36, primals_37, primals_40, primals_42, primals_43, primals_46, primals_47, primals_49, primals_50, primals_53, primals_55, primals_56, primals_59, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_86, primals_88, primals_89, primals_92, primals_94, primals_95, primals_98, primals_99, primals_101, primals_102, primals_105, primals_107, primals_108, primals_111, primals_112, primals_114, primals_115, primals_118, primals_120, primals_121, primals_124, primals_125, primals_127, primals_128, primals_131, primals_133, primals_134, primals_137, primals_138, primals_140, primals_141, primals_144, primals_146, primals_147, primals_150, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_162, reinterpret_tensor(buf0, (1, 1024), (1024, 1), 0), buf8, buf12, buf20, buf23, buf30, buf34, buf36, buf41, buf45, buf53, buf56, buf63, buf67, buf69, buf74, buf79, buf87, buf90, buf97, buf101, buf103, buf108, buf112, buf120, buf123, buf130, buf134, buf136, buf141, buf146, buf154, buf157, buf164, buf168, buf170, buf175, buf179, buf187, buf190, buf197, buf201, buf203, buf208, buf213, buf221, buf224, buf231, buf235, buf237, buf242, buf246, buf254, buf257, buf264, buf268, buf270, buf275, buf280, buf288, buf291, buf298, buf302, buf304, buf309, buf313, buf321, buf324, buf331, buf335, buf337, buf342, buf347, buf355, buf358, buf365, buf369, buf371, buf376, buf380, buf388, buf391, buf398, buf402, buf404, buf409, buf414, buf420, buf421, buf425, buf429, buf430, buf432, reinterpret_tensor(buf415, (768, 4096), (1, 768), 0), buf433, reinterpret_tensor(buf405, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf403, (768, 4096), (1, 768), 0), buf434, reinterpret_tensor(buf394, (768, 4096), (1, 768), 0), reinterpret_tensor(buf435, (48, 64, 1024), (65536, 1024, 1), 0), buf437, reinterpret_tensor(buf438, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf439, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf381, (768, 4096), (1, 768), 0), buf440, reinterpret_tensor(buf372, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf370, (768, 4096), (1, 768), 0), buf441, reinterpret_tensor(buf361, (768, 4096), (1, 768), 0), reinterpret_tensor(buf442, (48, 64, 1024), (65536, 1024, 1), 0), buf444, reinterpret_tensor(buf445, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf446, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf348, (768, 4096), (1, 768), 0), buf447, reinterpret_tensor(buf338, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf336, (768, 4096), (1, 768), 0), buf448, reinterpret_tensor(buf327, (768, 4096), (1, 768), 0), reinterpret_tensor(buf449, (48, 64, 1024), (65536, 1024, 1), 0), buf451, reinterpret_tensor(buf452, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf453, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf314, (768, 4096), (1, 768), 0), buf454, reinterpret_tensor(buf305, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf303, (768, 4096), (1, 768), 0), buf455, reinterpret_tensor(buf294, (768, 4096), (1, 768), 0), reinterpret_tensor(buf456, (48, 64, 1024), (65536, 1024, 1), 0), buf458, reinterpret_tensor(buf459, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf460, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf281, (768, 4096), (1, 768), 0), buf461, reinterpret_tensor(buf271, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf269, (768, 4096), (1, 768), 0), buf462, reinterpret_tensor(buf260, (768, 4096), (1, 768), 0), reinterpret_tensor(buf463, (48, 64, 1024), (65536, 1024, 1), 0), buf465, reinterpret_tensor(buf466, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf467, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf247, (768, 4096), (1, 768), 0), buf468, reinterpret_tensor(buf238, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf236, (768, 4096), (1, 768), 0), buf469, reinterpret_tensor(buf227, (768, 4096), (1, 768), 0), reinterpret_tensor(buf470, (48, 64, 1024), (65536, 1024, 1), 0), buf472, reinterpret_tensor(buf473, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf474, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf214, (768, 4096), (1, 768), 0), buf475, reinterpret_tensor(buf204, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf202, (768, 4096), (1, 768), 0), buf476, reinterpret_tensor(buf193, (768, 4096), (1, 768), 0), reinterpret_tensor(buf477, (48, 64, 1024), (65536, 1024, 1), 0), buf479, reinterpret_tensor(buf480, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf481, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf180, (768, 4096), (1, 768), 0), buf482, reinterpret_tensor(buf171, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf169, (768, 4096), (1, 768), 0), buf483, reinterpret_tensor(buf160, (768, 4096), (1, 768), 0), reinterpret_tensor(buf484, (48, 64, 1024), (65536, 1024, 1), 0), buf486, reinterpret_tensor(buf487, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf488, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf147, (768, 4096), (1, 768), 0), buf489, reinterpret_tensor(buf137, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf135, (768, 4096), (1, 768), 0), buf490, reinterpret_tensor(buf126, (768, 4096), (1, 768), 0), reinterpret_tensor(buf491, (48, 64, 1024), (65536, 1024, 1), 0), buf493, reinterpret_tensor(buf494, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf495, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf113, (768, 4096), (1, 768), 0), buf496, reinterpret_tensor(buf104, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf102, (768, 4096), (1, 768), 0), buf497, reinterpret_tensor(buf93, (768, 4096), (1, 768), 0), reinterpret_tensor(buf498, (48, 64, 1024), (65536, 1024, 1), 0), buf500, reinterpret_tensor(buf501, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf502, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf80, (768, 4096), (1, 768), 0), buf503, reinterpret_tensor(buf70, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf68, (768, 4096), (1, 768), 0), buf504, reinterpret_tensor(buf59, (768, 4096), (1, 768), 0), reinterpret_tensor(buf505, (48, 64, 1024), (65536, 1024, 1), 0), buf507, reinterpret_tensor(buf508, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf509, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf46, (768, 4096), (1, 768), 0), buf510, reinterpret_tensor(buf37, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf35, (768, 4096), (1, 768), 0), buf511, reinterpret_tensor(buf26, (768, 4096), (1, 768), 0), reinterpret_tensor(buf512, (48, 64, 1024), (65536, 1024, 1), 0), buf514, reinterpret_tensor(buf515, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf516, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf13, (768, 4096), (1, 768), 0), buf517, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((4, 1024), (1024, 1), device='npu:0', dtype=torch.int64)
    primals_2 = rand_strided((50257, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_9 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_19 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_22 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_26 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_29 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_32 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_34 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_35 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_39 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_42 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_45 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_48 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_52 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_55 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_58 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_61 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_71 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_73 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_74 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_78 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_84 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_86 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_87 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_94 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_97 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_98 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_99 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_100 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_104 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_110 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_112 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_113 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_117 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_120 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_123 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_125 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_126 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_130 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_133 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_136 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_138 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_139 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_143 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_144 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_146 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_149 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_152 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_156 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_159 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_162 = rand_strided((2, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_163 = rand_strided((4, ), (1, ), device='npu:0', dtype=torch.int64)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('GPT2ForSequenceClassification', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
