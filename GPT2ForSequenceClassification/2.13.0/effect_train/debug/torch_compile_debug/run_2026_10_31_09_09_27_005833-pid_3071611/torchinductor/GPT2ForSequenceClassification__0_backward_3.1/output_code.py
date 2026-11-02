# AOT ID: ['0_backward']
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/hw/chwn42rhcrxn57qnyu7gpcjkhimjcl73vbh2p3vxyc6bjxp2bsz7.py
# Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward]
# Source node to ATen node mapping:
#    => scalar_tensor_default, scalar_tensor_default_1, where_self
#   div_13 => div_13
#   loss => full_default_25
#   mul_98 => mul_98
#   where_15 => where_15
# Graph fragment:
#   %ne_3 : Tensor "b8[4, 1][1, 1]npu:0" = PlaceHolder[target=ne_3]
#   %tangents_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_13 : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type_13]
#   %div_13 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_13), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %where_self : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_15 : Tensor "f32[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_13, %full_default_25), kwargs = {})
#   %mul_98 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_15), kwargs = {})
#   return %buf0
triton_unk_fused_nll_loss_backward_nll_loss_forward_scalar_tensor_0 = async_compile.triton('triton_unk_fused_nll_loss_backward_nll_loss_forward_scalar_tensor_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_backward_nll_loss_forward_scalar_tensor_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_backward_nll_loss_forward_scalar_tensor_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (0))
        tmp2 = tmp1
        tmp3 = tl.load(in_ptr2 + (0))
        tmp4 = tmp3
        tmp5 = (tmp2 / tmp4)
        tmp6 = tl.full([1, 1], 0.0, tl.float32)
        tmp7 = tl.where(tmp0, tmp5, tmp6)
        tl.store(out_ptr0 + (x0 + 2*x1), tmp7, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/zs/czsjch2rq3jtphoxkukfpfcntz5rsbwm4wkhyrlx6qdmyl3yzpvf.py
# Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98, exp, sum_3, mul_99], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#    => scalar_tensor_default, scalar_tensor_default_1, where_self
#   div_13 => div_13
#   exp => exp
#   loss => full_default_25
#   mul_98 => mul_98
#   mul_99 => mul_99
#   sum_3 => sum_3
#   where_15 => where_15
# Graph fragment:
#   %eq_tensor : Tensor "b8[4, 2][2, 1]npu:0" = PlaceHolder[target=eq_tensor]
#   %buf0 : Tensor "f32[4, 2][2, 1]npu:0" = PlaceHolder[target=buf0]
#   %div_13 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_13), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %where_self : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_15 : Tensor "f32[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_13, %full_default_25), kwargs = {})
#   %mul_98 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_15), kwargs = {})
#   %exp : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%_log_softmax,), kwargs = {})
#   %sum_3 : Tensor "f32[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_98, [1], True), kwargs = {})
#   %mul_99 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %sum_3), kwargs = {})
#   return %buf1
triton_unk_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_scalar_tensor_1 = async_compile.triton('triton_unk_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_scalar_tensor_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_scalar_tensor_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_scalar_tensor_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + 2 * x1, x1mask, eviction_policy='evict_last') != 0
        _es_lane0 = tl.arange(0, 2)[None, :]
        _es_full0 = tl.load(in_ptr1 + 2 * x1 + _es_lane0, x1mask, eviction_policy='evict_last')
        tmp4 = extract_slice(_es_full0, [0, 0], [real_block_x1, 1], [1, 1])
        tmp6 = tl.load(in_ptr0 + (1 + 2 * x1), x1mask, eviction_policy='evict_last') != 0
        _es_lane1 = tl.arange(0, 2)[None, :]
        _es_full1 = tl.load(in_ptr1 + (1 + 2 * x1) + _es_lane1, x1mask, eviction_policy='evict_last')
        tmp8 = extract_slice(_es_full1, [0, 0], [real_block_x1, 1], [1, 1])
        tmp1 = tl.full([1, 1], -1.0, tl.float32)
        tmp2 = tl.full([1, 1], 0.0, tl.float32)
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp5 = tmp3 * tmp4
        tmp7 = tl.where(tmp6, tmp1, tmp2)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp5 + tmp9
        tl.store(out_ptr0 + (x0 + 2*x1), tmp10, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/wt/cwtqvr2qpszvtlvtliqqmveny5m5bj2po6q2fyao5vuwgsk47ilu.py
# Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98, exp, sum_3, mul_99, view_222, add_99], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward, aten._log_softmax_backward_data, aten.add]
# Source node to ATen node mapping:
#    => scalar_tensor_default, scalar_tensor_default_1, where_self
#   add_99 => add_99
#   div_13 => div_13
#   exp => exp
#   loss => full_default_25
#   mul_98 => mul_98
#   mul_99 => mul_99
#   sum_3 => sum_3
#   view_222 => sub_26
#   where_15 => where_15
# Graph fragment:
#   %tangents_2 : Tensor "f32[4, 2][2, 1]npu:0" = PlaceHolder[target=tangents_2]
#   %eq_tensor : Tensor "b8[4, 2][2, 1]npu:0" = PlaceHolder[target=eq_tensor]
#   %buf0 : Tensor "f32[4, 2][2, 1]npu:0" = PlaceHolder[target=buf0]
#   %_log_softmax : Tensor "f32[4, 2][2, 1]npu:0" = PlaceHolder[target=_log_softmax]
#   %buf1 : Tensor "f32[4, 2][2, 1]npu:0" = PlaceHolder[target=buf1]
#   %div_13 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_13), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %where_self : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_15 : Tensor "f32[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_13, %full_default_25), kwargs = {})
#   %mul_98 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_15), kwargs = {})
#   %exp : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%_log_softmax,), kwargs = {})
#   %sum_3 : Tensor "f32[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_98, [1], True), kwargs = {})
#   %mul_99 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %sum_3), kwargs = {})
#   %sub_26 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_98, %mul_99), kwargs = {})
#   %add_99 : Tensor "f32[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %sub_26), kwargs = {})
#   return %add_99
triton_unk_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_scalar_tensor_2 = async_compile.triton('triton_unk_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_scalar_tensor_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [8, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_scalar_tensor_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_scalar_tensor_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 8
    x_g_tile0 : tl.constexpr = (8) if (8) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 8
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
        tmp1 = tl.load(in_ptr1 + x0, x0mask) != 0
        tmp5 = tl.load(in_out_ptr0 + (x0), x0mask)
        tmp7 = tl.load(in_ptr2 + (x0), x0mask)
        tmp9 = tl.load(in_ptr3 + (x0), x0mask)
        tmp2 = tl.full([1], -1.0, tl.float32)
        tmp3 = tl.full([1], 0.0, tl.float32)
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp6 = tmp4 * tmp5
        tmp8 = libdevice.exp(tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp6 - tmp10
        tmp12 = tmp0 + tmp11
        tl.store(in_out_ptr0 + (x0), tmp12, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/fk/cfk5v2cmdjvzpsty5ctzfh2h2vi6sfj7w2mzoquzacw6wbvu4qgs.py
# Topologically Sorted Source Nodes: [full_25], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   full_25 => full_default_29
# Graph fragment:
#   %full_default_29 : Tensor "f32[4, 1024, 2][2048, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 1024, 2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %index_put
triton_unk_fused_new_zeros_3 = async_compile.triton('triton_unk_fused_new_zeros_3', '''
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
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [8192, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 8192, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_new_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_new_zeros_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 8192
    x_g_tile0 : tl.constexpr = (8192) if (8192) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 8192
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
        tmp0 = tl.full([1], 0.0, tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/sw/cswyascbfcl2svfglbolr4kqw5jolpczwf6p4iff7autbprryusi.py
# Topologically Sorted Source Nodes: [permute_63], Original ATen: [aten.t]
# Source node to ATen node mapping:
#   permute_63 => permute_63
# Graph fragment:
#   %mm_2 : Tensor "f32[768, 2][2, 1]npu:0" = PlaceHolder[target=mm_2]
#   %permute_63 : Tensor "f32[2, 768][1, 2]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mm_2, [1, 0]), kwargs = {})
#   return %permute_63
triton_unk_fused_t_4 = async_compile.triton('triton_unk_fused_t_4', '''
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
    size_hints={'x': 1536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1536, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1536, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_t_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_t_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1536
    x_g_tile0 : tl.constexpr = (1536) if (1536) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 1536
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tl.store(in_out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/yl/cylfujkdxgw6byuhmejnhieqzwnoktd4q6ats7twhoxshw4xfion.py
# Topologically Sorted Source Nodes: [view_227, mul_106, sum_6, sum_7], Original ATen: [aten.matmul_backward, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_106 => mul_106
#   sum_6 => sum_6
#   sum_7 => sum_7
#   view_227 => view_224
# Graph fragment:
#   %mm_1 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_1]
#   %mul_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_96]
#   %view_224 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 1024, 768]), kwargs = {})
#   %mul_106 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_224, %mul_96), kwargs = {})
#   %sum_6 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_106, [0, 1]), kwargs = {})
#   %sum_7 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_224, [0, 1]), kwargs = {})
#   return %sum_6,%sum_7
triton_unk_fused_matmul_backward_native_layer_norm_backward_5 = async_compile.triton('triton_unk_fused_matmul_backward_native_layer_norm_backward_5', '''
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
    size_hints={'x': 768, 'r0_': 4096},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [4096, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_matmul_backward_native_layer_norm_backward_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 4096
    x_g_tile0 : tl.constexpr = (768) if (768) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 768
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
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp4 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        _tmp7 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_ptr1 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, real_block_x0])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
            tmp6 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp8 = _tmp7 + tmp6
            _tmp7 = tmp8
        tmp4 = tl.sum(_tmp4, 0)[None, :]
        tmp7 = tl.sum(_tmp7, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp4, x0mask)
        tl.store(out_ptr1 + (x0), tmp7, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/ym/cymzvxzkdm3yjwmcss7hrxb272ze7w5xhbb5x2lsnxdleo367yv5.py
# Topologically Sorted Source Nodes: [view_227, mul_101, mul_102, sum_4, mul_103, sum_5, mul_104, sub_28, sub_29, mul_105], Original ATen: [aten.matmul_backward, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_101 => mul_101
#   mul_102 => mul_102
#   mul_103 => mul_103
#   mul_104 => mul_104
#   mul_105 => mul_105
#   sub_28 => sub_28
#   sub_29 => sub_29
#   sum_4 => sum_4
#   sum_5 => sum_5
#   view_227 => view_224
# Graph fragment:
#   %mm_1 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_1]
#   %primals_160 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_160]
#   %mul_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_96]
#   %div_14 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0" = PlaceHolder[target=div_14]
#   %sum_4 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=sum_4]
#   %sum_5 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=sum_5]
#   %view_224 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 1024, 768]), kwargs = {})
#   %mul_101 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_224, %primals_160), kwargs = {})
#   %mul_102 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_101, 768), kwargs = {})
#   %sum_4 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_101, [2], True), kwargs = {})
#   %mul_103 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_101, %mul_96), kwargs = {})
#   %sum_5 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_103, [2], True), kwargs = {})
#   %mul_104 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %sum_5), kwargs = {})
#   %sub_28 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_102, %sum_4), kwargs = {})
#   %sub_29 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_28, %mul_104), kwargs = {})
#   %mul_105 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_14, %sub_29), kwargs = {})
#   return %sum_4,%sum_5,%mul_105
triton_unk_fused_matmul_backward_native_layer_norm_backward_6 = async_compile.triton('triton_unk_fused_matmul_backward_native_layer_norm_backward_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_native_layer_norm_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_matmul_backward_native_layer_norm_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp9 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp6 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
            tmp7 = tmp2 * tmp6
            tmp8 = tl.broadcast_to(tmp7, [real_block_x0, R0_BLOCK])
            tmp10 = _tmp9 + tmp8
            _tmp9 = tmp10
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp9 = tl.sum(_tmp9, 1)[:, None]
        tmp11 = tl.load(in_ptr2 + (x0), x0mask, eviction_policy='evict_last')
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp13 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp14 = tmp12 * tmp13
            tmp15 = tl.full([1, 1], 768.0, tl.float32)
            tmp16 = tmp14 * tmp15
            tmp17 = tmp16 - tmp4
            tmp19 = tmp18 * tmp9
            tmp20 = tmp17 - tmp19
            tmp21 = tmp11 * tmp20
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp21, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/xh/cxhf3lsug55zxpsbnah5p2qejtvranpgondh5tx7no4qun2lb5uw.py
# Topologically Sorted Source Nodes: [view_228, sum_8], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_8 => sum_8
#   view_228 => view_228
# Graph fragment:
#   %npu_dropout_backward : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=npu_dropout_backward]
#   %view_228 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%npu_dropout_backward, [4096, 768]), kwargs = {})
#   %sum_8 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_228, [0], True), kwargs = {})
#   return %sum_8
triton_unk_fused_sum_view_7 = async_compile.triton('triton_unk_fused_sum_view_7', '''
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
    size_hints={'x': 768, 'r0_': 4096},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [4096, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': True, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_sum_view_7(in_ptr0, out_ptr0, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 4096
    x_g_tile0 : tl.constexpr = (768) if (768) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 768
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_base = 0
    group_size = x0_blocks
    rsplit_nsplit = total_thread
    rsplit_id = group_id
    r0__per_core = (r0_numel + rsplit_nsplit - 1) // rsplit_nsplit
    r0__lo = rsplit_id * r0__per_core
    r0__hi = tl.minimum(r0__lo + r0__per_core, r0_numel)
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in range(r0__lo, r0__hi, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0__hi
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(ws_ptr + (rsplit_id * (768)) + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/cp/ccpb7lvq4acbjy2zlji2ajridrisgzcqaomddglhrbaxy6fgiimt.py
# Topologically Sorted Source Nodes: [view_228, sum_8], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_8 => sum_8
#   view_228 => view_228
# Graph fragment:
#   %npu_dropout_backward : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=npu_dropout_backward]
#   %view_228 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%npu_dropout_backward, [4096, 768]), kwargs = {})
#   %sum_8 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_228, [0], True), kwargs = {})
#   return %sum_8
triton_unk_fused_sum_view_8 = async_compile.triton('triton_unk_fused_sum_view_8', '''
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
    size_hints={'x': 768, 'r0_': 40},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'r0_numel': 40}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_unk_fused_sum_view_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    x0numel : tl.constexpr = 768
    real_block_x0 : tl.constexpr = x0numel if x0numel <= XBLOCK else XBLOCK
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    group_base = group_id * group_size + group_tail
    if group_id < group_tail:
        group_size = group_size + 1
    if group_id < group_tail:
        group_base = group_id * group_size
    for i in range(group_size):
        x0offset = (group_base + i) * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        _acc = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:, None]
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_index), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)  # noqa: B950
            tmp1 = _acc + tmp0
            _acc = tl.where(r0_mask & x0mask, tmp1, _acc)
        tmp2 = tl.sum(_acc, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/hi/chihst6giytpimzixg5k27gjfuzphfmjki4ccayy4b3ha6rqvqoc.py
# Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
# Source node to ATen node mapping:
#   add_100 => add_100
#   add_101 => add_101
#   add_46 => add_94
#   add_47 => add_95
#   mul_107 => mul_107
#   mul_108 => mul_108
#   mul_109 => mul_109
#   mul_110 => mul_110
#   mul_111 => mul_111
#   mul_112 => mul_112
#   mul_113 => mul_113
#   mul_114 => mul_114
#   mul_115 => mul_115
#   mul_44 => mul_92
#   mul_45 => mul_93
#   mul_46 => mul_94
#   pow_12 => pow_12
#   pow_13 => pow_13
#   sub_30 => sub_30
#   tanh_11 => tanh_11
#   view_230 => view_230
#   x_93 => view_214
# Graph fragment:
#   %mm_3 : Tensor "f32[4096, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_3]
#   %addmm_46 : Tensor "f32[4096, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_46]
#   %view_230 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [4, 1024, 3072]), kwargs = {})
#   %view_214 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_46, [4, 1024, 3072]), kwargs = {})
#   %mul_92 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_214, 0.5), kwargs = {})
#   %mul_107 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_230, %mul_92), kwargs = {})
#   %pow_12 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_214, 3.0), kwargs = {})
#   %mul_93 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_12, 0.044715), kwargs = {})
#   %add_94 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_214, %mul_93), kwargs = {})
#   %mul_94 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, 0.7978845608028654), kwargs = {})
#   %tanh_11 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_94,), kwargs = {})
#   %add_95 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_11, 1.0), kwargs = {})
#   %mul_108 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_230, %add_95), kwargs = {})
#   %mul_109 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_11, %tanh_11), kwargs = {})
#   %sub_30 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_109), kwargs = {})
#   %mul_110 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %sub_30), kwargs = {})
#   %mul_111 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, 0.7978845608028654), kwargs = {})
#   %mul_112 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, 0.044715), kwargs = {})
#   %pow_13 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_214, 2.0), kwargs = {})
#   %mul_113 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_13, 3.0), kwargs = {})
#   %mul_114 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %mul_113), kwargs = {})
#   %add_100 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %mul_114), kwargs = {})
#   %mul_115 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_108, 0.5), kwargs = {})
#   %add_101 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, %mul_115), kwargs = {})
#   return %add_101
triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9 = async_compile.triton('triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [12582912, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 12582912, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tmp1 = tl.load(in_ptr0 + (x0), x0mask)
        tmp2 = tl.full([1], 0.5, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tmp1 * tmp1
        tmp6 = tmp5 * tmp1
        tmp7 = tl.full([1], 0.044715, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp1 + tmp8
        tmp10 = tl.full([1], 0.7978845608028654, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = libdevice.tanh(tmp11)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.full([1], 1.0, tl.float32)
        tmp15 = tmp14 - tmp13
        tmp16 = tmp4 * tmp15
        tmp17 = tmp16 * tmp10
        tmp18 = tmp17 * tmp7
        tmp19 = tl.full([1], 3.0, tl.float32)
        tmp20 = tmp5 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tmp17 + tmp21
        tmp23 = tmp12 + tmp14
        tmp24 = tmp0 * tmp23
        tmp25 = tmp24 * tmp2
        tmp26 = tmp22 + tmp25
        tl.store(in_out_ptr0 + (x0), tmp26, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/e3/ce3tpocw7gwpus4npbmmmynr434amy4w4anw2fxxcysddl3dnbui.py
# Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101, view_231, sum_9], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
# Source node to ATen node mapping:
#   add_100 => add_100
#   add_101 => add_101
#   add_46 => add_94
#   add_47 => add_95
#   mul_107 => mul_107
#   mul_108 => mul_108
#   mul_109 => mul_109
#   mul_110 => mul_110
#   mul_111 => mul_111
#   mul_112 => mul_112
#   mul_113 => mul_113
#   mul_114 => mul_114
#   mul_115 => mul_115
#   mul_44 => mul_92
#   mul_45 => mul_93
#   mul_46 => mul_94
#   pow_12 => pow_12
#   pow_13 => pow_13
#   sub_30 => sub_30
#   sum_9 => sum_9
#   tanh_11 => tanh_11
#   view_230 => view_230
#   view_231 => view_231
#   x_93 => view_214
# Graph fragment:
#   %add_101 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0" = PlaceHolder[target=add_101]
#   %view_230 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [4, 1024, 3072]), kwargs = {})
#   %view_214 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_46, [4, 1024, 3072]), kwargs = {})
#   %mul_92 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_214, 0.5), kwargs = {})
#   %mul_107 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_230, %mul_92), kwargs = {})
#   %pow_12 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_214, 3.0), kwargs = {})
#   %mul_93 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_12, 0.044715), kwargs = {})
#   %add_94 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_214, %mul_93), kwargs = {})
#   %mul_94 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, 0.7978845608028654), kwargs = {})
#   %tanh_11 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_94,), kwargs = {})
#   %add_95 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_11, 1.0), kwargs = {})
#   %mul_108 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_230, %add_95), kwargs = {})
#   %mul_109 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_11, %tanh_11), kwargs = {})
#   %sub_30 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_109), kwargs = {})
#   %mul_110 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %sub_30), kwargs = {})
#   %mul_111 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, 0.7978845608028654), kwargs = {})
#   %mul_112 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, 0.044715), kwargs = {})
#   %pow_13 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_214, 2.0), kwargs = {})
#   %mul_113 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_13, 3.0), kwargs = {})
#   %mul_114 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %mul_113), kwargs = {})
#   %add_100 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %mul_114), kwargs = {})
#   %mul_115 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_108, 0.5), kwargs = {})
#   %add_101 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, %mul_115), kwargs = {})
#   %view_231 : Tensor "f32[4096, 3072][3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_101, [4096, 3072]), kwargs = {})
#   %sum_9 : Tensor "f32[1, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_231, [0], True), kwargs = {})
#   return %sum_9
triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10 = async_compile.triton('triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10', '''
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
    size_hints={'x': 3072, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [3072, {'divisors': [1]}], 'R0_BLOCK_HINT': [4096, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 3072, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3072
    r0_numel = 4096
    x_g_tile0 : tl.constexpr = (3072) if (3072) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 3072
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
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/7a/c7ao7mndmcjncuyha4cw7rfda4mnlveblnxafamwd63uwndzgvdy.py
# Topologically Sorted Source Nodes: [view_233, mul_117, mul_118, sum_10, mul_119, sum_11, mul_120, sub_32, sub_33, mul_121, add_102], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_102 => add_102
#   mul_117 => mul_117
#   mul_118 => mul_118
#   mul_119 => mul_119
#   mul_120 => mul_120
#   mul_121 => mul_121
#   sub_32 => sub_32
#   sub_33 => sub_33
#   sum_10 => sum_10
#   sum_11 => sum_11
#   view_233 => view_233
# Graph fragment:
#   %mm_5 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_5]
#   %primals_154 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_154]
#   %mul_90 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_90]
#   %mul_105 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=mul_105]
#   %div_15 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0" = PlaceHolder[target=div_15]
#   %sum_10 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=sum_10]
#   %sum_11 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=sum_11]
#   %view_233 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [4, 1024, 768]), kwargs = {})
#   %mul_117 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_233, %primals_154), kwargs = {})
#   %mul_118 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_117, 768), kwargs = {})
#   %sum_10 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_117, [2], True), kwargs = {})
#   %mul_119 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_117, %mul_90), kwargs = {})
#   %sum_11 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_119, [2], True), kwargs = {})
#   %mul_120 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %sum_11), kwargs = {})
#   %sub_32 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_118, %sum_10), kwargs = {})
#   %sub_33 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %mul_120), kwargs = {})
#   %mul_121 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_15, %sub_33), kwargs = {})
#   %add_102 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_121), kwargs = {})
#   return %sum_10,%sum_11,%add_102
triton_unk_fused_add_native_layer_norm_backward_view_11 = async_compile.triton('triton_unk_fused_add_native_layer_norm_backward_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_backward_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_backward_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp9 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp6 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
            tmp7 = tmp2 * tmp6
            tmp8 = tl.broadcast_to(tmp7, [real_block_x0, R0_BLOCK])
            tmp10 = _tmp9 + tmp8
            _tmp9 = tmp10
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp9 = tl.sum(_tmp9, 1)[:, None]
        tmp12 = tl.load(in_ptr3 + (x0), x0mask, eviction_policy='evict_last')
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp13 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp19 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp15 = tmp13 * tmp14
            tmp16 = tl.full([1, 1], 768.0, tl.float32)
            tmp17 = tmp15 * tmp16
            tmp18 = tmp17 - tmp4
            tmp20 = tmp19 * tmp9
            tmp21 = tmp18 - tmp20
            tmp22 = tmp12 * tmp21
            tmp23 = tmp11 + tmp22
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp23, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/rn/crnwa7zq5c7dib452kgacytcj25w37ooc5tdg2imrqocx76ppz5d.py
# Topologically Sorted Source Nodes: [view_236, view_237, permute_70, clone_48], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
# Source node to ATen node mapping:
#   clone_48 => clone_48
#   permute_70 => permute_70
#   view_236 => view_236
#   view_237 => view_237
# Graph fragment:
#   %mm_7 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_7]
#   %view_236 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [4, 1024, 768]), kwargs = {})
#   %view_237 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_236, [4, 1024, 12, 64]), kwargs = {})
#   %permute_70 : Tensor "f32[4, 12, 1024, 64][786432, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_237, [0, 2, 1, 3]), kwargs = {})
#   %clone_48 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_48,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_48
triton_unk_fused_matmul_backward_permute_view_12 = async_compile.triton('triton_unk_fused_matmul_backward_permute_view_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_permute_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_permute_view_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 768*x1 + 786432*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/3d/c3dxytu7ux35hpbn3m36hl3coaydxcnb4xd7r75kr3lqupkwgnyw.py
# Topologically Sorted Source Nodes: [view_243, add_103], Original ATen: [aten.matmul_backward, aten.add]
# Source node to ATen node mapping:
#   add_103 => add_103
#   view_243 => view_243
# Graph fragment:
#   %tangents_26 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0" = PlaceHolder[target=tangents_26]
#   %bmm_25 : Tensor "f32[48, 1024, 64][65536, 64, 1]npu:0" = PlaceHolder[target=bmm_25]
#   %view_243 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_25, [4, 12, 1024, 64]), kwargs = {})
#   %add_103 : Tensor "f32[4, 12, 1024, 64][786432, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_26, %view_243), kwargs = {})
#   return %add_103
triton_unk_fused_add_matmul_backward_13 = async_compile.triton('triton_unk_fused_add_matmul_backward_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_matmul_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_matmul_backward_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp1 = tl.load(in_out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), x0mask & x1mask & x2mask & x3mask)
        tmp2 = tmp0 + tmp1
        tl.store(in_out_ptr0 + (x0 + 64*x1 + 65536*x2 + 786432*x3), tmp2, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/2t/c2tw5drvsljbhpiudfjysilccj5veko7z3lskftwuhj47kfugheb.py
# Topologically Sorted Source Nodes: [view_243, add_103, view_246, view_249, permute_75, add_104, permute_76, view_250, permute_77, view_251, permute_78, clone_53, view_252, cat], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
# Source node to ATen node mapping:
#   add_103 => add_103
#   add_104 => add_104
#   cat => cat
#   clone_53 => clone_53
#   permute_75 => permute_75
#   permute_76 => permute_76
#   permute_77 => permute_77
#   permute_78 => permute_78
#   view_243 => view_243
#   view_246 => view_246
#   view_249 => view_249
#   view_250 => view_250
#   view_251 => view_251
#   view_252 => view_252
# Graph fragment:
#   %add_103 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0" = PlaceHolder[target=add_103]
#   %view_243 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_25, [4, 12, 1024, 64]), kwargs = {})
#   %add_103 : Tensor "f32[4, 12, 1024, 64][786432, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_26, %view_243), kwargs = {})
#   %view_246 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_26, [4, 12, 1024, 64]), kwargs = {})
#   %view_249 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_27, [4, 12, 64, 1024]), kwargs = {})
#   %permute_75 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 1, 1024]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_249, [0, 1, 3, 2]), kwargs = {})
#   %add_104 : Tensor "f32[4, 12, 1024, 64][786432, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_25, %permute_75), kwargs = {})
#   %permute_76 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_103, [0, 2, 1, 3]), kwargs = {})
#   %view_250 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_76, [4, 1024, 768]), kwargs = {})
#   %permute_77 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_104, [0, 2, 1, 3]), kwargs = {})
#   %view_251 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_77, [4, 1024, 768]), kwargs = {})
#   %permute_78 : Tensor "f32[4, 1024, 12, 64][786432, 64, 65536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_246, [0, 2, 1, 3]), kwargs = {})
#   %clone_53 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_78,), kwargs = {memory_format: torch.contiguous_format})
#   %view_252 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_53, [4, 1024, 768]), kwargs = {})
#   %cat : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_252, %view_251, %view_250], 2), kwargs = {})
#   return %buf45
triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14 = async_compile.triton('triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, 4, 64, 12, {'divisors': [768, 786432, 1, 64]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x1', 'length': 1024, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 4, 'divisor': 786432, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x4', 'length': 12, 'divisor': 64, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    x1numel : tl.constexpr = 1024
    x2numel : tl.constexpr = 4
    x3numel : tl.constexpr = 64
    x4numel : tl.constexpr = 12
    real_block_x1 : tl.constexpr = x_g_tile2
    real_block_x2 : tl.constexpr = x_g_tile3
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x4 : tl.constexpr = x_g_tile1
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x4_blocks : tl.constexpr = (x4numel + real_block_x4 - 1) // real_block_x4
    x_cumblk_1 = x1_blocks
    x_cumblk_2 = x1_blocks * x2_blocks
    x_cumblk_3 = x1_blocks * x2_blocks * x3_blocks
    total_blocks = x1_blocks * x2_blocks * x3_blocks * x4_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x1offset = (group_base + i) % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_1 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_2 % x3_blocks * real_block_x3
        x4offset = (group_base + i) // x_cumblk_3 % x4_blocks * real_block_x4
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        x4index = x4offset + tl.arange(0, real_block_x4)[None, None, :, None]
        x4 = x4index
        x4mask = x4index < x4numel
        xmask = x1mask & x2mask & x3mask & x4mask
        x0 = x3 + 64*x4
        x0mask = x3mask & x4mask
        tmp0 = tl.load(in_ptr0 + (x3 + 64*x1 + 65536*x4 + 786432*x2), x1mask & x2mask & x3mask & x4mask)
        tl.store(out_ptr0 + (x3 + 64*x4 + 768*x1 + 786432*x2), tmp0, x1mask & x2mask & x3mask & x4mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/zq/czqaycu7jb7x26ne46opy2hkwnppthnqwwuhc6ae3dx7hc6earms.py
# Topologically Sorted Source Nodes: [loss, mul_123, sum_15, mul_124, sub_34, where_16, full, div_16], Original ATen: [aten.nll_loss_forward, aten._softmax_backward_data, aten.where, aten.full, aten.div]
# Source node to ATen node mapping:
#   div_16 => div_16
#   full => full_default
#   loss => full_default_25
#   mul_123 => mul_123
#   mul_124 => mul_124
#   sub_34 => sub_34
#   sum_15 => sum_15
#   where_16 => where_16
# Graph fragment:
#   %npu_dropout_backward_2 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0" = PlaceHolder[target=npu_dropout_backward_2]
#   %_softmax_11 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0" = PlaceHolder[target=_softmax_11]
#   %primals_151 : Tensor "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]npu:0" = PlaceHolder[target=primals_151]
#   %sum_15 : Tensor "f32[4, 12, 1024, 1][12288, 1024, 1, 49152]npu:0" = PlaceHolder[target=sum_15]
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %mul_123 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%npu_dropout_backward_2, %_softmax_11), kwargs = {})
#   %sum_15 : Tensor "f32[4, 12, 1024, 1][12288, 1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_123, [-1], True), kwargs = {})
#   %mul_124 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_softmax_11, %sum_15), kwargs = {})
#   %sub_34 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_123, %mul_124), kwargs = {})
#   %where_16 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%primals_151, %sub_34, %full_default_25), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 8.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %div_16 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_16, %full_default), kwargs = {})
#   return %sum_15,%expand_52
triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15 = async_compile.triton('triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15', '''
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
    size_hints={'x': 49152, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, 48, {'divisors': [1, 1024]}], 'R0_BLOCK_HINT': [1024, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'x3', 'length': 48, 'divisor': 1024, 'seed': 2}, {'name': 'r0_1', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 49152
    r0_numel = 1024
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (48) if (48) < (x_g_rem1) else (x_g_rem1)
    x2numel : tl.constexpr = 1024
    x3numel : tl.constexpr = 48
    real_block_x2 : tl.constexpr = x_g_tile0
    real_block_x3 : tl.constexpr = x_g_tile1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x2_blocks
    total_blocks = x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, None, :]
        x2offset = (group_base + i) % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_1 % x3_blocks * real_block_x3
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x2mask & x3mask
        x0 = x2 + 1024*x3
        x0mask = x2mask & x3mask
        r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp4 = tl.full([real_block_x3, real_block_x2, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 1024*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x3, real_block_x2, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp6 = tl.load(in_ptr1 + (r0_1 + 1024 * x2), r0_mask & x2mask, eviction_policy='evict_last', other=0.0) != 0
            tmp7 = tl.load(in_out_ptr0 + (r0_1 + 1024*x2 + 1048576*x3), r0_mask & x2mask & x3mask, eviction_policy='evict_first', other=0.0)
            tmp8 = tl.load(in_ptr0 + (r0_1 + 1024*x2 + 1048576*x3), r0_mask & x2mask & x3mask, eviction_policy='evict_first', other=0.0)
            tmp9 = tmp7 * tmp8
            tmp10 = tmp8 * tmp4
            tmp11 = tmp9 - tmp10
            tmp12 = tl.full([1, 1, 1], 0.0, tl.float32)
            tmp13 = tl.where(tmp6, tmp11, tmp12)
            tmp14 = tl.full([1, 1, 1], 0.125, tl.float32)
            tmp15 = tmp13 * tmp14
            tl.store(in_out_ptr0 + (r0_1 + 1024*x2 + 1048576*x3), tmp15, r0_mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/cw/ccwqo4elurpvg4apiip4aptyd6dj24yyhz7u3c5ysmqee5siwaaw.py
# Topologically Sorted Source Nodes: [view_246, permute_78, clone_53], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   clone_53 => clone_53
#   permute_78 => permute_78
#   view_246 => view_246
# Graph fragment:
#   %bmm_26 : Tensor "f32[48, 1024, 64][65536, 64, 1]npu:0" = PlaceHolder[target=bmm_26]
#   %view_246 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_26, [4, 12, 1024, 64]), kwargs = {})
#   %permute_78 : Tensor "f32[4, 1024, 12, 64][786432, 64, 65536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_246, [0, 2, 1, 3]), kwargs = {})
#   %clone_53 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_78,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_53
triton_unk_fused_clone_matmul_backward_permute_16 = async_compile.triton('triton_unk_fused_clone_matmul_backward_permute_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_backward_permute_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_backward_permute_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/qz/cqzdmfbawnix5olbjhtubhkjfwjdu5pm4x5ogyxb7czkmhj3hozt.py
# Topologically Sorted Source Nodes: [view_249, permute_75, add_104], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   add_104 => add_104
#   permute_75 => permute_75
#   view_249 => view_249
# Graph fragment:
#   %tangents_25 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0" = PlaceHolder[target=tangents_25]
#   %bmm_27 : Tensor "f32[48, 64, 1024][65536, 1024, 1]npu:0" = PlaceHolder[target=bmm_27]
#   %view_249 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_27, [4, 12, 64, 1024]), kwargs = {})
#   %permute_75 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 1, 1024]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_249, [0, 1, 3, 2]), kwargs = {})
#   %add_104 : Tensor "f32[4, 12, 1024, 64][786432, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_25, %permute_75), kwargs = {})
#   return %add_104
triton_unk_fused_add_matmul_backward_transpose_17 = async_compile.triton('triton_unk_fused_add_matmul_backward_transpose_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 12, 4, {'divisors': [1, 1024, 12288]}], 'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 1024, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 12288, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_matmul_backward_transpose_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_matmul_backward_transpose_17(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x3 + 64*y1 + 2304*y0 + 2359296*y2), x3mask & y0mask & y1mask & y2mask)
        tmp1 = tl.load(in_ptr1 + (y0 + 1024*x3 + 65536*y1 + 786432*y2), x3mask & y0mask & y1mask & y2mask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x3 + 64*y0 + 65536*y1 + 786432*y2), tmp2, x3mask & y0mask & y1mask & y2mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/op/cop4e222mze4hhjjks7jzuyzvra5bpqbe35j5x2evjogimk4lfc6.py
# Topologically Sorted Source Nodes: [view_253, sum_16], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_16 => sum_16
#   view_253 => view_253
# Graph fragment:
#   %cat : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0" = PlaceHolder[target=cat]
#   %view_253 : Tensor "f32[4096, 2304][2304, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [4096, 2304]), kwargs = {})
#   %sum_16 : Tensor "f32[1, 2304][2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_253, [0], True), kwargs = {})
#   return %sum_16
triton_unk_fused_sum_view_18 = async_compile.triton('triton_unk_fused_sum_view_18', '''
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
    size_hints={'x': 2304, 'r0_': 4096},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2304, {'divisors': [1]}], 'R0_BLOCK_HINT': [4096, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 2304, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': True, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_sum_view_18(in_ptr0, out_ptr0, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 2304
    r0_numel = 4096
    x_g_tile0 : tl.constexpr = (2304) if (2304) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 2304
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_base = 0
    group_size = x0_blocks
    rsplit_nsplit = total_thread
    rsplit_id = group_id
    r0__per_core = (r0_numel + rsplit_nsplit - 1) // rsplit_nsplit
    r0__lo = rsplit_id * r0__per_core
    r0__hi = tl.minimum(r0__lo + r0__per_core, r0_numel)
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        xoffset = x0offset
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in range(r0__lo, r0__hi, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0__hi
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 2304*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(ws_ptr + (rsplit_id * (2304)) + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/ns/cnsbb7kdksc7l5xmpb76c2ppifkuzl446d2ba2fiy3jvxcz77w53.py
# Topologically Sorted Source Nodes: [view_253, sum_16], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_16 => sum_16
#   view_253 => view_253
# Graph fragment:
#   %cat : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0" = PlaceHolder[target=cat]
#   %view_253 : Tensor "f32[4096, 2304][2304, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [4096, 2304]), kwargs = {})
#   %sum_16 : Tensor "f32[1, 2304][2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_253, [0], True), kwargs = {})
#   return %sum_16
triton_unk_fused_sum_view_19 = async_compile.triton('triton_unk_fused_sum_view_19', '''
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
    size_hints={'x': 2304, 'r0_': 40},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'r0_numel': 40}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_unk_fused_sum_view_19(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 2304
    x0numel : tl.constexpr = 2304
    real_block_x0 : tl.constexpr = x0numel if x0numel <= XBLOCK else XBLOCK
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    group_base = group_id * group_size + group_tail
    if group_id < group_tail:
        group_size = group_size + 1
    if group_id < group_tail:
        group_base = group_id * group_size
    for i in range(group_size):
        x0offset = (group_base + i) * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        _acc = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:, None]
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 2304*r0_index), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)  # noqa: B950
            tmp1 = _acc + tmp0
            _acc = tl.where(r0_mask & x0mask, tmp1, _acc)
        tmp2 = tl.sum(_acc, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_train/cache/an/canzwfwxulzpqpztutoyky23dcfafgyr45wqcys3ewiy5wz6z554.py
# Topologically Sorted Source Nodes: [sum_164, embedding_dense_backward], Original ATen: [aten.sum, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   embedding_dense_backward => embedding_dense_backward
#   sum_164 => sum_164
# Graph fragment:
#   %npu_dropout_backward_36 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=npu_dropout_backward_36]
#   %sum_164 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%npu_dropout_backward_36, [0], True), kwargs = {})
#   %embedding_dense_backward : Tensor "f32[1024, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding_dense_backward.default](args = (%sum_164, %unsqueeze, 1024, -1, False), kwargs = {})
#   return %buf531
triton_unk_fused_embedding_dense_backward_sum_20 = async_compile.triton('triton_unk_fused_embedding_dense_backward_sum_20', '''
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
    size_hints={'x': 786432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [786432, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 786432, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_embedding_dense_backward_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_embedding_dense_backward_sum_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 786432
    x_g_tile0 : tl.constexpr = (786432) if (786432) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 786432
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
        tmp1 = tl.load(in_ptr0 + (786432 + x0), x0mask)
        tmp3 = tl.load(in_ptr0 + (1572864 + x0), x0mask)
        tmp5 = tl.load(in_ptr0 + (2359296 + x0), x0mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tl.store(out_ptr0 + (x0), tmp6, x0mask)
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
        primals_1, primals_4, primals_7, primals_8, primals_10, primals_11, primals_14, primals_16, primals_17, primals_20, primals_21, primals_23, primals_24, primals_27, primals_29, primals_30, primals_33, primals_34, primals_36, primals_37, primals_40, primals_42, primals_43, primals_46, primals_47, primals_49, primals_50, primals_53, primals_55, primals_56, primals_59, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_86, primals_88, primals_89, primals_92, primals_94, primals_95, primals_98, primals_99, primals_101, primals_102, primals_105, primals_107, primals_108, primals_111, primals_112, primals_114, primals_115, primals_118, primals_120, primals_121, primals_124, primals_125, primals_127, primals_128, primals_131, primals_133, primals_134, primals_137, primals_138, primals_140, primals_141, primals_144, primals_146, primals_147, primals_150, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_162, unsqueeze, getitem_1, mul, _softmax, getitem_8, getitem_10, mul_2, addmm_2, getitem_14, mul_8, _softmax_1, getitem_21, getitem_23, mul_10, addmm_6, getitem_27, mul_16, _softmax_2, getitem_34, getitem_36, mul_18, addmm_10, getitem_40, mul_24, _softmax_3, getitem_47, getitem_49, mul_26, addmm_14, getitem_53, mul_32, _softmax_4, getitem_60, getitem_62, mul_34, addmm_18, getitem_66, mul_40, _softmax_5, getitem_73, getitem_75, mul_42, addmm_22, getitem_79, mul_48, _softmax_6, getitem_86, getitem_88, mul_50, addmm_26, getitem_92, mul_56, _softmax_7, getitem_99, getitem_101, mul_58, addmm_30, getitem_105, mul_64, _softmax_8, getitem_112, getitem_114, mul_66, addmm_34, getitem_118, mul_72, _softmax_9, getitem_125, getitem_127, mul_74, addmm_38, getitem_131, mul_80, _softmax_10, getitem_138, getitem_140, mul_82, addmm_42, getitem_144, mul_88, _softmax_11, getitem_151, getitem_153, mul_90, addmm_46, getitem_157, mul_96, sub_25, iota_1, _log_softmax, convert_element_type_13, ne_3, eq_tensor, permute_62, div_14, permute_65, permute_67, div_15, permute_69, view_239, view_241, view_245, view_247, permute_80, div_17, permute_82, permute_84, div_18, permute_86, view_267, view_269, view_273, view_275, permute_97, div_20, permute_99, permute_101, div_21, permute_103, view_295, view_297, view_301, view_303, permute_114, div_23, permute_116, permute_118, div_24, permute_120, view_323, view_325, view_329, view_331, permute_131, div_26, permute_133, permute_135, div_27, permute_137, view_351, view_353, view_357, view_359, permute_148, div_29, permute_150, permute_152, div_30, permute_154, view_379, view_381, view_385, view_387, permute_165, div_32, permute_167, permute_169, div_33, permute_171, view_407, view_409, view_413, view_415, permute_182, div_35, permute_184, permute_186, div_36, permute_188, view_435, view_437, view_441, view_443, permute_199, div_38, permute_201, permute_203, div_39, permute_205, view_463, view_465, view_469, view_471, permute_216, div_41, permute_218, permute_220, div_42, permute_222, view_491, view_493, view_497, view_499, permute_233, div_44, permute_235, permute_237, div_45, permute_239, view_519, view_521, view_525, view_527, permute_250, div_47, permute_252, permute_254, div_48, permute_256, view_547, view_549, view_553, view_555, permute_267, div_50, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((4, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_backward_nll_loss_forward_scalar_tensor_0.run(ne_3, tangents_1, convert_element_type_13, buf0, 8, stream=raw_stream0)
            del convert_element_type_13
            del ne_3
            del tangents_1
            buf1 = empty_strided_npu((4, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98, exp, sum_3, mul_99], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward, aten._log_softmax_backward_data]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_scalar_tensor_1.run(eq_tensor, buf0, buf1, 8, stream=raw_stream0)
            tangents_2 = copy_if_misaligned(tangents_2)
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [div_13, , loss, where_15, mul_98, exp, sum_3, mul_99, view_222, add_99], Original ATen: [aten.nll_loss_backward, aten.scalar_tensor, aten.nll_loss_forward, aten._log_softmax_backward_data, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_scalar_tensor_2.run(buf3, tangents_2, eq_tensor, _log_softmax, buf1, 8, stream=raw_stream0)
            del _log_softmax
            del buf1
            del eq_tensor
            del tangents_2
            buf2 = empty_strided_npu((4, 1024, 2), (2048, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_25], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_new_zeros_3.run(buf2, 8192, stream=raw_stream0)
            aten.index_put_(buf2, [iota_1, sub_25], buf3, True)
            del buf3
            del iota_1
            del sub_25
            buf6 = empty_strided_npu((768, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_223, view_226], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(permute_62, reinterpret_tensor(buf2, (4096, 2), (2, 1), 0), out=buf6)
            del permute_62
            buf7 = reinterpret_tensor(buf6, (2, 768), (1, 2), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [permute_63], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_4.run(buf7, 1536, stream=raw_stream0)
            buf5 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_223, logits, permute_61, mm_1], Original ATen: [aten.matmul_backward, aten.t]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(buf2, (4096, 2), (2, 1), 0), primals_162, out=buf5)
            del buf2
            del primals_162
            buf11 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf12 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_227, mul_106, sum_6, sum_7], Original ATen: [aten.matmul_backward, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf5, mul_96, buf11, buf12, 768, 4096, stream=raw_stream0)
            buf10 = reinterpret_tensor(buf5, (4, 1024, 768), (786432, 768, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [view_227, mul_101, mul_102, sum_4, mul_103, sum_5, mul_104, sub_28, sub_29, mul_105], Original ATen: [aten.matmul_backward, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_6.run(buf10, primals_160, mul_96, div_14, 4096, 768, stream=raw_stream0)
            del div_14
            del mul_96
            del primals_160
            # Topologically Sorted Source Nodes: [npu_dropout_backward], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:3
            buf13 = torch.ops.npu.npu_dropout_backward.default(buf10, getitem_157, 0.10000000000000009)
            del getitem_157
            buf14 = buf13
            assert_alignment(buf14, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf13
            buf16 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_228, mm_4], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(permute_65, reinterpret_tensor(buf14, (4096, 768), (768, 1), 0), out=buf16)
            del permute_65
            buf17 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_228, sum_8], Original ATen: [aten.view, aten.sum]
            workspace_0 = empty_strided_npu((30720, ), (1, ), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf14, buf17, workspace_0, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_0, buf17, 768, 40, stream=raw_stream0)
            buf15 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_228, permute_64, mm_3], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf14, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 3072), (1, 768), 0), out=buf15)
            del primals_159
            buf18 = reinterpret_tensor(buf15, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf18, addmm_46, 12582912, stream=raw_stream0)
            del addmm_46
            buf20 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101, view_231, mm_6], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(permute_67, reinterpret_tensor(buf18, (4096, 3072), (3072, 1), 0), out=buf20)
            del permute_67
            buf21 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101, view_231, sum_9], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf18, buf21, 3072, 4096, stream=raw_stream0)
            buf19 = reinterpret_tensor(buf14, (4096, 768), (768, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [view_230, x_93, mul_44, mul_107, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, mul_108, mul_109, sub_30, mul_110, mul_111, mul_112, pow_13, mul_113, mul_114, add_100, mul_115, add_101, view_231, permute_66, mm_5], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf18, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_157, (3072, 768), (1, 3072), 0), out=buf19)
            del buf18
            del primals_157
            buf26 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [view_233, mul_117, mul_118, sum_10, mul_119, sum_11, mul_120, sub_32, sub_33, mul_121, add_102], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf26, buf19, primals_154, mul_90, div_15, 4096, 768, stream=raw_stream0)
            del div_15
            del primals_154
            buf24 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf25 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_233, mul_122, sum_12, sum_13], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf19, mul_90, buf24, buf25, 768, 4096, stream=raw_stream0)
            del mul_90
            # Topologically Sorted Source Nodes: [npu_dropout_backward_1], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:8
            buf27 = torch.ops.npu.npu_dropout_backward.default(buf26, getitem_153, 0.10000000000000009)
            del getitem_153
            buf28 = buf27
            assert_alignment(buf28, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf27
            buf30 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_234, mm_8], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(permute_69, reinterpret_tensor(buf28, (4096, 768), (768, 1), 0), out=buf30)
            del permute_69
            buf31 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_234, sum_14], Original ATen: [aten.view, aten.sum]
            workspace_1 = workspace_0; del workspace_0  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf28, buf31, workspace_1, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_1, buf31, 768, 40, stream=raw_stream0)
            del workspace_1
            buf29 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [view_234, permute_68, mm_7], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf28, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_153, (768, 768), (1, 768), 0), out=buf29)
            del primals_153
            buf32 = reinterpret_tensor(buf28, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [view_236, view_237, permute_70, clone_48], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf29, buf32, 3145728, stream=raw_stream0)
            buf34 = reinterpret_tensor(buf29, (48, 1024, 64), (65536, 64, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [view_236, view_237, permute_70, clone_48, view_238, bmm_25], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:11
            extern_kernels.bmm(view_241, reinterpret_tensor(buf32, (48, 1024, 64), (65536, 64, 1), 0), out=buf34)
            del view_241
            tangents_26 = copy_if_misaligned(tangents_26)
            buf44 = reinterpret_tensor(buf34, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [view_243, add_103], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf44, tangents_26, 3145728, stream=raw_stream0)
            del tangents_26
            buf45 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_243, add_103, view_246, view_249, permute_75, add_104, permute_76, view_250, permute_77, view_251, permute_78, clone_53, view_252, cat], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf44, buf45, 3145728, stream=raw_stream0)
            del buf44
            buf33 = empty_strided_npu((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_236, view_237, permute_70, clone_48, view_238, bmm_24], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:12
            extern_kernels.bmm(reinterpret_tensor(buf32, (48, 1024, 64), (65536, 64, 1), 0), view_239, out=buf33)
            del buf32
            del view_239
            # Topologically Sorted Source Nodes: [view_240, npu_dropout_backward_2], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:13
            buf35 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf33, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_151, 0.10000000000000009)
            del buf33
            del getitem_151
            buf36 = buf35
            assert_alignment(buf36, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf35
            buf38 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [loss, mul_123, sum_15, mul_124, sub_34, where_16, full, div_16], Original ATen: [aten.nll_loss_forward, aten._softmax_backward_data, aten.where, aten.full, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf38, _softmax_11, primals_151, 49152, 1024, stream=raw_stream0)
            del _softmax_11
            del primals_151
            buf39 = empty_strided_npu((48, 1024, 64), (65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, mul_123, mul_124, sub_34, where_16, full, div_16, view_244, bmm_26], Original ATen: [aten.nll_loss_forward, aten._softmax_backward_data, aten.where, aten.full, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:14
            extern_kernels.bmm(reinterpret_tensor(buf38, (48, 1024, 1024), (1048576, 1024, 1), 0), view_245, out=buf39)
            del view_245
            buf40 = empty_strided_npu((48, 64, 1024), (65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, mul_123, mul_124, sub_34, where_16, full, div_16, view_244, bmm_27], Original ATen: [aten.nll_loss_forward, aten._softmax_backward_data, aten.where, aten.full, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:15
            extern_kernels.bmm(view_247, reinterpret_tensor(buf38, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf40)
            del view_247
            buf41 = empty_strided_npu((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_246, permute_78, clone_53], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf39, buf41, 3145728, stream=raw_stream0)
            tangents_25 = copy_if_misaligned(tangents_25)
            buf42 = reinterpret_tensor(buf39, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [view_249, permute_75, add_104], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_25, buf40, buf42, 49152, 64, stream=raw_stream0)
            del tangents_25
            buf43 = reinterpret_tensor(buf40, (4, 1024, 768), (786432, 768, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [view_243, add_103, view_246, view_249, permute_75, add_104, permute_76, view_250, permute_77, view_251, permute_78, clone_53, view_252, cat], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf42, buf43, 3145728, stream=raw_stream0)
            del buf42
            # Topologically Sorted Source Nodes: [view_243, add_103, view_246, view_249, permute_75, add_104, permute_76, view_250, permute_77, view_251, permute_78, clone_53, view_252, cat], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:16
            buf46 = torch.ops.aten.cat.default([reinterpret_tensor(buf41, (4, 1024, 768), (786432, 768, 1), 0), buf43, buf45], 2)
            del buf41
            buf47 = buf46
            assert_alignment(buf47, 16, 'torch.ops.aten.cat.default')
            del buf46
            buf49 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_253, mm_10], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(permute_80, reinterpret_tensor(buf47, (4096, 2304), (2304, 1), 0), out=buf49)
            del permute_80
            buf50 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_253, sum_16], Original ATen: [aten.view, aten.sum]
            workspace_2 = empty_strided_npu((92160, ), (1, ), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf47, buf50, workspace_2, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_2, buf50, 2304, 40, stream=raw_stream0)
            buf48 = reinterpret_tensor(buf45, (4096, 768), (768, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [view_253, permute_79, mm_9], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:18
            extern_kernels.mm(reinterpret_tensor(buf47, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_150, (2304, 768), (1, 2304), 0), out=buf48)
            del buf47
            del primals_150
            buf55 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [view_255, mul_126, mul_127, sum_17, mul_128, sum_18, mul_129, sub_36, sub_37, mul_130, add_105], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf55, buf48, primals_147, mul_88, div_17, 4096, 768, stream=raw_stream0)
            del div_17
            del primals_147
            buf53 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf54 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_255, mul_131, sum_19, sum_20], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf48, mul_88, buf53, buf54, 768, 4096, stream=raw_stream0)
            del mul_88
            # Topologically Sorted Source Nodes: [npu_dropout_backward_3], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:19
            buf56 = torch.ops.npu.npu_dropout_backward.default(buf55, getitem_144, 0.10000000000000009)
            del getitem_144
            buf57 = buf56
            assert_alignment(buf57, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf56
            buf59 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_256, mm_12], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(permute_82, reinterpret_tensor(buf57, (4096, 768), (768, 1), 0), out=buf59)
            del permute_82
            buf60 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_256, sum_21], Original ATen: [aten.view, aten.sum]
            workspace_3 = empty_strided_npu((30720, ), (1, ), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf57, buf60, workspace_3, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_3, buf60, 768, 40, stream=raw_stream0)
            buf58 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_256, permute_81, mm_11], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf57, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_146, (768, 3072), (1, 768), 0), out=buf58)
            del primals_146
            buf61 = reinterpret_tensor(buf58, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [view_258, x_85, mul_40, mul_132, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, mul_133, mul_134, sub_38, mul_135, mul_136, mul_137, pow_14, mul_138, mul_139, add_106, mul_140, add_107], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf61, addmm_42, 12582912, stream=raw_stream0)
            del addmm_42
            buf63 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_258, x_85, mul_40, mul_132, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, mul_133, mul_134, sub_38, mul_135, mul_136, mul_137, pow_14, mul_138, mul_139, add_106, mul_140, add_107, view_259, mm_14], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(permute_84, reinterpret_tensor(buf61, (4096, 3072), (3072, 1), 0), out=buf63)
            del permute_84
            buf64 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_258, x_85, mul_40, mul_132, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, mul_133, mul_134, sub_38, mul_135, mul_136, mul_137, pow_14, mul_138, mul_139, add_106, mul_140, add_107, view_259, sum_22], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf61, buf64, 3072, 4096, stream=raw_stream0)
            buf62 = reinterpret_tensor(buf57, (4096, 768), (768, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [view_258, x_85, mul_40, mul_132, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, mul_133, mul_134, sub_38, mul_135, mul_136, mul_137, pow_14, mul_138, mul_139, add_106, mul_140, add_107, view_259, permute_83, mm_13], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:23
            extern_kernels.mm(reinterpret_tensor(buf61, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (1, 3072), 0), out=buf62)
            del primals_144
            buf69 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [view_261, mul_142, mul_143, sum_23, mul_144, sum_24, mul_145, sub_40, sub_41, mul_146, add_108], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf69, buf62, primals_141, mul_82, div_18, 4096, 768, stream=raw_stream0)
            del div_18
            del primals_141
            buf67 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf68 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_261, mul_147, sum_25, sum_26], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf62, mul_82, buf67, buf68, 768, 4096, stream=raw_stream0)
            del mul_82
            # Topologically Sorted Source Nodes: [npu_dropout_backward_4], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:24
            buf70 = torch.ops.npu.npu_dropout_backward.default(buf69, getitem_140, 0.10000000000000009)
            del getitem_140
            buf71 = buf70
            assert_alignment(buf71, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf70
            buf73 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_262, mm_16], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(permute_86, reinterpret_tensor(buf71, (4096, 768), (768, 1), 0), out=buf73)
            del permute_86
            buf74 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_262, sum_27], Original ATen: [aten.view, aten.sum]
            workspace_4 = workspace_3; del workspace_3  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf71, buf74, workspace_4, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_4, buf74, 768, 40, stream=raw_stream0)
            buf72 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [view_262, permute_85, mm_15], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:26
            extern_kernels.mm(reinterpret_tensor(buf71, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), out=buf72)
            del primals_140
            buf75 = reinterpret_tensor(buf71, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [view_264, view_265, permute_87, clone_54], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf72, buf75, 3145728, stream=raw_stream0)
            buf77 = reinterpret_tensor(buf72, (48, 1024, 64), (65536, 64, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [view_264, view_265, permute_87, clone_54, view_266, bmm_29], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:27
            extern_kernels.bmm(view_269, reinterpret_tensor(buf75, (48, 1024, 64), (65536, 64, 1), 0), out=buf77)
            del view_269
            tangents_24 = copy_if_misaligned(tangents_24)
            buf87 = reinterpret_tensor(buf77, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [view_271, add_109], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf87, tangents_24, 3145728, stream=raw_stream0)
            del tangents_24
            buf88 = reinterpret_tensor(buf48, (4, 1024, 768), (786432, 768, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [view_271, add_109, view_274, view_277, permute_92, add_110, permute_93, view_278, permute_94, view_279, permute_95, clone_59, view_280, cat_1], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf87, buf88, 3145728, stream=raw_stream0)
            buf76 = reinterpret_tensor(buf38, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [view_264, view_265, permute_87, clone_54, view_266, bmm_28], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:28
            extern_kernels.bmm(reinterpret_tensor(buf75, (48, 1024, 64), (65536, 64, 1), 0), view_267, out=buf76)
            del view_267
            # Topologically Sorted Source Nodes: [view_268, npu_dropout_backward_5], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:29
            buf78 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf76, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_138, 0.10000000000000009)
            del buf76
            del getitem_138
            buf79 = buf78
            assert_alignment(buf79, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf78
            buf81 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_148, sum_28, mul_149, sub_42, where_17, div_19], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf81, _softmax_10, primals_138, 49152, 1024, stream=raw_stream0)
            del _softmax_10
            del primals_138
            buf82 = reinterpret_tensor(buf75, (48, 1024, 64), (65536, 64, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_148, mul_149, sub_42, where_17, div_19, view_272, bmm_30], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:30
            extern_kernels.bmm(reinterpret_tensor(buf81, (48, 1024, 1024), (1048576, 1024, 1), 0), view_273, out=buf82)
            del view_273
            buf83 = reinterpret_tensor(buf87, (48, 64, 1024), (65536, 1024, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_148, mul_149, sub_42, where_17, div_19, view_272, bmm_31], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:31
            extern_kernels.bmm(view_275, reinterpret_tensor(buf81, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf83)
            del view_275
            buf84 = reinterpret_tensor(buf43, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [view_274, permute_95, clone_59], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf82, buf84, 3145728, stream=raw_stream0)
            tangents_23 = copy_if_misaligned(tangents_23)
            buf85 = reinterpret_tensor(buf82, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [view_277, permute_92, add_110], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_23, buf83, buf85, 49152, 64, stream=raw_stream0)
            del tangents_23
            buf86 = reinterpret_tensor(buf83, (4, 1024, 768), (786432, 768, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [view_271, add_109, view_274, view_277, permute_92, add_110, permute_93, view_278, permute_94, view_279, permute_95, clone_59, view_280, cat_1], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf85, buf86, 3145728, stream=raw_stream0)
            del buf85
            # Topologically Sorted Source Nodes: [view_271, add_109, view_274, view_277, permute_92, add_110, permute_93, view_278, permute_94, view_279, permute_95, clone_59, view_280, cat_1], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:32
            buf89 = torch.ops.aten.cat.default([reinterpret_tensor(buf84, (4, 1024, 768), (786432, 768, 1), 0), buf86, buf88], 2)
            del buf84
            buf90 = buf89
            assert_alignment(buf90, 16, 'torch.ops.aten.cat.default')
            del buf89
            buf92 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_281, mm_18], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:33
            extern_kernels.mm(permute_97, reinterpret_tensor(buf90, (4096, 2304), (2304, 1), 0), out=buf92)
            del permute_97
            buf93 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_281, sum_29], Original ATen: [aten.view, aten.sum]
            workspace_5 = workspace_2; del workspace_2  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf90, buf93, workspace_5, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_5, buf93, 2304, 40, stream=raw_stream0)
            buf91 = reinterpret_tensor(buf88, (4096, 768), (768, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [view_281, permute_96, mm_17], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(reinterpret_tensor(buf90, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_137, (2304, 768), (1, 2304), 0), out=buf91)
            del buf90
            del primals_137
            buf98 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [view_283, mul_151, mul_152, sum_30, mul_153, sum_31, mul_154, sub_44, sub_45, mul_155, add_111], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf98, buf91, primals_134, mul_80, div_20, 4096, 768, stream=raw_stream0)
            del div_20
            del primals_134
            buf96 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf97 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_283, mul_156, sum_32, sum_33], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf91, mul_80, buf96, buf97, 768, 4096, stream=raw_stream0)
            del mul_80
            # Topologically Sorted Source Nodes: [npu_dropout_backward_6], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:35
            buf99 = torch.ops.npu.npu_dropout_backward.default(buf98, getitem_131, 0.10000000000000009)
            del getitem_131
            buf100 = buf99
            assert_alignment(buf100, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf99
            buf102 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_284, mm_20], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:36
            extern_kernels.mm(permute_99, reinterpret_tensor(buf100, (4096, 768), (768, 1), 0), out=buf102)
            del permute_99
            buf103 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_284, sum_34], Original ATen: [aten.view, aten.sum]
            workspace_6 = workspace_4; del workspace_4  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf100, buf103, workspace_6, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_6, buf103, 768, 40, stream=raw_stream0)
            buf101 = reinterpret_tensor(buf61, (4096, 3072), (3072, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [view_284, permute_98, mm_19], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf100, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 3072), (1, 768), 0), out=buf101)
            del primals_133
            buf104 = reinterpret_tensor(buf101, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [view_286, x_77, mul_36, mul_157, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, mul_158, mul_159, sub_46, mul_160, mul_161, mul_162, pow_15, mul_163, mul_164, add_112, mul_165, add_113], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf104, addmm_38, 12582912, stream=raw_stream0)
            del addmm_38
            buf106 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_286, x_77, mul_36, mul_157, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, mul_158, mul_159, sub_46, mul_160, mul_161, mul_162, pow_15, mul_163, mul_164, add_112, mul_165, add_113, view_287, mm_22], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:38
            extern_kernels.mm(permute_101, reinterpret_tensor(buf104, (4096, 3072), (3072, 1), 0), out=buf106)
            del permute_101
            buf107 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_286, x_77, mul_36, mul_157, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, mul_158, mul_159, sub_46, mul_160, mul_161, mul_162, pow_15, mul_163, mul_164, add_112, mul_165, add_113, view_287, sum_35], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf104, buf107, 3072, 4096, stream=raw_stream0)
            buf105 = reinterpret_tensor(buf100, (4096, 768), (768, 1), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [view_286, x_77, mul_36, mul_157, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, mul_158, mul_159, sub_46, mul_160, mul_161, mul_162, pow_15, mul_163, mul_164, add_112, mul_165, add_113, view_287, permute_100, mm_21], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:39
            extern_kernels.mm(reinterpret_tensor(buf104, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (1, 3072), 0), out=buf105)
            del primals_131
            buf112 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [view_289, mul_167, mul_168, sum_36, mul_169, sum_37, mul_170, sub_48, sub_49, mul_171, add_114], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf112, buf105, primals_128, mul_74, div_21, 4096, 768, stream=raw_stream0)
            del div_21
            del primals_128
            buf110 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf111 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_289, mul_172, sum_38, sum_39], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf105, mul_74, buf110, buf111, 768, 4096, stream=raw_stream0)
            del mul_74
            # Topologically Sorted Source Nodes: [npu_dropout_backward_7], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:40
            buf113 = torch.ops.npu.npu_dropout_backward.default(buf112, getitem_127, 0.10000000000000009)
            del getitem_127
            buf114 = buf113
            assert_alignment(buf114, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf113
            buf116 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_290, mm_24], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:41
            extern_kernels.mm(permute_103, reinterpret_tensor(buf114, (4096, 768), (768, 1), 0), out=buf116)
            del permute_103
            buf117 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_290, sum_40], Original ATen: [aten.view, aten.sum]
            workspace_7 = workspace_6; del workspace_6  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf114, buf117, workspace_7, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_7, buf117, 768, 40, stream=raw_stream0)
            buf115 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [view_290, permute_102, mm_23], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf114, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_127, (768, 768), (1, 768), 0), out=buf115)
            del primals_127
            buf118 = reinterpret_tensor(buf114, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [view_292, view_293, permute_104, clone_60], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf115, buf118, 3145728, stream=raw_stream0)
            buf120 = reinterpret_tensor(buf115, (48, 1024, 64), (65536, 64, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [view_292, view_293, permute_104, clone_60, view_294, bmm_33], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:43
            extern_kernels.bmm(view_297, reinterpret_tensor(buf118, (48, 1024, 64), (65536, 64, 1), 0), out=buf120)
            del view_297
            tangents_22 = copy_if_misaligned(tangents_22)
            buf130 = reinterpret_tensor(buf120, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [view_299, add_115], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf130, tangents_22, 3145728, stream=raw_stream0)
            del tangents_22
            buf131 = reinterpret_tensor(buf91, (4, 1024, 768), (786432, 768, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [view_299, add_115, view_302, view_305, permute_109, add_116, permute_110, view_306, permute_111, view_307, permute_112, clone_65, view_308, cat_2], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf130, buf131, 3145728, stream=raw_stream0)
            buf119 = reinterpret_tensor(buf81, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [view_292, view_293, permute_104, clone_60, view_294, bmm_32], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:44
            extern_kernels.bmm(reinterpret_tensor(buf118, (48, 1024, 64), (65536, 64, 1), 0), view_295, out=buf119)
            del view_295
            # Topologically Sorted Source Nodes: [view_296, npu_dropout_backward_8], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:45
            buf121 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf119, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_125, 0.10000000000000009)
            del buf119
            del getitem_125
            buf122 = buf121
            assert_alignment(buf122, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf121
            buf124 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_173, sum_41, mul_174, sub_50, where_18, div_22], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf124, _softmax_9, primals_125, 49152, 1024, stream=raw_stream0)
            del _softmax_9
            del primals_125
            buf125 = reinterpret_tensor(buf118, (48, 1024, 64), (65536, 64, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_173, mul_174, sub_50, where_18, div_22, view_300, bmm_34], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:46
            extern_kernels.bmm(reinterpret_tensor(buf124, (48, 1024, 1024), (1048576, 1024, 1), 0), view_301, out=buf125)
            del view_301
            buf126 = reinterpret_tensor(buf130, (48, 64, 1024), (65536, 1024, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_173, mul_174, sub_50, where_18, div_22, view_300, bmm_35], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:47
            extern_kernels.bmm(view_303, reinterpret_tensor(buf124, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf126)
            del view_303
            buf127 = reinterpret_tensor(buf86, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [view_302, permute_112, clone_65], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf125, buf127, 3145728, stream=raw_stream0)
            tangents_21 = copy_if_misaligned(tangents_21)
            buf128 = reinterpret_tensor(buf125, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [view_305, permute_109, add_116], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_21, buf126, buf128, 49152, 64, stream=raw_stream0)
            del tangents_21
            buf129 = reinterpret_tensor(buf126, (4, 1024, 768), (786432, 768, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [view_299, add_115, view_302, view_305, permute_109, add_116, permute_110, view_306, permute_111, view_307, permute_112, clone_65, view_308, cat_2], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf128, buf129, 3145728, stream=raw_stream0)
            del buf128
            # Topologically Sorted Source Nodes: [view_299, add_115, view_302, view_305, permute_109, add_116, permute_110, view_306, permute_111, view_307, permute_112, clone_65, view_308, cat_2], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:48
            buf132 = torch.ops.aten.cat.default([reinterpret_tensor(buf127, (4, 1024, 768), (786432, 768, 1), 0), buf129, buf131], 2)
            del buf127
            buf133 = buf132
            assert_alignment(buf133, 16, 'torch.ops.aten.cat.default')
            del buf132
            buf135 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_309, mm_26], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:49
            extern_kernels.mm(permute_114, reinterpret_tensor(buf133, (4096, 2304), (2304, 1), 0), out=buf135)
            del permute_114
            buf136 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_309, sum_42], Original ATen: [aten.view, aten.sum]
            workspace_8 = workspace_5; del workspace_5  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf133, buf136, workspace_8, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_8, buf136, 2304, 40, stream=raw_stream0)
            buf134 = reinterpret_tensor(buf131, (4096, 768), (768, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [view_309, permute_113, mm_25], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf133, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_124, (2304, 768), (1, 2304), 0), out=buf134)
            del buf133
            del primals_124
            buf141 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [view_311, mul_176, mul_177, sum_43, mul_178, sum_44, mul_179, sub_52, sub_53, mul_180, add_117], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf141, buf134, primals_121, mul_72, div_23, 4096, 768, stream=raw_stream0)
            del div_23
            del primals_121
            buf139 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf140 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_311, mul_181, sum_45, sum_46], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf134, mul_72, buf139, buf140, 768, 4096, stream=raw_stream0)
            del mul_72
            # Topologically Sorted Source Nodes: [npu_dropout_backward_9], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:51
            buf142 = torch.ops.npu.npu_dropout_backward.default(buf141, getitem_118, 0.10000000000000009)
            del getitem_118
            buf143 = buf142
            assert_alignment(buf143, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf142
            buf145 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_312, mm_28], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:52
            extern_kernels.mm(permute_116, reinterpret_tensor(buf143, (4096, 768), (768, 1), 0), out=buf145)
            del permute_116
            buf146 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_312, sum_47], Original ATen: [aten.view, aten.sum]
            workspace_9 = workspace_7; del workspace_7  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf143, buf146, workspace_9, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_9, buf146, 768, 40, stream=raw_stream0)
            buf144 = reinterpret_tensor(buf104, (4096, 3072), (3072, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [view_312, permute_115, mm_27], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:53
            extern_kernels.mm(reinterpret_tensor(buf143, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 3072), (1, 768), 0), out=buf144)
            del primals_120
            buf147 = reinterpret_tensor(buf144, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [view_314, x_69, mul_32, mul_182, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, mul_183, mul_184, sub_54, mul_185, mul_186, mul_187, pow_16, mul_188, mul_189, add_118, mul_190, add_119], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf147, addmm_34, 12582912, stream=raw_stream0)
            del addmm_34
            buf149 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_314, x_69, mul_32, mul_182, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, mul_183, mul_184, sub_54, mul_185, mul_186, mul_187, pow_16, mul_188, mul_189, add_118, mul_190, add_119, view_315, mm_30], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:54
            extern_kernels.mm(permute_118, reinterpret_tensor(buf147, (4096, 3072), (3072, 1), 0), out=buf149)
            del permute_118
            buf150 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_314, x_69, mul_32, mul_182, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, mul_183, mul_184, sub_54, mul_185, mul_186, mul_187, pow_16, mul_188, mul_189, add_118, mul_190, add_119, view_315, sum_48], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf147, buf150, 3072, 4096, stream=raw_stream0)
            buf148 = reinterpret_tensor(buf143, (4096, 768), (768, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [view_314, x_69, mul_32, mul_182, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, mul_183, mul_184, sub_54, mul_185, mul_186, mul_187, pow_16, mul_188, mul_189, add_118, mul_190, add_119, view_315, permute_117, mm_29], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:55
            extern_kernels.mm(reinterpret_tensor(buf147, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_118, (3072, 768), (1, 3072), 0), out=buf148)
            del primals_118
            buf155 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [view_317, mul_192, mul_193, sum_49, mul_194, sum_50, mul_195, sub_56, sub_57, mul_196, add_120], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf155, buf148, primals_115, mul_66, div_24, 4096, 768, stream=raw_stream0)
            del div_24
            del primals_115
            buf153 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf154 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_317, mul_197, sum_51, sum_52], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf148, mul_66, buf153, buf154, 768, 4096, stream=raw_stream0)
            del mul_66
            # Topologically Sorted Source Nodes: [npu_dropout_backward_10], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:56
            buf156 = torch.ops.npu.npu_dropout_backward.default(buf155, getitem_114, 0.10000000000000009)
            del getitem_114
            buf157 = buf156
            assert_alignment(buf157, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf156
            buf159 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_318, mm_32], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(permute_120, reinterpret_tensor(buf157, (4096, 768), (768, 1), 0), out=buf159)
            del permute_120
            buf160 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_318, sum_53], Original ATen: [aten.view, aten.sum]
            workspace_10 = workspace_9; del workspace_9  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf157, buf160, workspace_10, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_10, buf160, 768, 40, stream=raw_stream0)
            buf158 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [view_318, permute_119, mm_31], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:58
            extern_kernels.mm(reinterpret_tensor(buf157, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_114, (768, 768), (1, 768), 0), out=buf158)
            del primals_114
            buf161 = reinterpret_tensor(buf157, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [view_320, view_321, permute_121, clone_66], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf158, buf161, 3145728, stream=raw_stream0)
            buf163 = reinterpret_tensor(buf158, (48, 1024, 64), (65536, 64, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [view_320, view_321, permute_121, clone_66, view_322, bmm_37], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:59
            extern_kernels.bmm(view_325, reinterpret_tensor(buf161, (48, 1024, 64), (65536, 64, 1), 0), out=buf163)
            del view_325
            tangents_20 = copy_if_misaligned(tangents_20)
            buf173 = reinterpret_tensor(buf163, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [view_327, add_121], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf173, tangents_20, 3145728, stream=raw_stream0)
            del tangents_20
            buf174 = reinterpret_tensor(buf134, (4, 1024, 768), (786432, 768, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [view_327, add_121, view_330, view_333, permute_126, add_122, permute_127, view_334, permute_128, view_335, permute_129, clone_71, view_336, cat_3], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf173, buf174, 3145728, stream=raw_stream0)
            buf162 = reinterpret_tensor(buf124, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [view_320, view_321, permute_121, clone_66, view_322, bmm_36], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:60
            extern_kernels.bmm(reinterpret_tensor(buf161, (48, 1024, 64), (65536, 64, 1), 0), view_323, out=buf162)
            del view_323
            # Topologically Sorted Source Nodes: [view_324, npu_dropout_backward_11], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:61
            buf164 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf162, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_112, 0.10000000000000009)
            del buf162
            del getitem_112
            buf165 = buf164
            assert_alignment(buf165, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf164
            buf167 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_198, sum_54, mul_199, sub_58, where_19, div_25], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf167, _softmax_8, primals_112, 49152, 1024, stream=raw_stream0)
            del _softmax_8
            del primals_112
            buf168 = reinterpret_tensor(buf161, (48, 1024, 64), (65536, 64, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_198, mul_199, sub_58, where_19, div_25, view_328, bmm_38], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:62
            extern_kernels.bmm(reinterpret_tensor(buf167, (48, 1024, 1024), (1048576, 1024, 1), 0), view_329, out=buf168)
            del view_329
            buf169 = reinterpret_tensor(buf173, (48, 64, 1024), (65536, 1024, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_198, mul_199, sub_58, where_19, div_25, view_328, bmm_39], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:63
            extern_kernels.bmm(view_331, reinterpret_tensor(buf167, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf169)
            del view_331
            buf170 = reinterpret_tensor(buf129, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [view_330, permute_129, clone_71], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf168, buf170, 3145728, stream=raw_stream0)
            tangents_19 = copy_if_misaligned(tangents_19)
            buf171 = reinterpret_tensor(buf168, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [view_333, permute_126, add_122], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_19, buf169, buf171, 49152, 64, stream=raw_stream0)
            del tangents_19
            buf172 = reinterpret_tensor(buf169, (4, 1024, 768), (786432, 768, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [view_327, add_121, view_330, view_333, permute_126, add_122, permute_127, view_334, permute_128, view_335, permute_129, clone_71, view_336, cat_3], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf171, buf172, 3145728, stream=raw_stream0)
            del buf171
            # Topologically Sorted Source Nodes: [view_327, add_121, view_330, view_333, permute_126, add_122, permute_127, view_334, permute_128, view_335, permute_129, clone_71, view_336, cat_3], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:64
            buf175 = torch.ops.aten.cat.default([reinterpret_tensor(buf170, (4, 1024, 768), (786432, 768, 1), 0), buf172, buf174], 2)
            del buf170
            buf176 = buf175
            assert_alignment(buf176, 16, 'torch.ops.aten.cat.default')
            del buf175
            buf178 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_337, mm_34], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:65
            extern_kernels.mm(permute_131, reinterpret_tensor(buf176, (4096, 2304), (2304, 1), 0), out=buf178)
            del permute_131
            buf179 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_337, sum_55], Original ATen: [aten.view, aten.sum]
            workspace_11 = workspace_8; del workspace_8  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf176, buf179, workspace_11, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_11, buf179, 2304, 40, stream=raw_stream0)
            buf177 = reinterpret_tensor(buf174, (4096, 768), (768, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [view_337, permute_130, mm_33], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:66
            extern_kernels.mm(reinterpret_tensor(buf176, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_111, (2304, 768), (1, 2304), 0), out=buf177)
            del buf176
            del primals_111
            buf184 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [view_339, mul_201, mul_202, sum_56, mul_203, sum_57, mul_204, sub_60, sub_61, mul_205, add_123], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf184, buf177, primals_108, mul_64, div_26, 4096, 768, stream=raw_stream0)
            del div_26
            del primals_108
            buf182 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf183 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_339, mul_206, sum_58, sum_59], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf177, mul_64, buf182, buf183, 768, 4096, stream=raw_stream0)
            del mul_64
            # Topologically Sorted Source Nodes: [npu_dropout_backward_12], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:67
            buf185 = torch.ops.npu.npu_dropout_backward.default(buf184, getitem_105, 0.10000000000000009)
            del getitem_105
            buf186 = buf185
            assert_alignment(buf186, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf185
            buf188 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_340, mm_36], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:68
            extern_kernels.mm(permute_133, reinterpret_tensor(buf186, (4096, 768), (768, 1), 0), out=buf188)
            del permute_133
            buf189 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_340, sum_60], Original ATen: [aten.view, aten.sum]
            workspace_12 = workspace_10; del workspace_10  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf186, buf189, workspace_12, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_12, buf189, 768, 40, stream=raw_stream0)
            buf187 = reinterpret_tensor(buf147, (4096, 3072), (3072, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [view_340, permute_132, mm_35], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:69
            extern_kernels.mm(reinterpret_tensor(buf186, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), out=buf187)
            del primals_107
            buf190 = reinterpret_tensor(buf187, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [view_342, x_61, mul_28, mul_207, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, mul_208, mul_209, sub_62, mul_210, mul_211, mul_212, pow_17, mul_213, mul_214, add_124, mul_215, add_125], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf190, addmm_30, 12582912, stream=raw_stream0)
            del addmm_30
            buf192 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_342, x_61, mul_28, mul_207, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, mul_208, mul_209, sub_62, mul_210, mul_211, mul_212, pow_17, mul_213, mul_214, add_124, mul_215, add_125, view_343, mm_38], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(permute_135, reinterpret_tensor(buf190, (4096, 3072), (3072, 1), 0), out=buf192)
            del permute_135
            buf193 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_342, x_61, mul_28, mul_207, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, mul_208, mul_209, sub_62, mul_210, mul_211, mul_212, pow_17, mul_213, mul_214, add_124, mul_215, add_125, view_343, sum_61], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf190, buf193, 3072, 4096, stream=raw_stream0)
            buf191 = reinterpret_tensor(buf186, (4096, 768), (768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [view_342, x_61, mul_28, mul_207, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, mul_208, mul_209, sub_62, mul_210, mul_211, mul_212, pow_17, mul_213, mul_214, add_124, mul_215, add_125, view_343, permute_134, mm_37], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:71
            extern_kernels.mm(reinterpret_tensor(buf190, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_105, (3072, 768), (1, 3072), 0), out=buf191)
            del primals_105
            buf198 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [view_345, mul_217, mul_218, sum_62, mul_219, sum_63, mul_220, sub_64, sub_65, mul_221, add_126], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf198, buf191, primals_102, mul_58, div_27, 4096, 768, stream=raw_stream0)
            del div_27
            del primals_102
            buf196 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf197 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_345, mul_222, sum_64, sum_65], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf191, mul_58, buf196, buf197, 768, 4096, stream=raw_stream0)
            del mul_58
            # Topologically Sorted Source Nodes: [npu_dropout_backward_13], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:72
            buf199 = torch.ops.npu.npu_dropout_backward.default(buf198, getitem_101, 0.10000000000000009)
            del getitem_101
            buf200 = buf199
            assert_alignment(buf200, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf199
            buf202 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_346, mm_40], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:73
            extern_kernels.mm(permute_137, reinterpret_tensor(buf200, (4096, 768), (768, 1), 0), out=buf202)
            del permute_137
            buf203 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_346, sum_66], Original ATen: [aten.view, aten.sum]
            workspace_13 = workspace_12; del workspace_12  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf200, buf203, workspace_13, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_13, buf203, 768, 40, stream=raw_stream0)
            buf201 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [view_346, permute_136, mm_39], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:74
            extern_kernels.mm(reinterpret_tensor(buf200, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 768), (1, 768), 0), out=buf201)
            del primals_101
            buf204 = reinterpret_tensor(buf200, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [view_348, view_349, permute_138, clone_72], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf201, buf204, 3145728, stream=raw_stream0)
            buf206 = reinterpret_tensor(buf201, (48, 1024, 64), (65536, 64, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [view_348, view_349, permute_138, clone_72, view_350, bmm_41], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:75
            extern_kernels.bmm(view_353, reinterpret_tensor(buf204, (48, 1024, 64), (65536, 64, 1), 0), out=buf206)
            del view_353
            tangents_18 = copy_if_misaligned(tangents_18)
            buf216 = reinterpret_tensor(buf206, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [view_355, add_127], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf216, tangents_18, 3145728, stream=raw_stream0)
            del tangents_18
            buf217 = reinterpret_tensor(buf177, (4, 1024, 768), (786432, 768, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [view_355, add_127, view_358, view_361, permute_143, add_128, permute_144, view_362, permute_145, view_363, permute_146, clone_77, view_364, cat_4], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf216, buf217, 3145728, stream=raw_stream0)
            buf205 = reinterpret_tensor(buf167, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [view_348, view_349, permute_138, clone_72, view_350, bmm_40], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:76
            extern_kernels.bmm(reinterpret_tensor(buf204, (48, 1024, 64), (65536, 64, 1), 0), view_351, out=buf205)
            del view_351
            # Topologically Sorted Source Nodes: [view_352, npu_dropout_backward_14], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:77
            buf207 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf205, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_99, 0.10000000000000009)
            del buf205
            del getitem_99
            buf208 = buf207
            assert_alignment(buf208, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf207
            buf210 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_223, sum_67, mul_224, sub_66, where_20, div_28], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf210, _softmax_7, primals_99, 49152, 1024, stream=raw_stream0)
            del _softmax_7
            del primals_99
            buf211 = reinterpret_tensor(buf204, (48, 1024, 64), (65536, 64, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_223, mul_224, sub_66, where_20, div_28, view_356, bmm_42], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:78
            extern_kernels.bmm(reinterpret_tensor(buf210, (48, 1024, 1024), (1048576, 1024, 1), 0), view_357, out=buf211)
            del view_357
            buf212 = reinterpret_tensor(buf216, (48, 64, 1024), (65536, 1024, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_223, mul_224, sub_66, where_20, div_28, view_356, bmm_43], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:79
            extern_kernels.bmm(view_359, reinterpret_tensor(buf210, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf212)
            del view_359
            buf213 = reinterpret_tensor(buf172, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [view_358, permute_146, clone_77], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf211, buf213, 3145728, stream=raw_stream0)
            tangents_17 = copy_if_misaligned(tangents_17)
            buf214 = reinterpret_tensor(buf211, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [view_361, permute_143, add_128], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_17, buf212, buf214, 49152, 64, stream=raw_stream0)
            del tangents_17
            buf215 = reinterpret_tensor(buf212, (4, 1024, 768), (786432, 768, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [view_355, add_127, view_358, view_361, permute_143, add_128, permute_144, view_362, permute_145, view_363, permute_146, clone_77, view_364, cat_4], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf214, buf215, 3145728, stream=raw_stream0)
            del buf214
            # Topologically Sorted Source Nodes: [view_355, add_127, view_358, view_361, permute_143, add_128, permute_144, view_362, permute_145, view_363, permute_146, clone_77, view_364, cat_4], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:80
            buf218 = torch.ops.aten.cat.default([reinterpret_tensor(buf213, (4, 1024, 768), (786432, 768, 1), 0), buf215, buf217], 2)
            del buf213
            buf219 = buf218
            assert_alignment(buf219, 16, 'torch.ops.aten.cat.default')
            del buf218
            buf221 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_365, mm_42], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:81
            extern_kernels.mm(permute_148, reinterpret_tensor(buf219, (4096, 2304), (2304, 1), 0), out=buf221)
            del permute_148
            buf222 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_365, sum_68], Original ATen: [aten.view, aten.sum]
            workspace_14 = workspace_11; del workspace_11  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf219, buf222, workspace_14, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_14, buf222, 2304, 40, stream=raw_stream0)
            buf220 = reinterpret_tensor(buf217, (4096, 768), (768, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [view_365, permute_147, mm_41], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:82
            extern_kernels.mm(reinterpret_tensor(buf219, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_98, (2304, 768), (1, 2304), 0), out=buf220)
            del buf219
            del primals_98
            buf227 = buf198; del buf198  # reuse
            # Topologically Sorted Source Nodes: [view_367, mul_226, mul_227, sum_69, mul_228, sum_70, mul_229, sub_68, sub_69, mul_230, add_129], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf227, buf220, primals_95, mul_56, div_29, 4096, 768, stream=raw_stream0)
            del div_29
            del primals_95
            buf225 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf226 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_367, mul_231, sum_71, sum_72], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf220, mul_56, buf225, buf226, 768, 4096, stream=raw_stream0)
            del mul_56
            # Topologically Sorted Source Nodes: [npu_dropout_backward_15], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:83
            buf228 = torch.ops.npu.npu_dropout_backward.default(buf227, getitem_92, 0.10000000000000009)
            del getitem_92
            buf229 = buf228
            assert_alignment(buf229, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf228
            buf231 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_368, mm_44], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:84
            extern_kernels.mm(permute_150, reinterpret_tensor(buf229, (4096, 768), (768, 1), 0), out=buf231)
            del permute_150
            buf232 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_368, sum_73], Original ATen: [aten.view, aten.sum]
            workspace_15 = workspace_13; del workspace_13  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf229, buf232, workspace_15, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_15, buf232, 768, 40, stream=raw_stream0)
            buf230 = reinterpret_tensor(buf190, (4096, 3072), (3072, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [view_368, permute_149, mm_43], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:85
            extern_kernels.mm(reinterpret_tensor(buf229, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 3072), (1, 768), 0), out=buf230)
            del primals_94
            buf233 = reinterpret_tensor(buf230, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [view_370, x_53, mul_24, mul_232, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, mul_233, mul_234, sub_70, mul_235, mul_236, mul_237, pow_18, mul_238, mul_239, add_130, mul_240, add_131], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf233, addmm_26, 12582912, stream=raw_stream0)
            del addmm_26
            buf235 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_370, x_53, mul_24, mul_232, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, mul_233, mul_234, sub_70, mul_235, mul_236, mul_237, pow_18, mul_238, mul_239, add_130, mul_240, add_131, view_371, mm_46], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(permute_152, reinterpret_tensor(buf233, (4096, 3072), (3072, 1), 0), out=buf235)
            del permute_152
            buf236 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_370, x_53, mul_24, mul_232, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, mul_233, mul_234, sub_70, mul_235, mul_236, mul_237, pow_18, mul_238, mul_239, add_130, mul_240, add_131, view_371, sum_74], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf233, buf236, 3072, 4096, stream=raw_stream0)
            buf234 = reinterpret_tensor(buf229, (4096, 768), (768, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [view_370, x_53, mul_24, mul_232, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, mul_233, mul_234, sub_70, mul_235, mul_236, mul_237, pow_18, mul_238, mul_239, add_130, mul_240, add_131, view_371, permute_151, mm_45], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:87
            extern_kernels.mm(reinterpret_tensor(buf233, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_92, (3072, 768), (1, 3072), 0), out=buf234)
            del primals_92
            buf241 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [view_373, mul_242, mul_243, sum_75, mul_244, sum_76, mul_245, sub_72, sub_73, mul_246, add_132], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf241, buf234, primals_89, mul_50, div_30, 4096, 768, stream=raw_stream0)
            del div_30
            del primals_89
            buf239 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf240 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_373, mul_247, sum_77, sum_78], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf234, mul_50, buf239, buf240, 768, 4096, stream=raw_stream0)
            del mul_50
            # Topologically Sorted Source Nodes: [npu_dropout_backward_16], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:88
            buf242 = torch.ops.npu.npu_dropout_backward.default(buf241, getitem_88, 0.10000000000000009)
            del getitem_88
            buf243 = buf242
            assert_alignment(buf243, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf242
            buf245 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_374, mm_48], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:89
            extern_kernels.mm(permute_154, reinterpret_tensor(buf243, (4096, 768), (768, 1), 0), out=buf245)
            del permute_154
            buf246 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_374, sum_79], Original ATen: [aten.view, aten.sum]
            workspace_16 = workspace_15; del workspace_15  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf243, buf246, workspace_16, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_16, buf246, 768, 40, stream=raw_stream0)
            buf244 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [view_374, permute_153, mm_47], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:90
            extern_kernels.mm(reinterpret_tensor(buf243, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf244)
            del primals_88
            buf247 = reinterpret_tensor(buf243, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [view_376, view_377, permute_155, clone_78], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf244, buf247, 3145728, stream=raw_stream0)
            buf249 = reinterpret_tensor(buf244, (48, 1024, 64), (65536, 64, 1), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [view_376, view_377, permute_155, clone_78, view_378, bmm_45], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:91
            extern_kernels.bmm(view_381, reinterpret_tensor(buf247, (48, 1024, 64), (65536, 64, 1), 0), out=buf249)
            del view_381
            tangents_16 = copy_if_misaligned(tangents_16)
            buf259 = reinterpret_tensor(buf249, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [view_383, add_133], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf259, tangents_16, 3145728, stream=raw_stream0)
            del tangents_16
            buf260 = reinterpret_tensor(buf220, (4, 1024, 768), (786432, 768, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [view_383, add_133, view_386, view_389, permute_160, add_134, permute_161, view_390, permute_162, view_391, permute_163, clone_83, view_392, cat_5], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf259, buf260, 3145728, stream=raw_stream0)
            buf248 = reinterpret_tensor(buf210, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [view_376, view_377, permute_155, clone_78, view_378, bmm_44], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:92
            extern_kernels.bmm(reinterpret_tensor(buf247, (48, 1024, 64), (65536, 64, 1), 0), view_379, out=buf248)
            del view_379
            # Topologically Sorted Source Nodes: [view_380, npu_dropout_backward_17], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:93
            buf250 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf248, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_86, 0.10000000000000009)
            del buf248
            del getitem_86
            buf251 = buf250
            assert_alignment(buf251, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf250
            buf253 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_248, sum_80, mul_249, sub_74, where_21, div_31], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf253, _softmax_6, primals_86, 49152, 1024, stream=raw_stream0)
            del _softmax_6
            del primals_86
            buf254 = reinterpret_tensor(buf247, (48, 1024, 64), (65536, 64, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_248, mul_249, sub_74, where_21, div_31, view_384, bmm_46], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:94
            extern_kernels.bmm(reinterpret_tensor(buf253, (48, 1024, 1024), (1048576, 1024, 1), 0), view_385, out=buf254)
            del view_385
            buf255 = reinterpret_tensor(buf259, (48, 64, 1024), (65536, 1024, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_248, mul_249, sub_74, where_21, div_31, view_384, bmm_47], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:95
            extern_kernels.bmm(view_387, reinterpret_tensor(buf253, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf255)
            del view_387
            buf256 = reinterpret_tensor(buf215, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [view_386, permute_163, clone_83], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf254, buf256, 3145728, stream=raw_stream0)
            tangents_15 = copy_if_misaligned(tangents_15)
            buf257 = reinterpret_tensor(buf254, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [view_389, permute_160, add_134], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_15, buf255, buf257, 49152, 64, stream=raw_stream0)
            del tangents_15
            buf258 = reinterpret_tensor(buf255, (4, 1024, 768), (786432, 768, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [view_383, add_133, view_386, view_389, permute_160, add_134, permute_161, view_390, permute_162, view_391, permute_163, clone_83, view_392, cat_5], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf257, buf258, 3145728, stream=raw_stream0)
            del buf257
            # Topologically Sorted Source Nodes: [view_383, add_133, view_386, view_389, permute_160, add_134, permute_161, view_390, permute_162, view_391, permute_163, clone_83, view_392, cat_5], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:96
            buf261 = torch.ops.aten.cat.default([reinterpret_tensor(buf256, (4, 1024, 768), (786432, 768, 1), 0), buf258, buf260], 2)
            del buf256
            buf262 = buf261
            assert_alignment(buf262, 16, 'torch.ops.aten.cat.default')
            del buf261
            buf264 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_393, mm_50], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:97
            extern_kernels.mm(permute_165, reinterpret_tensor(buf262, (4096, 2304), (2304, 1), 0), out=buf264)
            del permute_165
            buf265 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_393, sum_81], Original ATen: [aten.view, aten.sum]
            workspace_17 = workspace_14; del workspace_14  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf262, buf265, workspace_17, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_17, buf265, 2304, 40, stream=raw_stream0)
            buf263 = reinterpret_tensor(buf260, (4096, 768), (768, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [view_393, permute_164, mm_49], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:98
            extern_kernels.mm(reinterpret_tensor(buf262, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_85, (2304, 768), (1, 2304), 0), out=buf263)
            del buf262
            del primals_85
            buf270 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [view_395, mul_251, mul_252, sum_82, mul_253, sum_83, mul_254, sub_76, sub_77, mul_255, add_135], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf270, buf263, primals_82, mul_48, div_32, 4096, 768, stream=raw_stream0)
            del div_32
            del primals_82
            buf268 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf269 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_395, mul_256, sum_84, sum_85], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf263, mul_48, buf268, buf269, 768, 4096, stream=raw_stream0)
            del mul_48
            # Topologically Sorted Source Nodes: [npu_dropout_backward_18], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:99
            buf271 = torch.ops.npu.npu_dropout_backward.default(buf270, getitem_79, 0.10000000000000009)
            del getitem_79
            buf272 = buf271
            assert_alignment(buf272, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf271
            buf274 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_396, mm_52], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:100
            extern_kernels.mm(permute_167, reinterpret_tensor(buf272, (4096, 768), (768, 1), 0), out=buf274)
            del permute_167
            buf275 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_396, sum_86], Original ATen: [aten.view, aten.sum]
            workspace_18 = workspace_16; del workspace_16  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf272, buf275, workspace_18, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_18, buf275, 768, 40, stream=raw_stream0)
            buf273 = reinterpret_tensor(buf233, (4096, 3072), (3072, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [view_396, permute_166, mm_51], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:101
            extern_kernels.mm(reinterpret_tensor(buf272, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 3072), (1, 768), 0), out=buf273)
            del primals_81
            buf276 = reinterpret_tensor(buf273, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf273  # reuse
            # Topologically Sorted Source Nodes: [view_398, x_45, mul_20, mul_257, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, mul_258, mul_259, sub_78, mul_260, mul_261, mul_262, pow_19, mul_263, mul_264, add_136, mul_265, add_137], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf276, addmm_22, 12582912, stream=raw_stream0)
            del addmm_22
            buf278 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_398, x_45, mul_20, mul_257, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, mul_258, mul_259, sub_78, mul_260, mul_261, mul_262, pow_19, mul_263, mul_264, add_136, mul_265, add_137, view_399, mm_54], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:102
            extern_kernels.mm(permute_169, reinterpret_tensor(buf276, (4096, 3072), (3072, 1), 0), out=buf278)
            del permute_169
            buf279 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_398, x_45, mul_20, mul_257, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, mul_258, mul_259, sub_78, mul_260, mul_261, mul_262, pow_19, mul_263, mul_264, add_136, mul_265, add_137, view_399, sum_87], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf276, buf279, 3072, 4096, stream=raw_stream0)
            buf277 = reinterpret_tensor(buf272, (4096, 768), (768, 1), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [view_398, x_45, mul_20, mul_257, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, mul_258, mul_259, sub_78, mul_260, mul_261, mul_262, pow_19, mul_263, mul_264, add_136, mul_265, add_137, view_399, permute_168, mm_53], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:103
            extern_kernels.mm(reinterpret_tensor(buf276, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 768), (1, 3072), 0), out=buf277)
            del primals_79
            buf284 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [view_401, mul_267, mul_268, sum_88, mul_269, sum_89, mul_270, sub_80, sub_81, mul_271, add_138], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf284, buf277, primals_76, mul_42, div_33, 4096, 768, stream=raw_stream0)
            del div_33
            del primals_76
            buf282 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf283 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_401, mul_272, sum_90, sum_91], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf277, mul_42, buf282, buf283, 768, 4096, stream=raw_stream0)
            del mul_42
            # Topologically Sorted Source Nodes: [npu_dropout_backward_19], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:104
            buf285 = torch.ops.npu.npu_dropout_backward.default(buf284, getitem_75, 0.10000000000000009)
            del getitem_75
            buf286 = buf285
            assert_alignment(buf286, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf285
            buf288 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_402, mm_56], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:105
            extern_kernels.mm(permute_171, reinterpret_tensor(buf286, (4096, 768), (768, 1), 0), out=buf288)
            del permute_171
            buf289 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_402, sum_92], Original ATen: [aten.view, aten.sum]
            workspace_19 = workspace_18; del workspace_18  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf286, buf289, workspace_19, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_19, buf289, 768, 40, stream=raw_stream0)
            buf287 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [view_402, permute_170, mm_55], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:106
            extern_kernels.mm(reinterpret_tensor(buf286, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), out=buf287)
            del primals_75
            buf290 = reinterpret_tensor(buf286, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [view_404, view_405, permute_172, clone_84], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf287, buf290, 3145728, stream=raw_stream0)
            buf292 = reinterpret_tensor(buf287, (48, 1024, 64), (65536, 64, 1), 0); del buf287  # reuse
            # Topologically Sorted Source Nodes: [view_404, view_405, permute_172, clone_84, view_406, bmm_49], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:107
            extern_kernels.bmm(view_409, reinterpret_tensor(buf290, (48, 1024, 64), (65536, 64, 1), 0), out=buf292)
            del view_409
            tangents_14 = copy_if_misaligned(tangents_14)
            buf302 = reinterpret_tensor(buf292, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [view_411, add_139], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf302, tangents_14, 3145728, stream=raw_stream0)
            del tangents_14
            buf303 = reinterpret_tensor(buf263, (4, 1024, 768), (786432, 768, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [view_411, add_139, view_414, view_417, permute_177, add_140, permute_178, view_418, permute_179, view_419, permute_180, clone_89, view_420, cat_6], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf302, buf303, 3145728, stream=raw_stream0)
            buf291 = reinterpret_tensor(buf253, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [view_404, view_405, permute_172, clone_84, view_406, bmm_48], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:108
            extern_kernels.bmm(reinterpret_tensor(buf290, (48, 1024, 64), (65536, 64, 1), 0), view_407, out=buf291)
            del view_407
            # Topologically Sorted Source Nodes: [view_408, npu_dropout_backward_20], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:109
            buf293 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf291, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_73, 0.10000000000000009)
            del buf291
            del getitem_73
            buf294 = buf293
            assert_alignment(buf294, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf293
            buf296 = buf294; del buf294  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_273, sum_93, mul_274, sub_82, where_22, div_34], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf296, _softmax_5, primals_73, 49152, 1024, stream=raw_stream0)
            del _softmax_5
            del primals_73
            buf297 = reinterpret_tensor(buf290, (48, 1024, 64), (65536, 64, 1), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_273, mul_274, sub_82, where_22, div_34, view_412, bmm_50], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:110
            extern_kernels.bmm(reinterpret_tensor(buf296, (48, 1024, 1024), (1048576, 1024, 1), 0), view_413, out=buf297)
            del view_413
            buf298 = reinterpret_tensor(buf302, (48, 64, 1024), (65536, 1024, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_273, mul_274, sub_82, where_22, div_34, view_412, bmm_51], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:111
            extern_kernels.bmm(view_415, reinterpret_tensor(buf296, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf298)
            del view_415
            buf299 = reinterpret_tensor(buf258, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [view_414, permute_180, clone_89], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf297, buf299, 3145728, stream=raw_stream0)
            tangents_13 = copy_if_misaligned(tangents_13)
            buf300 = reinterpret_tensor(buf297, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [view_417, permute_177, add_140], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_13, buf298, buf300, 49152, 64, stream=raw_stream0)
            del tangents_13
            buf301 = reinterpret_tensor(buf298, (4, 1024, 768), (786432, 768, 1), 0); del buf298  # reuse
            # Topologically Sorted Source Nodes: [view_411, add_139, view_414, view_417, permute_177, add_140, permute_178, view_418, permute_179, view_419, permute_180, clone_89, view_420, cat_6], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf300, buf301, 3145728, stream=raw_stream0)
            del buf300
            # Topologically Sorted Source Nodes: [view_411, add_139, view_414, view_417, permute_177, add_140, permute_178, view_418, permute_179, view_419, permute_180, clone_89, view_420, cat_6], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:112
            buf304 = torch.ops.aten.cat.default([reinterpret_tensor(buf299, (4, 1024, 768), (786432, 768, 1), 0), buf301, buf303], 2)
            del buf299
            buf305 = buf304
            assert_alignment(buf305, 16, 'torch.ops.aten.cat.default')
            del buf304
            buf307 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_421, mm_58], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:113
            extern_kernels.mm(permute_182, reinterpret_tensor(buf305, (4096, 2304), (2304, 1), 0), out=buf307)
            del permute_182
            buf308 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_421, sum_94], Original ATen: [aten.view, aten.sum]
            workspace_20 = workspace_17; del workspace_17  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf305, buf308, workspace_20, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_20, buf308, 2304, 40, stream=raw_stream0)
            buf306 = reinterpret_tensor(buf303, (4096, 768), (768, 1), 0); del buf303  # reuse
            # Topologically Sorted Source Nodes: [view_421, permute_181, mm_57], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:114
            extern_kernels.mm(reinterpret_tensor(buf305, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_72, (2304, 768), (1, 2304), 0), out=buf306)
            del buf305
            del primals_72
            buf313 = buf284; del buf284  # reuse
            # Topologically Sorted Source Nodes: [view_423, mul_276, mul_277, sum_95, mul_278, sum_96, mul_279, sub_84, sub_85, mul_280, add_141], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf313, buf306, primals_69, mul_40, div_35, 4096, 768, stream=raw_stream0)
            del div_35
            del primals_69
            buf311 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf312 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_423, mul_281, sum_97, sum_98], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf306, mul_40, buf311, buf312, 768, 4096, stream=raw_stream0)
            del mul_40
            # Topologically Sorted Source Nodes: [npu_dropout_backward_21], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:115
            buf314 = torch.ops.npu.npu_dropout_backward.default(buf313, getitem_66, 0.10000000000000009)
            del getitem_66
            buf315 = buf314
            assert_alignment(buf315, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf314
            buf317 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_424, mm_60], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:116
            extern_kernels.mm(permute_184, reinterpret_tensor(buf315, (4096, 768), (768, 1), 0), out=buf317)
            del permute_184
            buf318 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_424, sum_99], Original ATen: [aten.view, aten.sum]
            workspace_21 = workspace_19; del workspace_19  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf315, buf318, workspace_21, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_21, buf318, 768, 40, stream=raw_stream0)
            buf316 = reinterpret_tensor(buf276, (4096, 3072), (3072, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [view_424, permute_183, mm_59], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:117
            extern_kernels.mm(reinterpret_tensor(buf315, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_68, (768, 3072), (1, 768), 0), out=buf316)
            del primals_68
            buf319 = reinterpret_tensor(buf316, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf316  # reuse
            # Topologically Sorted Source Nodes: [view_426, x_37, mul_16, mul_282, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, mul_283, mul_284, sub_86, mul_285, mul_286, mul_287, pow_20, mul_288, mul_289, add_142, mul_290, add_143], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf319, addmm_18, 12582912, stream=raw_stream0)
            del addmm_18
            buf321 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_426, x_37, mul_16, mul_282, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, mul_283, mul_284, sub_86, mul_285, mul_286, mul_287, pow_20, mul_288, mul_289, add_142, mul_290, add_143, view_427, mm_62], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:118
            extern_kernels.mm(permute_186, reinterpret_tensor(buf319, (4096, 3072), (3072, 1), 0), out=buf321)
            del permute_186
            buf322 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_426, x_37, mul_16, mul_282, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, mul_283, mul_284, sub_86, mul_285, mul_286, mul_287, pow_20, mul_288, mul_289, add_142, mul_290, add_143, view_427, sum_100], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf319, buf322, 3072, 4096, stream=raw_stream0)
            buf320 = reinterpret_tensor(buf315, (4096, 768), (768, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [view_426, x_37, mul_16, mul_282, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, mul_283, mul_284, sub_86, mul_285, mul_286, mul_287, pow_20, mul_288, mul_289, add_142, mul_290, add_143, view_427, permute_185, mm_61], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:119
            extern_kernels.mm(reinterpret_tensor(buf319, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), out=buf320)
            del primals_66
            buf327 = buf313; del buf313  # reuse
            # Topologically Sorted Source Nodes: [view_429, mul_292, mul_293, sum_101, mul_294, sum_102, mul_295, sub_88, sub_89, mul_296, add_144], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf327, buf320, primals_63, mul_34, div_36, 4096, 768, stream=raw_stream0)
            del div_36
            del primals_63
            buf325 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf326 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_429, mul_297, sum_103, sum_104], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf320, mul_34, buf325, buf326, 768, 4096, stream=raw_stream0)
            del mul_34
            # Topologically Sorted Source Nodes: [npu_dropout_backward_22], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:120
            buf328 = torch.ops.npu.npu_dropout_backward.default(buf327, getitem_62, 0.10000000000000009)
            del getitem_62
            buf329 = buf328
            assert_alignment(buf329, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf328
            buf331 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_430, mm_64], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:121
            extern_kernels.mm(permute_188, reinterpret_tensor(buf329, (4096, 768), (768, 1), 0), out=buf331)
            del permute_188
            buf332 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_430, sum_105], Original ATen: [aten.view, aten.sum]
            workspace_22 = workspace_21; del workspace_21  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf329, buf332, workspace_22, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_22, buf332, 768, 40, stream=raw_stream0)
            buf330 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [view_430, permute_187, mm_63], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:122
            extern_kernels.mm(reinterpret_tensor(buf329, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_62, (768, 768), (1, 768), 0), out=buf330)
            del primals_62
            buf333 = reinterpret_tensor(buf329, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf329  # reuse
            # Topologically Sorted Source Nodes: [view_432, view_433, permute_189, clone_90], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf330, buf333, 3145728, stream=raw_stream0)
            buf335 = reinterpret_tensor(buf330, (48, 1024, 64), (65536, 64, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [view_432, view_433, permute_189, clone_90, view_434, bmm_53], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:123
            extern_kernels.bmm(view_437, reinterpret_tensor(buf333, (48, 1024, 64), (65536, 64, 1), 0), out=buf335)
            del view_437
            tangents_12 = copy_if_misaligned(tangents_12)
            buf345 = reinterpret_tensor(buf335, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [view_439, add_145], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf345, tangents_12, 3145728, stream=raw_stream0)
            del tangents_12
            buf346 = reinterpret_tensor(buf306, (4, 1024, 768), (786432, 768, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [view_439, add_145, view_442, view_445, permute_194, add_146, permute_195, view_446, permute_196, view_447, permute_197, clone_95, view_448, cat_7], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf345, buf346, 3145728, stream=raw_stream0)
            buf334 = reinterpret_tensor(buf296, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [view_432, view_433, permute_189, clone_90, view_434, bmm_52], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:124
            extern_kernels.bmm(reinterpret_tensor(buf333, (48, 1024, 64), (65536, 64, 1), 0), view_435, out=buf334)
            del view_435
            # Topologically Sorted Source Nodes: [view_436, npu_dropout_backward_23], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:125
            buf336 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf334, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_60, 0.10000000000000009)
            del buf334
            del getitem_60
            buf337 = buf336
            assert_alignment(buf337, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf336
            buf339 = buf337; del buf337  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_298, sum_106, mul_299, sub_90, where_23, div_37], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf339, _softmax_4, primals_60, 49152, 1024, stream=raw_stream0)
            del _softmax_4
            del primals_60
            buf340 = reinterpret_tensor(buf333, (48, 1024, 64), (65536, 64, 1), 0); del buf333  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_298, mul_299, sub_90, where_23, div_37, view_440, bmm_54], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:126
            extern_kernels.bmm(reinterpret_tensor(buf339, (48, 1024, 1024), (1048576, 1024, 1), 0), view_441, out=buf340)
            del view_441
            buf341 = reinterpret_tensor(buf345, (48, 64, 1024), (65536, 1024, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_298, mul_299, sub_90, where_23, div_37, view_440, bmm_55], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:127
            extern_kernels.bmm(view_443, reinterpret_tensor(buf339, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf341)
            del view_443
            buf342 = reinterpret_tensor(buf301, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [view_442, permute_197, clone_95], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf340, buf342, 3145728, stream=raw_stream0)
            tangents_11 = copy_if_misaligned(tangents_11)
            buf343 = reinterpret_tensor(buf340, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [view_445, permute_194, add_146], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_11, buf341, buf343, 49152, 64, stream=raw_stream0)
            del tangents_11
            buf344 = reinterpret_tensor(buf341, (4, 1024, 768), (786432, 768, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [view_439, add_145, view_442, view_445, permute_194, add_146, permute_195, view_446, permute_196, view_447, permute_197, clone_95, view_448, cat_7], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf343, buf344, 3145728, stream=raw_stream0)
            del buf343
            # Topologically Sorted Source Nodes: [view_439, add_145, view_442, view_445, permute_194, add_146, permute_195, view_446, permute_196, view_447, permute_197, clone_95, view_448, cat_7], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:128
            buf347 = torch.ops.aten.cat.default([reinterpret_tensor(buf342, (4, 1024, 768), (786432, 768, 1), 0), buf344, buf346], 2)
            del buf342
            buf348 = buf347
            assert_alignment(buf348, 16, 'torch.ops.aten.cat.default')
            del buf347
            buf350 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_449, mm_66], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:129
            extern_kernels.mm(permute_199, reinterpret_tensor(buf348, (4096, 2304), (2304, 1), 0), out=buf350)
            del permute_199
            buf351 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_449, sum_107], Original ATen: [aten.view, aten.sum]
            workspace_23 = workspace_20; del workspace_20  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf348, buf351, workspace_23, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_23, buf351, 2304, 40, stream=raw_stream0)
            buf349 = reinterpret_tensor(buf346, (4096, 768), (768, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [view_449, permute_198, mm_65], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:130
            extern_kernels.mm(reinterpret_tensor(buf348, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_59, (2304, 768), (1, 2304), 0), out=buf349)
            del buf348
            del primals_59
            buf356 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [view_451, mul_301, mul_302, sum_108, mul_303, sum_109, mul_304, sub_92, sub_93, mul_305, add_147], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf356, buf349, primals_56, mul_32, div_38, 4096, 768, stream=raw_stream0)
            del div_38
            del primals_56
            buf354 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf355 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_451, mul_306, sum_110, sum_111], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf349, mul_32, buf354, buf355, 768, 4096, stream=raw_stream0)
            del mul_32
            # Topologically Sorted Source Nodes: [npu_dropout_backward_24], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:131
            buf357 = torch.ops.npu.npu_dropout_backward.default(buf356, getitem_53, 0.10000000000000009)
            del getitem_53
            buf358 = buf357
            assert_alignment(buf358, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf357
            buf360 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_452, mm_68], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:132
            extern_kernels.mm(permute_201, reinterpret_tensor(buf358, (4096, 768), (768, 1), 0), out=buf360)
            del permute_201
            buf361 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_452, sum_112], Original ATen: [aten.view, aten.sum]
            workspace_24 = workspace_22; del workspace_22  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf358, buf361, workspace_24, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_24, buf361, 768, 40, stream=raw_stream0)
            buf359 = reinterpret_tensor(buf319, (4096, 3072), (3072, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [view_452, permute_200, mm_67], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:133
            extern_kernels.mm(reinterpret_tensor(buf358, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_55, (768, 3072), (1, 768), 0), out=buf359)
            del primals_55
            buf362 = reinterpret_tensor(buf359, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [view_454, x_29, mul_12, mul_307, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, mul_308, mul_309, sub_94, mul_310, mul_311, mul_312, pow_21, mul_313, mul_314, add_148, mul_315, add_149], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf362, addmm_14, 12582912, stream=raw_stream0)
            del addmm_14
            buf364 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_454, x_29, mul_12, mul_307, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, mul_308, mul_309, sub_94, mul_310, mul_311, mul_312, pow_21, mul_313, mul_314, add_148, mul_315, add_149, view_455, mm_70], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:134
            extern_kernels.mm(permute_203, reinterpret_tensor(buf362, (4096, 3072), (3072, 1), 0), out=buf364)
            del permute_203
            buf365 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_454, x_29, mul_12, mul_307, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, mul_308, mul_309, sub_94, mul_310, mul_311, mul_312, pow_21, mul_313, mul_314, add_148, mul_315, add_149, view_455, sum_113], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf362, buf365, 3072, 4096, stream=raw_stream0)
            buf363 = reinterpret_tensor(buf358, (4096, 768), (768, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [view_454, x_29, mul_12, mul_307, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, mul_308, mul_309, sub_94, mul_310, mul_311, mul_312, pow_21, mul_313, mul_314, add_148, mul_315, add_149, view_455, permute_202, mm_69], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:135
            extern_kernels.mm(reinterpret_tensor(buf362, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_53, (3072, 768), (1, 3072), 0), out=buf363)
            del primals_53
            buf370 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [view_457, mul_317, mul_318, sum_114, mul_319, sum_115, mul_320, sub_96, sub_97, mul_321, add_150], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf370, buf363, primals_50, mul_26, div_39, 4096, 768, stream=raw_stream0)
            del div_39
            del primals_50
            buf368 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf369 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_457, mul_322, sum_116, sum_117], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf363, mul_26, buf368, buf369, 768, 4096, stream=raw_stream0)
            del mul_26
            # Topologically Sorted Source Nodes: [npu_dropout_backward_25], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:136
            buf371 = torch.ops.npu.npu_dropout_backward.default(buf370, getitem_49, 0.10000000000000009)
            del getitem_49
            buf372 = buf371
            assert_alignment(buf372, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf371
            buf374 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_458, mm_72], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:137
            extern_kernels.mm(permute_205, reinterpret_tensor(buf372, (4096, 768), (768, 1), 0), out=buf374)
            del permute_205
            buf375 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_458, sum_118], Original ATen: [aten.view, aten.sum]
            workspace_25 = workspace_24; del workspace_24  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf372, buf375, workspace_25, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_25, buf375, 768, 40, stream=raw_stream0)
            buf373 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [view_458, permute_204, mm_71], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:138
            extern_kernels.mm(reinterpret_tensor(buf372, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 768), (1, 768), 0), out=buf373)
            del primals_49
            buf376 = reinterpret_tensor(buf372, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [view_460, view_461, permute_206, clone_96], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf373, buf376, 3145728, stream=raw_stream0)
            buf378 = reinterpret_tensor(buf373, (48, 1024, 64), (65536, 64, 1), 0); del buf373  # reuse
            # Topologically Sorted Source Nodes: [view_460, view_461, permute_206, clone_96, view_462, bmm_57], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:139
            extern_kernels.bmm(view_465, reinterpret_tensor(buf376, (48, 1024, 64), (65536, 64, 1), 0), out=buf378)
            del view_465
            tangents_10 = copy_if_misaligned(tangents_10)
            buf388 = reinterpret_tensor(buf378, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [view_467, add_151], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf388, tangents_10, 3145728, stream=raw_stream0)
            del tangents_10
            buf389 = reinterpret_tensor(buf349, (4, 1024, 768), (786432, 768, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [view_467, add_151, view_470, view_473, permute_211, add_152, permute_212, view_474, permute_213, view_475, permute_214, clone_101, view_476, cat_8], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf388, buf389, 3145728, stream=raw_stream0)
            buf377 = reinterpret_tensor(buf339, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [view_460, view_461, permute_206, clone_96, view_462, bmm_56], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:140
            extern_kernels.bmm(reinterpret_tensor(buf376, (48, 1024, 64), (65536, 64, 1), 0), view_463, out=buf377)
            del view_463
            # Topologically Sorted Source Nodes: [view_464, npu_dropout_backward_26], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:141
            buf379 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf377, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_47, 0.10000000000000009)
            del buf377
            del getitem_47
            buf380 = buf379
            assert_alignment(buf380, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf379
            buf382 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_323, sum_119, mul_324, sub_98, where_24, div_40], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf382, _softmax_3, primals_47, 49152, 1024, stream=raw_stream0)
            del _softmax_3
            del primals_47
            buf383 = reinterpret_tensor(buf376, (48, 1024, 64), (65536, 64, 1), 0); del buf376  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_323, mul_324, sub_98, where_24, div_40, view_468, bmm_58], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:142
            extern_kernels.bmm(reinterpret_tensor(buf382, (48, 1024, 1024), (1048576, 1024, 1), 0), view_469, out=buf383)
            del view_469
            buf384 = reinterpret_tensor(buf388, (48, 64, 1024), (65536, 1024, 1), 0); del buf388  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_323, mul_324, sub_98, where_24, div_40, view_468, bmm_59], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:143
            extern_kernels.bmm(view_471, reinterpret_tensor(buf382, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf384)
            del view_471
            buf385 = reinterpret_tensor(buf344, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf344  # reuse
            # Topologically Sorted Source Nodes: [view_470, permute_214, clone_101], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf383, buf385, 3145728, stream=raw_stream0)
            tangents_9 = copy_if_misaligned(tangents_9)
            buf386 = reinterpret_tensor(buf383, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [view_473, permute_211, add_152], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_9, buf384, buf386, 49152, 64, stream=raw_stream0)
            del tangents_9
            buf387 = reinterpret_tensor(buf384, (4, 1024, 768), (786432, 768, 1), 0); del buf384  # reuse
            # Topologically Sorted Source Nodes: [view_467, add_151, view_470, view_473, permute_211, add_152, permute_212, view_474, permute_213, view_475, permute_214, clone_101, view_476, cat_8], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf386, buf387, 3145728, stream=raw_stream0)
            del buf386
            # Topologically Sorted Source Nodes: [view_467, add_151, view_470, view_473, permute_211, add_152, permute_212, view_474, permute_213, view_475, permute_214, clone_101, view_476, cat_8], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:144
            buf390 = torch.ops.aten.cat.default([reinterpret_tensor(buf385, (4, 1024, 768), (786432, 768, 1), 0), buf387, buf389], 2)
            del buf385
            buf391 = buf390
            assert_alignment(buf391, 16, 'torch.ops.aten.cat.default')
            del buf390
            buf393 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_477, mm_74], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:145
            extern_kernels.mm(permute_216, reinterpret_tensor(buf391, (4096, 2304), (2304, 1), 0), out=buf393)
            del permute_216
            buf394 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_477, sum_120], Original ATen: [aten.view, aten.sum]
            workspace_26 = workspace_23; del workspace_23  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf391, buf394, workspace_26, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_26, buf394, 2304, 40, stream=raw_stream0)
            buf392 = reinterpret_tensor(buf389, (4096, 768), (768, 1), 0); del buf389  # reuse
            # Topologically Sorted Source Nodes: [view_477, permute_215, mm_73], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:146
            extern_kernels.mm(reinterpret_tensor(buf391, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_46, (2304, 768), (1, 2304), 0), out=buf392)
            del buf391
            del primals_46
            buf399 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [view_479, mul_326, mul_327, sum_121, mul_328, sum_122, mul_329, sub_100, sub_101, mul_330, add_153], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf399, buf392, primals_43, mul_24, div_41, 4096, 768, stream=raw_stream0)
            del div_41
            del primals_43
            buf397 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf398 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_479, mul_331, sum_123, sum_124], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf392, mul_24, buf397, buf398, 768, 4096, stream=raw_stream0)
            del mul_24
            # Topologically Sorted Source Nodes: [npu_dropout_backward_27], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:147
            buf400 = torch.ops.npu.npu_dropout_backward.default(buf399, getitem_40, 0.10000000000000009)
            del getitem_40
            buf401 = buf400
            assert_alignment(buf401, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf400
            buf403 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_480, mm_76], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:148
            extern_kernels.mm(permute_218, reinterpret_tensor(buf401, (4096, 768), (768, 1), 0), out=buf403)
            del permute_218
            buf404 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_480, sum_125], Original ATen: [aten.view, aten.sum]
            workspace_27 = workspace_25; del workspace_25  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf401, buf404, workspace_27, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_27, buf404, 768, 40, stream=raw_stream0)
            buf402 = reinterpret_tensor(buf362, (4096, 3072), (3072, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [view_480, permute_217, mm_75], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:149
            extern_kernels.mm(reinterpret_tensor(buf401, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 3072), (1, 768), 0), out=buf402)
            del primals_42
            buf405 = reinterpret_tensor(buf402, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf402  # reuse
            # Topologically Sorted Source Nodes: [view_482, x_21, mul_8, mul_332, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, mul_333, mul_334, sub_102, mul_335, mul_336, mul_337, pow_22, mul_338, mul_339, add_154, mul_340, add_155], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf405, addmm_10, 12582912, stream=raw_stream0)
            del addmm_10
            buf407 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_482, x_21, mul_8, mul_332, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, mul_333, mul_334, sub_102, mul_335, mul_336, mul_337, pow_22, mul_338, mul_339, add_154, mul_340, add_155, view_483, mm_78], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:150
            extern_kernels.mm(permute_220, reinterpret_tensor(buf405, (4096, 3072), (3072, 1), 0), out=buf407)
            del permute_220
            buf408 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_482, x_21, mul_8, mul_332, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, mul_333, mul_334, sub_102, mul_335, mul_336, mul_337, pow_22, mul_338, mul_339, add_154, mul_340, add_155, view_483, sum_126], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf405, buf408, 3072, 4096, stream=raw_stream0)
            buf406 = reinterpret_tensor(buf401, (4096, 768), (768, 1), 0); del buf401  # reuse
            # Topologically Sorted Source Nodes: [view_482, x_21, mul_8, mul_332, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, mul_333, mul_334, sub_102, mul_335, mul_336, mul_337, pow_22, mul_338, mul_339, add_154, mul_340, add_155, view_483, permute_219, mm_77], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:151
            extern_kernels.mm(reinterpret_tensor(buf405, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_40, (3072, 768), (1, 3072), 0), out=buf406)
            del primals_40
            buf413 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [view_485, mul_342, mul_343, sum_127, mul_344, sum_128, mul_345, sub_104, sub_105, mul_346, add_156], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf413, buf406, primals_37, mul_18, div_42, 4096, 768, stream=raw_stream0)
            del div_42
            del primals_37
            buf411 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf412 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_485, mul_347, sum_129, sum_130], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf406, mul_18, buf411, buf412, 768, 4096, stream=raw_stream0)
            del mul_18
            # Topologically Sorted Source Nodes: [npu_dropout_backward_28], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:152
            buf414 = torch.ops.npu.npu_dropout_backward.default(buf413, getitem_36, 0.10000000000000009)
            del getitem_36
            buf415 = buf414
            assert_alignment(buf415, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf414
            buf417 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_486, mm_80], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:153
            extern_kernels.mm(permute_222, reinterpret_tensor(buf415, (4096, 768), (768, 1), 0), out=buf417)
            del permute_222
            buf418 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_486, sum_131], Original ATen: [aten.view, aten.sum]
            workspace_28 = workspace_27; del workspace_27  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf415, buf418, workspace_28, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_28, buf418, 768, 40, stream=raw_stream0)
            buf416 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [view_486, permute_221, mm_79], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:154
            extern_kernels.mm(reinterpret_tensor(buf415, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_36, (768, 768), (1, 768), 0), out=buf416)
            del primals_36
            buf419 = reinterpret_tensor(buf415, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [view_488, view_489, permute_223, clone_102], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf416, buf419, 3145728, stream=raw_stream0)
            buf421 = reinterpret_tensor(buf416, (48, 1024, 64), (65536, 64, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [view_488, view_489, permute_223, clone_102, view_490, bmm_61], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:155
            extern_kernels.bmm(view_493, reinterpret_tensor(buf419, (48, 1024, 64), (65536, 64, 1), 0), out=buf421)
            del view_493
            tangents_8 = copy_if_misaligned(tangents_8)
            buf431 = reinterpret_tensor(buf421, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf421  # reuse
            # Topologically Sorted Source Nodes: [view_495, add_157], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf431, tangents_8, 3145728, stream=raw_stream0)
            del tangents_8
            buf432 = reinterpret_tensor(buf392, (4, 1024, 768), (786432, 768, 1), 0); del buf392  # reuse
            # Topologically Sorted Source Nodes: [view_495, add_157, view_498, view_501, permute_228, add_158, permute_229, view_502, permute_230, view_503, permute_231, clone_107, view_504, cat_9], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf431, buf432, 3145728, stream=raw_stream0)
            buf420 = reinterpret_tensor(buf382, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf382  # reuse
            # Topologically Sorted Source Nodes: [view_488, view_489, permute_223, clone_102, view_490, bmm_60], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:156
            extern_kernels.bmm(reinterpret_tensor(buf419, (48, 1024, 64), (65536, 64, 1), 0), view_491, out=buf420)
            del view_491
            # Topologically Sorted Source Nodes: [view_492, npu_dropout_backward_29], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:157
            buf422 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf420, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_34, 0.10000000000000009)
            del buf420
            del getitem_34
            buf423 = buf422
            assert_alignment(buf423, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf422
            buf425 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_348, sum_132, mul_349, sub_106, where_25, div_43], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf425, _softmax_2, primals_34, 49152, 1024, stream=raw_stream0)
            del _softmax_2
            del primals_34
            buf426 = reinterpret_tensor(buf419, (48, 1024, 64), (65536, 64, 1), 0); del buf419  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_348, mul_349, sub_106, where_25, div_43, view_496, bmm_62], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:158
            extern_kernels.bmm(reinterpret_tensor(buf425, (48, 1024, 1024), (1048576, 1024, 1), 0), view_497, out=buf426)
            del view_497
            buf427 = reinterpret_tensor(buf431, (48, 64, 1024), (65536, 1024, 1), 0); del buf431  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_348, mul_349, sub_106, where_25, div_43, view_496, bmm_63], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:159
            extern_kernels.bmm(view_499, reinterpret_tensor(buf425, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf427)
            del view_499
            buf428 = reinterpret_tensor(buf387, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [view_498, permute_231, clone_107], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf426, buf428, 3145728, stream=raw_stream0)
            tangents_7 = copy_if_misaligned(tangents_7)
            buf429 = reinterpret_tensor(buf426, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [view_501, permute_228, add_158], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_7, buf427, buf429, 49152, 64, stream=raw_stream0)
            del tangents_7
            buf430 = reinterpret_tensor(buf427, (4, 1024, 768), (786432, 768, 1), 0); del buf427  # reuse
            # Topologically Sorted Source Nodes: [view_495, add_157, view_498, view_501, permute_228, add_158, permute_229, view_502, permute_230, view_503, permute_231, clone_107, view_504, cat_9], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf429, buf430, 3145728, stream=raw_stream0)
            del buf429
            # Topologically Sorted Source Nodes: [view_495, add_157, view_498, view_501, permute_228, add_158, permute_229, view_502, permute_230, view_503, permute_231, clone_107, view_504, cat_9], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:160
            buf433 = torch.ops.aten.cat.default([reinterpret_tensor(buf428, (4, 1024, 768), (786432, 768, 1), 0), buf430, buf432], 2)
            del buf428
            buf434 = buf433
            assert_alignment(buf434, 16, 'torch.ops.aten.cat.default')
            del buf433
            buf436 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_505, mm_82], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:161
            extern_kernels.mm(permute_233, reinterpret_tensor(buf434, (4096, 2304), (2304, 1), 0), out=buf436)
            del permute_233
            buf437 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_505, sum_133], Original ATen: [aten.view, aten.sum]
            workspace_29 = workspace_26; del workspace_26  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf434, buf437, workspace_29, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_29, buf437, 2304, 40, stream=raw_stream0)
            buf435 = reinterpret_tensor(buf432, (4096, 768), (768, 1), 0); del buf432  # reuse
            # Topologically Sorted Source Nodes: [view_505, permute_232, mm_81], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:162
            extern_kernels.mm(reinterpret_tensor(buf434, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_33, (2304, 768), (1, 2304), 0), out=buf435)
            del buf434
            del primals_33
            buf442 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [view_507, mul_351, mul_352, sum_134, mul_353, sum_135, mul_354, sub_108, sub_109, mul_355, add_159], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf442, buf435, primals_30, mul_16, div_44, 4096, 768, stream=raw_stream0)
            del div_44
            del primals_30
            buf440 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf441 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_507, mul_356, sum_136, sum_137], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf435, mul_16, buf440, buf441, 768, 4096, stream=raw_stream0)
            del mul_16
            # Topologically Sorted Source Nodes: [npu_dropout_backward_30], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:163
            buf443 = torch.ops.npu.npu_dropout_backward.default(buf442, getitem_27, 0.10000000000000009)
            del getitem_27
            buf444 = buf443
            assert_alignment(buf444, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf443
            buf446 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_508, mm_84], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:164
            extern_kernels.mm(permute_235, reinterpret_tensor(buf444, (4096, 768), (768, 1), 0), out=buf446)
            del permute_235
            buf447 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_508, sum_138], Original ATen: [aten.view, aten.sum]
            workspace_30 = workspace_28; del workspace_28  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf444, buf447, workspace_30, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_30, buf447, 768, 40, stream=raw_stream0)
            buf445 = reinterpret_tensor(buf405, (4096, 3072), (3072, 1), 0); del buf405  # reuse
            # Topologically Sorted Source Nodes: [view_508, permute_234, mm_83], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:165
            extern_kernels.mm(reinterpret_tensor(buf444, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 3072), (1, 768), 0), out=buf445)
            del primals_29
            buf448 = reinterpret_tensor(buf445, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf445  # reuse
            # Topologically Sorted Source Nodes: [view_510, x_13, mul_4, mul_357, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, mul_358, mul_359, sub_110, mul_360, mul_361, mul_362, pow_23, mul_363, mul_364, add_160, mul_365, add_161], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf448, addmm_6, 12582912, stream=raw_stream0)
            del addmm_6
            buf450 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_510, x_13, mul_4, mul_357, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, mul_358, mul_359, sub_110, mul_360, mul_361, mul_362, pow_23, mul_363, mul_364, add_160, mul_365, add_161, view_511, mm_86], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:166
            extern_kernels.mm(permute_237, reinterpret_tensor(buf448, (4096, 3072), (3072, 1), 0), out=buf450)
            del permute_237
            buf451 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_510, x_13, mul_4, mul_357, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, mul_358, mul_359, sub_110, mul_360, mul_361, mul_362, pow_23, mul_363, mul_364, add_160, mul_365, add_161, view_511, sum_139], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf448, buf451, 3072, 4096, stream=raw_stream0)
            buf449 = reinterpret_tensor(buf444, (4096, 768), (768, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [view_510, x_13, mul_4, mul_357, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, mul_358, mul_359, sub_110, mul_360, mul_361, mul_362, pow_23, mul_363, mul_364, add_160, mul_365, add_161, view_511, permute_236, mm_85], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:167
            extern_kernels.mm(reinterpret_tensor(buf448, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_27, (3072, 768), (1, 3072), 0), out=buf449)
            del primals_27
            buf456 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [view_513, mul_367, mul_368, sum_140, mul_369, sum_141, mul_370, sub_112, sub_113, mul_371, add_162], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf456, buf449, primals_24, mul_10, div_45, 4096, 768, stream=raw_stream0)
            del div_45
            del primals_24
            buf454 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf455 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_513, mul_372, sum_142, sum_143], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf449, mul_10, buf454, buf455, 768, 4096, stream=raw_stream0)
            del mul_10
            # Topologically Sorted Source Nodes: [npu_dropout_backward_31], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:168
            buf457 = torch.ops.npu.npu_dropout_backward.default(buf456, getitem_23, 0.10000000000000009)
            del getitem_23
            buf458 = buf457
            assert_alignment(buf458, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf457
            buf460 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_514, mm_88], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:169
            extern_kernels.mm(permute_239, reinterpret_tensor(buf458, (4096, 768), (768, 1), 0), out=buf460)
            del permute_239
            buf461 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_514, sum_144], Original ATen: [aten.view, aten.sum]
            workspace_31 = workspace_30; del workspace_30  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf458, buf461, workspace_31, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_31, buf461, 768, 40, stream=raw_stream0)
            buf459 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [view_514, permute_238, mm_87], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:170
            extern_kernels.mm(reinterpret_tensor(buf458, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), out=buf459)
            del primals_23
            buf462 = reinterpret_tensor(buf458, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf458  # reuse
            # Topologically Sorted Source Nodes: [view_516, view_517, permute_240, clone_108], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf459, buf462, 3145728, stream=raw_stream0)
            buf464 = reinterpret_tensor(buf459, (48, 1024, 64), (65536, 64, 1), 0); del buf459  # reuse
            # Topologically Sorted Source Nodes: [view_516, view_517, permute_240, clone_108, view_518, bmm_65], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:171
            extern_kernels.bmm(view_521, reinterpret_tensor(buf462, (48, 1024, 64), (65536, 64, 1), 0), out=buf464)
            del view_521
            tangents_6 = copy_if_misaligned(tangents_6)
            buf474 = reinterpret_tensor(buf464, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf464  # reuse
            # Topologically Sorted Source Nodes: [view_523, add_163], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf474, tangents_6, 3145728, stream=raw_stream0)
            del tangents_6
            buf475 = reinterpret_tensor(buf435, (4, 1024, 768), (786432, 768, 1), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [view_523, add_163, view_526, view_529, permute_245, add_164, permute_246, view_530, permute_247, view_531, permute_248, clone_113, view_532, cat_10], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf474, buf475, 3145728, stream=raw_stream0)
            buf463 = reinterpret_tensor(buf425, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf425  # reuse
            # Topologically Sorted Source Nodes: [view_516, view_517, permute_240, clone_108, view_518, bmm_64], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:172
            extern_kernels.bmm(reinterpret_tensor(buf462, (48, 1024, 64), (65536, 64, 1), 0), view_519, out=buf463)
            del view_519
            # Topologically Sorted Source Nodes: [view_520, npu_dropout_backward_32], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:173
            buf465 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf463, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_21, 0.10000000000000009)
            del buf463
            del getitem_21
            buf466 = buf465
            assert_alignment(buf466, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf465
            buf468 = buf466; del buf466  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_373, sum_145, mul_374, sub_114, where_26, div_46], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf468, _softmax_1, primals_21, 49152, 1024, stream=raw_stream0)
            del _softmax_1
            del primals_21
            buf469 = reinterpret_tensor(buf462, (48, 1024, 64), (65536, 64, 1), 0); del buf462  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_373, mul_374, sub_114, where_26, div_46, view_524, bmm_66], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:174
            extern_kernels.bmm(reinterpret_tensor(buf468, (48, 1024, 1024), (1048576, 1024, 1), 0), view_525, out=buf469)
            del view_525
            buf470 = reinterpret_tensor(buf474, (48, 64, 1024), (65536, 1024, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_373, mul_374, sub_114, where_26, div_46, view_524, bmm_67], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:175
            extern_kernels.bmm(view_527, reinterpret_tensor(buf468, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf470)
            del view_527
            buf471 = reinterpret_tensor(buf430, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf430  # reuse
            # Topologically Sorted Source Nodes: [view_526, permute_248, clone_113], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf469, buf471, 3145728, stream=raw_stream0)
            tangents_5 = copy_if_misaligned(tangents_5)
            buf472 = reinterpret_tensor(buf469, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf469  # reuse
            # Topologically Sorted Source Nodes: [view_529, permute_245, add_164], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_5, buf470, buf472, 49152, 64, stream=raw_stream0)
            del tangents_5
            buf473 = reinterpret_tensor(buf470, (4, 1024, 768), (786432, 768, 1), 0); del buf470  # reuse
            # Topologically Sorted Source Nodes: [view_523, add_163, view_526, view_529, permute_245, add_164, permute_246, view_530, permute_247, view_531, permute_248, clone_113, view_532, cat_10], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf472, buf473, 3145728, stream=raw_stream0)
            del buf472
            # Topologically Sorted Source Nodes: [view_523, add_163, view_526, view_529, permute_245, add_164, permute_246, view_530, permute_247, view_531, permute_248, clone_113, view_532, cat_10], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:176
            buf476 = torch.ops.aten.cat.default([reinterpret_tensor(buf471, (4, 1024, 768), (786432, 768, 1), 0), buf473, buf475], 2)
            del buf471
            buf477 = buf476
            assert_alignment(buf477, 16, 'torch.ops.aten.cat.default')
            del buf476
            buf479 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_533, mm_90], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:177
            extern_kernels.mm(permute_250, reinterpret_tensor(buf477, (4096, 2304), (2304, 1), 0), out=buf479)
            del permute_250
            buf480 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_533, sum_146], Original ATen: [aten.view, aten.sum]
            workspace_32 = workspace_29; del workspace_29  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf477, buf480, workspace_32, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_32, buf480, 2304, 40, stream=raw_stream0)
            buf478 = reinterpret_tensor(buf475, (4096, 768), (768, 1), 0); del buf475  # reuse
            # Topologically Sorted Source Nodes: [view_533, permute_249, mm_89], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:178
            extern_kernels.mm(reinterpret_tensor(buf477, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_20, (2304, 768), (1, 2304), 0), out=buf478)
            del buf477
            del primals_20
            buf485 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [view_535, mul_376, mul_377, sum_147, mul_378, sum_148, mul_379, sub_116, sub_117, mul_380, add_165], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf485, buf478, primals_17, mul_8, div_47, 4096, 768, stream=raw_stream0)
            del div_47
            del primals_17
            buf483 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf484 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_535, mul_381, sum_149, sum_150], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf478, mul_8, buf483, buf484, 768, 4096, stream=raw_stream0)
            del mul_8
            # Topologically Sorted Source Nodes: [npu_dropout_backward_33], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:179
            buf486 = torch.ops.npu.npu_dropout_backward.default(buf485, getitem_14, 0.10000000000000009)
            del getitem_14
            buf487 = buf486
            assert_alignment(buf487, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf486
            buf489 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_536, mm_92], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:180
            extern_kernels.mm(permute_252, reinterpret_tensor(buf487, (4096, 768), (768, 1), 0), out=buf489)
            del permute_252
            buf490 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_536, sum_151], Original ATen: [aten.view, aten.sum]
            workspace_33 = workspace_31; del workspace_31  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf487, buf490, workspace_33, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_33, buf490, 768, 40, stream=raw_stream0)
            buf488 = reinterpret_tensor(buf448, (4096, 3072), (3072, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [view_536, permute_251, mm_91], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:181
            extern_kernels.mm(reinterpret_tensor(buf487, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), out=buf488)
            del primals_16
            buf491 = reinterpret_tensor(buf488, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [view_538, x_5, mul, mul_382, pow_1, mul_1, add_2, mul_2, tanh, add_3, mul_383, mul_384, sub_118, mul_385, mul_386, mul_387, pow_24, mul_388, mul_389, add_166, mul_390, add_167], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_tanh_backward_view_9.run(buf491, addmm_2, 12582912, stream=raw_stream0)
            del addmm_2
            buf493 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_538, x_5, mul, mul_382, pow_1, mul_1, add_2, mul_2, tanh, add_3, mul_383, mul_384, sub_118, mul_385, mul_386, mul_387, pow_24, mul_388, mul_389, add_166, mul_390, add_167, view_539, mm_94], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:182
            extern_kernels.mm(permute_254, reinterpret_tensor(buf491, (4096, 3072), (3072, 1), 0), out=buf493)
            del permute_254
            buf494 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_538, x_5, mul, mul_382, pow_1, mul_1, add_2, mul_2, tanh, add_3, mul_383, mul_384, sub_118, mul_385, mul_386, mul_387, pow_24, mul_388, mul_389, add_166, mul_390, add_167, view_539, sum_152], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_sum_tanh_tanh_backward_view_10.run(buf491, buf494, 3072, 4096, stream=raw_stream0)
            buf492 = reinterpret_tensor(buf487, (4096, 768), (768, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [view_538, x_5, mul, mul_382, pow_1, mul_1, add_2, mul_2, tanh, add_3, mul_383, mul_384, sub_118, mul_385, mul_386, mul_387, pow_24, mul_388, mul_389, add_166, mul_390, add_167, view_539, permute_253, mm_93], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:183
            extern_kernels.mm(reinterpret_tensor(buf491, (4096, 3072), (3072, 1), 0), reinterpret_tensor(primals_14, (3072, 768), (1, 3072), 0), out=buf492)
            del buf491
            del primals_14
            buf499 = buf485; del buf485  # reuse
            # Topologically Sorted Source Nodes: [view_541, mul_392, mul_393, sum_153, mul_394, sum_154, mul_395, sub_120, sub_121, mul_396, add_168], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf499, buf492, primals_11, mul_2, div_48, 4096, 768, stream=raw_stream0)
            del div_48
            del primals_11
            buf497 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf498 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_541, mul_397, sum_155, sum_156], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf492, mul_2, buf497, buf498, 768, 4096, stream=raw_stream0)
            del mul_2
            # Topologically Sorted Source Nodes: [npu_dropout_backward_34], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:184
            buf500 = torch.ops.npu.npu_dropout_backward.default(buf499, getitem_10, 0.10000000000000009)
            del getitem_10
            buf501 = buf500
            assert_alignment(buf501, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf500
            buf503 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_542, mm_96], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:185
            extern_kernels.mm(permute_256, reinterpret_tensor(buf501, (4096, 768), (768, 1), 0), out=buf503)
            del permute_256
            buf504 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_542, sum_157], Original ATen: [aten.view, aten.sum]
            workspace_34 = workspace_33; del workspace_33  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_7.run(buf501, buf504, workspace_34, 768, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_8.run(workspace_34, buf504, 768, 40, stream=raw_stream0)
            del workspace_34
            buf502 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [view_542, permute_255, mm_95], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:186
            extern_kernels.mm(reinterpret_tensor(buf501, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf502)
            del primals_10
            buf505 = reinterpret_tensor(buf501, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf501  # reuse
            # Topologically Sorted Source Nodes: [view_544, view_545, permute_257, clone_114], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_view_12.run(buf502, buf505, 3145728, stream=raw_stream0)
            buf507 = reinterpret_tensor(buf502, (48, 1024, 64), (65536, 64, 1), 0); del buf502  # reuse
            # Topologically Sorted Source Nodes: [view_544, view_545, permute_257, clone_114, view_546, bmm_69], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:187
            extern_kernels.bmm(view_549, reinterpret_tensor(buf505, (48, 1024, 64), (65536, 64, 1), 0), out=buf507)
            del view_549
            tangents_4 = copy_if_misaligned(tangents_4)
            buf517 = reinterpret_tensor(buf507, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf507  # reuse
            # Topologically Sorted Source Nodes: [view_551, add_169], Original ATen: [aten.matmul_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_13.run(buf517, tangents_4, 3145728, stream=raw_stream0)
            del tangents_4
            buf518 = reinterpret_tensor(buf478, (4, 1024, 768), (786432, 768, 1), 0); del buf478  # reuse
            # Topologically Sorted Source Nodes: [view_551, add_169, view_554, view_557, permute_262, add_170, permute_263, view_558, permute_264, view_559, permute_265, clone_119, view_560, cat_11], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf517, buf518, 3145728, stream=raw_stream0)
            buf506 = reinterpret_tensor(buf468, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [view_544, view_545, permute_257, clone_114, view_546, bmm_68], Original ATen: [aten.view, aten.permute, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:188
            extern_kernels.bmm(reinterpret_tensor(buf505, (48, 1024, 64), (65536, 64, 1), 0), view_547, out=buf506)
            del view_547
            # Topologically Sorted Source Nodes: [view_548, npu_dropout_backward_35], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:189
            buf508 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf506, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0), getitem_8, 0.10000000000000009)
            del buf506
            del getitem_8
            buf509 = buf508
            assert_alignment(buf509, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf508
            buf511 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_398, sum_158, mul_399, sub_122, where_27, div_49], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_div_full_nll_loss_forward_where_15.run(buf511, _softmax, primals_8, 49152, 1024, stream=raw_stream0)
            del _softmax
            del primals_8
            buf512 = reinterpret_tensor(buf505, (48, 1024, 64), (65536, 64, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_398, mul_399, sub_122, where_27, div_49, view_552, bmm_70], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:190
            extern_kernels.bmm(reinterpret_tensor(buf511, (48, 1024, 1024), (1048576, 1024, 1), 0), view_553, out=buf512)
            del view_553
            buf513 = reinterpret_tensor(buf517, (48, 64, 1024), (65536, 1024, 1), 0); del buf517  # reuse
            # Topologically Sorted Source Nodes: [loss, full, mul_398, mul_399, sub_122, where_27, div_49, view_552, bmm_71], Original ATen: [aten.nll_loss_forward, aten.full, aten._softmax_backward_data, aten.where, aten.div, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:191
            extern_kernels.bmm(view_555, reinterpret_tensor(buf511, (48, 1024, 1024), (1048576, 1024, 1), 0), out=buf513)
            del buf511
            del view_555
            buf514 = reinterpret_tensor(buf473, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [view_554, permute_265, clone_119], Original ATen: [aten.matmul_backward, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_backward_permute_16.run(buf512, buf514, 3145728, stream=raw_stream0)
            tangents_3 = copy_if_misaligned(tangents_3)
            buf515 = reinterpret_tensor(buf512, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [view_557, permute_262, add_170], Original ATen: [aten.matmul_backward, aten.transpose, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_matmul_backward_transpose_17.run(tangents_3, buf513, buf515, 49152, 64, stream=raw_stream0)
            del tangents_3
            buf516 = reinterpret_tensor(buf513, (4, 1024, 768), (786432, 768, 1), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [view_551, add_169, view_554, view_557, permute_262, add_170, permute_263, view_558, permute_264, view_559, permute_265, clone_119, view_560, cat_11], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_add_cat_clone_matmul_backward_permute_transpose_view_14.run(buf515, buf516, 3145728, stream=raw_stream0)
            del buf515
            # Topologically Sorted Source Nodes: [view_551, add_169, view_554, view_557, permute_262, add_170, permute_263, view_558, permute_264, view_559, permute_265, clone_119, view_560, cat_11], Original ATen: [aten.matmul_backward, aten.add, aten.transpose, aten.permute, aten.view, aten.clone, aten._unsafe_view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:192
            buf519 = torch.ops.aten.cat.default([reinterpret_tensor(buf514, (4, 1024, 768), (786432, 768, 1), 0), buf516, buf518], 2)
            del buf514
            del buf516
            buf520 = buf519
            assert_alignment(buf520, 16, 'torch.ops.aten.cat.default')
            del buf519
            buf522 = empty_strided_npu((768, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_561, mm_98], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:193
            extern_kernels.mm(permute_267, reinterpret_tensor(buf520, (4096, 2304), (2304, 1), 0), out=buf522)
            del permute_267
            buf523 = empty_strided_npu((1, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_561, sum_159], Original ATen: [aten.view, aten.sum]
            workspace_35 = workspace_32; del workspace_32  # reuse
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_18.run(buf520, buf523, workspace_35, 2304, 4096, stream=raw_stream0)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_19.run(workspace_35, buf523, 2304, 40, stream=raw_stream0)
            del workspace_35
            buf521 = reinterpret_tensor(buf518, (4096, 768), (768, 1), 0); del buf518  # reuse
            # Topologically Sorted Source Nodes: [view_561, permute_266, mm_97], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:194
            extern_kernels.mm(reinterpret_tensor(buf520, (4096, 2304), (2304, 1), 0), reinterpret_tensor(primals_7, (2304, 768), (1, 2304), 0), out=buf521)
            del buf520
            del primals_7
            buf528 = buf499; del buf499  # reuse
            # Topologically Sorted Source Nodes: [view_563, mul_401, mul_402, sum_160, mul_403, sum_161, mul_404, sub_124, sub_125, mul_405, add_171], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_11.run(buf528, buf521, primals_4, mul, div_50, 4096, 768, stream=raw_stream0)
            del div_50
            del primals_4
            buf526 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf527 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_563, mul_406, sum_162, sum_163], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_native_layer_norm_backward_5.run(buf521, mul, buf526, buf527, 768, 4096, stream=raw_stream0)
            del buf521
            del mul
            # Topologically Sorted Source Nodes: [view_563, mul_401, mul_402, mul_404, sub_124, sub_125, mul_405, add_171, npu_dropout_backward_36], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:195
            buf529 = torch.ops.npu.npu_dropout_backward.default(buf528, getitem_1, 0.10000000000000009)
            del buf528
            del getitem_1
            buf530 = buf529
            assert_alignment(buf530, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf529
            buf531 = empty_strided_npu((1, 1024, 768), (786432, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_164, embedding_dense_backward], Original ATen: [aten.sum, aten.embedding_dense_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_embedding_dense_backward_sum_20.run(buf530, buf531, 786432, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [sum_164, embedding_dense_backward], Original ATen: [aten.sum, aten.embedding_dense_backward]
            # [Provenance debug handles] torch.ops.aten.embedding_dense_backward.default:196
            buf532 = torch.ops.aten.embedding_dense_backward.default(buf531, unsqueeze, 1024, -1, False)
            del buf531
            del unsqueeze
            buf533 = buf532
            assert_alignment(buf533, 16, 'torch.ops.aten.embedding_dense_backward.default')
            del buf532
            # Topologically Sorted Source Nodes: [embedding_dense_backward_1], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] torch.ops.aten.embedding_dense_backward.default:197
            buf534 = torch.ops.aten.embedding_dense_backward.default(buf530, primals_1, 50257, -1, False)
            del buf530
            del primals_1
            buf535 = buf534
            assert_alignment(buf535, 16, 'torch.ops.aten.embedding_dense_backward.default')
            del buf534
        return (None, buf535, buf533, buf526, buf527, reinterpret_tensor(buf523, (2304, ), (1, ), 0), buf522, None, reinterpret_tensor(buf504, (768, ), (1, ), 0), buf503, buf497, buf498, reinterpret_tensor(buf494, (3072, ), (1, ), 0), buf493, reinterpret_tensor(buf490, (768, ), (1, ), 0), buf489, buf483, buf484, reinterpret_tensor(buf480, (2304, ), (1, ), 0), buf479, None, reinterpret_tensor(buf461, (768, ), (1, ), 0), buf460, buf454, buf455, reinterpret_tensor(buf451, (3072, ), (1, ), 0), buf450, reinterpret_tensor(buf447, (768, ), (1, ), 0), buf446, buf440, buf441, reinterpret_tensor(buf437, (2304, ), (1, ), 0), buf436, None, reinterpret_tensor(buf418, (768, ), (1, ), 0), buf417, buf411, buf412, reinterpret_tensor(buf408, (3072, ), (1, ), 0), buf407, reinterpret_tensor(buf404, (768, ), (1, ), 0), buf403, buf397, buf398, reinterpret_tensor(buf394, (2304, ), (1, ), 0), buf393, None, reinterpret_tensor(buf375, (768, ), (1, ), 0), buf374, buf368, buf369, reinterpret_tensor(buf365, (3072, ), (1, ), 0), buf364, reinterpret_tensor(buf361, (768, ), (1, ), 0), buf360, buf354, buf355, reinterpret_tensor(buf351, (2304, ), (1, ), 0), buf350, None, reinterpret_tensor(buf332, (768, ), (1, ), 0), buf331, buf325, buf326, reinterpret_tensor(buf322, (3072, ), (1, ), 0), buf321, reinterpret_tensor(buf318, (768, ), (1, ), 0), buf317, buf311, buf312, reinterpret_tensor(buf308, (2304, ), (1, ), 0), buf307, None, reinterpret_tensor(buf289, (768, ), (1, ), 0), buf288, buf282, buf283, reinterpret_tensor(buf279, (3072, ), (1, ), 0), buf278, reinterpret_tensor(buf275, (768, ), (1, ), 0), buf274, buf268, buf269, reinterpret_tensor(buf265, (2304, ), (1, ), 0), buf264, None, reinterpret_tensor(buf246, (768, ), (1, ), 0), buf245, buf239, buf240, reinterpret_tensor(buf236, (3072, ), (1, ), 0), buf235, reinterpret_tensor(buf232, (768, ), (1, ), 0), buf231, buf225, buf226, reinterpret_tensor(buf222, (2304, ), (1, ), 0), buf221, None, reinterpret_tensor(buf203, (768, ), (1, ), 0), buf202, buf196, buf197, reinterpret_tensor(buf193, (3072, ), (1, ), 0), buf192, reinterpret_tensor(buf189, (768, ), (1, ), 0), buf188, buf182, buf183, reinterpret_tensor(buf179, (2304, ), (1, ), 0), buf178, None, reinterpret_tensor(buf160, (768, ), (1, ), 0), buf159, buf153, buf154, reinterpret_tensor(buf150, (3072, ), (1, ), 0), buf149, reinterpret_tensor(buf146, (768, ), (1, ), 0), buf145, buf139, buf140, reinterpret_tensor(buf136, (2304, ), (1, ), 0), buf135, None, reinterpret_tensor(buf117, (768, ), (1, ), 0), buf116, buf110, buf111, reinterpret_tensor(buf107, (3072, ), (1, ), 0), buf106, reinterpret_tensor(buf103, (768, ), (1, ), 0), buf102, buf96, buf97, reinterpret_tensor(buf93, (2304, ), (1, ), 0), buf92, None, reinterpret_tensor(buf74, (768, ), (1, ), 0), buf73, buf67, buf68, reinterpret_tensor(buf64, (3072, ), (1, ), 0), buf63, reinterpret_tensor(buf60, (768, ), (1, ), 0), buf59, buf53, buf54, reinterpret_tensor(buf50, (2304, ), (1, ), 0), buf49, None, reinterpret_tensor(buf31, (768, ), (1, ), 0), buf30, buf24, buf25, reinterpret_tensor(buf21, (3072, ), (1, ), 0), buf20, reinterpret_tensor(buf17, (768, ), (1, ), 0), buf16, buf11, buf12, buf7, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    _shared_storage_304 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_305 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_306 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_307 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_308 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_309 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_310 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_311 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_312 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_313 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_314 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    _shared_storage_315 = rand_strided((9437184,), (1,), device='npu:0', dtype=torch.float32)
    primals_1 = rand_strided((4, 1024), (1024, 1), device='npu:0', dtype=torch.int64)
    primals_4 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_10 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_23 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_29 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_34 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_36 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_42 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_49 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_55 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_62 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_73 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_75 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_86 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_88 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_94 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_98 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_99 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_101 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_112 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_114 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_120 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_125 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_127 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_133 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_138 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_140 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_144 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_146 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    primals_153 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_159 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_162 = rand_strided((2, 768), (768, 1), device='npu:0', dtype=torch.float32)
    unsqueeze = rand_strided((1, 1024), (1024, 1), device='npu:0', dtype=torch.int64)
    getitem_1 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_8 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_10 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_2 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_2 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_14 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_8 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_1 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_21 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_23 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_10 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_6 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_27 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_16 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_2 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_34 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_36 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_18 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_10 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_40 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_24 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_3 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_47 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_49 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_26 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_14 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_53 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_32 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_4 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_60 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_62 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_34 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_18 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_66 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_40 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_5 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_73 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_75 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_42 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_22 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_79 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_48 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_6 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_86 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_88 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_50 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_26 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_92 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_56 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_7 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_99 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_101 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_58 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_30 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_105 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_64 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_8 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_112 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_114 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_66 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_34 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_118 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_72 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_9 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_125 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_127 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_74 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_38 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_131 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_80 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_10 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_138 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_140 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_82 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_42 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_144 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_88 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    _softmax_11 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='npu:0', dtype=torch.float32)
    getitem_151 = rand_strided((6291456, ), (1, ), device='npu:0', dtype=torch.uint8)
    getitem_153 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_90 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    addmm_46 = rand_strided((4096, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_157 = rand_strided((393216, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_96 = rand_strided((4, 1024, 768), (786432, 768, 1), device='npu:0', dtype=torch.float32)
    sub_25 = rand_strided((4, ), (1, ), device='npu:0', dtype=torch.int64)
    iota_1 = rand_strided((4, ), (1, ), device='npu:0', dtype=torch.int64)
    _log_softmax = rand_strided((4, 2), (2, 1), device='npu:0', dtype=torch.float32)
    convert_element_type_13 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    ne_3 = rand_strided((4, 1), (1, 1), device='npu:0', dtype=torch.bool)
    eq_tensor = rand_strided((4, 2), (2, 1), device='npu:0', dtype=torch.bool)
    permute_62 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_14 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_65 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_67 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_15 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_69 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_239 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_241 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_245 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_247 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_80 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_17 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_82 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_84 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_18 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_86 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_267 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_269 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_273 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_275 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_97 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_20 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_99 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_101 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_21 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_103 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_295 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_297 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_301 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_303 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_114 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_23 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_116 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_118 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_24 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_120 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_323 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_325 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_329 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_331 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_131 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_26 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_133 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_135 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_27 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_351 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_353 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_357 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_359 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_148 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_29 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_150 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_152 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_30 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_154 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_379 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_381 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_385 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_387 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_165 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_32 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_167 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_169 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_33 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_407 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_409 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_413 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_415 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_182 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_35 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_184 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_186 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_36 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_188 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_435 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_437 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_441 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_443 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_38 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_201 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_39 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_205 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_463 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_465 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_469 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_471 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_41 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_218 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_220 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_42 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_491 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_493 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_497 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_499 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_44 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_235 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_45 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_519 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_521 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_525 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_527 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_250 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_47 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_252 = rand_strided((3072, 4096), (1, 3072), device='npu:0', dtype=torch.float32)
    permute_254 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_48 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    permute_256 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    view_547 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    view_549 = rand_strided((48, 1024, 1024), (1048576, 1, 1024), device='npu:0', dtype=torch.float32)
    view_553 = rand_strided((48, 1024, 64), (65536, 64, 1), device='npu:0', dtype=torch.float32)
    view_555 = rand_strided((48, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
    permute_267 = rand_strided((768, 4096), (1, 768), device='npu:0', dtype=torch.float32)
    div_50 = rand_strided((4, 1024, 1), (1024, 1, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    tangents_2 = rand_strided((4, 2), (2, 1), device='npu:0', dtype=torch.float32)
    tangents_3 = torch.as_strided(_shared_storage_304, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_4 = torch.as_strided(_shared_storage_304, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_5 = torch.as_strided(_shared_storage_305, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_6 = torch.as_strided(_shared_storage_305, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_7 = torch.as_strided(_shared_storage_306, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_8 = torch.as_strided(_shared_storage_306, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_9 = torch.as_strided(_shared_storage_307, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_10 = torch.as_strided(_shared_storage_307, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_11 = torch.as_strided(_shared_storage_308, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_12 = torch.as_strided(_shared_storage_308, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_13 = torch.as_strided(_shared_storage_309, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_14 = torch.as_strided(_shared_storage_309, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_15 = torch.as_strided(_shared_storage_310, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_16 = torch.as_strided(_shared_storage_310, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_17 = torch.as_strided(_shared_storage_311, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_18 = torch.as_strided(_shared_storage_311, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_19 = torch.as_strided(_shared_storage_312, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_20 = torch.as_strided(_shared_storage_312, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_21 = torch.as_strided(_shared_storage_313, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_22 = torch.as_strided(_shared_storage_313, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_23 = torch.as_strided(_shared_storage_314, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_24 = torch.as_strided(_shared_storage_314, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_25 = torch.as_strided(_shared_storage_315, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    tangents_26 = torch.as_strided(_shared_storage_315, (4, 12, 1024, 64), (2359296, 64, 2304, 1))
    return [primals_1, primals_4, primals_7, primals_8, primals_10, primals_11, primals_14, primals_16, primals_17, primals_20, primals_21, primals_23, primals_24, primals_27, primals_29, primals_30, primals_33, primals_34, primals_36, primals_37, primals_40, primals_42, primals_43, primals_46, primals_47, primals_49, primals_50, primals_53, primals_55, primals_56, primals_59, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_86, primals_88, primals_89, primals_92, primals_94, primals_95, primals_98, primals_99, primals_101, primals_102, primals_105, primals_107, primals_108, primals_111, primals_112, primals_114, primals_115, primals_118, primals_120, primals_121, primals_124, primals_125, primals_127, primals_128, primals_131, primals_133, primals_134, primals_137, primals_138, primals_140, primals_141, primals_144, primals_146, primals_147, primals_150, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_162, unsqueeze, getitem_1, mul, _softmax, getitem_8, getitem_10, mul_2, addmm_2, getitem_14, mul_8, _softmax_1, getitem_21, getitem_23, mul_10, addmm_6, getitem_27, mul_16, _softmax_2, getitem_34, getitem_36, mul_18, addmm_10, getitem_40, mul_24, _softmax_3, getitem_47, getitem_49, mul_26, addmm_14, getitem_53, mul_32, _softmax_4, getitem_60, getitem_62, mul_34, addmm_18, getitem_66, mul_40, _softmax_5, getitem_73, getitem_75, mul_42, addmm_22, getitem_79, mul_48, _softmax_6, getitem_86, getitem_88, mul_50, addmm_26, getitem_92, mul_56, _softmax_7, getitem_99, getitem_101, mul_58, addmm_30, getitem_105, mul_64, _softmax_8, getitem_112, getitem_114, mul_66, addmm_34, getitem_118, mul_72, _softmax_9, getitem_125, getitem_127, mul_74, addmm_38, getitem_131, mul_80, _softmax_10, getitem_138, getitem_140, mul_82, addmm_42, getitem_144, mul_88, _softmax_11, getitem_151, getitem_153, mul_90, addmm_46, getitem_157, mul_96, sub_25, iota_1, _log_softmax, convert_element_type_13, ne_3, eq_tensor, permute_62, div_14, permute_65, permute_67, div_15, permute_69, view_239, view_241, view_245, view_247, permute_80, div_17, permute_82, permute_84, div_18, permute_86, view_267, view_269, view_273, view_275, permute_97, div_20, permute_99, permute_101, div_21, permute_103, view_295, view_297, view_301, view_303, permute_114, div_23, permute_116, permute_118, div_24, permute_120, view_323, view_325, view_329, view_331, permute_131, div_26, permute_133, permute_135, div_27, permute_137, view_351, view_353, view_357, view_359, permute_148, div_29, permute_150, permute_152, div_30, permute_154, view_379, view_381, view_385, view_387, permute_165, div_32, permute_167, permute_169, div_33, permute_171, view_407, view_409, view_413, view_415, permute_182, div_35, permute_184, permute_186, div_36, permute_188, view_435, view_437, view_441, view_443, permute_199, div_38, permute_201, permute_203, div_39, permute_205, view_463, view_465, view_469, view_471, permute_216, div_41, permute_218, permute_220, div_42, permute_222, view_491, view_493, view_497, view_499, permute_233, div_44, permute_235, permute_237, div_45, permute_239, view_519, view_521, view_525, view_527, permute_250, div_47, permute_252, permute_254, div_48, permute_256, view_547, view_549, view_553, view_555, permute_267, div_50, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('GPT2ForSequenceClassification', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
