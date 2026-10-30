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
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/5n/c5no6mqdcwz6h2xzsykxxpl7gs634ascalmyxbea5vpbmvtd2od6.py
# Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, sum_9, mul_57, sub_20, view_144, add_55], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add]
# Source node to ATen node mapping:
#    => eq_tensor, expand_default, iota_default, scalar_tensor_default, scalar_tensor_default_1, view_default, where_self
#   add_55 => add_55
#   div_20 => div_20
#   exp_13 => exp_13
#   mul_56 => mul_56
#   mul_57 => mul_57
#   ne_3 => ne_3
#   scalar_tensor_2 => full_default_8
#   scalar_tensor_3 => full_default_10
#   sub_20 => sub_20
#   sum_9 => sum_9
#   unsqueeze_1 => unsqueeze_1
#   view_144 => view_144
#   view_31 => view_143
#   where_8 => where_8
#   where_9 => where_9
# Graph fragment:
#   %primals_108 : Tensor "i64[1, 128][128, 1]npu:0" = PlaceHolder[target=primals_108]
#   %tangents_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type]
#   %tangents_2 : Tensor "f32[1, 128, 30522][3906816, 30522, 1]npu:0" = PlaceHolder[target=tangents_2]
#   %_log_softmax : Tensor "f32[128, 30522][30522, 1]npu:0" = PlaceHolder[target=_log_softmax]
#   %sum_9 : Tensor "f32[128, 1][1, 128]npu:0" = PlaceHolder[target=sum_9]
#   %div_20 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type), kwargs = {})
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[128, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_143, 1), kwargs = {})
#   %ne_3 : Tensor "b8[128, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
#   %full_default_8 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_8 : Tensor "i64[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default_8), kwargs = {})
#   %iota_default : Tensor "i64[30522][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (30522,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 30522]), kwargs = {})
#   %expand_default : Tensor "i64[128, 30522][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_8, [128, 30522]), kwargs = {})
#   %eq_tensor : Tensor "b8[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %where_self : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_10 : Tensor "f32[][]npu:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_9 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_20, %full_default_10), kwargs = {})
#   %mul_56 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_9), kwargs = {})
#   %exp_13 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%_log_softmax,), kwargs = {})
#   %sum_9 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1], True), kwargs = {})
#   %mul_57 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_13, %sum_9), kwargs = {})
#   %sub_20 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %view_144 : Tensor "f32[1, 128, 30522][3906816, 30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_20, [1, 128, 30522]), kwargs = {})
#   %add_55 : Tensor "f32[1, 128, 30522][3906816, 30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %view_144), kwargs = {})
#   return %sum_9,%add_55
triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0 = async_compile.triton('triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0', '''
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
    size_hints={'x': 128, 'r0_': 30522},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [30522, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 30522, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 30522
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr1 + (0))
        tmp11 = tmp10
        tmp12 = tl.load(in_ptr2 + (0))
        tmp13 = tmp12
        _tmp18 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp1 = tl.full([1, 1], -100, tl.int32)
            tmp2 = tmp0 != tmp1
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tl.where(tmp2, tmp0, tmp3)
            tmp5 = (r0_1).to(tl.int32)
            tmp6 = tmp4 == tmp5
            tmp7 = tl.full([1, 1], -1.0, tl.float32)
            tmp8 = tl.full([1, 1], 0.0, tl.float32)
            tmp9 = tl.where(tmp6, tmp7, tmp8)
            tmp14 = (tmp11 / tmp13)
            tmp15 = tl.where(tmp2, tmp14, tmp8)
            tmp16 = tmp9 * tmp15
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        tmp30 = tl.load(in_ptr1 + (0))
        tmp31 = tl.broadcast_to(tmp30, [1])
        tmp32 = tl.load(in_ptr2 + (0))
        tmp33 = tl.broadcast_to(tmp32, [1])
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_ptr3 + (r0_1 + 30522*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp37 = tl.load(in_ptr4 + (r0_1 + 30522*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp21 = tl.full([1, 1], -100, tl.int32)
            tmp22 = tmp0 != tmp21
            tmp23 = tl.full([1, 1], 0, tl.int32)
            tmp24 = tl.where(tmp22, tmp0, tmp23)
            tmp25 = (r0_1).to(tl.int32)
            tmp26 = tmp24 == tmp25
            tmp27 = tl.full([1, 1], -1.0, tl.float32)
            tmp28 = tl.full([1, 1], 0.0, tl.float32)
            tmp29 = tl.where(tmp26, tmp27, tmp28)
            tmp34 = (tmp31 / tmp33)
            tmp35 = tl.where(tmp22, tmp34, tmp28)
            tmp36 = tmp29 * tmp35
            tmp38 = libdevice.exp(tmp37)
            tmp39 = tmp38 * tmp18
            tmp40 = tmp36 - tmp39
            tmp41 = tmp20 + tmp40
            tl.store(out_ptr1 + (r0_1 + 30528*x0), tmp41, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/xh/cxhwusorzogzuw5nkgd2xieyuyi6x7r74iwixg66zblusurn2bif.py
# Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, mul_57, sub_20, view_144, add_55, view_145, sum_10], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add, aten.sum]
# Source node to ATen node mapping:
#    => eq_tensor, expand_default, iota_default, scalar_tensor_default, scalar_tensor_default_1, view_default, where_self
#   add_55 => add_55
#   div_20 => div_20
#   exp_13 => exp_13
#   mul_56 => mul_56
#   mul_57 => mul_57
#   ne_3 => ne_3
#   scalar_tensor_2 => full_default_8
#   scalar_tensor_3 => full_default_10
#   sub_20 => sub_20
#   sum_10 => sum_10
#   unsqueeze_1 => unsqueeze_1
#   view_144 => view_144
#   view_145 => view_145
#   view_31 => view_143
#   where_8 => where_8
#   where_9 => where_9
# Graph fragment:
#   %add_55 : Tensor "f32[1, 128, 30522][3907584, 30528, 1]npu:0" = PlaceHolder[target=add_55]
#   %div_20 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type), kwargs = {})
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[128, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_143, 1), kwargs = {})
#   %ne_3 : Tensor "b8[128, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
#   %full_default_8 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_8 : Tensor "i64[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default_8), kwargs = {})
#   %iota_default : Tensor "i64[30522][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (30522,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 30522]), kwargs = {})
#   %expand_default : Tensor "i64[128, 30522][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_8, [128, 30522]), kwargs = {})
#   %eq_tensor : Tensor "b8[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0})
#   %where_self : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_10 : Tensor "f32[][]npu:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_9 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_20, %full_default_10), kwargs = {})
#   %mul_56 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_9), kwargs = {})
#   %exp_13 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%_log_softmax,), kwargs = {})
#   %mul_57 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_13, %sum_9), kwargs = {})
#   %sub_20 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %view_144 : Tensor "f32[1, 128, 30522][3906816, 30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_20, [1, 128, 30522]), kwargs = {})
#   %add_55 : Tensor "f32[1, 128, 30522][3906816, 30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %view_144), kwargs = {})
#   %view_145 : Tensor "f32[128, 30522][30522, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_55, [128, 30522]), kwargs = {})
#   %sum_10 : Tensor "f32[1, 30522][30522, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_145, [0], True), kwargs = {})
#   return %sum_10
triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_sum_view_1 = async_compile.triton('triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_sum_view_1', '''
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
    size_hints={'x': 30522, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [30522, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 30522, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 30522
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (30522) if (30522) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 30522
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
            tmp0 = tl.load(in_ptr0 + (x0 + 30528*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/y2/cy2gko6zbwef7gymoiaov6z4tphjnrdm746ffzulql5cxepokqjf.py
# Topologically Sorted Source Nodes: [view_147, prediction_logits, prediction_logits_1, prediction_logits_2, mul_64, sum_13, sum_14, mul_66, add_56, mul_67, exp_14, mul_68, add_57, mul_69, add_58, mul_70, mul_71, mul_72, mul_73], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_56 => add_56
#   add_57 => add_57
#   add_58 => add_58
#   exp_14 => exp_14
#   mul_64 => mul_64
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_70 => reciprocal
#   mul_71 => mul_71
#   mul_72 => mul_72
#   mul_73 => mul_73
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => mul_54, sub_19
#   sum_13 => sum_13
#   sum_14 => sum_14
#   view_147 => view_147
# Graph fragment:
#   %addmm_36 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=addmm_36]
#   %mm : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm]
#   %getitem_53 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=getitem_53]
#   %rsqrt_13 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=rsqrt_13]
#   %view_147 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 128, 768]), kwargs = {})
#   %view_139 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_36, [1, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %sub_19 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_53), kwargs = {})
#   %mul_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_64 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_147, %mul_54), kwargs = {})
#   %sum_13 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_64, [0, 1]), kwargs = {})
#   %sum_14 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_147, [0, 1]), kwargs = {})
#   %mul_66 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, -0.07135481627260025), kwargs = {})
#   %add_56 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, -1.5957691216057308), kwargs = {})
#   %mul_67 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %view_139), kwargs = {})
#   %exp_14 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_67,), kwargs = {})
#   %mul_68 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, 0.2140644488178007), kwargs = {})
#   %add_57 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, 1.5957691216057308), kwargs = {})
#   %mul_69 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %view_139), kwargs = {})
#   %add_58 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_14, 1.0), kwargs = {})
#   %reciprocal : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_58,), kwargs = {})
#   %mul_71 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_14, %reciprocal), kwargs = {})
#   %mul_72 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %mul_69), kwargs = {})
#   %mul_73 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_72, %reciprocal), kwargs = {})
#   return %mul_73,%sum_13,%sum_14
triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2 = async_compile.triton('triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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
        _tmp34 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        _tmp37 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp19 = tl.load(in_ptr1 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tmp0 * tmp0
            tmp2 = tl.full([1, 1], -0.07135481627260025, tl.float32)
            tmp3 = tmp1 * tmp2
            tmp4 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tmp5 * tmp0
            tmp7 = libdevice.exp(tmp6)
            tmp8 = tl.full([1, 1], 1.0, tl.float32)
            tmp9 = tmp7 + tmp8
            tmp10 = (tmp8 / tmp9)
            tmp11 = tmp7 * tmp10
            tmp12 = tl.full([1, 1], 0.2140644488178007, tl.float32)
            tmp13 = tmp1 * tmp12
            tmp14 = tl.full([1, 1], 1.5957691216057308, tl.float32)
            tmp15 = tmp13 + tmp14
            tmp16 = tmp15 * tmp0
            tmp17 = tmp11 * tmp16
            tmp18 = tmp17 * tmp10
            tmp20 = tmp1 * tmp0
            tmp21 = tl.full([1, 1], 0.044715, tl.float32)
            tmp22 = tmp20 * tmp21
            tmp23 = tmp0 + tmp22
            tmp24 = tmp23 * tmp4
            tmp25 = libdevice.exp(tmp24)
            tmp26 = tmp25 + tmp8
            tmp27 = (tmp0 / tmp26)
            tmp29 = tmp27 - tmp28
            tmp31 = tmp29 * tmp30
            tmp32 = tmp19 * tmp31
            tmp33 = tl.broadcast_to(tmp32, [R0_BLOCK, real_block_x0])
            tmp35 = _tmp34 + tmp33
            _tmp34 = tmp35
            tmp36 = tl.broadcast_to(tmp19, [R0_BLOCK, real_block_x0])
            tmp38 = _tmp37 + tmp36
            _tmp37 = tmp38
            tl.store(out_ptr0 + (x0 + 768*r0_1), tmp18, r0_mask & x0mask)
        tmp34 = tl.sum(_tmp34, 0)[None, :]
        tmp37 = tl.sum(_tmp37, 0)[None, :]
        tl.store(out_ptr1 + (x0), tmp34, x0mask)
        tl.store(out_ptr2 + (x0), tmp37, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/45/c45pghdptxc7ljawntuz4knuxn5wauefb5flmwvisfqbhggzzort.py
# Topologically Sorted Source Nodes: [view_147, mul_59, mul_60, sum_11, prediction_logits, prediction_logits_1, prediction_logits_2, mul_61, sum_12, mul_62, sub_22, sub_23, div_21, mul_63, mul_66, add_56, mul_67, exp_14, add_58, mul_70, full_2, where_10, add_59, mul_74], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_56 => add_56
#   add_58 => add_58
#   add_59 => add_59
#   div_21 => div_21
#   exp_14 => exp_14
#   full_2 => full_default_11
#   mul_59 => mul_59
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_62
#   mul_63 => mul_63
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_70 => reciprocal
#   mul_74 => mul_74
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => mul_54, sub_19
#   sub_22 => sub_22
#   sub_23 => sub_23
#   sum_11 => sum_11
#   sum_12 => sum_12
#   view_147 => view_147
#   where_10 => where_10
# Graph fragment:
#   %mm : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm]
#   %primals_105 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_105]
#   %addmm_36 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=addmm_36]
#   %getitem_53 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=getitem_53]
#   %rsqrt_13 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=rsqrt_13]
#   %sum_11 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_11]
#   %sum_12 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_12]
#   %isnan : Tensor "b8[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=isnan]
#   %mul_73 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_73]
#   %view_147 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 128, 768]), kwargs = {})
#   %mul_59 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_147, %primals_105), kwargs = {})
#   %mul_60 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, 768), kwargs = {})
#   %sum_11 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [2], True), kwargs = {})
#   %view_139 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_36, [1, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %sub_19 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_53), kwargs = {})
#   %mul_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_61 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, %mul_54), kwargs = {})
#   %sum_12 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [2], True), kwargs = {})
#   %mul_62 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %sum_12), kwargs = {})
#   %sub_22 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_60, %sum_11), kwargs = {})
#   %sub_23 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_22, %mul_62), kwargs = {})
#   %div_21 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_13, 768), kwargs = {})
#   %mul_63 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_21, %sub_23), kwargs = {})
#   %mul_66 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, -0.07135481627260025), kwargs = {})
#   %add_56 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, -1.5957691216057308), kwargs = {})
#   %mul_67 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %view_139), kwargs = {})
#   %exp_14 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_67,), kwargs = {})
#   %add_58 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_14, 1.0), kwargs = {})
#   %reciprocal : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_58,), kwargs = {})
#   %full_default_11 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_10 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_11, %mul_73), kwargs = {})
#   %add_59 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_10, %reciprocal), kwargs = {})
#   %mul_74 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %add_59), kwargs = {})
#   return %sum_11,%sum_12,%mul_74
triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_3', '''
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
    size_hints={'x': 128, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i1', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        tmp18 = tl.load(in_ptr2 + (x0), x0mask, eviction_policy='evict_last')
        tmp20 = tl.load(in_ptr3 + (x0), x0mask, eviction_policy='evict_last')
        _tmp24 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
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
            tmp7 = tmp6 * tmp6
            tmp8 = tmp7 * tmp6
            tmp9 = tl.full([1, 1], 0.044715, tl.float32)
            tmp10 = tmp8 * tmp9
            tmp11 = tmp6 + tmp10
            tmp12 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp13 = tmp11 * tmp12
            tmp14 = libdevice.exp(tmp13)
            tmp15 = tl.full([1, 1], 1.0, tl.float32)
            tmp16 = tmp14 + tmp15
            tmp17 = (tmp6 / tmp16)
            tmp19 = tmp17 - tmp18
            tmp21 = tmp19 * tmp20
            tmp22 = tmp2 * tmp21
            tmp23 = tl.broadcast_to(tmp22, [real_block_x0, R0_BLOCK])
            tmp25 = _tmp24 + tmp23
            _tmp24 = tmp25
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp24 = tl.sum(_tmp24, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp28 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp29 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp34 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp51 = tl.load(in_ptr4 + (r0_1 + 768 * x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0) != 0
            tmp52 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp26 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
            tmp27 = tmp20 * tmp26
            tmp30 = tmp28 * tmp29
            tmp31 = tl.full([1, 1], 768.0, tl.float32)
            tmp32 = tmp30 * tmp31
            tmp33 = tmp32 - tmp4
            tmp35 = tmp34 * tmp34
            tmp36 = tmp35 * tmp34
            tmp37 = tl.full([1, 1], 0.044715, tl.float32)
            tmp38 = tmp36 * tmp37
            tmp39 = tmp34 + tmp38
            tmp40 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp41 = tmp39 * tmp40
            tmp42 = libdevice.exp(tmp41)
            tmp43 = tl.full([1, 1], 1.0, tl.float32)
            tmp44 = tmp42 + tmp43
            tmp45 = (tmp34 / tmp44)
            tmp46 = tmp45 - tmp18
            tmp47 = tmp46 * tmp20
            tmp48 = tmp47 * tmp24
            tmp49 = tmp33 - tmp48
            tmp50 = tmp27 * tmp49
            tmp53 = tl.full([1, 1], 0.0, tl.float32)
            tmp54 = tl.where(tmp51, tmp53, tmp52)
            tmp55 = tl.full([1, 1], -0.07135481627260025, tl.float32)
            tmp56 = tmp35 * tmp55
            tmp57 = tmp56 + tmp40
            tmp58 = tmp57 * tmp34
            tmp59 = libdevice.exp(tmp58)
            tmp60 = tmp59 + tmp43
            tmp61 = (tmp43 / tmp60)
            tmp62 = tmp54 + tmp61
            tmp63 = tmp50 * tmp62
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp63, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/xp/cxplsknmzi5y5ulzhgvglz3imok7fxselopf3hsahx3h7we3zrd6.py
# Topologically Sorted Source Nodes: [view_148, sum_15], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_15 => sum_15
#   view_148 => view_148
# Graph fragment:
#   %mul_74 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_74]
#   %view_148 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_74, [128, 768]), kwargs = {})
#   %sum_15 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_148, [0], True), kwargs = {})
#   return %sum_15
triton_unk_fused_sum_view_4 = async_compile.triton('triton_unk_fused_sum_view_4', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_sum_view_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/fs/cfsfyafhlayby4n5w3vgwlrrf2ky6sbdzm7eub5ywhm46htidrin.py
# Topologically Sorted Source Nodes: [view_150, mul_76, mul_77, sum_16, mul_78, sum_17, mul_79, sub_25, sub_26, mul_80], Original ATen: [aten.view, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_76 => mul_76
#   mul_77 => mul_77
#   mul_78 => mul_78
#   mul_79 => mul_79
#   mul_80 => mul_80
#   sub_25 => sub_25
#   sub_26 => sub_26
#   sum_16 => sum_16
#   sum_17 => sum_17
#   view_150 => view_150
# Graph fragment:
#   %mm_2 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_2]
#   %primals_101 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_101]
#   %mul_48 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_48]
#   %div_22 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=div_22]
#   %sum_16 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_16]
#   %sum_17 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_17]
#   %view_150 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 128, 768]), kwargs = {})
#   %mul_76 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_150, %primals_101), kwargs = {})
#   %mul_77 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, 768), kwargs = {})
#   %sum_16 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_76, [2], True), kwargs = {})
#   %mul_78 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %mul_48), kwargs = {})
#   %sum_17 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_78, [2], True), kwargs = {})
#   %mul_79 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %sum_17), kwargs = {})
#   %sub_25 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_77, %sum_16), kwargs = {})
#   %sub_26 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_25, %mul_79), kwargs = {})
#   %mul_80 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_22, %sub_26), kwargs = {})
#   return %sum_16,%sum_17,%mul_80
triton_unk_fused_native_layer_norm_backward_view_5 = async_compile.triton('triton_unk_fused_native_layer_norm_backward_view_5', '''
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
    size_hints={'x': 128, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_native_layer_norm_backward_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_native_layer_norm_backward_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        tmp11 = tl.load(in_ptr3 + (x0), x0mask, eviction_policy='evict_last')
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp14 = tmp12 * tmp13
            tmp15 = tl.full([1, 1], 768.0, tl.float32)
            tmp16 = tmp14 * tmp15
            tmp17 = tmp16 - tmp4
            tmp19 = tmp18 * tmp9
            tmp20 = tmp17 - tmp19
            tmp21 = tmp11 * tmp20
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp21, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/cm/ccm3aefz5qlsywptekjagbpl73fpadba6cbrq7n52xsrf63mcf2c.py
# Topologically Sorted Source Nodes: [view_150, mul_81, sum_18, sum_19], Original ATen: [aten.view, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_81 => mul_81
#   sum_18 => sum_18
#   sum_19 => sum_19
#   view_150 => view_150
# Graph fragment:
#   %mm_2 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_2]
#   %mul_48 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_48]
#   %view_150 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 128, 768]), kwargs = {})
#   %mul_81 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_150, %mul_48), kwargs = {})
#   %sum_18 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_81, [0, 1]), kwargs = {})
#   %sum_19 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_150, [0, 1]), kwargs = {})
#   return %sum_18,%sum_19
triton_unk_fused_native_layer_norm_backward_view_6 = async_compile.triton('triton_unk_fused_native_layer_norm_backward_view_6', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_native_layer_norm_backward_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_native_layer_norm_backward_view_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/s2/cs2bqz3rmwxk2unpemfdxhlswgwz6w377qoxujmaghbtfiieh4u3.py
# Topologically Sorted Source Nodes: [x_20, x_21, mul_83, add_60, mul_84, exp_15, mul_85, add_61, mul_86, add_62, mul_87, mul_88, mul_89, mul_90], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_60 => add_60
#   add_61 => add_61
#   add_62 => add_62
#   exp_15 => exp_15
#   mul_83 => mul_83
#   mul_84 => mul_84
#   mul_85 => mul_85
#   mul_86 => mul_86
#   mul_87 => reciprocal_1
#   mul_88 => mul_88
#   mul_89 => mul_89
#   mul_90 => mul_90
#   x_20 => view_135
#   x_21 => mul_44
# Graph fragment:
#   %addmm_34 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_34]
#   %view_135 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_34, [1, 128, 3072]), kwargs = {})
#   %mul_44 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_135, %view_135), kwargs = {})
#   %mul_83 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, -0.07135481627260025), kwargs = {})
#   %add_60 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, -1.5957691216057308), kwargs = {})
#   %mul_84 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %view_135), kwargs = {})
#   %exp_15 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_84,), kwargs = {})
#   %mul_85 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, 0.2140644488178007), kwargs = {})
#   %add_61 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, 1.5957691216057308), kwargs = {})
#   %mul_86 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, %view_135), kwargs = {})
#   %add_62 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_15, 1.0), kwargs = {})
#   %reciprocal_1 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_62,), kwargs = {})
#   %mul_88 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_15, %reciprocal_1), kwargs = {})
#   %mul_89 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %mul_86), kwargs = {})
#   %mul_90 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %reciprocal_1), kwargs = {})
#   return %mul_90
triton_unk_fused_gelu_gelu_backward_view_7 = async_compile.triton('triton_unk_fused_gelu_gelu_backward_view_7', '''
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
    size_hints={'x': 393216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [393216, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 393216, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_gelu_backward_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_gelu_gelu_backward_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 393216
    x_g_tile0 : tl.constexpr = (393216) if (393216) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 393216
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
        tmp1 = tmp0 * tmp0
        tmp2 = tl.full([1], -0.07135481627260025, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.full([1], -1.5957691216057308, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5 * tmp0
        tmp7 = libdevice.exp(tmp6)
        tmp8 = tl.full([1], 1.0, tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = (tmp8 / tmp9)
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full([1], 0.2140644488178007, tl.float32)
        tmp13 = tmp1 * tmp12
        tmp14 = tl.full([1], 1.5957691216057308, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15 * tmp0
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17 * tmp10
        tl.store(out_ptr0 + (x0), tmp18, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/do/cdo5333yo4hmlh3zd7fvazdcriqx4hynr27gi42lkwdazcjj7ugv.py
# Topologically Sorted Source Nodes: [view_153, x_20, x_21, mul_83, add_60, mul_84, exp_15, add_62, mul_87, full_3, where_11, add_63, mul_91, view_154, sum_21], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_60 => add_60
#   add_62 => add_62
#   add_63 => add_63
#   exp_15 => exp_15
#   full_3 => full_default_12
#   mul_83 => mul_83
#   mul_84 => mul_84
#   mul_87 => reciprocal_1
#   mul_91 => mul_91
#   sum_21 => sum_21
#   view_153 => view_153
#   view_154 => view_154
#   where_11 => where_11
#   x_20 => view_135
#   x_21 => mul_44
# Graph fragment:
#   %mm_4 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_4]
#   %isnan_1 : Tensor "b8[1, 128, 3072][393216, 3072, 1]npu:0" = PlaceHolder[target=isnan_1]
#   %mul_90 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0" = PlaceHolder[target=mul_90]
#   %addmm_34 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_34]
#   %mul_91 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0" = PlaceHolder[target=mul_91]
#   %view_153 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 128, 3072]), kwargs = {})
#   %view_135 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_34, [1, 128, 3072]), kwargs = {})
#   %mul_44 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_135, %view_135), kwargs = {})
#   %mul_83 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, -0.07135481627260025), kwargs = {})
#   %add_60 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, -1.5957691216057308), kwargs = {})
#   %mul_84 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %view_135), kwargs = {})
#   %exp_15 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_84,), kwargs = {})
#   %add_62 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_15, 1.0), kwargs = {})
#   %reciprocal_1 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_62,), kwargs = {})
#   %full_default_12 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 3072], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_11 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan_1, %full_default_12, %mul_90), kwargs = {})
#   %add_63 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_11, %reciprocal_1), kwargs = {})
#   %mul_91 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_153, %add_63), kwargs = {})
#   %view_154 : Tensor "f32[128, 3072][3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_91, [128, 3072]), kwargs = {})
#   %sum_21 : Tensor "f32[1, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_154, [0], True), kwargs = {})
#   return %mul_91,%sum_21
triton_unk_fused_gelu_gelu_backward_sum_view_8 = async_compile.triton('triton_unk_fused_gelu_gelu_backward_sum_view_8', '''
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
    size_hints={'x': 3072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [3072, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 3072, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_gelu_backward_sum_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_gelu_gelu_backward_sum_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 3072
    r0_numel = 128
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
        _tmp19 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (x0 + 3072*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_ptr0 + (x0 + 3072 * r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0) != 0
            tmp2 = tl.load(in_ptr1 + (x0 + 3072*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp5 = tl.load(in_ptr2 + (x0 + 3072*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.full([1, 1], 0.0, tl.float32)
            tmp4 = tl.where(tmp1, tmp3, tmp2)
            tmp6 = tmp5 * tmp5
            tmp7 = tl.full([1, 1], -0.07135481627260025, tl.float32)
            tmp8 = tmp6 * tmp7
            tmp9 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10 * tmp5
            tmp12 = libdevice.exp(tmp11)
            tmp13 = tl.full([1, 1], 1.0, tl.float32)
            tmp14 = tmp12 + tmp13
            tmp15 = (tmp13 / tmp14)
            tmp16 = tmp4 + tmp15
            tmp17 = tmp0 * tmp16
            tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK, real_block_x0])
            tmp20 = _tmp19 + tmp18
            _tmp19 = tmp20
            tl.store(in_out_ptr0 + (x0 + 3072*r0_1), tmp17, r0_mask & x0mask)
        tmp19 = tl.sum(_tmp19, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp19, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/ej/cejeebkrrmwb7n2l7rs2wmode7u2ugovwk3juj55dceu24xzcdfi.py
# Topologically Sorted Source Nodes: [view_156, add_64, mul_93, mul_94, sum_22, mul_95, sum_23, mul_96, sub_28, sub_29, mul_97], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_64 => add_64
#   mul_93 => mul_93
#   mul_94 => mul_94
#   mul_95 => mul_95
#   mul_96 => mul_96
#   mul_97 => mul_97
#   sub_28 => sub_28
#   sub_29 => sub_29
#   sum_22 => sum_22
#   sum_23 => sum_23
#   view_156 => view_156
# Graph fragment:
#   %mul_80 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_80]
#   %mm_6 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_6]
#   %primals_95 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_95]
#   %mul_42 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_42]
#   %div_23 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=div_23]
#   %sum_22 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_22]
#   %sum_23 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_23]
#   %view_156 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 128, 768]), kwargs = {})
#   %add_64 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %view_156), kwargs = {})
#   %mul_93 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %primals_95), kwargs = {})
#   %mul_94 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, 768), kwargs = {})
#   %sum_22 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_93, [2], True), kwargs = {})
#   %mul_95 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %mul_42), kwargs = {})
#   %sum_23 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_95, [2], True), kwargs = {})
#   %mul_96 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %sum_23), kwargs = {})
#   %sub_28 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_94, %sum_22), kwargs = {})
#   %sub_29 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_28, %mul_96), kwargs = {})
#   %mul_97 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_23, %sub_29), kwargs = {})
#   return %sum_22,%sum_23,%mul_97
triton_unk_fused_add_native_layer_norm_backward_view_9 = async_compile.triton('triton_unk_fused_add_native_layer_norm_backward_view_9', '''
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
    size_hints={'x': 128, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_backward_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_backward_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        _tmp11 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 * tmp3
            tmp5 = tl.broadcast_to(tmp4, [real_block_x0, R0_BLOCK])
            tmp7 = _tmp6 + tmp5
            _tmp6 = tmp7
            tmp9 = tmp4 * tmp8
            tmp10 = tl.broadcast_to(tmp9, [real_block_x0, R0_BLOCK])
            tmp12 = _tmp11 + tmp10
            _tmp11 = tmp12
        tmp6 = tl.sum(_tmp6, 1)[:, None]
        tmp11 = tl.sum(_tmp11, 1)[:, None]
        tmp13 = tl.load(in_ptr4 + (x0), x0mask, eviction_policy='evict_last')
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp14 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp15 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp22 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp16 = tmp14 + tmp15
            tmp18 = tmp16 * tmp17
            tmp19 = tl.full([1, 1], 768.0, tl.float32)
            tmp20 = tmp18 * tmp19
            tmp21 = tmp20 - tmp6
            tmp23 = tmp22 * tmp11
            tmp24 = tmp21 - tmp23
            tmp25 = tmp13 * tmp24
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp25, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/ks/ckscwhr6aho5ln6djs3bp5wv7h2jybaiwbvhld42c7mkxeexahhh.py
# Topologically Sorted Source Nodes: [view_156, add_64, mul_98, sum_24, sum_25], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_64 => add_64
#   mul_98 => mul_98
#   sum_24 => sum_24
#   sum_25 => sum_25
#   view_156 => view_156
# Graph fragment:
#   %mul_80 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_80]
#   %mm_6 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_6]
#   %mul_42 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_42]
#   %view_156 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 128, 768]), kwargs = {})
#   %add_64 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %view_156), kwargs = {})
#   %mul_98 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %mul_42), kwargs = {})
#   %sum_24 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_98, [0, 1]), kwargs = {})
#   %sum_25 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_64, [0, 1]), kwargs = {})
#   return %sum_24,%sum_25
triton_unk_fused_add_native_layer_norm_backward_view_10 = async_compile.triton('triton_unk_fused_add_native_layer_norm_backward_view_10', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_backward_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_backward_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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
        _tmp6 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        _tmp9 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_ptr1 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr2 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 * tmp3
            tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK, real_block_x0])
            tmp7 = _tmp6 + tmp5
            _tmp6 = tmp7
            tmp8 = tl.broadcast_to(tmp2, [R0_BLOCK, real_block_x0])
            tmp10 = _tmp9 + tmp8
            _tmp9 = tmp10
        tmp6 = tl.sum(_tmp6, 0)[None, :]
        tmp9 = tl.sum(_tmp9, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp6, x0mask)
        tl.store(out_ptr1 + (x0), tmp9, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/pc/cpc4kxggcxzgyp2kybpeyu4rfinkjqf5ozyprhcjmtdvhifkmjlc.py
# Topologically Sorted Source Nodes: [view_166, permute_94, clone_6, view_173, view_174, sum_28], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
# Source node to ATen node mapping:
#   clone_6 => clone_6
#   permute_94 => permute_94
#   sum_28 => sum_28
#   view_166 => view_166
#   view_173 => view_173
#   view_174 => view_174
# Graph fragment:
#   %bmm_13 : Tensor "f32[12, 128, 64][8192, 64, 1]npu:0" = PlaceHolder[target=bmm_13]
#   %view_174 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=view_174]
#   %view_166 : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [1, 12, 128, 64]), kwargs = {})
#   %permute_94 : Tensor "f32[1, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_166, [0, 2, 1, 3]), kwargs = {})
#   %clone_6 : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_94,), kwargs = {memory_format: torch.contiguous_format})
#   %view_173 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [1, 128, 768]), kwargs = {})
#   %view_174 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_173, [128, 768]), kwargs = {})
#   %sum_28 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_174, [0], True), kwargs = {})
#   return %view_174,%sum_28
triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11 = async_compile.triton('triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, {'divisors': [1, 64]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x2', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x3', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (12) if (12) < (x_g_rem1) else (x_g_rem1)
    x2numel : tl.constexpr = 64
    x3numel : tl.constexpr = 12
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
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
        rbase = tl.arange(0, R0_BLOCK)[:, None, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None, None]
        x2offset = (group_base + i) % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_1 % x3_blocks * real_block_x3
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, :, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x2mask & x3mask
        x0 = x2 + 64*x3
        x0mask = x2mask & x3mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([R0_BLOCK, real_block_x3, real_block_x2], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x2 + 64*r0_1 + 8192*x3), r0_mask & x2mask & x3mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x3, real_block_x2])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
            tl.store(out_ptr0 + (x2 + 64*x3 + 768*r0_1), tmp0, r0_mask & x2mask & x3mask)
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(out_ptr1 + (x2 + 64*x3), tmp2, x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/io/ciopvghinmjylhvwjc5ycdapm7g3dvx3u736rtxdseq55jmiiyme.py
# Topologically Sorted Source Nodes: [scalar_tensor_3, mul_99, sum_27, mul_100, sub_30, attention_mask, eq, view_3, mask, where_12], Original ATen: [aten.nll_loss_backward, aten._softmax_backward_data, aten.ones, aten.eq, aten.view, aten.expand, aten.masked_fill]
# Source node to ATen node mapping:
#   attention_mask => full
#   eq => eq
#   mask => expand_2
#   mul_100 => mul_100
#   mul_99 => mul_99
#   scalar_tensor_3 => full_default_10
#   sub_30 => sub_30
#   sum_27 => sum_27
#   view_3 => view_12
#   where_12 => where_12
# Graph fragment:
#   %npu_dropout_backward_1 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=npu_dropout_backward_1]
#   %div_16 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=div_16]
#   %sum_27 : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1536]npu:0" = PlaceHolder[target=sum_27]
#   %full_default_10 : Tensor "f32[][]npu:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %mul_99 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%npu_dropout_backward_1, %div_16), kwargs = {})
#   %sum_27 : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_99, [-1], True), kwargs = {})
#   %mul_100 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %sum_27), kwargs = {})
#   %sub_30 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_99, %mul_100), kwargs = {})
#   %full : Tensor "f32[1, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %eq : Tensor "b8[1, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%full, 0), kwargs = {})
#   %view_12 : Tensor "b8[1, 1, 1, 128][128, 128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [1, 1, 1, 128]), kwargs = {})
#   %expand_2 : Tensor "b8[1, 12, 128, 128][128, 0, 0, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view_12, [1, 12, 128, 128]), kwargs = {})
#   %where_12 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default_10, %sub_30), kwargs = {})
#   return %sum_27,%expand_34
triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12 = async_compile.triton('triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12', '''
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
    size_hints={'x': 1536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1536, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 1536, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1536
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (1536) if (1536) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 1536
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
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp9 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp10 = tl.load(in_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp6 = tl.full([1, 1], 1.0, tl.float32)
            tmp7 = tl.full([1, 1], 0.0, tl.float32)
            tmp8 = tmp6 == tmp7
            tmp11 = tmp9 * tmp10
            tmp12 = tmp10 * tmp4
            tmp13 = tmp11 - tmp12
            tmp14 = tl.where(tmp8, tmp7, tmp13)
            tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp14, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/w5/cw5dm33ywb6qapjsmgvy37al4ixivpg3f4tde3q7mjzhdyrooaz7.py
# Topologically Sorted Source Nodes: [view_169, div_24, permute_104, clone_7, view_181, view_182, sum_30], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
# Source node to ATen node mapping:
#   clone_7 => clone_7
#   div_24 => div_24
#   permute_104 => permute_104
#   sum_30 => sum_30
#   view_169 => view_169
#   view_181 => view_181
#   view_182 => view_182
# Graph fragment:
#   %bmm_14 : Tensor "f32[12, 128, 64][8192, 64, 1]npu:0" = PlaceHolder[target=bmm_14]
#   %view_182 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=view_182]
#   %view_169 : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_14, [1, 12, 128, 64]), kwargs = {})
#   %div_24 : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_169, 8.0), kwargs = {})
#   %permute_104 : Tensor "f32[1, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_24, [0, 2, 1, 3]), kwargs = {})
#   %clone_7 : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_104,), kwargs = {memory_format: torch.contiguous_format})
#   %view_181 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [1, 128, 768]), kwargs = {})
#   %view_182 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_181, [128, 768]), kwargs = {})
#   %sum_30 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_182, [0], True), kwargs = {})
#   return %view_182,%sum_30
triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13 = async_compile.triton('triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, {'divisors': [1, 64]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x2', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x3', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (12) if (12) < (x_g_rem1) else (x_g_rem1)
    x2numel : tl.constexpr = 64
    x3numel : tl.constexpr = 12
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
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
        rbase = tl.arange(0, R0_BLOCK)[:, None, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None, None]
        x2offset = (group_base + i) % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_1 % x3_blocks * real_block_x3
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, :, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x2mask & x3mask
        x0 = x2 + 64*x3
        x0mask = x2mask & x3mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp4 = tl.full([R0_BLOCK, real_block_x3, real_block_x2], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x2 + 64*r0_1 + 8192*x3), r0_mask & x2mask & x3mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.full([1, 1, 1], 0.125, tl.float32)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, real_block_x3, real_block_x2])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
            tl.store(out_ptr0 + (x2 + 64*x3 + 768*r0_1), tmp2, r0_mask & x2mask & x3mask)
        tmp4 = tl.sum(_tmp4, 0)[None, :]
        tl.store(out_ptr1 + (x2 + 64*x3), tmp4, x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/pe/cpe6csglnymmylc52gmdc5dm4gfbnixjlpogxmlwcj62lc5e3ose.py
# Topologically Sorted Source Nodes: [view_157, sum_26, view_176, add_65, view_180, add_66, view_184, add_67, mul_107, sum_33, sum_34], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_65 => add_65
#   add_66 => add_66
#   add_67 => add_67
#   mul_107 => mul_107
#   sum_26 => sum_26
#   sum_33 => sum_33
#   sum_34 => sum_34
#   view_157 => view_157
#   view_176 => view_176
#   view_180 => view_180
#   view_184 => view_184
# Graph fragment:
#   %mul_97 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_97]
#   %mm_10 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_10]
#   %mm_12 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_12]
#   %mm_14 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_14]
#   %mul_40 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_40]
#   %view_157 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_97, [128, 768]), kwargs = {})
#   %sum_26 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_157, [0], True), kwargs = {})
#   %view_176 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [1, 128, 768]), kwargs = {})
#   %add_65 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %view_176), kwargs = {})
#   %view_180 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [1, 128, 768]), kwargs = {})
#   %add_66 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %view_180), kwargs = {})
#   %view_184 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_14, [1, 128, 768]), kwargs = {})
#   %add_67 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %view_184), kwargs = {})
#   %mul_107 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %mul_40), kwargs = {})
#   %sum_33 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_107, [0, 1]), kwargs = {})
#   %sum_34 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_67, [0, 1]), kwargs = {})
#   return %sum_26,%sum_33,%sum_34
triton_unk_fused_add_native_layer_norm_backward_sum_view_14 = async_compile.triton('triton_unk_fused_add_native_layer_norm_backward_sum_view_14', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_backward_sum_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_backward_sum_view_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        _tmp13 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        _tmp16 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp4 = tl.load(in_ptr1 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp6 = tl.load(in_ptr2 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp8 = tl.load(in_ptr3 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp10 = tl.load(in_ptr4 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
            tmp5 = tmp0 + tmp4
            tmp7 = tmp5 + tmp6
            tmp9 = tmp7 + tmp8
            tmp11 = tmp9 * tmp10
            tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK, real_block_x0])
            tmp14 = _tmp13 + tmp12
            _tmp13 = tmp14
            tmp15 = tl.broadcast_to(tmp9, [R0_BLOCK, real_block_x0])
            tmp17 = _tmp16 + tmp15
            _tmp16 = tmp17
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tmp13 = tl.sum(_tmp13, 0)[None, :]
        tmp16 = tl.sum(_tmp16, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
        tl.store(out_ptr1 + (x0), tmp13, x0mask)
        tl.store(out_ptr2 + (x0), tmp16, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/xy/cxycho2omzixwg25viuadlcgxgyhf257xenx5ccwd2jl4t52iyq7.py
# Topologically Sorted Source Nodes: [permute_96], Original ATen: [aten.t]
# Source node to ATen node mapping:
#   permute_96 => permute_96
# Graph fragment:
#   %view_174 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=view_174]
#   %permute_96 : Tensor "f32[768, 128][1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_174, [1, 0]), kwargs = {})
#   return %permute_96
triton_unk_fused_t_15 = async_compile.triton('triton_unk_fused_t_15', '''
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
    size_hints={'x': 98304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [98304, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 98304, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_t_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_t_15(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 98304
    x_g_tile0 : tl.constexpr = (98304) if (98304) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 98304
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/bg/cbgdhxn77bjrymml4hvme5a2p6uahsr3guv4kib7r2zfaa3mgrz6.py
# Topologically Sorted Source Nodes: [view_172, permute_93, permute_99, view_177, view_178, sum_29], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
# Source node to ATen node mapping:
#   permute_93 => permute_93
#   permute_99 => permute_99
#   sum_29 => sum_29
#   view_172 => view_172
#   view_177 => view_177
#   view_178 => view_178
# Graph fragment:
#   %bmm_15 : Tensor "f32[12, 64, 128][8192, 128, 1]npu:0" = PlaceHolder[target=bmm_15]
#   %view_172 : Tensor "f32[1, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_15, [1, 12, 64, 128]), kwargs = {})
#   %permute_93 : Tensor "f32[1, 12, 128, 64][98304, 8192, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_172, [0, 1, 3, 2]), kwargs = {})
#   %permute_99 : Tensor "f32[1, 128, 12, 64][98304, 1, 8192, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_93, [0, 2, 1, 3]), kwargs = {})
#   %view_177 : Tensor "f32[1, 128, 768][98304, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_99, [1, 128, 768]), kwargs = {})
#   %view_178 : Tensor "f32[128, 768][1, 128]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_177, [128, 768]), kwargs = {})
#   %sum_29 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_178, [0], True), kwargs = {})
#   return %sum_29
triton_unk_fused_matmul_backward_sum_transpose_view_16 = async_compile.triton('triton_unk_fused_matmul_backward_sum_transpose_view_16', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_sum_transpose_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_matmul_backward_sum_transpose_view_16(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (768) if (768) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 768
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
            tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [real_block_x0, R0_BLOCK])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 1)[:, None]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/bc/cbcl5zwc3nbjirbwgna5lv4zkib7kkxi7bwjeu4cgynlvgtblgd4.py
# Topologically Sorted Source Nodes: [view_176, add_65, view_180, add_66, view_184, add_67, mul_102, mul_103, sum_31, mul_104, sum_32, mul_105, sub_32, sub_33, mul_106], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_65 => add_65
#   add_66 => add_66
#   add_67 => add_67
#   mul_102 => mul_102
#   mul_103 => mul_103
#   mul_104 => mul_104
#   mul_105 => mul_105
#   mul_106 => mul_106
#   sub_32 => sub_32
#   sub_33 => sub_33
#   sum_31 => sum_31
#   sum_32 => sum_32
#   view_176 => view_176
#   view_180 => view_180
#   view_184 => view_184
# Graph fragment:
#   %mul_97 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_97]
#   %mm_10 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_10]
#   %mm_12 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_12]
#   %mm_14 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_14]
#   %primals_85 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_85]
#   %mul_102 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_102]
#   %mul_40 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_40]
#   %div_25 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=div_25]
#   %sum_31 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_31]
#   %sum_32 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=sum_32]
#   %view_176 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [1, 128, 768]), kwargs = {})
#   %add_65 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %view_176), kwargs = {})
#   %view_180 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [1, 128, 768]), kwargs = {})
#   %add_66 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %view_180), kwargs = {})
#   %view_184 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_14, [1, 128, 768]), kwargs = {})
#   %add_67 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %view_184), kwargs = {})
#   %mul_102 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %primals_85), kwargs = {})
#   %mul_103 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, 768), kwargs = {})
#   %sum_31 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_102, [2], True), kwargs = {})
#   %mul_104 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %mul_40), kwargs = {})
#   %sum_32 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_104, [2], True), kwargs = {})
#   %mul_105 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %sum_32), kwargs = {})
#   %sub_32 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_103, %sum_31), kwargs = {})
#   %sub_33 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %mul_105), kwargs = {})
#   %mul_106 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_25, %sub_33), kwargs = {})
#   return %mul_102,%sum_31,%sum_32,%mul_106
triton_unk_fused_add_native_layer_norm_backward_view_17 = async_compile.triton('triton_unk_fused_add_native_layer_norm_backward_view_17', '''
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
    size_hints={'x': 128, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_backward_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_backward_view_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        _tmp15 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp5 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp7 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp12 = tl.load(in_ptr4 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            tmp6 = tmp4 + tmp5
            tmp8 = tmp6 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tmp11
            tmp13 = tmp8 * tmp12
            tmp14 = tl.broadcast_to(tmp13, [real_block_x0, R0_BLOCK])
            tmp16 = _tmp15 + tmp14
            _tmp15 = tmp16
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & x0mask)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tmp15 = tl.sum(_tmp15, 1)[:, None]
        tmp17 = tl.load(in_ptr5 + (x0), x0mask, eviction_policy='evict_last')
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp18 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp22 = tl.load(in_ptr4 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp19 = tl.full([1, 1], 768.0, tl.float32)
            tmp20 = tmp18 * tmp19
            tmp21 = tmp20 - tmp10
            tmp23 = tmp22 * tmp15
            tmp24 = tmp21 - tmp23
            tmp25 = tmp17 * tmp24
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/mb/cmbvtb7spxx7wprau55lqq4bfaewsygb3w2akymqnw6leyfog34q.py
# Topologically Sorted Source Nodes: [view_327, sum_101, view_346, add_105, view_350, add_106, view_354, add_107], Original ATen: [aten.view, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_105 => add_105
#   add_106 => add_106
#   add_107 => add_107
#   sum_101 => sum_101
#   view_327 => view_327
#   view_346 => view_346
#   view_350 => view_350
#   view_354 => view_354
# Graph fragment:
#   %mul_227 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_227]
#   %mm_70 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_70]
#   %mm_72 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_72]
#   %mm_74 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_74]
#   %view_327 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_227, [128, 768]), kwargs = {})
#   %sum_101 : Tensor "f32[1, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_327, [0], True), kwargs = {})
#   %view_346 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_70, [1, 128, 768]), kwargs = {})
#   %add_105 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %view_346), kwargs = {})
#   %view_350 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_72, [1, 128, 768]), kwargs = {})
#   %add_106 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %view_350), kwargs = {})
#   %view_354 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_74, [1, 128, 768]), kwargs = {})
#   %add_107 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %view_354), kwargs = {})
#   return %sum_101,%add_107
triton_unk_fused_add_sum_view_18 = async_compile.triton('triton_unk_fused_add_sum_view_18', '''
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
    size_hints={'x': 768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [768, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_sum_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_sum_view_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 768
    r0_numel = 128
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
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp4 = tl.load(in_out_ptr0 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp6 = tl.load(in_ptr1 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp8 = tl.load(in_ptr2 + (x0 + 768*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
            tmp5 = tmp0 + tmp4
            tmp7 = tmp5 + tmp6
            tmp9 = tmp7 + tmp8
            tl.store(in_out_ptr0 + (x0 + 768*r0_1), tmp9, r0_mask & x0mask)
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/cu/ccup7bexgbjwcqptzxbbiycntlsqz43iv6djltmilonikk32rjis.py
# Topologically Sorted Source Nodes: [add_108], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_108 => add_108
# Graph fragment:
#   %mm_1 : Tensor "f32[30522, 768][768, 1]npu:0" = PlaceHolder[target=mm_1]
#   %embedding_dense_backward_1 : Tensor "f32[30522, 768][768, 1]npu:0" = PlaceHolder[target=embedding_dense_backward_1]
#   %add_108 : Tensor "f32[30522, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %embedding_dense_backward_1), kwargs = {})
#   return %add_108
triton_unk_fused_add_19 = async_compile.triton('triton_unk_fused_add_19', '''
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
    size_hints={'x': 23440896}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [23440896, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 23440896, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 23440896
    x_g_tile0 : tl.constexpr = (23440896) if (23440896) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 23440896
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
        tmp2 = tmp0 + tmp1
        tl.store(in_out_ptr0 + (x0), tmp2, x0mask)
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
        primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, mul, getitem_3, view, div_1, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_8, view_23, div_4, getitem_13, view_40, mul_10, view_42, addmm_10, view_44, getitem_17, mul_16, view_46, div_7, getitem_21, view_63, mul_18, view_65, addmm_16, view_67, getitem_25, mul_24, view_69, div_10, getitem_29, view_86, mul_26, view_88, addmm_22, view_90, getitem_33, mul_32, view_92, div_13, getitem_37, view_109, mul_34, view_111, addmm_28, view_113, getitem_41, mul_40, view_115, div_16, getitem_45, view_132, mul_42, view_134, addmm_34, view_136, getitem_49, mul_48, view_138, addmm_36, getitem_53, rsqrt_13, view_140, _log_softmax, convert_element_type, div_22, div_23, view_162, view_164, view_168, view_170, div_25, div_26, view_196, view_198, view_202, view_204, div_28, div_29, view_230, view_232, view_236, view_238, div_31, div_32, view_264, view_266, view_270, view_272, div_34, div_35, view_298, view_300, view_304, view_306, div_37, div_38, view_332, view_334, view_338, view_340, div_40, tangents_1, tangents_2 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            tangents_2 = copy_if_misaligned(tangents_2)
            buf1 = empty_strided_npu((1, 128, 30522), (3907584, 30528, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, sum_9, mul_57, sub_20, view_144, add_55], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0.run(primals_108, tangents_1, convert_element_type, tangents_2, _log_softmax, buf1, 128, 30522, stream=raw_stream0)
            del _log_softmax
            del convert_element_type
            del primals_108
            del tangents_1
            del tangents_2
            buf2 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, mul_57, sub_20, view_144, add_55, view_145, prediction_logits_3, permute_68, mm], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(reinterpret_tensor(buf1, (128, 30522), (30528, 1), 0), primals_2, out=buf2)
            del primals_2
            buf3 = empty_strided_npu((30522, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, mul_57, sub_20, view_144, add_55, view_145, permute_69, permute_71], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(buf1, (30522, 128), (1, 30528), 0), view_140, out=buf3)
            del view_140
            buf4 = empty_strided_npu((1, 30522), (30522, 1), torch.float32)
            # Topologically Sorted Source Nodes: [div_20, view_31, unsqueeze_1, ne_3, scalar_tensor_2, where_8, , scalar_tensor_3, where_9, mul_56, exp_13, mul_57, sub_20, view_144, add_55, view_145, sum_10], Original ATen: [aten.nll_loss_backward, aten.view, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._log_softmax_backward_data, aten.add, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_scalar_tensor_sum_view_1.run(buf1, buf4, 30522, 128, stream=raw_stream0)
            del buf1
            buf9 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf7 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf8 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_147, prediction_logits, prediction_logits_1, prediction_logits_2, mul_64, sum_13, sum_14, mul_66, add_56, mul_67, exp_14, mul_68, add_57, mul_69, add_58, mul_70, mul_71, mul_72, mul_73], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2.run(addmm_36, buf2, getitem_53, rsqrt_13, buf9, buf7, buf8, 768, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, mul_66, add_56, mul_67, exp_14, mul_68, add_57, mul_69, add_58, mul_70, mul_71, mul_72, mul_73, isnan], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:3
            buf10 = torch.ops.aten.isnan.default(buf9)
            assert_alignment(buf10, 16, 'torch.ops.aten.isnan.default')
            buf11 = reinterpret_tensor(buf2, (1, 128, 768), (98304, 768, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [view_147, mul_59, mul_60, sum_11, prediction_logits, prediction_logits_1, prediction_logits_2, mul_61, sum_12, mul_62, sub_22, sub_23, div_21, mul_63, mul_66, add_56, mul_67, exp_14, add_58, mul_70, full_2, where_10, add_59, mul_74], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_3.run(buf11, primals_105, addmm_36, getitem_53, rsqrt_13, buf10, buf9, 128, 768, stream=raw_stream0)
            del addmm_36
            del buf10
            del getitem_53
            del primals_105
            del rsqrt_13
            buf12 = reinterpret_tensor(buf9, (128, 768), (768, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [view_148, prediction_logits, permute_72, mm_2], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf11, (128, 768), (768, 1), 0), primals_103, out=buf12)
            del primals_103
            buf13 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_148, permute_73, permute_75], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf11, (768, 128), (1, 768), 0), view_138, out=buf13)
            del view_138
            buf14 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_148, sum_15], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf11, buf14, 768, 128, stream=raw_stream0)
            buf17 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [view_150, mul_76, mul_77, sum_16, mul_78, sum_17, mul_79, sub_25, sub_26, mul_80], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_backward_view_5.run(buf12, primals_101, mul_48, div_22, buf17, 128, 768, stream=raw_stream0)
            del div_22
            del primals_101
            buf18 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf19 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_150, mul_81, sum_18, sum_19], Original ATen: [aten.view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_backward_view_6.run(buf12, mul_48, buf18, buf19, 768, 128, stream=raw_stream0)
            del mul_48
            # Topologically Sorted Source Nodes: [npu_dropout_backward], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:6
            buf20 = torch.ops.npu.npu_dropout_backward.default(buf17, getitem_49, 0.10000000000000009)
            assert_alignment(buf20, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_49
            buf21 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_151, x_22, permute_76, mm_4], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf20, (128, 768), (768, 1), 0), primals_99, out=buf21)
            del primals_99
            buf22 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_151, permute_77, permute_79], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf20, (768, 128), (1, 768), 0), view_136, out=buf22)
            del view_136
            buf23 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_151, sum_20], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf20, buf23, 768, 128, stream=raw_stream0)
            buf24 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_83, add_60, mul_84, exp_15, mul_85, add_61, mul_86, add_62, mul_87, mul_88, mul_89, mul_90], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_34, buf24, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_83, add_60, mul_84, exp_15, mul_85, add_61, mul_86, add_62, mul_87, mul_88, mul_89, mul_90, isnan_1], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:9
            buf25 = torch.ops.aten.isnan.default(buf24)
            assert_alignment(buf25, 16, 'torch.ops.aten.isnan.default')
            buf26 = reinterpret_tensor(buf21, (1, 128, 3072), (393216, 3072, 1), 0); del buf21  # reuse
            buf29 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_153, x_20, x_21, mul_83, add_60, mul_84, exp_15, add_62, mul_87, full_3, where_11, add_63, mul_91, view_154, sum_21], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf26, buf25, buf24, addmm_34, buf29, 3072, 128, stream=raw_stream0)
            del addmm_34
            del buf25
            buf27 = reinterpret_tensor(buf20, (128, 768), (768, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [view_153, x_20, x_21, mul_83, add_60, mul_84, exp_15, add_62, mul_87, full_3, where_11, add_63, mul_91, view_154, permute_80, mm_6], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf26, (128, 3072), (3072, 1), 0), primals_97, out=buf27)
            del primals_97
            buf28 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_153, x_20, x_21, mul_83, add_60, mul_84, exp_15, add_62, mul_87, full_3, where_11, add_63, mul_91, view_154, permute_81, permute_83], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:11
            extern_kernels.mm(reinterpret_tensor(buf26, (3072, 128), (1, 3072), 0), view_134, out=buf28)
            del view_134
            buf32 = reinterpret_tensor(buf12, (1, 128, 768), (98304, 768, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [view_156, add_64, mul_93, mul_94, sum_22, mul_95, sum_23, mul_96, sub_28, sub_29, mul_97], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf17, buf27, primals_95, mul_42, div_23, buf32, 128, 768, stream=raw_stream0)
            del div_23
            del primals_95
            buf33 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf34 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_156, add_64, mul_98, sum_24, sum_25], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf17, buf27, mul_42, buf33, buf34, 768, 128, stream=raw_stream0)
            del mul_42
            buf35 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [view_157, context_17, permute_84, mm_8], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:12
            extern_kernels.mm(reinterpret_tensor(buf32, (128, 768), (768, 1), 0), primals_93, out=buf35)
            del primals_93
            buf36 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_157, permute_85, permute_87], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(reinterpret_tensor(buf32, (768, 128), (1, 768), 0), view_132, out=buf36)
            del view_132
            buf39 = reinterpret_tensor(buf17, (12, 128, 64), (8192, 64, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [view_159, view_160, permute_88, view_161, bmm_13], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:14
            extern_kernels.bmm(view_164, reinterpret_tensor(buf35, (12, 128, 64), (64, 768, 1), 0), out=buf39)
            del view_164
            buf45 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            buf49 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_166, permute_94, clone_6, view_173, view_174, sum_28], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf39, buf45, buf49, 768, 128, stream=raw_stream0)
            buf46 = reinterpret_tensor(buf39, (128, 768), (768, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_32, permute_95, mm_10], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(buf45, primals_91, out=buf46)
            del primals_91
            buf38 = empty_strided_npu((12, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_159, view_160, permute_88, view_161, bmm_12], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:16
            extern_kernels.bmm(reinterpret_tensor(buf35, (12, 128, 64), (64, 768, 1), 0), view_162, out=buf38)
            del view_162
            # Topologically Sorted Source Nodes: [view_163, npu_dropout_backward_1], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:17
            buf40 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf38, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_45, 0.10000000000000009)
            assert_alignment(buf40, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf38
            del getitem_45
            buf42 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, mul_99, sum_27, mul_100, sub_30, attention_mask, eq, view_3, mask, where_12], Original ATen: [aten.nll_loss_backward, aten._softmax_backward_data, aten.ones, aten.eq, aten.view, aten.expand, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf42, div_16, 1536, 128, stream=raw_stream0)
            del div_16
            buf44 = reinterpret_tensor(buf35, (12, 64, 128), (8192, 128, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, mul_99, mul_100, sub_30, attention_mask, eq, view_3, mask, where_12, view_167, bmm_15], Original ATen: [aten.nll_loss_backward, aten._softmax_backward_data, aten.ones, aten.eq, aten.view, aten.expand, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:18
            extern_kernels.bmm(view_170, reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0), out=buf44)
            del view_170
            buf50 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_172, permute_93, permute_99, view_177, view_178, linear_31, permute_100, mm_12], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf44, (128, 768), (1, 128), 0), primals_89, out=buf50)
            del primals_89
            buf43 = empty_strided_npu((12, 128, 64), (8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [scalar_tensor_3, mul_99, mul_100, sub_30, attention_mask, eq, view_3, mask, where_12, view_167, bmm_14], Original ATen: [aten.nll_loss_backward, aten._softmax_backward_data, aten.ones, aten.eq, aten.view, aten.expand, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:20
            extern_kernels.bmm(reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0), view_168, out=buf43)
            del view_168
            buf53 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            buf57 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_169, div_24, permute_104, clone_7, view_181, view_182, sum_30], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf43, buf53, buf57, 768, 128, stream=raw_stream0)
            buf54 = reinterpret_tensor(buf43, (128, 768), (768, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_30, permute_105, mm_14], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(buf53, primals_87, out=buf54)
            del primals_87
            buf37 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf62 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf63 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_157, sum_26, view_176, add_65, view_180, add_66, view_184, add_67, mul_107, sum_33, sum_34], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_sum_view_14.run(buf32, buf46, buf50, buf54, mul_40, buf37, buf62, buf63, 768, 128, stream=raw_stream0)
            buf47 = reinterpret_tensor(buf45, (768, 128), (1, 768), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [permute_96], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf47, 98304, stream=raw_stream0)
            buf48 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_98], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(buf47, view_115, out=buf48)
            del buf47
            buf51 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_172, permute_93, permute_99, view_177, view_178, permute_101, permute_103], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:23
            extern_kernels.mm(reinterpret_tensor(buf44, (768, 128), (128, 1), 0), view_115, out=buf51)
            buf52 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_172, permute_93, permute_99, view_177, view_178, sum_29], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf44, buf52, 768, 128, stream=raw_stream0)
            buf55 = reinterpret_tensor(buf53, (768, 128), (1, 768), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [permute_106], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf55, 98304, stream=raw_stream0)
            buf56 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_108], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:24
            extern_kernels.mm(buf55, view_115, out=buf56)
            del view_115
            buf58 = buf32; del buf32  # reuse
            buf61 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [view_176, add_65, view_180, add_66, view_184, add_67, mul_102, mul_103, sum_31, mul_104, sum_32, mul_105, sub_32, sub_33, mul_106], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_17.run(buf61, buf46, buf50, buf54, primals_85, mul_40, div_25, 128, 768, stream=raw_stream0)
            del div_25
            del mul_40
            del primals_85
            # Topologically Sorted Source Nodes: [npu_dropout_backward_2], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:25
            buf64 = torch.ops.npu.npu_dropout_backward.default(buf61, getitem_41, 0.10000000000000009)
            assert_alignment(buf64, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_41
            buf65 = reinterpret_tensor(buf26, (128, 3072), (3072, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [view_185, x_18, permute_109, mm_16], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:26
            extern_kernels.mm(reinterpret_tensor(buf64, (128, 768), (768, 1), 0), primals_83, out=buf65)
            del primals_83
            buf66 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_185, permute_110, permute_112], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:27
            extern_kernels.mm(reinterpret_tensor(buf64, (768, 128), (1, 768), 0), view_113, out=buf66)
            del view_113
            buf67 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_185, sum_35], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf64, buf67, 768, 128, stream=raw_stream0)
            buf68 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, mul_109, add_68, mul_110, exp_16, mul_111, add_69, mul_112, add_70, mul_113, mul_114, mul_115, mul_116], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_28, buf68, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_16, x_17, mul_109, add_68, mul_110, exp_16, mul_111, add_69, mul_112, add_70, mul_113, mul_114, mul_115, mul_116, isnan_2], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:28
            buf69 = torch.ops.aten.isnan.default(buf68)
            assert_alignment(buf69, 16, 'torch.ops.aten.isnan.default')
            buf70 = reinterpret_tensor(buf65, (1, 128, 3072), (393216, 3072, 1), 0); del buf65  # reuse
            buf73 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_187, x_16, x_17, mul_109, add_68, mul_110, exp_16, add_70, mul_113, where_13, add_71, mul_117, view_188, sum_36], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf70, buf69, buf68, addmm_28, buf73, 3072, 128, stream=raw_stream0)
            del addmm_28
            del buf69
            buf71 = reinterpret_tensor(buf64, (128, 768), (768, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [full_3, view_187, x_16, x_17, mul_109, add_68, mul_110, exp_16, add_70, mul_113, where_13, add_71, mul_117, view_188, permute_113, mm_18], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:29
            extern_kernels.mm(reinterpret_tensor(buf70, (128, 3072), (3072, 1), 0), primals_81, out=buf71)
            del primals_81
            buf72 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_187, x_16, x_17, mul_109, add_68, mul_110, exp_16, add_70, mul_113, where_13, add_71, mul_117, view_188, permute_114, permute_116], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf70, (3072, 128), (1, 3072), 0), view_111, out=buf72)
            del view_111
            buf76 = reinterpret_tensor(buf54, (1, 128, 768), (98304, 768, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [view_190, add_72, mul_119, mul_120, sum_37, mul_121, sum_38, mul_122, sub_35, sub_36, mul_123], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf61, buf71, primals_79, mul_34, div_26, buf76, 128, 768, stream=raw_stream0)
            del div_26
            del primals_79
            buf77 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf78 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_190, add_72, mul_124, sum_39, sum_40], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf61, buf71, mul_34, buf77, buf78, 768, 128, stream=raw_stream0)
            del mul_34
            buf79 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [view_191, context_14, permute_117, mm_20], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:31
            extern_kernels.mm(reinterpret_tensor(buf76, (128, 768), (768, 1), 0), primals_77, out=buf79)
            del primals_77
            buf80 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_191, permute_118, permute_120], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(reinterpret_tensor(buf76, (768, 128), (1, 768), 0), view_109, out=buf80)
            del view_109
            buf83 = reinterpret_tensor(buf61, (12, 128, 64), (8192, 64, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [view_193, view_194, permute_121, view_195, bmm_17], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:33
            extern_kernels.bmm(view_198, reinterpret_tensor(buf79, (12, 128, 64), (64, 768, 1), 0), out=buf83)
            del view_198
            buf89 = buf50; del buf50  # reuse
            buf93 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_200, permute_127, clone_8, view_207, view_208, sum_43], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf83, buf89, buf93, 768, 128, stream=raw_stream0)
            buf90 = reinterpret_tensor(buf83, (128, 768), (768, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [linear_26, permute_128, mm_22], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(buf89, primals_75, out=buf90)
            del primals_75
            buf82 = reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [view_193, view_194, permute_121, view_195, bmm_16], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:35
            extern_kernels.bmm(reinterpret_tensor(buf79, (12, 128, 64), (64, 768, 1), 0), view_196, out=buf82)
            del view_196
            # Topologically Sorted Source Nodes: [view_197, npu_dropout_backward_3], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:36
            buf84 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf82, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_37, 0.10000000000000009)
            assert_alignment(buf84, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf82
            del getitem_37
            buf86 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_125, sum_42, mul_126, sub_37, where_14], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf86, div_13, 1536, 128, stream=raw_stream0)
            del div_13
            buf88 = reinterpret_tensor(buf79, (12, 64, 128), (8192, 128, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_125, mul_126, sub_37, where_14, view_201, bmm_19], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:37
            extern_kernels.bmm(view_204, reinterpret_tensor(buf86, (12, 128, 128), (16384, 128, 1), 0), out=buf88)
            del view_204
            buf94 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [view_206, permute_126, permute_132, view_211, view_212, linear_25, permute_133, mm_24], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:38
            extern_kernels.mm(reinterpret_tensor(buf88, (128, 768), (1, 128), 0), primals_73, out=buf94)
            del primals_73
            buf87 = reinterpret_tensor(buf55, (12, 128, 64), (8192, 64, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_125, mul_126, sub_37, where_14, view_201, bmm_18], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf86, (12, 128, 128), (16384, 128, 1), 0), view_202, out=buf87)
            del view_202
            buf97 = reinterpret_tensor(buf44, (128, 768), (768, 1), 0); del buf44  # reuse
            buf101 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_203, div_27, permute_137, clone_9, view_215, view_216, sum_45], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf87, buf97, buf101, 768, 128, stream=raw_stream0)
            buf98 = reinterpret_tensor(buf87, (128, 768), (768, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [linear_24, permute_138, mm_26], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:40
            extern_kernels.mm(buf97, primals_71, out=buf98)
            del primals_71
            buf81 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf106 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf107 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_191, sum_41, view_210, add_73, view_214, add_74, view_218, add_75, mul_133, sum_48, sum_49], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_sum_view_14.run(buf76, buf90, buf94, buf98, mul_32, buf81, buf106, buf107, 768, 128, stream=raw_stream0)
            buf91 = reinterpret_tensor(buf89, (768, 128), (1, 768), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [permute_129], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf91, 98304, stream=raw_stream0)
            buf92 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_131], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:41
            extern_kernels.mm(buf91, view_92, out=buf92)
            del buf91
            buf95 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_206, permute_126, permute_132, view_211, view_212, permute_134, permute_136], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf88, (768, 128), (128, 1), 0), view_92, out=buf95)
            buf96 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_206, permute_126, permute_132, view_211, view_212, sum_44], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf88, buf96, 768, 128, stream=raw_stream0)
            buf99 = reinterpret_tensor(buf97, (768, 128), (1, 768), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [permute_139], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf99, 98304, stream=raw_stream0)
            buf100 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_141], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:43
            extern_kernels.mm(buf99, view_92, out=buf100)
            del view_92
            buf102 = buf76; del buf76  # reuse
            buf105 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [view_210, add_73, view_214, add_74, view_218, add_75, mul_128, mul_129, sum_46, mul_130, sum_47, mul_131, sub_39, sub_40, mul_132], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_17.run(buf105, buf90, buf94, buf98, primals_69, mul_32, div_28, 128, 768, stream=raw_stream0)
            del div_28
            del mul_32
            del primals_69
            # Topologically Sorted Source Nodes: [npu_dropout_backward_4], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:44
            buf108 = torch.ops.npu.npu_dropout_backward.default(buf105, getitem_33, 0.10000000000000009)
            assert_alignment(buf108, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_33
            buf109 = reinterpret_tensor(buf70, (128, 3072), (3072, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [view_219, x_14, permute_142, mm_28], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:45
            extern_kernels.mm(reinterpret_tensor(buf108, (128, 768), (768, 1), 0), primals_67, out=buf109)
            del primals_67
            buf110 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_219, permute_143, permute_145], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:46
            extern_kernels.mm(reinterpret_tensor(buf108, (768, 128), (1, 768), 0), view_90, out=buf110)
            del view_90
            buf111 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_219, sum_50], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf108, buf111, 768, 128, stream=raw_stream0)
            buf112 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_135, add_76, mul_136, exp_17, mul_137, add_77, mul_138, add_78, mul_139, mul_140, mul_141, mul_142], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_22, buf112, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_135, add_76, mul_136, exp_17, mul_137, add_77, mul_138, add_78, mul_139, mul_140, mul_141, mul_142, isnan_3], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:47
            buf113 = torch.ops.aten.isnan.default(buf112)
            assert_alignment(buf113, 16, 'torch.ops.aten.isnan.default')
            buf114 = reinterpret_tensor(buf109, (1, 128, 3072), (393216, 3072, 1), 0); del buf109  # reuse
            buf117 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_221, x_12, x_13, mul_135, add_76, mul_136, exp_17, add_78, mul_139, where_15, add_79, mul_143, view_222, sum_51], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf114, buf113, buf112, addmm_22, buf117, 3072, 128, stream=raw_stream0)
            del addmm_22
            del buf113
            buf115 = reinterpret_tensor(buf108, (128, 768), (768, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [full_3, view_221, x_12, x_13, mul_135, add_76, mul_136, exp_17, add_78, mul_139, where_15, add_79, mul_143, view_222, permute_146, mm_30], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:48
            extern_kernels.mm(reinterpret_tensor(buf114, (128, 3072), (3072, 1), 0), primals_65, out=buf115)
            del primals_65
            buf116 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_221, x_12, x_13, mul_135, add_76, mul_136, exp_17, add_78, mul_139, where_15, add_79, mul_143, view_222, permute_147, permute_149], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:49
            extern_kernels.mm(reinterpret_tensor(buf114, (3072, 128), (1, 3072), 0), view_88, out=buf116)
            del view_88
            buf120 = reinterpret_tensor(buf98, (1, 128, 768), (98304, 768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [view_224, add_80, mul_145, mul_146, sum_52, mul_147, sum_53, mul_148, sub_42, sub_43, mul_149], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf105, buf115, primals_63, mul_26, div_29, buf120, 128, 768, stream=raw_stream0)
            del div_29
            del primals_63
            buf121 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf122 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_224, add_80, mul_150, sum_54, sum_55], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf105, buf115, mul_26, buf121, buf122, 768, 128, stream=raw_stream0)
            del mul_26
            buf123 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [view_225, context_11, permute_150, mm_32], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf120, (128, 768), (768, 1), 0), primals_61, out=buf123)
            del primals_61
            buf124 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_225, permute_151, permute_153], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf120, (768, 128), (1, 768), 0), view_86, out=buf124)
            del view_86
            buf127 = reinterpret_tensor(buf105, (12, 128, 64), (8192, 64, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [view_227, view_228, permute_154, view_229, bmm_21], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:52
            extern_kernels.bmm(view_232, reinterpret_tensor(buf123, (12, 128, 64), (64, 768, 1), 0), out=buf127)
            del view_232
            buf133 = buf94; del buf94  # reuse
            buf137 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_234, permute_160, clone_10, view_241, view_242, sum_58], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf127, buf133, buf137, 768, 128, stream=raw_stream0)
            buf134 = reinterpret_tensor(buf127, (128, 768), (768, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [linear_20, permute_161, mm_34], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:53
            extern_kernels.mm(buf133, primals_59, out=buf134)
            del primals_59
            buf126 = reinterpret_tensor(buf86, (12, 128, 128), (16384, 128, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [view_227, view_228, permute_154, view_229, bmm_20], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:54
            extern_kernels.bmm(reinterpret_tensor(buf123, (12, 128, 64), (64, 768, 1), 0), view_230, out=buf126)
            del view_230
            # Topologically Sorted Source Nodes: [view_231, npu_dropout_backward_5], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:55
            buf128 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf126, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_29, 0.10000000000000009)
            assert_alignment(buf128, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf126
            del getitem_29
            buf130 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_151, sum_57, mul_152, sub_44, where_16], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf130, div_10, 1536, 128, stream=raw_stream0)
            del div_10
            buf132 = reinterpret_tensor(buf123, (12, 64, 128), (8192, 128, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_151, mul_152, sub_44, where_16, view_235, bmm_23], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:56
            extern_kernels.bmm(view_238, reinterpret_tensor(buf130, (12, 128, 128), (16384, 128, 1), 0), out=buf132)
            del view_238
            buf138 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [view_240, permute_159, permute_165, view_245, view_246, linear_19, permute_166, mm_36], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(reinterpret_tensor(buf132, (128, 768), (1, 128), 0), primals_57, out=buf138)
            del primals_57
            buf131 = reinterpret_tensor(buf99, (12, 128, 64), (8192, 64, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_151, mul_152, sub_44, where_16, view_235, bmm_22], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:58
            extern_kernels.bmm(reinterpret_tensor(buf130, (12, 128, 128), (16384, 128, 1), 0), view_236, out=buf131)
            del view_236
            buf141 = reinterpret_tensor(buf88, (128, 768), (768, 1), 0); del buf88  # reuse
            buf145 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_237, div_30, permute_170, clone_11, view_249, view_250, sum_60], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf131, buf141, buf145, 768, 128, stream=raw_stream0)
            buf142 = reinterpret_tensor(buf131, (128, 768), (768, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_18, permute_171, mm_38], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:59
            extern_kernels.mm(buf141, primals_55, out=buf142)
            del primals_55
            buf125 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf150 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf151 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_225, sum_56, view_244, add_81, view_248, add_82, view_252, add_83, mul_159, sum_63, sum_64], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_sum_view_14.run(buf120, buf134, buf138, buf142, mul_24, buf125, buf150, buf151, 768, 128, stream=raw_stream0)
            buf135 = reinterpret_tensor(buf133, (768, 128), (1, 768), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [permute_162], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf135, 98304, stream=raw_stream0)
            buf136 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_164], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:60
            extern_kernels.mm(buf135, view_69, out=buf136)
            del buf135
            buf139 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_240, permute_159, permute_165, view_245, view_246, permute_167, permute_169], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:61
            extern_kernels.mm(reinterpret_tensor(buf132, (768, 128), (128, 1), 0), view_69, out=buf139)
            buf140 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_240, permute_159, permute_165, view_245, view_246, sum_59], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf132, buf140, 768, 128, stream=raw_stream0)
            buf143 = reinterpret_tensor(buf141, (768, 128), (1, 768), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [permute_172], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf143, 98304, stream=raw_stream0)
            buf144 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_174], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:62
            extern_kernels.mm(buf143, view_69, out=buf144)
            del view_69
            buf146 = buf120; del buf120  # reuse
            buf149 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [view_244, add_81, view_248, add_82, view_252, add_83, mul_154, mul_155, sum_61, mul_156, sum_62, mul_157, sub_46, sub_47, mul_158], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_17.run(buf149, buf134, buf138, buf142, primals_53, mul_24, div_31, 128, 768, stream=raw_stream0)
            del div_31
            del mul_24
            del primals_53
            # Topologically Sorted Source Nodes: [npu_dropout_backward_6], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:63
            buf152 = torch.ops.npu.npu_dropout_backward.default(buf149, getitem_25, 0.10000000000000009)
            assert_alignment(buf152, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_25
            buf153 = reinterpret_tensor(buf114, (128, 3072), (3072, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [view_253, x_10, permute_175, mm_40], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:64
            extern_kernels.mm(reinterpret_tensor(buf152, (128, 768), (768, 1), 0), primals_51, out=buf153)
            del primals_51
            buf154 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_253, permute_176, permute_178], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:65
            extern_kernels.mm(reinterpret_tensor(buf152, (768, 128), (1, 768), 0), view_67, out=buf154)
            del view_67
            buf155 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_253, sum_65], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf152, buf155, 768, 128, stream=raw_stream0)
            buf156 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9, mul_161, add_84, mul_162, exp_18, mul_163, add_85, mul_164, add_86, mul_165, mul_166, mul_167, mul_168], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_16, buf156, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_8, x_9, mul_161, add_84, mul_162, exp_18, mul_163, add_85, mul_164, add_86, mul_165, mul_166, mul_167, mul_168, isnan_4], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:66
            buf157 = torch.ops.aten.isnan.default(buf156)
            assert_alignment(buf157, 16, 'torch.ops.aten.isnan.default')
            buf158 = reinterpret_tensor(buf153, (1, 128, 3072), (393216, 3072, 1), 0); del buf153  # reuse
            buf161 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_255, x_8, x_9, mul_161, add_84, mul_162, exp_18, add_86, mul_165, where_17, add_87, mul_169, view_256, sum_66], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf158, buf157, buf156, addmm_16, buf161, 3072, 128, stream=raw_stream0)
            del addmm_16
            del buf157
            buf159 = reinterpret_tensor(buf152, (128, 768), (768, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [full_3, view_255, x_8, x_9, mul_161, add_84, mul_162, exp_18, add_86, mul_165, where_17, add_87, mul_169, view_256, permute_179, mm_42], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:67
            extern_kernels.mm(reinterpret_tensor(buf158, (128, 3072), (3072, 1), 0), primals_49, out=buf159)
            del primals_49
            buf160 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_255, x_8, x_9, mul_161, add_84, mul_162, exp_18, add_86, mul_165, where_17, add_87, mul_169, view_256, permute_180, permute_182], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:68
            extern_kernels.mm(reinterpret_tensor(buf158, (3072, 128), (1, 3072), 0), view_65, out=buf160)
            del view_65
            buf164 = reinterpret_tensor(buf142, (1, 128, 768), (98304, 768, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [view_258, add_88, mul_171, mul_172, sum_67, mul_173, sum_68, mul_174, sub_49, sub_50, mul_175], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf149, buf159, primals_47, mul_18, div_32, buf164, 128, 768, stream=raw_stream0)
            del div_32
            del primals_47
            buf165 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf166 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_258, add_88, mul_176, sum_69, sum_70], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf149, buf159, mul_18, buf165, buf166, 768, 128, stream=raw_stream0)
            del mul_18
            buf167 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [view_259, context_8, permute_183, mm_44], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:69
            extern_kernels.mm(reinterpret_tensor(buf164, (128, 768), (768, 1), 0), primals_45, out=buf167)
            del primals_45
            buf168 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_259, permute_184, permute_186], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(reinterpret_tensor(buf164, (768, 128), (1, 768), 0), view_63, out=buf168)
            del view_63
            buf171 = reinterpret_tensor(buf149, (12, 128, 64), (8192, 64, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [view_261, view_262, permute_187, view_263, bmm_25], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:71
            extern_kernels.bmm(view_266, reinterpret_tensor(buf167, (12, 128, 64), (64, 768, 1), 0), out=buf171)
            del view_266
            buf177 = buf138; del buf138  # reuse
            buf181 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_268, permute_193, clone_12, view_275, view_276, sum_73], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf171, buf177, buf181, 768, 128, stream=raw_stream0)
            buf178 = reinterpret_tensor(buf171, (128, 768), (768, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [linear_14, permute_194, mm_46], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:72
            extern_kernels.mm(buf177, primals_43, out=buf178)
            del primals_43
            buf170 = reinterpret_tensor(buf130, (12, 128, 128), (16384, 128, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [view_261, view_262, permute_187, view_263, bmm_24], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:73
            extern_kernels.bmm(reinterpret_tensor(buf167, (12, 128, 64), (64, 768, 1), 0), view_264, out=buf170)
            del view_264
            # Topologically Sorted Source Nodes: [view_265, npu_dropout_backward_7], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:74
            buf172 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf170, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_21, 0.10000000000000009)
            assert_alignment(buf172, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf170
            del getitem_21
            buf174 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_177, sum_72, mul_178, sub_51, where_18], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf174, div_7, 1536, 128, stream=raw_stream0)
            del div_7
            buf176 = reinterpret_tensor(buf167, (12, 64, 128), (8192, 128, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_177, mul_178, sub_51, where_18, view_269, bmm_27], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:75
            extern_kernels.bmm(view_272, reinterpret_tensor(buf174, (12, 128, 128), (16384, 128, 1), 0), out=buf176)
            del view_272
            buf182 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [view_274, permute_192, permute_198, view_279, view_280, linear_13, permute_199, mm_48], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:76
            extern_kernels.mm(reinterpret_tensor(buf176, (128, 768), (1, 128), 0), primals_41, out=buf182)
            del primals_41
            buf175 = reinterpret_tensor(buf143, (12, 128, 64), (8192, 64, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_177, mul_178, sub_51, where_18, view_269, bmm_26], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:77
            extern_kernels.bmm(reinterpret_tensor(buf174, (12, 128, 128), (16384, 128, 1), 0), view_270, out=buf175)
            del view_270
            buf185 = reinterpret_tensor(buf132, (128, 768), (768, 1), 0); del buf132  # reuse
            buf189 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_271, div_33, permute_203, clone_13, view_283, view_284, sum_75], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf175, buf185, buf189, 768, 128, stream=raw_stream0)
            buf186 = reinterpret_tensor(buf175, (128, 768), (768, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_12, permute_204, mm_50], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:78
            extern_kernels.mm(buf185, primals_39, out=buf186)
            del primals_39
            buf169 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf194 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf195 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_259, sum_71, view_278, add_89, view_282, add_90, view_286, add_91, mul_185, sum_78, sum_79], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_sum_view_14.run(buf164, buf178, buf182, buf186, mul_16, buf169, buf194, buf195, 768, 128, stream=raw_stream0)
            buf179 = reinterpret_tensor(buf177, (768, 128), (1, 768), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [permute_195], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf179, 98304, stream=raw_stream0)
            buf180 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_197], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:79
            extern_kernels.mm(buf179, view_46, out=buf180)
            del buf179
            buf183 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_274, permute_192, permute_198, view_279, view_280, permute_200, permute_202], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:80
            extern_kernels.mm(reinterpret_tensor(buf176, (768, 128), (128, 1), 0), view_46, out=buf183)
            buf184 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_274, permute_192, permute_198, view_279, view_280, sum_74], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf176, buf184, 768, 128, stream=raw_stream0)
            buf187 = reinterpret_tensor(buf185, (768, 128), (1, 768), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [permute_205], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf187, 98304, stream=raw_stream0)
            buf188 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_207], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:81
            extern_kernels.mm(buf187, view_46, out=buf188)
            del view_46
            buf190 = buf164; del buf164  # reuse
            buf193 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [view_278, add_89, view_282, add_90, view_286, add_91, mul_180, mul_181, sum_76, mul_182, sum_77, mul_183, sub_53, sub_54, mul_184], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_17.run(buf193, buf178, buf182, buf186, primals_37, mul_16, div_34, 128, 768, stream=raw_stream0)
            del div_34
            del mul_16
            del primals_37
            # Topologically Sorted Source Nodes: [npu_dropout_backward_8], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:82
            buf196 = torch.ops.npu.npu_dropout_backward.default(buf193, getitem_17, 0.10000000000000009)
            assert_alignment(buf196, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_17
            buf197 = reinterpret_tensor(buf158, (128, 3072), (3072, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [view_287, x_6, permute_208, mm_52], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:83
            extern_kernels.mm(reinterpret_tensor(buf196, (128, 768), (768, 1), 0), primals_35, out=buf197)
            del primals_35
            buf198 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_287, permute_209, permute_211], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:84
            extern_kernels.mm(reinterpret_tensor(buf196, (768, 128), (1, 768), 0), view_44, out=buf198)
            del view_44
            buf199 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_287, sum_80], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf196, buf199, 768, 128, stream=raw_stream0)
            buf200 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, mul_187, add_92, mul_188, exp_19, mul_189, add_93, mul_190, add_94, mul_191, mul_192, mul_193, mul_194], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_10, buf200, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_4, x_5, mul_187, add_92, mul_188, exp_19, mul_189, add_93, mul_190, add_94, mul_191, mul_192, mul_193, mul_194, isnan_5], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:85
            buf201 = torch.ops.aten.isnan.default(buf200)
            assert_alignment(buf201, 16, 'torch.ops.aten.isnan.default')
            buf202 = reinterpret_tensor(buf197, (1, 128, 3072), (393216, 3072, 1), 0); del buf197  # reuse
            buf205 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_289, x_4, x_5, mul_187, add_92, mul_188, exp_19, add_94, mul_191, where_19, add_95, mul_195, view_290, sum_81], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf202, buf201, buf200, addmm_10, buf205, 3072, 128, stream=raw_stream0)
            del addmm_10
            del buf201
            buf203 = reinterpret_tensor(buf196, (128, 768), (768, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [full_3, view_289, x_4, x_5, mul_187, add_92, mul_188, exp_19, add_94, mul_191, where_19, add_95, mul_195, view_290, permute_212, mm_54], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(reinterpret_tensor(buf202, (128, 3072), (3072, 1), 0), primals_33, out=buf203)
            del primals_33
            buf204 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_289, x_4, x_5, mul_187, add_92, mul_188, exp_19, add_94, mul_191, where_19, add_95, mul_195, view_290, permute_213, permute_215], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:87
            extern_kernels.mm(reinterpret_tensor(buf202, (3072, 128), (1, 3072), 0), view_42, out=buf204)
            del view_42
            buf208 = reinterpret_tensor(buf186, (1, 128, 768), (98304, 768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [view_292, add_96, mul_197, mul_198, sum_82, mul_199, sum_83, mul_200, sub_56, sub_57, mul_201], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf193, buf203, primals_31, mul_10, div_35, buf208, 128, 768, stream=raw_stream0)
            del div_35
            del primals_31
            buf209 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf210 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_292, add_96, mul_202, sum_84, sum_85], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf193, buf203, mul_10, buf209, buf210, 768, 128, stream=raw_stream0)
            del mul_10
            buf211 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [view_293, context_5, permute_216, mm_56], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:88
            extern_kernels.mm(reinterpret_tensor(buf208, (128, 768), (768, 1), 0), primals_29, out=buf211)
            del primals_29
            buf212 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_293, permute_217, permute_219], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:89
            extern_kernels.mm(reinterpret_tensor(buf208, (768, 128), (1, 768), 0), view_40, out=buf212)
            del view_40
            buf215 = reinterpret_tensor(buf193, (12, 128, 64), (8192, 64, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [view_295, view_296, permute_220, view_297, bmm_29], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:90
            extern_kernels.bmm(view_300, reinterpret_tensor(buf211, (12, 128, 64), (64, 768, 1), 0), out=buf215)
            del view_300
            buf221 = buf182; del buf182  # reuse
            buf225 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_302, permute_226, clone_14, view_309, view_310, sum_88], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf215, buf221, buf225, 768, 128, stream=raw_stream0)
            buf222 = reinterpret_tensor(buf215, (128, 768), (768, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [linear_8, permute_227, mm_58], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:91
            extern_kernels.mm(buf221, primals_27, out=buf222)
            del primals_27
            buf214 = reinterpret_tensor(buf174, (12, 128, 128), (16384, 128, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [view_295, view_296, permute_220, view_297, bmm_28], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:92
            extern_kernels.bmm(reinterpret_tensor(buf211, (12, 128, 64), (64, 768, 1), 0), view_298, out=buf214)
            del view_298
            # Topologically Sorted Source Nodes: [view_299, npu_dropout_backward_9], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:93
            buf216 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf214, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_13, 0.10000000000000009)
            assert_alignment(buf216, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf214
            del getitem_13
            buf218 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_203, sum_87, mul_204, sub_58, where_20], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf218, div_4, 1536, 128, stream=raw_stream0)
            del div_4
            buf220 = reinterpret_tensor(buf211, (12, 64, 128), (8192, 128, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_203, mul_204, sub_58, where_20, view_303, bmm_31], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:94
            extern_kernels.bmm(view_306, reinterpret_tensor(buf218, (12, 128, 128), (16384, 128, 1), 0), out=buf220)
            del view_306
            buf226 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [view_308, permute_225, permute_231, view_313, view_314, linear_7, permute_232, mm_60], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:95
            extern_kernels.mm(reinterpret_tensor(buf220, (128, 768), (1, 128), 0), primals_25, out=buf226)
            del primals_25
            buf219 = reinterpret_tensor(buf187, (12, 128, 64), (8192, 64, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_203, mul_204, sub_58, where_20, view_303, bmm_30], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:96
            extern_kernels.bmm(reinterpret_tensor(buf218, (12, 128, 128), (16384, 128, 1), 0), view_304, out=buf219)
            del view_304
            buf229 = reinterpret_tensor(buf176, (128, 768), (768, 1), 0); del buf176  # reuse
            buf233 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_305, div_36, permute_236, clone_15, view_317, view_318, sum_90], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf219, buf229, buf233, 768, 128, stream=raw_stream0)
            buf230 = reinterpret_tensor(buf219, (128, 768), (768, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [linear_6, permute_237, mm_62], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:97
            extern_kernels.mm(buf229, primals_23, out=buf230)
            del primals_23
            buf213 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf238 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf239 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_293, sum_86, view_312, add_97, view_316, add_98, view_320, add_99, mul_211, sum_93, sum_94], Original ATen: [aten.view, aten.sum, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_sum_view_14.run(buf208, buf222, buf226, buf230, mul_8, buf213, buf238, buf239, 768, 128, stream=raw_stream0)
            buf223 = reinterpret_tensor(buf221, (768, 128), (1, 768), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [permute_228], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf223, 98304, stream=raw_stream0)
            buf224 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_230], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:98
            extern_kernels.mm(buf223, view_23, out=buf224)
            del buf223
            buf227 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_308, permute_225, permute_231, view_313, view_314, permute_233, permute_235], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:99
            extern_kernels.mm(reinterpret_tensor(buf220, (768, 128), (128, 1), 0), view_23, out=buf227)
            buf228 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_308, permute_225, permute_231, view_313, view_314, sum_89], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf220, buf228, 768, 128, stream=raw_stream0)
            buf231 = reinterpret_tensor(buf229, (768, 128), (1, 768), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [permute_238], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf231, 98304, stream=raw_stream0)
            buf232 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_240], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:100
            extern_kernels.mm(buf231, view_23, out=buf232)
            del view_23
            buf234 = buf208; del buf208  # reuse
            buf237 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [view_312, add_97, view_316, add_98, view_320, add_99, mul_206, mul_207, sum_91, mul_208, sum_92, mul_209, sub_60, sub_61, mul_210], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_17.run(buf237, buf222, buf226, buf230, primals_21, mul_8, div_37, 128, 768, stream=raw_stream0)
            del div_37
            del mul_8
            del primals_21
            # Topologically Sorted Source Nodes: [npu_dropout_backward_10], Original ATen: [aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:101
            buf240 = torch.ops.npu.npu_dropout_backward.default(buf237, getitem_9, 0.10000000000000009)
            assert_alignment(buf240, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_9
            buf241 = reinterpret_tensor(buf202, (128, 3072), (3072, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [view_321, x_2, permute_241, mm_64], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:102
            extern_kernels.mm(reinterpret_tensor(buf240, (128, 768), (768, 1), 0), primals_19, out=buf241)
            del primals_19
            buf242 = empty_strided_npu((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_321, permute_242, permute_244], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:103
            extern_kernels.mm(reinterpret_tensor(buf240, (768, 128), (1, 768), 0), view_21, out=buf242)
            del view_21
            buf243 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_321, sum_95], Original ATen: [aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sum_view_4.run(buf240, buf243, 768, 128, stream=raw_stream0)
            buf244 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, mul_213, add_100, mul_214, exp_20, mul_215, add_101, mul_216, add_102, mul_217, mul_218, mul_219, mul_220], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_view_7.run(addmm_4, buf244, 393216, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x, x_1, mul_213, add_100, mul_214, exp_20, mul_215, add_101, mul_216, add_102, mul_217, mul_218, mul_219, mul_220, isnan_6], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            # [Provenance debug handles] torch.ops.aten.isnan.default:104
            buf245 = torch.ops.aten.isnan.default(buf244)
            assert_alignment(buf245, 16, 'torch.ops.aten.isnan.default')
            buf246 = reinterpret_tensor(buf241, (1, 128, 3072), (393216, 3072, 1), 0); del buf241  # reuse
            buf249 = empty_strided_npu((1, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_323, x, x_1, mul_213, add_100, mul_214, exp_20, add_102, mul_217, where_21, add_103, mul_221, view_324, sum_96], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_gelu_backward_sum_view_8.run(buf246, buf245, buf244, addmm_4, buf249, 3072, 128, stream=raw_stream0)
            del addmm_4
            del buf244
            del buf245
            buf247 = reinterpret_tensor(buf240, (128, 768), (768, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [full_3, view_323, x, x_1, mul_213, add_100, mul_214, exp_20, add_102, mul_217, where_21, add_103, mul_221, view_324, permute_245, mm_66], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:105
            extern_kernels.mm(reinterpret_tensor(buf246, (128, 3072), (3072, 1), 0), primals_17, out=buf247)
            del primals_17
            buf248 = empty_strided_npu((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_3, view_323, x, x_1, mul_213, add_100, mul_214, exp_20, add_102, mul_217, where_21, add_103, mul_221, view_324, permute_246, permute_248], Original ATen: [aten.gelu_backward, aten.view, aten.gelu, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:106
            extern_kernels.mm(reinterpret_tensor(buf246, (3072, 128), (1, 3072), 0), view_19, out=buf248)
            del buf246
            del view_19
            buf252 = reinterpret_tensor(buf230, (1, 128, 768), (98304, 768, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [view_326, add_104, mul_223, mul_224, sum_97, mul_225, sum_98, mul_226, sub_63, sub_64, mul_227], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_9.run(buf237, buf247, primals_15, mul_2, div_38, buf252, 128, 768, stream=raw_stream0)
            del div_38
            del primals_15
            buf253 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf254 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_326, add_104, mul_228, sum_99, sum_100], Original ATen: [aten.view, aten.add, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_backward_view_10.run(buf237, buf247, mul_2, buf253, buf254, 768, 128, stream=raw_stream0)
            del mul_2
            buf255 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [view_327, context_2, permute_249, mm_68], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:107
            extern_kernels.mm(reinterpret_tensor(buf252, (128, 768), (768, 1), 0), primals_13, out=buf255)
            del primals_13
            buf256 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_327, permute_250, permute_252], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:108
            extern_kernels.mm(reinterpret_tensor(buf252, (768, 128), (1, 768), 0), view_17, out=buf256)
            del view_17
            buf259 = reinterpret_tensor(buf237, (12, 128, 64), (8192, 64, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [view_329, view_330, permute_253, view_331, bmm_33], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:109
            extern_kernels.bmm(view_334, reinterpret_tensor(buf255, (12, 128, 64), (64, 768, 1), 0), out=buf259)
            del view_334
            buf265 = buf226; del buf226  # reuse
            buf269 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_336, permute_259, clone_16, view_343, view_344, sum_103], Original ATen: [aten.matmul_backward, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_matmul_backward_sum_transpose_view_11.run(buf259, buf265, buf269, 768, 128, stream=raw_stream0)
            buf266 = reinterpret_tensor(buf259, (128, 768), (768, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [linear_2, permute_260, mm_70], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:110
            extern_kernels.mm(buf265, primals_11, out=buf266)
            del primals_11
            buf258 = reinterpret_tensor(buf218, (12, 128, 128), (16384, 128, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [view_329, view_330, permute_253, view_331, bmm_32], Original ATen: [aten.view, aten.transpose, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:111
            extern_kernels.bmm(reinterpret_tensor(buf255, (12, 128, 64), (64, 768, 1), 0), view_332, out=buf258)
            del view_332
            # Topologically Sorted Source Nodes: [view_333, npu_dropout_backward_11], Original ATen: [aten.matmul_backward, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:112
            buf260 = torch.ops.npu.npu_dropout_backward.default(reinterpret_tensor(buf258, (1, 12, 128, 128), (196608, 16384, 128, 1), 0), getitem_5, 0.10000000000000009)
            assert_alignment(buf260, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del buf258
            del getitem_5
            buf262 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_229, sum_102, mul_230, sub_65, where_22], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_eq_expand_masked_fill_nll_loss_backward_ones_view_12.run(buf262, div_1, 1536, 128, stream=raw_stream0)
            del div_1
            buf264 = reinterpret_tensor(buf255, (12, 64, 128), (8192, 128, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_229, mul_230, sub_65, where_22, view_337, bmm_35], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:113
            extern_kernels.bmm(view_340, reinterpret_tensor(buf262, (12, 128, 128), (16384, 128, 1), 0), out=buf264)
            del view_340
            buf270 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [view_342, permute_258, permute_264, view_347, view_348, linear_1, permute_265, mm_72], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:114
            extern_kernels.mm(reinterpret_tensor(buf264, (128, 768), (1, 128), 0), primals_9, out=buf270)
            del primals_9
            buf263 = reinterpret_tensor(buf231, (12, 128, 64), (8192, 64, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_3, attention_mask, eq, view_3, mask, mul_229, mul_230, sub_65, where_22, view_337, bmm_34], Original ATen: [aten.nll_loss_backward, aten.ones, aten.eq, aten.view, aten.expand, aten._softmax_backward_data, aten.masked_fill, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.bmm:115
            extern_kernels.bmm(reinterpret_tensor(buf262, (12, 128, 128), (16384, 128, 1), 0), view_338, out=buf263)
            del buf262
            del view_338
            buf273 = reinterpret_tensor(buf220, (128, 768), (768, 1), 0); del buf220  # reuse
            buf277 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_339, div_39, permute_269, clone_17, view_351, view_352, sum_105], Original ATen: [aten.matmul_backward, aten.div, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__unsafe_view_clone_div_matmul_backward_sum_transpose_view_13.run(buf263, buf273, buf277, 768, 128, stream=raw_stream0)
            buf274 = reinterpret_tensor(buf263, (128, 768), (768, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [linear, permute_270, mm_74], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:116
            extern_kernels.mm(buf273, primals_7, out=buf274)
            del primals_7
            buf257 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            buf278 = reinterpret_tensor(buf266, (1, 128, 768), (98304, 768, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [view_327, sum_101, view_346, add_105, view_350, add_106, view_354, add_107], Original ATen: [aten.view, aten.sum, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_sum_view_18.run(buf278, buf252, buf270, buf274, buf257, 768, 128, stream=raw_stream0)
            del buf252
            del buf270
            del buf274
            buf267 = reinterpret_tensor(buf265, (768, 128), (1, 768), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [permute_261], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf267, 98304, stream=raw_stream0)
            buf268 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_263], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:117
            extern_kernels.mm(buf267, view, out=buf268)
            del buf267
            buf271 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_342, permute_258, permute_264, view_347, view_348, permute_266, permute_268], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:118
            extern_kernels.mm(reinterpret_tensor(buf264, (768, 128), (128, 1), 0), view, out=buf271)
            buf272 = empty_strided_npu((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_342, permute_258, permute_264, view_347, view_348, sum_104], Original ATen: [aten.matmul_backward, aten.transpose, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_sum_transpose_view_16.run(buf264, buf272, 768, 128, stream=raw_stream0)
            del buf264
            buf275 = reinterpret_tensor(buf273, (768, 128), (1, 768), 0); del buf273  # reuse
            # Topologically Sorted Source Nodes: [permute_271], Original ATen: [aten.t]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_t_15.run(buf275, 98304, stream=raw_stream0)
            buf276 = empty_strided_npu((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_273], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:119
            extern_kernels.mm(buf275, view, out=buf276)
            del buf275
            del view
            # Topologically Sorted Source Nodes: [view_346, add_105, view_350, add_106, view_354, add_107, npu_dropout_backward_12], Original ATen: [aten.view, aten.add, aten.native_dropout_backward]
            # [Provenance debug handles] torch.ops.npu.npu_dropout_backward.default:120
            buf279 = torch.ops.npu.npu_dropout_backward.default(buf278, getitem_3, 0.10000000000000009)
            assert_alignment(buf279, 16, 'torch.ops.npu.npu_dropout_backward.default')
            del getitem_3
            buf282 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [mul_232, mul_233, sum_106, mul_234, sum_107, mul_235, sub_67, sub_68, mul_236], Original ATen: [aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_backward_view_5.run(buf279, primals_5, mul, div_40, buf282, 128, 768, stream=raw_stream0)
            del div_40
            del primals_5
            buf283 = empty_strided_npu((768, ), (1, ), torch.float32)
            buf284 = empty_strided_npu((768, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mul_237, sum_108, sum_109], Original ATen: [aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_backward_view_6.run(buf279, mul, buf283, buf284, 768, 128, stream=raw_stream0)
            del buf279
            del mul
            # Topologically Sorted Source Nodes: [position_ids, embedding_dense_backward], Original ATen: [aten.slice, aten.embedding_dense_backward]
            # [Provenance debug handles] torch.ops.aten.embedding_dense_backward.default:121
            buf285 = torch.ops.aten.embedding_dense_backward.default(buf282, reinterpret_tensor(primals_3, (1, 128), (128, 1), 0), 512, -1, False)
            assert_alignment(buf285, 16, 'torch.ops.aten.embedding_dense_backward.default')
            del primals_3
            # Topologically Sorted Source Nodes: [embedding_dense_backward_1], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] torch.ops.aten.embedding_dense_backward.default:122
            buf286 = torch.ops.aten.embedding_dense_backward.default(buf282, primals_1, 30522, 0, False)
            assert_alignment(buf286, 16, 'torch.ops.aten.embedding_dense_backward.default')
            del buf282
            del primals_1
            buf287 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [add_108], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_19.run(buf287, buf286, 23440896, stream=raw_stream0)
            del buf286
        return (None, buf287, None, buf285, buf283, buf284, buf276, reinterpret_tensor(buf277, (768, ), (1, ), 0), buf271, reinterpret_tensor(buf272, (768, ), (1, ), 0), buf268, reinterpret_tensor(buf269, (768, ), (1, ), 0), buf256, reinterpret_tensor(buf257, (768, ), (1, ), 0), buf253, buf254, buf248, reinterpret_tensor(buf249, (3072, ), (1, ), 0), buf242, reinterpret_tensor(buf243, (768, ), (1, ), 0), buf238, buf239, buf232, reinterpret_tensor(buf233, (768, ), (1, ), 0), buf227, reinterpret_tensor(buf228, (768, ), (1, ), 0), buf224, reinterpret_tensor(buf225, (768, ), (1, ), 0), buf212, reinterpret_tensor(buf213, (768, ), (1, ), 0), buf209, buf210, buf204, reinterpret_tensor(buf205, (3072, ), (1, ), 0), buf198, reinterpret_tensor(buf199, (768, ), (1, ), 0), buf194, buf195, buf188, reinterpret_tensor(buf189, (768, ), (1, ), 0), buf183, reinterpret_tensor(buf184, (768, ), (1, ), 0), buf180, reinterpret_tensor(buf181, (768, ), (1, ), 0), buf168, reinterpret_tensor(buf169, (768, ), (1, ), 0), buf165, buf166, buf160, reinterpret_tensor(buf161, (3072, ), (1, ), 0), buf154, reinterpret_tensor(buf155, (768, ), (1, ), 0), buf150, buf151, buf144, reinterpret_tensor(buf145, (768, ), (1, ), 0), buf139, reinterpret_tensor(buf140, (768, ), (1, ), 0), buf136, reinterpret_tensor(buf137, (768, ), (1, ), 0), buf124, reinterpret_tensor(buf125, (768, ), (1, ), 0), buf121, buf122, buf116, reinterpret_tensor(buf117, (3072, ), (1, ), 0), buf110, reinterpret_tensor(buf111, (768, ), (1, ), 0), buf106, buf107, buf100, reinterpret_tensor(buf101, (768, ), (1, ), 0), buf95, reinterpret_tensor(buf96, (768, ), (1, ), 0), buf92, reinterpret_tensor(buf93, (768, ), (1, ), 0), buf80, reinterpret_tensor(buf81, (768, ), (1, ), 0), buf77, buf78, buf72, reinterpret_tensor(buf73, (3072, ), (1, ), 0), buf66, reinterpret_tensor(buf67, (768, ), (1, ), 0), buf62, buf63, buf56, reinterpret_tensor(buf57, (768, ), (1, ), 0), buf51, reinterpret_tensor(buf52, (768, ), (1, ), 0), buf48, reinterpret_tensor(buf49, (768, ), (1, ), 0), buf36, reinterpret_tensor(buf37, (768, ), (1, ), 0), buf33, buf34, buf28, reinterpret_tensor(buf29, (3072, ), (1, ), 0), buf22, reinterpret_tensor(buf23, (768, ), (1, ), 0), buf18, buf19, buf13, reinterpret_tensor(buf14, (768, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf4, (30522, ), (1, ), 0), None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    primals_2 = rand_strided((30522, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 512), (512, 1), device='npu:0', dtype=torch.int64)
    primals_5 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_17 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_33 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_35 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_43 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_83 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_97 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_108 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    mul = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    getitem_3 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    view = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_1 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_5 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_17 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_19 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_4 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_21 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_9 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_8 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_23 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_4 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_13 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_40 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_10 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_42 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_10 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_44 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_17 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_16 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_46 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_7 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_21 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_63 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_18 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_65 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_16 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_67 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_25 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_24 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_69 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_10 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_29 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_86 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_26 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_88 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_22 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_90 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_33 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_32 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_92 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_37 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_109 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_34 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_111 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_28 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_113 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_41 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_40 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_115 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    div_16 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='npu:0', dtype=torch.float32)
    getitem_45 = rand_strided((24576, ), (1, ), device='npu:0', dtype=torch.uint8)
    view_132 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    mul_42 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_134 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_34 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    view_136 = rand_strided((128, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    getitem_49 = rand_strided((12288, ), (1, ), device='npu:0', dtype=torch.uint8)
    mul_48 = rand_strided((1, 128, 768), (98304, 768, 1), device='npu:0', dtype=torch.float32)
    view_138 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    addmm_36 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_140 = rand_strided((128, 768), (768, 1), device='npu:0', dtype=torch.float32)
    _log_softmax = rand_strided((128, 30522), (30522, 1), device='npu:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='npu:0', dtype=torch.float32)
    div_22 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_23 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_162 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_164 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_168 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_170 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_26 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_196 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_198 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_202 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_204 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_29 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_230 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_232 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_236 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_238 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_32 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_264 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_266 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_270 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_272 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_34 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_35 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_298 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_300 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_304 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_306 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_37 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    div_38 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    view_332 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    view_334 = rand_strided((12, 128, 128), (16384, 1, 128), device='npu:0', dtype=torch.float32)
    view_338 = rand_strided((12, 128, 64), (64, 768, 1), device='npu:0', dtype=torch.float32)
    view_340 = rand_strided((12, 64, 128), (64, 1, 768), device='npu:0', dtype=torch.float32)
    div_40 = rand_strided((1, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 30522), (3906816, 30522, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, mul, getitem_3, view, div_1, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_8, view_23, div_4, getitem_13, view_40, mul_10, view_42, addmm_10, view_44, getitem_17, mul_16, view_46, div_7, getitem_21, view_63, mul_18, view_65, addmm_16, view_67, getitem_25, mul_24, view_69, div_10, getitem_29, view_86, mul_26, view_88, addmm_22, view_90, getitem_33, mul_32, view_92, div_13, getitem_37, view_109, mul_34, view_111, addmm_28, view_113, getitem_41, mul_40, view_115, div_16, getitem_45, view_132, mul_42, view_134, addmm_34, view_136, getitem_49, mul_48, view_138, addmm_36, getitem_53, rsqrt_13, view_140, _log_softmax, convert_element_type, div_22, div_23, view_162, view_164, view_168, view_170, div_25, div_26, view_196, view_198, view_202, view_204, div_28, div_29, view_230, view_232, view_236, view_238, div_31, div_32, view_264, view_266, view_270, view_272, div_34, div_35, view_298, view_300, view_304, view_306, div_37, div_38, view_332, view_334, view_338, view_340, div_40, tangents_1, tangents_2]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('DistilBertForMaskedLM', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
