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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/j5/cj5bwopwca2thojfwgxurrrgqq4mjllwmul56tou7bgzfybs7apv.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => full_default_6, ne, where_6
#   view_31 => view_143
# Graph fragment:
#   %primals_108 : Tensor "i64[1, 128][128, 1]npu:0" = PlaceHolder[target=primals_108]
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %ne : Tensor "b8[128][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %full_default_6 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_6 : Tensor "i64[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_143, %full_default_6), kwargs = {})
#   return %where_6
triton_unk_fused_nll_loss_forward_view_0 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_0', '''
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
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 128
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        tmp1 = tl.full([1], -100, tl.int32)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tl.store(out_ptr0 + (x0), tmp4, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/v4/cv42qcasvawxoqyq2zzelbm5gnppfswcxi2pelbfdcmt4bdqeteq.py
# Topologically Sorted Source Nodes: [embeddings, embeddings_1, div_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_40 => div_40
#   embeddings => add
#   embeddings_1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf2 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf2]
#   %getitem_1 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_1]
#   %buf4 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf4]
#   %mul : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul]
#   %primals_5 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_5]
#   %primals_6 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_6]
#   %add : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_5), kwargs = {})
#   %add_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   %div_40 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 768), kwargs = {})
#   return %buf2,%getitem_1,%buf4,%mul,%add_2,%div_40
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
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
            tmp8 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10 - tmp7
            tmp12 = tmp11 * tmp11
            tmp13 = tl.broadcast_to(tmp12, [real_block_x0, R0_BLOCK])
            tmp15 = _tmp14 + tmp13
            _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp14 = tl.sum(_tmp14, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp16 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp26 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp7
            tmp20 = tl.full([1, 1], 768.0, tl.float32)
            tmp21 = (tmp14 / tmp20)
            tmp22 = tl.full([1, 1], 1e-12, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp25, r0_mask & x0mask)
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp29, r0_mask & x0mask)
        tmp30 = tl.full([1, 1], 768.0, tl.float32)
        tmp31 = (tmp14 / tmp30)
        tmp32 = tl.full([1, 1], 1e-12, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tl.rsqrt(tmp33)
        tmp35 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp36 = tmp34 * tmp35
        tl.store(out_ptr2 + (x0), tmp36, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/qs/cqssuwa5f3bnyckgeezkuhhqycuoiohuvkxaopusteulubou66zu.py
# Topologically Sorted Source Nodes: [, linear, view, q, q_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
# Source node to ATen node mapping:
#    => add_tensor_11
#   linear => view_1
#   q => permute_1
#   q_1 => div
#   view => view_2
# Graph fragment:
#   %primals_8 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_8]
#   %mm_default_11 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_11]
#   %add_tensor_11 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_8, %mm_default_11), kwargs = {})
#   %view_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_11, [1, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %div : Tensor "f32[1, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_1, 8.0), kwargs = {})
#   return %expand
triton_unk_fused_addmm_div_transpose_view_2 = async_compile.triton('triton_unk_fused_addmm_div_transpose_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 128, 12, {'divisors': [1, 64, 8192]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 8192, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_div_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_div_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 98304
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (128) if (128) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 128
    x2numel : tl.constexpr = 12
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks
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
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x0mask & x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2), x0mask & x2mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + 64*x2 + 768*x1), x0mask & x1mask & x2mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1, 1], 0.125, tl.float32)
        tmp4 = tmp2 * tmp3
        tl.store(out_ptr0 + (x0 + 64*x1 + 8192*x2), tmp4, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/5g/c5gea2kh4qnnyr36uj4cy6mjxd5fal5mj56dvdcumpl5gibxw4hp.py
# Topologically Sorted Source Nodes: [permute_257], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   permute_257 => permute_257
# Graph fragment:
#   %expand : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0" = PlaceHolder[target=expand]
#   %permute_257 : Tensor "f32[1, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 1, 3, 2]), kwargs = {})
#   return %expand_76
triton_unk_fused_matmul_backward_3 = async_compile.triton('triton_unk_fused_matmul_backward_3', '''
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
    size_hints={'y': 768, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, {'divisors': [1, 64]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 768
    xnumel = 128
    y_g_tile0 : tl.constexpr = (64) if (64) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 64
    y1numel : tl.constexpr = 12
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 8192*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 128*y0 + 8192*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/v2/cv2pk7rvhzph5h2q3cmlhygmjn7xcklombmlnmrkqpzayokng5ij.py
# Topologically Sorted Source Nodes: [view_340], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   view_340 => view_340
# Graph fragment:
#   %expand_76 : Tensor "f32[1, 12, 64, 128][98304, 8192, 128, 1]npu:0" = PlaceHolder[target=expand_76]
#   %view_340 : Tensor "f32[12, 64, 128][64, 1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_76, [12, 64, 128]), kwargs = {})
#   return %view_340
triton_unk_fused_matmul_backward_4 = async_compile.triton('triton_unk_fused_matmul_backward_4', '''
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
    size_hints={'y': 768, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, {'divisors': [1]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 768
    xnumel = 128
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y0numel : tl.constexpr = 768
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x1numel : tl.constexpr = 128
    real_block_x1 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_0 = y0_blocks
    total_blocks = y0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x1offset = (group_base + i) // x_cumblk_0 % x1_blocks * real_block_x1
        xoffset = x1offset
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x1mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        yoffset = y0offset
        y0index = y0offset + tl.arange(0, real_block_y0)[:, None]
        y0 = y0index
        y0mask = y0index < y0numel
        ymask = y0mask
        tmp0 = tl.load(in_ptr0 + (x1 + 128*y0), x1mask & y0mask)
        tl.store(out_ptr0 + (y0 + 768*x1), tmp0, x1mask & y0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/yf/cyftmvsbse7ad5nfjgybes5as7ri4seet6qqboylukilw2yk2u2d.py
# Topologically Sorted Source Nodes: [attention_mask, scores, eq, view_3, mask, tensor, scores_1, weights], Original ATen: [aten.ones, aten.matmul, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   attention_mask => full
#   eq => eq
#   mask => expand_2
#   scores => view_11
#   scores_1 => where
#   tensor => full_default
#   view_3 => view_12
#   weights => amax, div_1, exp, sub_1, sum_1
# Graph fragment:
#   %bmm : Tensor "f32[12, 128, 128][16384, 128, 1]npu:0" = PlaceHolder[target=bmm]
#   %amax : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1536]npu:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1536]npu:0" = PlaceHolder[target=sum_1]
#   %full : Tensor "f32[1, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_11 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 12, 128, 128]), kwargs = {})
#   %eq : Tensor "b8[1, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%full, 0), kwargs = {})
#   %view_12 : Tensor "b8[1, 1, 1, 128][128, 128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [1, 1, 1, 128]), kwargs = {})
#   %expand_2 : Tensor "b8[1, 12, 128, 128][128, 0, 0, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view_12, [1, 12, 128, 128]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %view_11), kwargs = {})
#   %amax : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax,%sum_1,%div_1
triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5 = async_compile.triton('triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1536, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 1536, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp7 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp3 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp0 = tl.full([1, 1], 1.0, tl.float32)
            tmp1 = tl.full([1, 1], 0.0, tl.float32)
            tmp2 = tmp0 == tmp1
            tmp4 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp5 = tl.where(tmp2, tmp4, tmp3)
            tmp6 = tl.broadcast_to(tmp5, [real_block_x0, R0_BLOCK])
            tmp8 = npu_triton_helpers.maximum(_tmp7, tmp6)
            _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
        _tmp18 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.full([1, 1], 1.0, tl.float32)
            tmp10 = tl.full([1, 1], 0.0, tl.float32)
            tmp11 = tmp9 == tmp10
            tmp13 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp14 = tl.where(tmp11, tmp13, tmp12)
            tmp15 = tmp14 - tmp7
            tmp16 = libdevice.exp(tmp15)
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp23 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp20 = tl.full([1, 1], 1.0, tl.float32)
            tmp21 = tl.full([1, 1], 0.0, tl.float32)
            tmp22 = tmp20 == tmp21
            tmp24 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp25 = tl.where(tmp22, tmp24, tmp23)
            tmp26 = tmp25 - tmp7
            tmp27 = libdevice.exp(tmp26)
            tmp28 = (tmp27 / tmp18)
            tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp28, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/it/citrdetobv6t7p63toeemztijrv6r2a2gatn6tbnxn4zdsq7n3uk.py
# Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   context => view_15
#   contiguous => clone
#   transpose_4 => permute_7
# Graph fragment:
#   %bmm_1 : Tensor "f32[12, 128, 64][8192, 64, 1]npu:0" = PlaceHolder[target=bmm_1]
#   %view_15 : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 12, 128, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[1, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_unk_fused_clone_matmul_transpose_6 = async_compile.triton('triton_unk_fused_clone_matmul_transpose_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, 128, {'divisors': [1, 64, 768]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 768, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_transpose_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_transpose_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 98304
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (12) if (12) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (128) if (128) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 12
    x2numel : tl.constexpr = 128
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks
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
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x0mask & x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 8192*x1), x0mask & x1mask & x2mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 768*x2), tmp0, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/i2/ci2mpbhcynmbghtm2f5rho7trsuqczjn5sdmngbodmhg626qa74q.py
# Topologically Sorted Source Nodes: [permute_255], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   permute_255 => permute_255
# Graph fragment:
#   %expand_3 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=expand_3]
#   %permute_255 : Tensor "f32[1, 12, 128, 128][196608, 16384, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_4, [0, 1, 3, 2]), kwargs = {})
#   return %expand_72
triton_unk_fused_matmul_backward_7 = async_compile.triton('triton_unk_fused_matmul_backward_7', '''
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
    size_hints={'y': 1536, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [128, 12, {'divisors': [1, 128]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 1536
    xnumel = 128
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 12
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 16384*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 128*y0 + 16384*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/uw/cuwealap5lojhdrlpn7w7d754uikbh7hdviuoswhxscoqx7nalhb.py
# Topologically Sorted Source Nodes: [view_334], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   view_334 => view_334
# Graph fragment:
#   %expand_72 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=expand_72]
#   %view_334 : Tensor "f32[12, 128, 128][16384, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_72, [12, 128, 128]), kwargs = {})
#   return %view_334
triton_unk_fused_matmul_backward_8 = async_compile.triton('triton_unk_fused_matmul_backward_8', '''
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
    size_hints={'y': 1536, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [128, 12, {'divisors': [1, 128]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 1536
    xnumel = 128
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 12
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x2 + 128*y0 + 16384*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (y0 + 128*x2 + 16384*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/cq/ccq7qgyd26kssqmawsccc6mwemilxskgcxgdj2grbbo2iiph3u3v.py
# Topologically Sorted Source Nodes: [, context_2, add_1, sa_output, div_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_10
#   add_1 => add_3
#   context_2 => view_18
#   div_38 => div_38
#   sa_output => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %primals_14 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_14]
#   %mm_default_10 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_10]
#   %getitem_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=getitem_2]
#   %buf24 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf24]
#   %getitem_7 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_7]
#   %buf26 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf26]
#   %mul_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_2]
#   %primals_15 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_15]
#   %primals_16 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_16]
#   %add_tensor_10 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_14, %mm_default_10), kwargs = {})
#   %view_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [1, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %getitem_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_7), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_15), kwargs = {})
#   %add_5 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_16), kwargs = {})
#   %div_38 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 768), kwargs = {})
#   return %buf24,%getitem_7,%buf26,%mul_2,%add_5,%div_38
triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
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
            tmp10 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp11 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp13 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp12 = tmp10 + tmp11
            tmp14 = tmp12 + tmp13
            tmp15 = tmp14 - tmp9
            tmp16 = tmp15 * tmp15
            tmp17 = tl.broadcast_to(tmp16, [real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp21 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp23 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp32 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp34 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp22 = tmp20 + tmp21
            tmp24 = tmp22 + tmp23
            tmp25 = tmp24 - tmp9
            tmp26 = tl.full([1, 1], 768.0, tl.float32)
            tmp27 = (tmp18 / tmp26)
            tmp28 = tl.full([1, 1], 1e-12, tl.float32)
            tmp29 = tmp27 + tmp28
            tmp30 = tl.rsqrt(tmp29)
            tmp31 = tmp25 * tmp30
            tmp33 = tmp31 * tmp32
            tmp35 = tmp33 + tmp34
            tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask & x0mask)
        tmp36 = tl.full([1, 1], 768.0, tl.float32)
        tmp37 = (tmp18 / tmp36)
        tmp38 = tl.full([1, 1], 1e-12, tl.float32)
        tmp39 = tmp37 + tmp38
        tmp40 = tl.rsqrt(tmp39)
        tmp41 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr2 + (x0), tmp42, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/iy/ciy4noutwugqowdszad2ffnvlxubqk3h65wy4jousj5lgprjtggx.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x => view_20
#   x_1 => add_6, add_7, div_2, exp_1, mul_4, mul_5, mul_6, mul_7
# Graph fragment:
#   %addmm_4 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_4]
#   %view_20 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %view_20), kwargs = {})
#   %mul_5 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %view_20), kwargs = {})
#   %mul_6 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, -1.5957691216057308), kwargs = {})
#   %exp_1 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1.0), kwargs = {})
#   %div_2 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, %add_7), kwargs = {})
#   return %div_2
triton_unk_fused_gelu_view_10 = async_compile.triton('triton_unk_fused_gelu_view_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_view_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_gelu_view_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp2 = tmp1 * tmp0
        tmp3 = tl.full([1], 0.044715, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 + tmp4
        tmp6 = tl.full([1], -1.5957691216057308, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = libdevice.exp(tmp7)
        tmp9 = tl.full([1], 1.0, tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = (tmp0 / tmp10)
        tl.store(out_ptr0 + (x0), tmp11, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/o4/co4vwsv54ms6huyzpq4qfab47so3hyplbkmapvadw4di372gotys.py
# Topologically Sorted Source Nodes: [add_2, ffn_output, div_37], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_2 => add_8
#   div_37 => div_37
#   ffn_output => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %getitem_8 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=getitem_8]
#   %add_5 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=add_5]
#   %buf35 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf35]
#   %getitem_11 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_11]
#   %buf37 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf37]
#   %mul_8 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_8]
#   %primals_21 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_21]
#   %primals_22 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_22]
#   %add_8 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, %add_5), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-12), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_11), kwargs = {})
#   %mul_8 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_21), kwargs = {})
#   %add_10 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_22), kwargs = {})
#   %div_37 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
#   return %buf35,%getitem_11,%buf37,%mul_8,%add_10,%div_37
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
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
            tmp22 = tl.full([1, 1], 1e-12, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp25, r0_mask & x0mask)
            tl.store(out_ptr2 + (r0_1 + 768*x0), tmp29, r0_mask & x0mask)
        tmp30 = tl.full([1, 1], 768.0, tl.float32)
        tmp31 = (tmp14 / tmp30)
        tmp32 = tl.full([1, 1], 1e-12, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tl.rsqrt(tmp33)
        tmp35 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp36 = tmp34 * tmp35
        tl.store(out_ptr3 + (x0), tmp36, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/gr/cgrov7gfwhh5tmtpmq37yk3dvfjfooemrdiqo23cy4qyfoholgzu.py
# Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => add_53, add_54, getitem_53, mul_54, mul_55, rsqrt_13, sub_19, var_mean_13
# Graph fragment:
#   %addmm_36 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=addmm_36]
#   %buf191 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf191]
#   %buf192 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf192]
#   %buf193 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf193]
#   %rsqrt_13 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=rsqrt_13]
#   %primals_105 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_105]
#   %primals_106 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_106]
#   %view_139 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_36, [1, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%div_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %getitem_53 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=operator.getitem](args = (%var_mean_13, 1), kwargs = {})
#   %add_53 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-12), kwargs = {})
#   %rsqrt_13 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %sub_19 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_53), kwargs = {})
#   %mul_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_55 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %primals_105), kwargs = {})
#   %add_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %primals_106), kwargs = {})
#   return %buf191,%buf192,%buf193,%rsqrt_13,%add_54
triton_unk_fused_gelu_native_layer_norm_view_12 = async_compile.triton('triton_unk_fused_gelu_native_layer_norm_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_native_layer_norm_view_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_gelu_native_layer_norm_view_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp13 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tmp0 * tmp0
            tmp2 = tmp1 * tmp0
            tmp3 = tl.full([1, 1], 0.044715, tl.float32)
            tmp4 = tmp2 * tmp3
            tmp5 = tmp0 + tmp4
            tmp6 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp7 = tmp5 * tmp6
            tmp8 = libdevice.exp(tmp7)
            tmp9 = tl.full([1, 1], 1.0, tl.float32)
            tmp10 = tmp8 + tmp9
            tmp11 = (tmp0 / tmp10)
            tmp12 = tl.broadcast_to(tmp11, [real_block_x0, R0_BLOCK])
            tmp14 = _tmp13 + tmp12
            _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp13 = tl.sum(_tmp13, 1)[:, None]
        tmp15 = tl.full([1, 1], 768.0, tl.float32)
        tmp16 = (tmp13 / tmp15)
        tl.store(in_out_ptr0 + (x0), tmp16, x0mask)
        _tmp32 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp17 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tmp17 * tmp17
            tmp19 = tmp18 * tmp17
            tmp20 = tl.full([1, 1], 0.044715, tl.float32)
            tmp21 = tmp19 * tmp20
            tmp22 = tmp17 + tmp21
            tmp23 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp24 = tmp22 * tmp23
            tmp25 = libdevice.exp(tmp24)
            tmp26 = tl.full([1, 1], 1.0, tl.float32)
            tmp27 = tmp25 + tmp26
            tmp28 = (tmp17 / tmp27)
            tmp29 = tmp28 - tmp16
            tmp30 = tmp29 * tmp29
            tmp31 = tl.broadcast_to(tmp30, [real_block_x0, R0_BLOCK])
            tmp33 = _tmp32 + tmp31
            _tmp32 = tl.where(r0_mask & xmask, tmp33, _tmp32)
        tmp32 = tl.sum(_tmp32, 1)[:, None]
        tmp34 = tl.full([1, 1], 768.0, tl.float32)
        tmp35 = (tmp32 / tmp34)
        tmp36 = tl.full([1, 1], 1e-12, tl.float32)
        tmp37 = tmp35 + tmp36
        tmp38 = tl.rsqrt(tmp37)
        tl.store(in_out_ptr1 + (x0), tmp38, x0mask)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp39 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp53 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp55 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp40 = tmp39 * tmp39
            tmp41 = tmp40 * tmp39
            tmp42 = tl.full([1, 1], 0.044715, tl.float32)
            tmp43 = tmp41 * tmp42
            tmp44 = tmp39 + tmp43
            tmp45 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp46 = tmp44 * tmp45
            tmp47 = libdevice.exp(tmp46)
            tmp48 = tl.full([1, 1], 1.0, tl.float32)
            tmp49 = tmp47 + tmp48
            tmp50 = (tmp39 / tmp49)
            tmp51 = tmp50 - tmp16
            tmp52 = tmp51 * tmp38
            tmp54 = tmp52 * tmp53
            tmp56 = tmp54 + tmp55
            tl.store(out_ptr0 + (r0_1 + 768*x0), tmp56, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.14.0/effect_train/cache/he/ched4wbtntzqmbjx57gzqygvrogb3vqdzf6kutmeofga4ya7d2cd.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => convert_element_type, div_19, full_default_7, ne, ne_2, neg, squeeze, sum_7, sum_8, where_7
#   view_31 => view_143
# Graph fragment:
#   %primals_108 : Tensor "i64[1, 128][128, 1]npu:0" = PlaceHolder[target=primals_108]
#   %gather : Tensor "f32[128, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %sum_7 : Tensor "i64[][]npu:0" = PlaceHolder[target=sum_7]
#   %sum_8 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_8]
#   %convert_element_type : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type]
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %ne : Tensor "b8[128][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %squeeze : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_7 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_7), kwargs = {})
#   %ne_2 : Tensor "b8[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %sum_7 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_7, torch.float32), kwargs = {})
#   %sum_8 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %div_19 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_8, %convert_element_type), kwargs = {})
#   return %sum_7,%sum_8,%convert_element_type,%div_19
triton_unk_fused_nll_loss_forward_view_13 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_13', '''
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
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'r0_0', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '67363117C7719F58FCE331F21785FA9D1E1EBF1DB623A8FDC8E5DE3A7EA90E6E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_13(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1
    r0_numel = 128
    group_size = 1 if group_id < 1 else 0
    group_base = group_id
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp5 = tl.full([R0_BLOCK, 1], 0, tl.int32)
        _tmp12 = tl.full([R0_BLOCK, 1], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_0 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
            tmp7 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.full([], -100, tl.int32)
            tmp2 = tmp0 != tmp1
            tmp3 = tmp2.to(tl.int32)
            tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK, 1])
            tmp6 = _tmp5 + tmp4
            _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
            tmp8 = -tmp7
            tmp9 = tl.full([], 0.0, tl.float32)
            tmp10 = tl.where(tmp2, tmp8, tmp9)
            tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK, 1])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tmp13
        tmp5 = tl.sum(_tmp5, 0)[None, :]
        tmp12 = tl.sum(_tmp12, 0)[None, :]
        tmp14 = tmp5.to(tl.float32)
        tmp15 = (tmp12 / tmp14)
        tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp14, None)
        tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp15, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_108 = copy_if_misaligned(primals_108)
            buf198 = empty_strided_npu((128, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_0.run(primals_108, buf198, 128, stream=raw_stream0)
            primals_1 = copy_if_misaligned(primals_1)
            # Topologically Sorted Source Nodes: [input_embeds], Original ATen: [aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:1
            buf0 = torch.ops.aten.embedding.default(primals_2, primals_1, 0)
            assert_alignment(buf0, 16, 'torch.ops.aten.embedding.default')
            # Topologically Sorted Source Nodes: [position_ids, position_embeddings], Original ATen: [aten.slice, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf1 = torch.ops.aten.embedding.default(primals_4, reinterpret_tensor(primals_3, (1, 128), (512, 1), 0))
            assert_alignment(buf1, 16, 'torch.ops.aten.embedding.default')
            del primals_4
            buf5 = buf0; del buf0  # reuse
            buf6 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf239 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embeddings, embeddings_1, div_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1.run(buf5, buf1, primals_5, primals_6, buf6, buf239, 128, 768, stream=raw_stream0)
            del primals_6
            # Topologically Sorted Source Nodes: [embeddings_1, embeddings_2], Original ATen: [aten.native_layer_norm, aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:3
            buf7 = torch.ops.npu._npu_dropout.default(buf6, 0.1)
            buf8 = buf7[0]
            assert_alignment(buf8, 16, 'torch.ops.npu._npu_dropout.default')
            buf9 = buf7[1]
            assert_alignment(buf9, 16, 'torch.ops.npu._npu_dropout.default')
            del buf7
            buf10 = reinterpret_tensor(buf6, (128, 768), (768, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf8, (128, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), out=buf10)
            buf13 = reinterpret_tensor(buf1, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [, linear, view, q, q_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_8, buf10, buf13, 98304, stream=raw_stream0)
            del primals_8
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear, linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(primals_10, reinterpret_tensor(buf8, (128, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf11)
            del primals_10
            buf12 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:6
            extern_kernels.addmm(primals_12, reinterpret_tensor(buf8, (128, 768), (768, 1), 0), reinterpret_tensor(primals_11, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf12)
            del primals_12
            buf237 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_257], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf13, buf237, 768, 128, stream=raw_stream0)
            buf238 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_340], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf237, buf238, 768, 128, stream=raw_stream0)
            buf14 = empty_strided_npu((12, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, linear_1, view_1, k, q_1, transpose_3, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:7
            extern_kernels.bmm(reinterpret_tensor(buf13, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf11, (12, 64, 128), (64, 1, 768), 0), out=buf14)
            buf17 = reinterpret_tensor(buf14, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, scores, eq, view_3, mask, tensor, scores_1, weights], Original ATen: [aten.ones, aten.matmul, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf17, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_1], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:8
            buf18 = torch.ops.npu._npu_dropout.default(buf17, 0.1)
            buf19 = buf18[0]
            assert_alignment(buf19, 16, 'torch.ops.npu._npu_dropout.default')
            buf20 = buf18[1]
            assert_alignment(buf20, 16, 'torch.ops.npu._npu_dropout.default')
            del buf18
            buf21 = reinterpret_tensor(buf13, (12, 128, 64), (8192, 64, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [linear_2, view_2, v, context], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:9
            extern_kernels.bmm(reinterpret_tensor(buf19, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf12, (12, 128, 64), (64, 768, 1), 0), out=buf21)
            buf22 = reinterpret_tensor(buf237, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf21, buf22, 98304, stream=raw_stream0)
            buf235 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_255], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf19, buf235, 1536, 128, stream=raw_stream0)
            buf236 = reinterpret_tensor(buf19, (12, 128, 128), (16384, 1, 128), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [view_334], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf235, buf236, 1536, 128, stream=raw_stream0)
            buf23 = reinterpret_tensor(buf21, (128, 768), (768, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous, context_1, context_2, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf22, (128, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 768), (1, 768), 0), out=buf23)
            buf27 = reinterpret_tensor(buf23, (1, 128, 768), (98304, 768, 1), 0); del buf23  # reuse
            buf28 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf234 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_2, add_1, sa_output, div_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf27, primals_14, buf8, primals_15, primals_16, buf28, buf234, 128, 768, stream=raw_stream0)
            del primals_14
            del primals_16
            buf29 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output, x], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:11
            extern_kernels.addmm(primals_18, reinterpret_tensor(buf28, (128, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf29)
            del primals_18
            buf30 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf29, buf30, 393216, stream=raw_stream0)
            buf31 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(primals_20, reinterpret_tensor(buf30, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf31)
            del primals_20
            # Unsorted Source Nodes: [x_3, x_2], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:13
            buf32 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf31, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf33 = buf32[0]
            assert_alignment(buf33, 16, 'torch.ops.npu._npu_dropout.default')
            buf34 = buf32[1]
            assert_alignment(buf34, 16, 'torch.ops.npu._npu_dropout.default')
            del buf32
            buf38 = reinterpret_tensor(buf31, (1, 128, 768), (98304, 768, 1), 0); del buf31  # reuse
            buf39 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf233 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_2, ffn_output, div_37], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf33, buf28, primals_21, primals_22, buf38, buf39, buf233, 128, 768, stream=raw_stream0)
            del primals_22
            buf40 = reinterpret_tensor(buf33, (128, 768), (768, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf39, (128, 768), (768, 1), 0), reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), out=buf40)
            buf43 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, q_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_24, buf40, buf43, 98304, stream=raw_stream0)
            del primals_24
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, linear_7], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:15
            extern_kernels.addmm(primals_26, reinterpret_tensor(buf39, (128, 768), (768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
            del primals_26
            buf42 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, linear_8], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:16
            extern_kernels.addmm(primals_28, reinterpret_tensor(buf39, (128, 768), (768, 1), 0), reinterpret_tensor(primals_27, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
            del primals_28
            buf231 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_224], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf43, buf231, 768, 128, stream=raw_stream0)
            buf232 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_306], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf231, buf232, 768, 128, stream=raw_stream0)
            buf44 = reinterpret_tensor(buf235, (12, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, linear_7, view_6, k_1, q_3, transpose_8, scores_2], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:17
            extern_kernels.bmm(reinterpret_tensor(buf43, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf41, (12, 64, 128), (64, 1, 768), 0), out=buf44)
            buf47 = reinterpret_tensor(buf44, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_2, scores_3, weights_2], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf47, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_3], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:18
            buf48 = torch.ops.npu._npu_dropout.default(buf47, 0.1)
            buf49 = buf48[0]
            assert_alignment(buf49, 16, 'torch.ops.npu._npu_dropout.default')
            buf50 = buf48[1]
            assert_alignment(buf50, 16, 'torch.ops.npu._npu_dropout.default')
            del buf48
            buf51 = reinterpret_tensor(buf43, (12, 128, 64), (8192, 64, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_8, view_7, v_1, context_3], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:19
            extern_kernels.bmm(reinterpret_tensor(buf49, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf42, (12, 128, 64), (64, 768, 1), 0), out=buf51)
            buf52 = reinterpret_tensor(buf231, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf51, buf52, 98304, stream=raw_stream0)
            buf229 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_222], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf49, buf229, 1536, 128, stream=raw_stream0)
            buf230 = reinterpret_tensor(buf49, (12, 128, 128), (16384, 1, 128), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [view_300], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf229, buf230, 1536, 128, stream=raw_stream0)
            buf53 = reinterpret_tensor(buf51, (128, 768), (768, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1, context_4, context_5, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(reinterpret_tensor(buf52, (128, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 768), (1, 768), 0), out=buf53)
            buf57 = reinterpret_tensor(buf53, (1, 128, 768), (98304, 768, 1), 0); del buf53  # reuse
            buf58 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf228 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_5, add_3, sa_output_1, div_35], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf57, primals_30, buf39, primals_31, primals_32, buf58, buf228, 128, 768, stream=raw_stream0)
            del primals_30
            del primals_32
            buf59 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_1, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:21
            extern_kernels.addmm(primals_34, reinterpret_tensor(buf58, (128, 768), (768, 1), 0), reinterpret_tensor(primals_33, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf59)
            del primals_34
            buf60 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf59, buf60, 393216, stream=raw_stream0)
            buf61 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(primals_36, reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf61)
            del primals_36
            # Unsorted Source Nodes: [x_7, x_6], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:23
            buf62 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf61, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf63 = buf62[0]
            assert_alignment(buf63, 16, 'torch.ops.npu._npu_dropout.default')
            buf64 = buf62[1]
            assert_alignment(buf64, 16, 'torch.ops.npu._npu_dropout.default')
            del buf62
            buf68 = reinterpret_tensor(buf61, (1, 128, 768), (98304, 768, 1), 0); del buf61  # reuse
            buf69 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf227 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_4, ffn_output_1, div_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf63, buf58, primals_37, primals_38, buf68, buf69, buf227, 128, 768, stream=raw_stream0)
            del primals_38
            buf70 = reinterpret_tensor(buf63, (128, 768), (768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:24
            extern_kernels.mm(reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), out=buf70)
            buf73 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, q_5], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_40, buf70, buf73, 98304, stream=raw_stream0)
            del primals_40
            buf71 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, linear_13], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(primals_42, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
            del primals_42
            buf72 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, linear_14], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:26
            extern_kernels.addmm(primals_44, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), reinterpret_tensor(primals_43, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
            del primals_44
            buf225 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_191], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf73, buf225, 768, 128, stream=raw_stream0)
            buf226 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_272], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf225, buf226, 768, 128, stream=raw_stream0)
            buf74 = reinterpret_tensor(buf229, (12, 128, 128), (16384, 128, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, linear_13, view_11, k_2, q_5, transpose_13, scores_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:27
            extern_kernels.bmm(reinterpret_tensor(buf73, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf71, (12, 64, 128), (64, 1, 768), 0), out=buf74)
            buf77 = reinterpret_tensor(buf74, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_4, scores_5, weights_4], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf77, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_5], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:28
            buf78 = torch.ops.npu._npu_dropout.default(buf77, 0.1)
            buf79 = buf78[0]
            assert_alignment(buf79, 16, 'torch.ops.npu._npu_dropout.default')
            buf80 = buf78[1]
            assert_alignment(buf80, 16, 'torch.ops.npu._npu_dropout.default')
            del buf78
            buf81 = reinterpret_tensor(buf73, (12, 128, 64), (8192, 64, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [linear_14, view_12, v_2, context_6], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:29
            extern_kernels.bmm(reinterpret_tensor(buf79, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf72, (12, 128, 64), (64, 768, 1), 0), out=buf81)
            buf82 = reinterpret_tensor(buf225, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf81, buf82, 98304, stream=raw_stream0)
            buf223 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_189], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf79, buf223, 1536, 128, stream=raw_stream0)
            buf224 = reinterpret_tensor(buf79, (12, 128, 128), (16384, 1, 128), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [view_266], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf223, buf224, 1536, 128, stream=raw_stream0)
            buf83 = reinterpret_tensor(buf81, (128, 768), (768, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2, context_7, context_8, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf82, (128, 768), (768, 1), 0), reinterpret_tensor(primals_45, (768, 768), (1, 768), 0), out=buf83)
            buf87 = reinterpret_tensor(buf83, (1, 128, 768), (98304, 768, 1), 0); del buf83  # reuse
            buf88 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf222 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_8, add_5, sa_output_2, div_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf87, primals_46, buf69, primals_47, primals_48, buf88, buf222, 128, 768, stream=raw_stream0)
            del primals_46
            del primals_48
            buf89 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_2, x_8], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:31
            extern_kernels.addmm(primals_50, reinterpret_tensor(buf88, (128, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf89)
            del primals_50
            buf90 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf89, buf90, 393216, stream=raw_stream0)
            buf91 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:32
            extern_kernels.addmm(primals_52, reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_51, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf91)
            del primals_52
            # Unsorted Source Nodes: [x_11, x_10], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:33
            buf92 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf91, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf93 = buf92[0]
            assert_alignment(buf93, 16, 'torch.ops.npu._npu_dropout.default')
            buf94 = buf92[1]
            assert_alignment(buf94, 16, 'torch.ops.npu._npu_dropout.default')
            del buf92
            buf98 = reinterpret_tensor(buf91, (1, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
            buf99 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf221 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_6, ffn_output_2, div_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf93, buf88, primals_53, primals_54, buf98, buf99, buf221, 128, 768, stream=raw_stream0)
            del primals_54
            buf100 = reinterpret_tensor(buf93, (128, 768), (768, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(reinterpret_tensor(buf99, (128, 768), (768, 1), 0), reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), out=buf100)
            buf103 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, q_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_56, buf100, buf103, 98304, stream=raw_stream0)
            del primals_56
            buf101 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, linear_19], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:35
            extern_kernels.addmm(primals_58, reinterpret_tensor(buf99, (128, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf101)
            del primals_58
            buf102 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, linear_20], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:36
            extern_kernels.addmm(primals_60, reinterpret_tensor(buf99, (128, 768), (768, 1), 0), reinterpret_tensor(primals_59, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
            del primals_60
            buf219 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_158], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf103, buf219, 768, 128, stream=raw_stream0)
            buf220 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_238], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf219, buf220, 768, 128, stream=raw_stream0)
            buf104 = reinterpret_tensor(buf223, (12, 128, 128), (16384, 128, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, linear_19, view_16, k_3, q_7, transpose_18, scores_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:37
            extern_kernels.bmm(reinterpret_tensor(buf103, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf101, (12, 64, 128), (64, 1, 768), 0), out=buf104)
            buf107 = reinterpret_tensor(buf104, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_6, scores_7, weights_6], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf107, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_7], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:38
            buf108 = torch.ops.npu._npu_dropout.default(buf107, 0.1)
            buf109 = buf108[0]
            assert_alignment(buf109, 16, 'torch.ops.npu._npu_dropout.default')
            buf110 = buf108[1]
            assert_alignment(buf110, 16, 'torch.ops.npu._npu_dropout.default')
            del buf108
            buf111 = reinterpret_tensor(buf103, (12, 128, 64), (8192, 64, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [linear_20, view_17, v_3, context_9], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf109, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf102, (12, 128, 64), (64, 768, 1), 0), out=buf111)
            buf112 = reinterpret_tensor(buf219, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf111, buf112, 98304, stream=raw_stream0)
            buf217 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_156], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf109, buf217, 1536, 128, stream=raw_stream0)
            buf218 = reinterpret_tensor(buf109, (12, 128, 128), (16384, 1, 128), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [view_232], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf217, buf218, 1536, 128, stream=raw_stream0)
            buf113 = reinterpret_tensor(buf111, (128, 768), (768, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3, context_10, context_11, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:40
            extern_kernels.mm(reinterpret_tensor(buf112, (128, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 768), (1, 768), 0), out=buf113)
            buf117 = reinterpret_tensor(buf113, (1, 128, 768), (98304, 768, 1), 0); del buf113  # reuse
            buf118 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf216 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_11, add_7, sa_output_3, div_29], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf117, primals_62, buf99, primals_63, primals_64, buf118, buf216, 128, 768, stream=raw_stream0)
            del primals_62
            del primals_64
            buf119 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_3, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:41
            extern_kernels.addmm(primals_66, reinterpret_tensor(buf118, (128, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf119)
            del primals_66
            buf120 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf119, buf120, 393216, stream=raw_stream0)
            buf121 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, x_14], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:42
            extern_kernels.addmm(primals_68, reinterpret_tensor(buf120, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_67, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf121)
            del primals_68
            # Unsorted Source Nodes: [x_15, x_14], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:43
            buf122 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf121, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf123 = buf122[0]
            assert_alignment(buf123, 16, 'torch.ops.npu._npu_dropout.default')
            buf124 = buf122[1]
            assert_alignment(buf124, 16, 'torch.ops.npu._npu_dropout.default')
            del buf122
            buf128 = reinterpret_tensor(buf121, (1, 128, 768), (98304, 768, 1), 0); del buf121  # reuse
            buf129 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf215 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_8, ffn_output_3, div_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf123, buf118, primals_69, primals_70, buf128, buf129, buf215, 128, 768, stream=raw_stream0)
            del primals_70
            buf130 = reinterpret_tensor(buf123, (128, 768), (768, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:44
            extern_kernels.mm(reinterpret_tensor(buf129, (128, 768), (768, 1), 0), reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), out=buf130)
            buf133 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, q_9], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_72, buf130, buf133, 98304, stream=raw_stream0)
            del primals_72
            buf131 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, linear_25], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:45
            extern_kernels.addmm(primals_74, reinterpret_tensor(buf129, (128, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
            del primals_74
            buf132 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, linear_26], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:46
            extern_kernels.addmm(primals_76, reinterpret_tensor(buf129, (128, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
            del primals_76
            buf213 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_125], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf133, buf213, 768, 128, stream=raw_stream0)
            buf214 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_204], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf213, buf214, 768, 128, stream=raw_stream0)
            buf134 = reinterpret_tensor(buf217, (12, 128, 128), (16384, 128, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, linear_25, view_21, k_4, q_9, transpose_23, scores_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:47
            extern_kernels.bmm(reinterpret_tensor(buf133, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf131, (12, 64, 128), (64, 1, 768), 0), out=buf134)
            buf137 = reinterpret_tensor(buf134, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_8, scores_9, weights_8], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf137, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_9], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:48
            buf138 = torch.ops.npu._npu_dropout.default(buf137, 0.1)
            buf139 = buf138[0]
            assert_alignment(buf139, 16, 'torch.ops.npu._npu_dropout.default')
            buf140 = buf138[1]
            assert_alignment(buf140, 16, 'torch.ops.npu._npu_dropout.default')
            del buf138
            buf141 = reinterpret_tensor(buf133, (12, 128, 64), (8192, 64, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [linear_26, view_22, v_4, context_12], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:49
            extern_kernels.bmm(reinterpret_tensor(buf139, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf132, (12, 128, 64), (64, 768, 1), 0), out=buf141)
            buf142 = reinterpret_tensor(buf213, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf141, buf142, 98304, stream=raw_stream0)
            buf211 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_123], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf139, buf211, 1536, 128, stream=raw_stream0)
            buf212 = reinterpret_tensor(buf139, (12, 128, 128), (16384, 1, 128), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [view_198], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf211, buf212, 1536, 128, stream=raw_stream0)
            buf143 = reinterpret_tensor(buf141, (128, 768), (768, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4, context_13, context_14, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf142, (128, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 768), (1, 768), 0), out=buf143)
            buf147 = reinterpret_tensor(buf143, (1, 128, 768), (98304, 768, 1), 0); del buf143  # reuse
            buf148 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf210 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_14, add_9, sa_output_4, div_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf147, primals_78, buf129, primals_79, primals_80, buf148, buf210, 128, 768, stream=raw_stream0)
            del primals_78
            del primals_80
            buf149 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_4, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:51
            extern_kernels.addmm(primals_82, reinterpret_tensor(buf148, (128, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf149)
            del primals_82
            buf150 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf149, buf150, 393216, stream=raw_stream0)
            buf151 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17, x_18], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(primals_84, reinterpret_tensor(buf150, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf151)
            del primals_84
            # Unsorted Source Nodes: [x_19, x_18], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:53
            buf152 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf151, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf153 = buf152[0]
            assert_alignment(buf153, 16, 'torch.ops.npu._npu_dropout.default')
            buf154 = buf152[1]
            assert_alignment(buf154, 16, 'torch.ops.npu._npu_dropout.default')
            del buf152
            buf158 = reinterpret_tensor(buf151, (1, 128, 768), (98304, 768, 1), 0); del buf151  # reuse
            buf159 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf209 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_10, ffn_output_4, div_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf153, buf148, primals_85, primals_86, buf158, buf159, buf209, 128, 768, stream=raw_stream0)
            del primals_86
            buf160 = reinterpret_tensor(buf153, (128, 768), (768, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:54
            extern_kernels.mm(reinterpret_tensor(buf159, (128, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), out=buf160)
            buf163 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, q_11], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_2.run(primals_88, buf160, buf163, 98304, stream=raw_stream0)
            del primals_88
            buf161 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, linear_31], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:55
            extern_kernels.addmm(primals_90, reinterpret_tensor(buf159, (128, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf161)
            del primals_90
            buf162 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, linear_32], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:56
            extern_kernels.addmm(primals_92, reinterpret_tensor(buf159, (128, 768), (768, 1), 0), reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf162)
            del primals_92
            buf207 = empty_strided_npu((1, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_92], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_3.run(buf163, buf207, 768, 128, stream=raw_stream0)
            buf208 = empty_strided_npu((12, 64, 128), (64, 1, 768), torch.float32)
            # Topologically Sorted Source Nodes: [view_170], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_4.run(buf207, buf208, 768, 128, stream=raw_stream0)
            buf164 = reinterpret_tensor(buf211, (12, 128, 128), (16384, 128, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, linear_31, view_26, k_5, q_11, transpose_28, scores_10], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:57
            extern_kernels.bmm(reinterpret_tensor(buf163, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf161, (12, 64, 128), (64, 1, 768), 0), out=buf164)
            buf167 = reinterpret_tensor(buf164, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_10, scores_11, weights_10], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf167, 1536, 128, stream=raw_stream0)
            # Unsorted Source Nodes: [weights_11], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:58
            buf168 = torch.ops.npu._npu_dropout.default(buf167, 0.1)
            buf169 = buf168[0]
            assert_alignment(buf169, 16, 'torch.ops.npu._npu_dropout.default')
            buf170 = buf168[1]
            assert_alignment(buf170, 16, 'torch.ops.npu._npu_dropout.default')
            del buf168
            buf171 = reinterpret_tensor(buf163, (12, 128, 64), (8192, 64, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [linear_32, view_27, v_5, context_15], Original ATen: [aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:59
            extern_kernels.bmm(reinterpret_tensor(buf169, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf162, (12, 128, 64), (64, 768, 1), 0), out=buf171)
            buf172 = reinterpret_tensor(buf207, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_6.run(buf171, buf172, 98304, stream=raw_stream0)
            buf205 = empty_strided_npu((1, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_90], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_7.run(buf169, buf205, 1536, 128, stream=raw_stream0)
            buf206 = reinterpret_tensor(buf169, (12, 128, 128), (16384, 1, 128), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [view_164], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf205, buf206, 1536, 128, stream=raw_stream0)
            del buf205
            buf173 = reinterpret_tensor(buf171, (128, 768), (768, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5, context_16, context_17, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:60
            extern_kernels.mm(reinterpret_tensor(buf172, (128, 768), (768, 1), 0), reinterpret_tensor(primals_93, (768, 768), (1, 768), 0), out=buf173)
            buf177 = reinterpret_tensor(buf173, (1, 128, 768), (98304, 768, 1), 0); del buf173  # reuse
            buf178 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            buf204 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_17, add_11, sa_output_5, div_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_9.run(buf177, primals_94, buf159, primals_95, primals_96, buf178, buf204, 128, 768, stream=raw_stream0)
            del primals_94
            del primals_96
            buf179 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_5, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:61
            extern_kernels.addmm(primals_98, reinterpret_tensor(buf178, (128, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf179)
            del primals_98
            buf180 = empty_strided_npu((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_10.run(buf179, buf180, 393216, stream=raw_stream0)
            buf181 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, x_22], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:62
            extern_kernels.addmm(primals_100, reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_99, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf181)
            del primals_100
            # Unsorted Source Nodes: [x_23, x_22], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:63
            buf182 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf181, (1, 128, 768), (98304, 768, 1), 0), 0.1)
            buf183 = buf182[0]
            assert_alignment(buf183, 16, 'torch.ops.npu._npu_dropout.default')
            buf184 = buf182[1]
            assert_alignment(buf184, 16, 'torch.ops.npu._npu_dropout.default')
            del buf182
            buf188 = buf183; del buf183  # reuse
            buf189 = reinterpret_tensor(buf181, (1, 128, 768), (98304, 768, 1), 0); del buf181  # reuse
            buf203 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_12, ffn_output_5, div_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_1.run(buf188, buf178, primals_101, primals_102, buf189, buf203, 128, 768, stream=raw_stream0)
            del primals_102
            buf190 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_5, prediction_logits], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:64
            extern_kernels.addmm(primals_104, reinterpret_tensor(buf189, (128, 768), (768, 1), 0), reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf190)
            del primals_104
            buf191 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf192 = buf191; del buf191  # reuse
            buf193 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf194 = reinterpret_tensor(buf193, (1, 128, 1), (128, 1, 1), 0); del buf193  # reuse
            buf195 = empty_strided_npu((1, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_native_layer_norm_view_12.run(buf192, buf194, buf190, primals_105, primals_106, buf195, 128, 768, stream=raw_stream0)
            del primals_106
            buf196 = empty_strided_npu((128, 30522), (30522, 1), torch.float32)
            # Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2, prediction_logits_3], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(primals_107, reinterpret_tensor(buf195, (128, 768), (768, 1), 0), reinterpret_tensor(primals_2, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf196)
            del primals_107
            # Topologically Sorted Source Nodes: [prediction_logits_3, view_30, mlm_loss], Original ATen: [aten.view, aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:66
            buf197 = torch.ops.aten._log_softmax.default(buf196, 1, False)
            assert_alignment(buf197, 16, 'torch.ops.aten._log_softmax.default')
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:67
            buf199 = torch.ops.aten.gather.default(buf197, 1, reinterpret_tensor(buf198, (128, 1), (1, 0), 0))
            assert_alignment(buf199, 16, 'torch.ops.aten.gather.default')
            del buf198
            buf202 = empty_strided_npu((), (), torch.float32)
            buf201 = empty_strided_npu((), (), torch.float32)
            buf240 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_13.run(buf240, primals_108, buf199, buf201, 1, 128, stream=raw_stream0)
            del buf199
        return (buf240, reinterpret_tensor(buf196, (1, 128, 30522), (3906816, 30522, 1), 0), primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, buf5, buf9, reinterpret_tensor(buf8, (128, 768), (768, 1), 0), buf17, buf20, reinterpret_tensor(buf22, (128, 768), (768, 1), 0), buf27, reinterpret_tensor(buf28, (128, 768), (768, 1), 0), buf29, reinterpret_tensor(buf30, (128, 3072), (3072, 1), 0), buf34, buf38, reinterpret_tensor(buf39, (128, 768), (768, 1), 0), buf47, buf50, reinterpret_tensor(buf52, (128, 768), (768, 1), 0), buf57, reinterpret_tensor(buf58, (128, 768), (768, 1), 0), buf59, reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0), buf64, buf68, reinterpret_tensor(buf69, (128, 768), (768, 1), 0), buf77, buf80, reinterpret_tensor(buf82, (128, 768), (768, 1), 0), buf87, reinterpret_tensor(buf88, (128, 768), (768, 1), 0), buf89, reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0), buf94, buf98, reinterpret_tensor(buf99, (128, 768), (768, 1), 0), buf107, buf110, reinterpret_tensor(buf112, (128, 768), (768, 1), 0), buf117, reinterpret_tensor(buf118, (128, 768), (768, 1), 0), buf119, reinterpret_tensor(buf120, (128, 3072), (3072, 1), 0), buf124, buf128, reinterpret_tensor(buf129, (128, 768), (768, 1), 0), buf137, buf140, reinterpret_tensor(buf142, (128, 768), (768, 1), 0), buf147, reinterpret_tensor(buf148, (128, 768), (768, 1), 0), buf149, reinterpret_tensor(buf150, (128, 3072), (3072, 1), 0), buf154, buf158, reinterpret_tensor(buf159, (128, 768), (768, 1), 0), buf167, buf170, reinterpret_tensor(buf172, (128, 768), (768, 1), 0), buf177, reinterpret_tensor(buf178, (128, 768), (768, 1), 0), buf179, reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0), buf184, buf188, reinterpret_tensor(buf189, (128, 768), (768, 1), 0), buf190, reinterpret_tensor(buf192, (1, 128, 1), (128, 1, 1), 0), buf194, reinterpret_tensor(buf195, (128, 768), (768, 1), 0), buf197, buf201, buf203, buf204, reinterpret_tensor(buf162, (12, 64, 128), (64, 1, 768), 0), buf206, reinterpret_tensor(buf161, (12, 128, 64), (64, 768, 1), 0), buf208, buf209, buf210, reinterpret_tensor(buf132, (12, 64, 128), (64, 1, 768), 0), buf212, reinterpret_tensor(buf131, (12, 128, 64), (64, 768, 1), 0), buf214, buf215, buf216, reinterpret_tensor(buf102, (12, 64, 128), (64, 1, 768), 0), buf218, reinterpret_tensor(buf101, (12, 128, 64), (64, 768, 1), 0), buf220, buf221, buf222, reinterpret_tensor(buf72, (12, 64, 128), (64, 1, 768), 0), buf224, reinterpret_tensor(buf71, (12, 128, 64), (64, 768, 1), 0), buf226, buf227, buf228, reinterpret_tensor(buf42, (12, 64, 128), (64, 1, 768), 0), buf230, reinterpret_tensor(buf41, (12, 128, 64), (64, 768, 1), 0), buf232, buf233, buf234, reinterpret_tensor(buf12, (12, 64, 128), (64, 1, 768), 0), buf236, reinterpret_tensor(buf11, (12, 128, 64), (64, 768, 1), 0), buf238, buf239, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    primals_2 = rand_strided((30522, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 512), (512, 1), device='npu:0', dtype=torch.int64)
    primals_4 = rand_strided((512, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_17 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_18 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_33 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_34 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_35 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_43 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_50 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_66 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_82 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_83 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_97 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_98 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_107 = rand_strided((30522, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_108 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('DistilBertForMaskedLM', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
