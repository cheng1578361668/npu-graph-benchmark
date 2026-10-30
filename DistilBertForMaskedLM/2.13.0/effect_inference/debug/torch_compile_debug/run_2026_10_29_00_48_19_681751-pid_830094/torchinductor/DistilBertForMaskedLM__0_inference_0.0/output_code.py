# AOT ID: ['0_inference']
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/ks/cks3k2ihizgkgpkclegjcabmjo7h6is47dolci2flhtsmcwwduwn.py
# Topologically Sorted Source Nodes: [embeddings, embeddings_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf4 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf4]
#   %getitem_1 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_1]
#   %buf6 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg5_1]
#   %add : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   return %buf4,%getitem_1,%buf6,%add_2
triton_unk_fused_add_native_layer_norm_0 = async_compile.triton('triton_unk_fused_add_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_0(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp29, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/3l/c3l3tkzthkl6sx2afbyjomfshpk4oybw4hv4y4p2w2vddvopj7h3.py
# Topologically Sorted Source Nodes: [, linear, view, q, q_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
# Source node to ATen node mapping:
#    => add_tensor_24
#   linear => view_1
#   q => permute_1
#   q_1 => div
#   view => view_2
# Graph fragment:
#   %arg7_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg7_1]
#   %mm_default_24 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_24]
#   %add_tensor_24 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg7_1, %mm_default_24), kwargs = {})
#   %view_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [1, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %div : Tensor "f32[1, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_1, 8.0), kwargs = {})
#   return %expand
triton_unk_fused_addmm_div_transpose_view_1 = async_compile.triton('triton_unk_fused_addmm_div_transpose_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_div_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_div_transpose_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/cu/ccu7znczadlutl6kgr5k4m3cj7zyh3oonjqic5zdanq4rpfgwz47.py
# Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores, scores_1, weights], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
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
#   %full : Tensor "f32[1, 128][128, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %eq : Tensor "b8[1, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%full, 0), kwargs = {})
#   %view_12 : Tensor "b8[1, 1, 1, 128][128, 128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [1, 1, 1, 128]), kwargs = {})
#   %expand_2 : Tensor "b8[1, 12, 128, 128][128, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_12, [1, 12, 128, 128]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_11 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 12, 128, 128]), kwargs = {})
#   %where : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %view_11), kwargs = {})
#   %amax : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[1, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[1, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax,%sum_1,%expand_3
triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2 = async_compile.triton('triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/4p/c4pexdtshz637atf3ss6gw6oqujtfbyljm5e4l4s22pfoenqymqx.py
# Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   context => view_15
#   contiguous => clone_2
#   transpose_4 => permute_7
# Graph fragment:
#   %bmm_1 : Tensor "f32[12, 128, 64][8192, 64, 1]npu:0" = PlaceHolder[target=bmm_1]
#   %view_15 : Tensor "f32[1, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 12, 128, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[1, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[1, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_unk_fused_clone_matmul_transpose_3 = async_compile.triton('triton_unk_fused_clone_matmul_transpose_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_transpose_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_transpose_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/wl/cwlvvftihujefd7nw42ksfywtswkuyress2cpnpf3tzv2sx4o4nj.py
# Topologically Sorted Source Nodes: [, context_2, add_1, sa_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_23
#   add_1 => add_3
#   context_2 => view_18
#   sa_output => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %arg13_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg13_1]
#   %mm_default_23 : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_23]
#   %add_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=add_2]
#   %buf19 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf19]
#   %getitem_3 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_3]
#   %buf21 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf21]
#   %arg14_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg15_1]
#   %add_tensor_23 : Tensor "f32[128, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, %mm_default_23), kwargs = {})
#   %view_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [1, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %add_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg14_1), kwargs = {})
#   %add_5 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   return %buf19,%getitem_3,%buf21,%add_5
triton_unk_fused_add_addmm_native_layer_norm_view_4 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_4(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/ux/cuxo4onizl7n6jeypcsncurjagitirq6x6p52otxwolk45j6ftqj.py
# Topologically Sorted Source Nodes: [, x, x_1], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#    => add_tensor_22
#   x => view_20
#   x_1 => add_6, add_7, div_2, exp_1, mul_4, mul_5, mul_6, mul_7
# Graph fragment:
#   %arg17_1 : Tensor "f32[3072][1]npu:0" = PlaceHolder[target=arg17_1]
#   %mm_default_22 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_default_22]
#   %add_tensor_22 : Tensor "f32[128, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg17_1, %mm_default_22), kwargs = {})
#   %view_20 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [1, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %view_20), kwargs = {})
#   %mul_5 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %view_20), kwargs = {})
#   %mul_6 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, -1.5957691216057308), kwargs = {})
#   %exp_1 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1.0), kwargs = {})
#   %div_2 : Tensor "f32[1, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, %add_7), kwargs = {})
#   return %div_2
triton_unk_fused_addmm_gelu_view_5 = async_compile.triton('triton_unk_fused_addmm_gelu_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [3072, 128, {'divisors': [1, 3072]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 3072, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 3072, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_gelu_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_gelu_view_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 393216
    x_g_tile0 : tl.constexpr = (3072) if (3072) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (128) if (128) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 3072
    x1numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + (x0 + 3072*x1), x0mask & x1mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3 * tmp2
        tmp5 = tl.full([1, 1], 0.044715, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 + tmp6
        tmp8 = tl.full([1, 1], -1.5957691216057308, tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = libdevice.exp(tmp9)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = (tmp2 / tmp12)
        tl.store(in_out_ptr0 + (x0 + 3072*x1), tmp13, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/u3/cu3gf37m7h4wb2knnwjtji576j72nmhmyldxhx2xaj3j237dwz5u.py
# Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => add_53, add_54, mul_54, mul_55, rsqrt_13, sub_19, var_mean_13
# Graph fragment:
#   %arg103_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg103_1]
#   %mm_default : Tensor "f32[128, 768][768, 1]npu:0" = PlaceHolder[target=mm_default]
#   %buf141 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf141]
#   %getitem_27 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_27]
#   %buf143 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf143]
#   %arg104_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg104_1]
#   %arg105_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg105_1]
#   %add_tensor : Tensor "f32[128, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg103_1, %mm_default), kwargs = {})
#   %view_139 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [1, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%div_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_19 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_27), kwargs = {})
#   %add_53 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-12), kwargs = {})
#   %rsqrt_13 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %mul_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_55 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %arg104_1), kwargs = {})
#   %add_54 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %arg105_1), kwargs = {})
#   return %buf141,%getitem_27,%buf143,%add_54
triton_unk_fused_addmm_gelu_native_layer_norm_view_6 = async_compile.triton('triton_unk_fused_addmm_gelu_native_layer_norm_view_6', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_gelu_native_layer_norm_view_6', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_addmm_gelu_native_layer_norm_view_6(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp15 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2 * tmp2
            tmp4 = tmp3 * tmp2
            tmp5 = tl.full([1, 1], 0.044715, tl.float32)
            tmp6 = tmp4 * tmp5
            tmp7 = tmp2 + tmp6
            tmp8 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp9 = tmp7 * tmp8
            tmp10 = libdevice.exp(tmp9)
            tmp11 = tl.full([1, 1], 1.0, tl.float32)
            tmp12 = tmp10 + tmp11
            tmp13 = (tmp2 / tmp12)
            tmp14 = tl.broadcast_to(tmp13, [real_block_x0, R0_BLOCK])
            tmp16 = _tmp15 + tmp14
            _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp15 = tl.sum(_tmp15, 1)[:, None]
        tmp17 = tl.full([1, 1], 768.0, tl.float32)
        tmp18 = (tmp15 / tmp17)
        _tmp36 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp19 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp20 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp21 = tmp19 + tmp20
            tmp22 = tmp21 * tmp21
            tmp23 = tmp22 * tmp21
            tmp24 = tl.full([1, 1], 0.044715, tl.float32)
            tmp25 = tmp23 * tmp24
            tmp26 = tmp21 + tmp25
            tmp27 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp28 = tmp26 * tmp27
            tmp29 = libdevice.exp(tmp28)
            tmp30 = tl.full([1, 1], 1.0, tl.float32)
            tmp31 = tmp29 + tmp30
            tmp32 = (tmp21 / tmp31)
            tmp33 = tmp32 - tmp18
            tmp34 = tmp33 * tmp33
            tmp35 = tl.broadcast_to(tmp34, [real_block_x0, R0_BLOCK])
            tmp37 = _tmp36 + tmp35
            _tmp36 = tl.where(r0_mask & xmask, tmp37, _tmp36)
        tmp36 = tl.sum(_tmp36, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp38 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp39 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp59 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp61 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp40 = tmp38 + tmp39
            tmp41 = tmp40 * tmp40
            tmp42 = tmp41 * tmp40
            tmp43 = tl.full([1, 1], 0.044715, tl.float32)
            tmp44 = tmp42 * tmp43
            tmp45 = tmp40 + tmp44
            tmp46 = tl.full([1, 1], -1.5957691216057308, tl.float32)
            tmp47 = tmp45 * tmp46
            tmp48 = libdevice.exp(tmp47)
            tmp49 = tl.full([1, 1], 1.0, tl.float32)
            tmp50 = tmp48 + tmp49
            tmp51 = (tmp40 / tmp50)
            tmp52 = tmp51 - tmp18
            tmp53 = tl.full([1, 1], 768.0, tl.float32)
            tmp54 = (tmp36 / tmp53)
            tmp55 = tl.full([1, 1], 1e-12, tl.float32)
            tmp56 = tmp54 + tmp55
            tmp57 = tl.rsqrt(tmp56)
            tmp58 = tmp52 * tmp57
            tmp60 = tmp58 * tmp59
            tmp62 = tmp60 + tmp61
            tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp62, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/kg/ckg74r5c5xtxjvyr7viofg7ist323bjojoiyvsy3zwfrxyhjon34.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => full_default_6, ne, where_6
#   view_31 => view_143
# Graph fragment:
#   %arg107_1 : Tensor "i64[1, 128][128, 1]npu:0" = PlaceHolder[target=arg107_1]
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%arg107_1, [-1]), kwargs = {})
#   %ne : Tensor "b8[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %full_default_6 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_6 : Tensor "i64[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_143, %full_default_6), kwargs = {})
#   return %where_6
triton_unk_fused_nll_loss_forward_view_7 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2/2.13/effect_inference/cache/ii/cii2jf4osi73culyxlehktume5n3j5cj4ncy27ehlnvabgbem3l6.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => convert_element_type, div_19, full_default_7, ne_1, ne_2, neg, squeeze, sum_7, sum_8, where_7
#   view_31 => view_143
# Graph fragment:
#   %arg107_1 : Tensor "i64[1, 128][128, 1]npu:0" = PlaceHolder[target=arg107_1]
#   %gather : Tensor "f32[128, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %sum_8 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_8]
#   %sum_7 : Tensor "i64[][]npu:0" = PlaceHolder[target=sum_7]
#   %view_143 : Tensor "i64[128][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%arg107_1, [-1]), kwargs = {})
#   %ne_1 : Tensor "b8[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %squeeze : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_7 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_7), kwargs = {})
#   %sum_8 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %ne_2 : Tensor "b8[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %sum_7 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_7, torch.float32), kwargs = {})
#   %div_19 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_8, %convert_element_type), kwargs = {})
#   return %sum_8,%sum_7,%div_19
triton_unk_fused_nll_loss_forward_view_8 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'r0_0', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp8 = tl.full([R0_BLOCK, 1], 0, tl.float32)
        _tmp12 = tl.full([R0_BLOCK, 1], 0, tl.int32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_0 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.full([], -100, tl.int32)
            tmp2 = tmp0 != tmp1
            tmp4 = -tmp3
            tmp5 = tl.full([], 0.0, tl.float32)
            tmp6 = tl.where(tmp2, tmp4, tmp5)
            tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK, 1])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tmp9
            tmp10 = tmp2.to(tl.int32)
            tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK, 1])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tmp8 = tl.sum(_tmp8, 0)[None, :]
        tmp12 = tl.sum(_tmp12, 0)[None, :]
        tmp14 = tmp12.to(tl.float32)
        tmp15 = (tmp8 / tmp14)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [input_embeds], Original ATen: [aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:1
            buf0 = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0)
            del arg0_1
            buf1 = buf0
            assert_alignment(buf1, 16, 'torch.ops.aten.embedding.default')
            del buf0
            # Topologically Sorted Source Nodes: [position_ids, position_embeddings], Original ATen: [aten.slice, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf2 = torch.ops.aten.embedding.default(arg3_1, reinterpret_tensor(arg2_1, (1, 128), (512, 1), 0))
            del arg2_1
            del arg3_1
            buf3 = buf2
            assert_alignment(buf3, 16, 'torch.ops.aten.embedding.default')
            del buf2
            buf7 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [embeddings, embeddings_1], Original ATen: [aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_0.run(buf7, buf3, arg4_1, arg5_1, 128, 768, stream=raw_stream0)
            del arg4_1
            del arg5_1
            buf8 = reinterpret_tensor(buf3, (128, 768), (768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), out=buf8)
            del arg6_1
            buf9 = empty_strided_npu((128, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:4
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf7, (128, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
            del arg8_1
            del arg9_1
            buf10 = empty_strided_npu((1, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, q_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg7_1, buf8, buf10, 98304, stream=raw_stream0)
            del arg7_1
            buf11 = empty_strided_npu((12, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, q_1, scores, linear_1, view_1, k, transpose_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:5
            extern_kernels.bmm(reinterpret_tensor(buf10, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf9, (12, 64, 128), (64, 1, 768), 0), out=buf11)
            buf15 = reinterpret_tensor(buf11, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores, scores_1, weights], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf15, 1536, 128, stream=raw_stream0)
            buf14 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:6
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf7, (128, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf14)
            del arg10_1
            del arg11_1
            buf16 = reinterpret_tensor(buf10, (12, 128, 64), (8192, 64, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores, scores_1, weights, context, linear_2, view_2, v], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:7
            extern_kernels.bmm(reinterpret_tensor(buf15, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf14, (12, 128, 64), (64, 768, 1), 0), out=buf16)
            buf17 = reinterpret_tensor(buf14, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf16, buf17, 98304, stream=raw_stream0)
            buf18 = reinterpret_tensor(buf16, (128, 768), (768, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous, context_1, context_2, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf17, (128, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf18)
            del arg12_1
            buf22 = reinterpret_tensor(buf18, (1, 128, 768), (98304, 768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [, context_2, add_1, sa_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf22, arg13_1, buf7, arg14_1, arg15_1, 128, 768, stream=raw_stream0)
            del arg13_1
            del arg14_1
            del arg15_1
            buf23 = empty_strided_npu((128, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(reinterpret_tensor(buf22, (128, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf23)
            del arg16_1
            buf24 = reinterpret_tensor(buf23, (1, 128, 3072), (393216, 3072, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [, x, x_1], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf24, arg17_1, 393216, stream=raw_stream0)
            del arg17_1
            buf25 = reinterpret_tensor(buf7, (128, 768), (768, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [, x, x_1, x_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf25)
            del arg18_1
            buf29 = reinterpret_tensor(buf25, (1, 128, 768), (98304, 768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [, x_2, add_2, ffn_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf29, arg19_1, buf22, arg20_1, arg21_1, 128, 768, stream=raw_stream0)
            del arg19_1
            del arg20_1
            del arg21_1
            buf30 = reinterpret_tensor(buf22, (128, 768), (768, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_6, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:11
            extern_kernels.mm(reinterpret_tensor(buf29, (128, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf30)
            del arg22_1
            buf31 = reinterpret_tensor(buf17, (128, 768), (768, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(arg25_1, reinterpret_tensor(buf29, (128, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
            del arg24_1
            del arg25_1
            buf32 = reinterpret_tensor(buf8, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, q_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg23_1, buf30, buf32, 98304, stream=raw_stream0)
            del arg23_1
            buf33 = reinterpret_tensor(buf15, (12, 128, 128), (16384, 128, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, q_3, scores_2, linear_7, view_6, k_1, transpose_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:13
            extern_kernels.bmm(reinterpret_tensor(buf32, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf31, (12, 64, 128), (64, 1, 768), 0), out=buf33)
            buf37 = reinterpret_tensor(buf33, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_1, view_8, mask_1, tensor_1, scores_2, scores_3, weights_2], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf37, 1536, 128, stream=raw_stream0)
            buf36 = reinterpret_tensor(buf32, (128, 768), (768, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:14
            extern_kernels.addmm(arg27_1, reinterpret_tensor(buf29, (128, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf36)
            del arg26_1
            del arg27_1
            buf38 = reinterpret_tensor(buf31, (12, 128, 64), (8192, 64, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_1, view_8, mask_1, tensor_1, scores_2, scores_3, weights_2, context_3, linear_8, view_7, v_1], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:15
            extern_kernels.bmm(reinterpret_tensor(buf37, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf36, (12, 128, 64), (64, 768, 1), 0), out=buf38)
            buf39 = reinterpret_tensor(buf36, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf38, buf39, 98304, stream=raw_stream0)
            buf40 = reinterpret_tensor(buf38, (128, 768), (768, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1, context_4, context_5, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf39, (128, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), out=buf40)
            del arg28_1
            buf44 = reinterpret_tensor(buf40, (1, 128, 768), (98304, 768, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [, context_5, add_3, sa_output_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf44, arg29_1, buf29, arg30_1, arg31_1, 128, 768, stream=raw_stream0)
            del arg29_1
            del arg30_1
            del arg31_1
            buf45 = reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [x_4, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(reinterpret_tensor(buf44, (128, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 3072), (1, 768), 0), out=buf45)
            del arg32_1
            buf46 = reinterpret_tensor(buf45, (1, 128, 3072), (393216, 3072, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [, x_4, x_5], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf46, arg33_1, 393216, stream=raw_stream0)
            del arg33_1
            buf47 = reinterpret_tensor(buf29, (128, 768), (768, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [, x_4, x_5, x_6], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:18
            extern_kernels.mm(reinterpret_tensor(buf46, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf47)
            del arg34_1
            buf51 = reinterpret_tensor(buf47, (1, 128, 768), (98304, 768, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [, x_6, add_4, ffn_output_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf51, arg35_1, buf44, arg36_1, arg37_1, 128, 768, stream=raw_stream0)
            del arg35_1
            del arg36_1
            del arg37_1
            buf52 = reinterpret_tensor(buf44, (128, 768), (768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [linear_12, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf51, (128, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), out=buf52)
            del arg38_1
            buf53 = reinterpret_tensor(buf39, (128, 768), (768, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:20
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf51, (128, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
            del arg40_1
            del arg41_1
            buf54 = reinterpret_tensor(buf30, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, q_5], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg39_1, buf52, buf54, 98304, stream=raw_stream0)
            del arg39_1
            buf55 = reinterpret_tensor(buf37, (12, 128, 128), (16384, 128, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, q_5, scores_4, linear_13, view_11, k_2, transpose_13], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:21
            extern_kernels.bmm(reinterpret_tensor(buf54, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf53, (12, 64, 128), (64, 1, 768), 0), out=buf55)
            buf59 = reinterpret_tensor(buf55, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_2, view_13, mask_2, tensor_2, scores_4, scores_5, weights_4], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf59, 1536, 128, stream=raw_stream0)
            buf58 = reinterpret_tensor(buf54, (128, 768), (768, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(arg43_1, reinterpret_tensor(buf51, (128, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf58)
            del arg42_1
            del arg43_1
            buf60 = reinterpret_tensor(buf53, (12, 128, 64), (8192, 64, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_2, view_13, mask_2, tensor_2, scores_4, scores_5, weights_4, context_6, linear_14, view_12, v_2], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:23
            extern_kernels.bmm(reinterpret_tensor(buf59, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf58, (12, 128, 64), (64, 768, 1), 0), out=buf60)
            buf61 = reinterpret_tensor(buf58, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf60, buf61, 98304, stream=raw_stream0)
            buf62 = reinterpret_tensor(buf60, (128, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2, context_7, context_8, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:24
            extern_kernels.mm(reinterpret_tensor(buf61, (128, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf62)
            del arg44_1
            buf66 = reinterpret_tensor(buf62, (1, 128, 768), (98304, 768, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [, context_8, add_5, sa_output_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf66, arg45_1, buf51, arg46_1, arg47_1, 128, 768, stream=raw_stream0)
            del arg45_1
            del arg46_1
            del arg47_1
            buf67 = reinterpret_tensor(buf46, (128, 3072), (3072, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [x_8, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(reinterpret_tensor(buf66, (128, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf67)
            del arg48_1
            buf68 = reinterpret_tensor(buf67, (1, 128, 3072), (393216, 3072, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [, x_8, x_9], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf68, arg49_1, 393216, stream=raw_stream0)
            del arg49_1
            buf69 = reinterpret_tensor(buf51, (128, 768), (768, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [, x_8, x_9, x_10], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:26
            extern_kernels.mm(reinterpret_tensor(buf68, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf69)
            del arg50_1
            buf73 = reinterpret_tensor(buf69, (1, 128, 768), (98304, 768, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [, x_10, add_6, ffn_output_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf73, arg51_1, buf66, arg52_1, arg53_1, 128, 768, stream=raw_stream0)
            del arg51_1
            del arg52_1
            del arg53_1
            buf74 = reinterpret_tensor(buf66, (128, 768), (768, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [linear_18, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:27
            extern_kernels.mm(reinterpret_tensor(buf73, (128, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), out=buf74)
            del arg54_1
            buf75 = reinterpret_tensor(buf61, (128, 768), (768, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:28
            extern_kernels.addmm(arg57_1, reinterpret_tensor(buf73, (128, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf75)
            del arg56_1
            del arg57_1
            buf76 = reinterpret_tensor(buf52, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, q_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg55_1, buf74, buf76, 98304, stream=raw_stream0)
            del arg55_1
            buf77 = reinterpret_tensor(buf59, (12, 128, 128), (16384, 128, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, q_7, scores_6, linear_19, view_16, k_3, transpose_18], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:29
            extern_kernels.bmm(reinterpret_tensor(buf76, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf75, (12, 64, 128), (64, 1, 768), 0), out=buf77)
            buf81 = reinterpret_tensor(buf77, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_3, view_18, mask_3, tensor_3, scores_6, scores_7, weights_6], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf81, 1536, 128, stream=raw_stream0)
            buf80 = reinterpret_tensor(buf76, (128, 768), (768, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:30
            extern_kernels.addmm(arg59_1, reinterpret_tensor(buf73, (128, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf80)
            del arg58_1
            del arg59_1
            buf82 = reinterpret_tensor(buf75, (12, 128, 64), (8192, 64, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_3, view_18, mask_3, tensor_3, scores_6, scores_7, weights_6, context_9, linear_20, view_17, v_3], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:31
            extern_kernels.bmm(reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf80, (12, 128, 64), (64, 768, 1), 0), out=buf82)
            buf83 = reinterpret_tensor(buf80, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf82, buf83, 98304, stream=raw_stream0)
            buf84 = reinterpret_tensor(buf82, (128, 768), (768, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3, context_10, context_11, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(reinterpret_tensor(buf83, (128, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), out=buf84)
            del arg60_1
            buf88 = reinterpret_tensor(buf84, (1, 128, 768), (98304, 768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [, context_11, add_7, sa_output_3], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf88, arg61_1, buf73, arg62_1, arg63_1, 128, 768, stream=raw_stream0)
            del arg61_1
            del arg62_1
            del arg63_1
            buf89 = reinterpret_tensor(buf68, (128, 3072), (3072, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [x_12, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:33
            extern_kernels.mm(reinterpret_tensor(buf88, (128, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 3072), (1, 768), 0), out=buf89)
            del arg64_1
            buf90 = reinterpret_tensor(buf89, (1, 128, 3072), (393216, 3072, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [, x_12, x_13], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf90, arg65_1, 393216, stream=raw_stream0)
            del arg65_1
            buf91 = reinterpret_tensor(buf73, (128, 768), (768, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [, x_12, x_13, x_14], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf91)
            del arg66_1
            buf95 = reinterpret_tensor(buf91, (1, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [, x_14, add_8, ffn_output_3], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf95, arg67_1, buf88, arg68_1, arg69_1, 128, 768, stream=raw_stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            buf96 = reinterpret_tensor(buf88, (128, 768), (768, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [linear_24, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:35
            extern_kernels.mm(reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), out=buf96)
            del arg70_1
            buf97 = reinterpret_tensor(buf83, (128, 768), (768, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:36
            extern_kernels.addmm(arg73_1, reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
            del arg72_1
            del arg73_1
            buf98 = reinterpret_tensor(buf74, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, q_9], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg71_1, buf96, buf98, 98304, stream=raw_stream0)
            del arg71_1
            buf99 = reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, q_9, scores_8, linear_25, view_21, k_4, transpose_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:37
            extern_kernels.bmm(reinterpret_tensor(buf98, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf97, (12, 64, 128), (64, 1, 768), 0), out=buf99)
            buf103 = reinterpret_tensor(buf99, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_4, view_23, mask_4, tensor_4, scores_8, scores_9, weights_8], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf103, 1536, 128, stream=raw_stream0)
            buf102 = reinterpret_tensor(buf98, (128, 768), (768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:38
            extern_kernels.addmm(arg75_1, reinterpret_tensor(buf95, (128, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
            del arg74_1
            del arg75_1
            buf104 = reinterpret_tensor(buf97, (12, 128, 64), (8192, 64, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_4, view_23, mask_4, tensor_4, scores_8, scores_9, weights_8, context_12, linear_26, view_22, v_4], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf103, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf102, (12, 128, 64), (64, 768, 1), 0), out=buf104)
            buf105 = reinterpret_tensor(buf102, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf104, buf105, 98304, stream=raw_stream0)
            buf106 = reinterpret_tensor(buf104, (128, 768), (768, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4, context_13, context_14, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:40
            extern_kernels.mm(reinterpret_tensor(buf105, (128, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf106)
            del arg76_1
            buf110 = reinterpret_tensor(buf106, (1, 128, 768), (98304, 768, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [, context_14, add_9, sa_output_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf110, arg77_1, buf95, arg78_1, arg79_1, 128, 768, stream=raw_stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            buf111 = reinterpret_tensor(buf90, (128, 3072), (3072, 1), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [x_16, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:41
            extern_kernels.mm(reinterpret_tensor(buf110, (128, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf111)
            del arg80_1
            buf112 = reinterpret_tensor(buf111, (1, 128, 3072), (393216, 3072, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [, x_16, x_17], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf112, arg81_1, 393216, stream=raw_stream0)
            del arg81_1
            buf113 = reinterpret_tensor(buf95, (128, 768), (768, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [, x_16, x_17, x_18], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf112, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf113)
            del arg82_1
            buf117 = reinterpret_tensor(buf113, (1, 128, 768), (98304, 768, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [, x_18, add_10, ffn_output_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf117, arg83_1, buf110, arg84_1, arg85_1, 128, 768, stream=raw_stream0)
            del arg83_1
            del arg84_1
            del arg85_1
            buf118 = reinterpret_tensor(buf110, (128, 768), (768, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [linear_30, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:43
            extern_kernels.mm(reinterpret_tensor(buf117, (128, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf118)
            del arg86_1
            buf119 = reinterpret_tensor(buf105, (128, 768), (768, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:44
            extern_kernels.addmm(arg89_1, reinterpret_tensor(buf117, (128, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
            del arg88_1
            del arg89_1
            buf120 = reinterpret_tensor(buf96, (1, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, q_11], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_transpose_view_1.run(arg87_1, buf118, buf120, 98304, stream=raw_stream0)
            del arg87_1
            del buf118
            buf121 = reinterpret_tensor(buf103, (12, 128, 128), (16384, 128, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, q_11, scores_10, linear_31, view_26, k_5, transpose_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:45
            extern_kernels.bmm(reinterpret_tensor(buf120, (12, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf119, (12, 64, 128), (64, 1, 768), 0), out=buf121)
            buf125 = reinterpret_tensor(buf121, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_5, view_28, mask_5, tensor_5, scores_10, scores_11, weights_10], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_2.run(buf125, 1536, 128, stream=raw_stream0)
            buf124 = reinterpret_tensor(buf120, (128, 768), (768, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:46
            extern_kernels.addmm(arg91_1, reinterpret_tensor(buf117, (128, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf124)
            del arg90_1
            del arg91_1
            buf126 = reinterpret_tensor(buf119, (12, 128, 64), (8192, 64, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq_5, view_28, mask_5, tensor_5, scores_10, scores_11, weights_10, context_15, linear_32, view_27, v_5], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:47
            extern_kernels.bmm(reinterpret_tensor(buf125, (12, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf124, (12, 128, 64), (64, 768, 1), 0), out=buf126)
            del buf125
            buf127 = reinterpret_tensor(buf124, (1, 128, 12, 64), (98304, 768, 64, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_3.run(buf126, buf127, 98304, stream=raw_stream0)
            buf128 = reinterpret_tensor(buf126, (128, 768), (768, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5, context_16, context_17, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:48
            extern_kernels.mm(reinterpret_tensor(buf127, (128, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf128)
            del arg92_1
            del buf127
            buf132 = reinterpret_tensor(buf128, (1, 128, 768), (98304, 768, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [, context_17, add_11, sa_output_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf132, arg93_1, buf117, arg94_1, arg95_1, 128, 768, stream=raw_stream0)
            del arg93_1
            del arg94_1
            del arg95_1
            buf133 = reinterpret_tensor(buf112, (128, 3072), (3072, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [x_20, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:49
            extern_kernels.mm(reinterpret_tensor(buf132, (128, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf133)
            del arg96_1
            buf134 = reinterpret_tensor(buf133, (1, 128, 3072), (393216, 3072, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [, x_20, x_21], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_5.run(buf134, arg97_1, 393216, stream=raw_stream0)
            del arg97_1
            buf135 = reinterpret_tensor(buf117, (128, 768), (768, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [, x_20, x_21, x_22], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf134, (128, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf135)
            del arg98_1
            del buf134
            buf139 = reinterpret_tensor(buf135, (1, 128, 768), (98304, 768, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [, x_22, add_12, ffn_output_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_4.run(buf139, arg99_1, buf132, arg100_1, arg101_1, 128, 768, stream=raw_stream0)
            del arg100_1
            del arg101_1
            del arg99_1
            buf140 = reinterpret_tensor(buf132, (128, 768), (768, 1), 0); del buf132  # reuse
            # Topologically Sorted Source Nodes: [, x_22, add_12, ffn_output_5, prediction_logits], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf139, (128, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf140)
            del arg102_1
            del buf139
            buf144 = reinterpret_tensor(buf140, (1, 128, 768), (98304, 768, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_native_layer_norm_view_6.run(buf144, arg103_1, arg104_1, arg105_1, 128, 768, stream=raw_stream0)
            del arg103_1
            del arg104_1
            del arg105_1
            buf145 = empty_strided_npu((128, 30522), (30522, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2, prediction_logits_3], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(arg106_1, reinterpret_tensor(buf144, (128, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf145)
            del arg106_1
            del arg1_1
            del buf144
            # Topologically Sorted Source Nodes: [prediction_logits_3, view_30, mlm_loss], Original ATen: [aten.view, aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:53
            buf146 = torch.ops.aten._log_softmax.default(buf145, 1, False)
            buf147 = buf146
            assert_alignment(buf147, 16, 'torch.ops.aten._log_softmax.default')
            del buf146
            arg107_1 = copy_if_misaligned(arg107_1)
            buf148 = empty_strided_npu((128, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_7.run(arg107_1, buf148, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:54
            buf149 = torch.ops.aten.gather.default(buf147, 1, reinterpret_tensor(buf148, (128, 1), (1, 0), 0))
            del buf147
            del buf148
            buf150 = buf149
            assert_alignment(buf150, 16, 'torch.ops.aten.gather.default')
            del buf149
            buf151 = empty_strided_npu((), (), torch.float32)
            buf153 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_8.run(buf153, arg107_1, buf150, 1, 128, stream=raw_stream0)
            del arg107_1
            del buf150
        return (buf153, reinterpret_tensor(buf145, (1, 128, 30522), (3906816, 30522, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    arg1_1 = rand_strided((30522, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 512), (512, 1), device='npu:0', dtype=torch.int64)
    arg3_1 = rand_strided((512, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg33_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg65_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg106_1 = rand_strided((30522, ), (1, ), device='npu:0', dtype=torch.float32)
    arg107_1 = rand_strided((1, 128), (128, 1), device='npu:0', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('DistilBertForMaskedLM', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
