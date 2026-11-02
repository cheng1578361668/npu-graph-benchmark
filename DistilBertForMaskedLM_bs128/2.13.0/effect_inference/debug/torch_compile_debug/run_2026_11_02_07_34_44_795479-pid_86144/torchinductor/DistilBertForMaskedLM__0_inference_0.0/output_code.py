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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/ur/curcl4pwzvljhk7e6bd5ss6rhayvtmcgco5zfhctnmwasnz7t3ac.py
# Topologically Sorted Source Nodes: [embeddings, embeddings_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf4 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf4]
#   %getitem_1 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_1]
#   %buf6 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg5_1]
#   %add : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
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
    size_hints={'x': 16384, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, 128, {'divisors': [1, 128]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 128, 'seed': 2}, {'name': 'r0_2', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_0(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 16384
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (128) if (128) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 128
    x1numel : tl.constexpr = 128
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
        rbase = tl.arange(0, R0_BLOCK)[None, None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :, None]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        x3 = x0 + 128*x1
        x3mask = x0mask & x1mask
        r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp4 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_out_ptr1 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 2)[:, :, None]
        tmp6 = tl.full([1, 1, 1], 768.0, tl.float32)
        tmp7 = (tmp4 / tmp6)
        _tmp14 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp8 = tl.load(in_out_ptr1 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.load(in_ptr0 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10 - tmp7
            tmp12 = tmp11 * tmp11
            tmp13 = tl.broadcast_to(tmp12, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp15 = _tmp14 + tmp13
            _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp14 = tl.sum(_tmp14, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp16 = tl.load(in_out_ptr1 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr0 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp26 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp28 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp7
            tmp20 = tl.full([1, 1, 1], 768.0, tl.float32)
            tmp21 = (tmp14 / tmp20)
            tmp22 = tl.full([1, 1, 1], 1e-12, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(in_out_ptr1 + (r0_2 + 768*x0 + 98304*x1), tmp29, r0_mask & x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/6z/c6zftpup6ewb676avuflh5lvucgyapcrr6km6azpl32q6tg2sgmn.py
# Topologically Sorted Source Nodes: [linear, view, , linear_1, view_1, linear_2, view_2, attention_mask], Original ATen: [aten.view, aten.transpose, aten.ones, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
# Source node to ATen node mapping:
#    => eq_scalar_5, expand_default_5, logical_not_default_5, npu_fusion_attention_v3_default_5, permute_default_15, permute_default_16, permute_default_17, view_default_5
#   attention_mask => full
#   linear => view_1
#   linear_1 => view_4
#   linear_2 => view_7
#   view => view_2
#   view_1 => view_5
#   view_2 => view_8
# Graph fragment:
#   %view_1 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, -1, 12, 64]), kwargs = {})
#   %permute_default_15 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 128, 768]), kwargs = {})
#   %view_5 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [128, -1, 12, 64]), kwargs = {})
#   %permute_default_16 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %view_7 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [128, 128, 768]), kwargs = {})
#   %view_8 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [128, -1, 12, 64]), kwargs = {})
#   %permute_default_17 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %full : Tensor "f32[128, 128][128, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %eq_scalar_5 : Tensor "b8[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%full, 1), kwargs = {})
#   %view_default_5 : Tensor "b8[128, 1, 1, 128][128, 128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq_scalar_5, [128, 1, 1, 128]), kwargs = {})
#   %expand_default_5 : Tensor "b8[128, 12, 128, 128][128, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_default_5, [128, 12, 128, 128]), kwargs = {})
#   %logical_not_default_5 : Tensor "b8[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%expand_default_5,), kwargs = {})
#   %npu_fusion_attention_v3_default_5 : [num_users=1] = call_function[target=torch.ops.npu.npu_fusion_attention_v3.default](args = (%permute_default_15, %permute_default_16, %permute_default_17, 12, BNSD, None, None, %logical_not_default_5, 0.125, 1.0, 2147483647, 2147483647, 2), kwargs = {})
#   return %buf11
triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1 = async_compile.triton('triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1', '''
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
    size_hints={'x': 25165824}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [25165824, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 25165824, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 25165824
    x_g_tile0 : tl.constexpr = (25165824) if (25165824) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 25165824
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
        tmp0 = tl.full([1], 1.0, tl.float32)
        tmp1 = tmp0 == tmp0
        tmp2 = tmp1 == 0
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/34/c34at4lcchqynweqenepsw7dhflgtgl4gmytuszynd3zgzvs4vz2.py
# Topologically Sorted Source Nodes: [transpose_4, contiguous], Original ATen: [aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_5
#   transpose_4 => permute_7
# Graph fragment:
#   %getitem_33 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0" = PlaceHolder[target=getitem_33]
#   %permute_7 : Tensor "f32[128, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_33, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_unk_fused_clone_transpose_2 = async_compile.triton('triton_unk_fused_clone_transpose_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, 128, 128, {'divisors': [1, 64, 768, 98304]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 128, 'divisor': 98304, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_transpose_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_transpose_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 12582912
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (12) if (12) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (128) if (128) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (128) if (128) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 12
    x2numel : tl.constexpr = 128
    x3numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 8192*x1 + 98304*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 64*x1 + 768*x2 + 98304*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/xz/cxzyulowjsjzjdoq5oatqnq4s6d3z67vxtzf7lbgvf22yjjyxeoi.py
# Topologically Sorted Source Nodes: [, context_2, add_1, sa_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_18
#   add_1 => add_3
#   context_2 => view_18
#   sa_output => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %arg13_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg13_1]
#   %mm_default_18 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_18]
#   %add_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=add_2]
#   %buf21 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf21]
#   %getitem_3 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_3]
#   %buf23 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf23]
#   %arg14_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg15_1]
#   %add_tensor_18 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, %mm_default_18), kwargs = {})
#   %view_18 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_18, [128, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %add_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg14_1), kwargs = {})
#   %add_5 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   return %buf21,%getitem_3,%buf23,%add_5
triton_unk_fused_add_addmm_native_layer_norm_view_3 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_3', '''
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
    size_hints={'x': 16384, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_3', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_3(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 16384
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (16384) if (16384) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 16384
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
            _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/qm/cqmkil67getjx2qbcjht7cbrrl6vlck7hzpvzmzrlas6mb2lpq6d.py
# Topologically Sorted Source Nodes: [, x, x_1], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#    => add_tensor_17
#   x => view_20
#   x_1 => add_6, add_7, div_2, exp_1, mul_4, mul_5, mul_6, mul_7
# Graph fragment:
#   %arg17_1 : Tensor "f32[3072][1]npu:0" = PlaceHolder[target=arg17_1]
#   %mm_default_17 : Tensor "f32[16384, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_default_17]
#   %add_tensor_17 : Tensor "f32[16384, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg17_1, %mm_default_17), kwargs = {})
#   %view_20 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_17, [128, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %view_20), kwargs = {})
#   %mul_5 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %view_20), kwargs = {})
#   %mul_6 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, -1.5957691216057308), kwargs = {})
#   %exp_1 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1.0), kwargs = {})
#   %div_2 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, %add_7), kwargs = {})
#   return %div_2
triton_unk_fused_addmm_gelu_view_4 = async_compile.triton('triton_unk_fused_addmm_gelu_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [3072, 16384, {'divisors': [1, 3072]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 3072, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 16384, 'divisor': 3072, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_gelu_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 50331648
    x_g_tile0 : tl.constexpr = (3072) if (3072) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (16384) if (16384) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 3072
    x1numel : tl.constexpr = 16384
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/2g/c2gpj6zcwet22mxnqsp2u5qrbfzp6l3p4eiium5akhh3iwf6gmjv.py
# Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => add_53, add_54, mul_54, mul_55, rsqrt_13, sub_19, var_mean_13
# Graph fragment:
#   %arg103_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg103_1]
#   %mm_default : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default]
#   %buf153 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf153]
#   %getitem_27 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_27]
#   %buf155 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf155]
#   %arg104_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg104_1]
#   %arg105_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg105_1]
#   %add_tensor : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg103_1, %mm_default), kwargs = {})
#   %view_139 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%div_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_19 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_27), kwargs = {})
#   %add_53 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-12), kwargs = {})
#   %rsqrt_13 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %mul_54 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_55 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %arg104_1), kwargs = {})
#   %add_54 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %arg105_1), kwargs = {})
#   return %buf153,%getitem_27,%buf155,%add_54
triton_unk_fused_addmm_gelu_native_layer_norm_view_5 = async_compile.triton('triton_unk_fused_addmm_gelu_native_layer_norm_view_5', '''
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
    size_hints={'x': 16384, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_gelu_native_layer_norm_view_5', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_addmm_gelu_native_layer_norm_view_5(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 16384
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (16384) if (16384) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 16384
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
            _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
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
            _tmp36 = tl.where(r0_mask, tmp37, _tmp36)
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/tv/ctvgwpidi27ffmj2vtdpjlojpqlpul3d5eqaul2e75qxgs67ukwv.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => full_default_6, ne, where_6
#   view_31 => view_143
# Graph fragment:
#   %arg107_1 : Tensor "i64[128, 128][128, 1]npu:0" = PlaceHolder[target=arg107_1]
#   %view_143 : Tensor "i64[16384][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%arg107_1, [-1]), kwargs = {})
#   %ne : Tensor "b8[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %full_default_6 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_6 : Tensor "i64[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_143, %full_default_6), kwargs = {})
#   return %where_6
triton_unk_fused_nll_loss_forward_view_6 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_6', '''
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 16384
    x_g_tile0 : tl.constexpr = (16384) if (16384) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 16384
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_inference/cache/76/c76nhy4thkltejv2fhxlpag7f2wlovgxjhgkjfony6hbvdibrbpa.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => convert_element_type, div_19, full_default_7, ne_1, ne_2, neg, squeeze, sum_7, sum_8, where_7
#   view_31 => view_143
# Graph fragment:
#   %arg107_1 : Tensor "i64[128, 128][128, 1]npu:0" = PlaceHolder[target=arg107_1]
#   %gather : Tensor "f32[16384, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %sum_8 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_8]
#   %sum_7 : Tensor "i64[][]npu:0" = PlaceHolder[target=sum_7]
#   %view_143 : Tensor "i64[16384][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%arg107_1, [-1]), kwargs = {})
#   %ne_1 : Tensor "b8[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %squeeze : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_7 : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_7), kwargs = {})
#   %sum_8 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %ne_2 : Tensor "b8[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %sum_7 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_7, torch.float32), kwargs = {})
#   %div_19 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_8, %convert_element_type), kwargs = {})
#   return %sum_8,%sum_7,%div_19
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

@npu_triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [16384, {'divisors': [1]}]}, 'axis_hints': [{'name': 'r0_0', 'length': 16384, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1
    r0_numel = 16384
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
            triton_unk_fused_add_native_layer_norm_0.run(buf7, buf3, arg4_1, arg5_1, 16384, 768, stream=raw_stream0)
            del arg4_1
            del arg5_1
            del buf3
            buf8 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:3
            extern_kernels.addmm(arg7_1, reinterpret_tensor(buf7, (16384, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
            del arg6_1
            del arg7_1
            buf9 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:4
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf7, (16384, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
            del arg8_1
            del arg9_1
            buf10 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf7, (16384, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
            del arg10_1
            del arg11_1
            buf11 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.bool)
            # Topologically Sorted Source Nodes: [linear, view, , linear_1, view_1, linear_2, view_2, attention_mask], Original ATen: [aten.view, aten.transpose, aten.ones, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf11, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [linear, view, , linear_1, view_1, linear_2, view_2, attention_mask], Original ATen: [aten.view, aten.transpose, aten.ones, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:6
            buf12 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf8, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf9, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf10, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf11, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf10
            buf13 = buf12[0]
            assert_alignment(buf13, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf12
            buf19 = reinterpret_tensor(buf9, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, contiguous], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf13, buf19, 12582912, stream=raw_stream0)
            buf20 = reinterpret_tensor(buf13, (16384, 768), (768, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, contiguous, context_1, context_2, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf19, (16384, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf20)
            del arg12_1
            buf24 = reinterpret_tensor(buf20, (128, 128, 768), (98304, 768, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [, context_2, add_1, sa_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf24, arg13_1, buf7, arg14_1, arg15_1, 16384, 768, stream=raw_stream0)
            del arg13_1
            del arg14_1
            del arg15_1
            buf25 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf24, (16384, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf25)
            del arg16_1
            buf26 = reinterpret_tensor(buf25, (128, 128, 3072), (393216, 3072, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [, x, x_1], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf26, arg17_1, 50331648, stream=raw_stream0)
            del arg17_1
            buf27 = reinterpret_tensor(buf7, (16384, 768), (768, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [, x, x_1, x_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(reinterpret_tensor(buf26, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf27)
            del arg18_1
            buf31 = reinterpret_tensor(buf27, (128, 128, 768), (98304, 768, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [, x_2, add_2, ffn_output], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf31, arg19_1, buf24, arg20_1, arg21_1, 16384, 768, stream=raw_stream0)
            del arg19_1
            del arg20_1
            del arg21_1
            buf32 = reinterpret_tensor(buf24, (16384, 768), (768, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:10
            extern_kernels.addmm(arg23_1, reinterpret_tensor(buf31, (16384, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
            del arg22_1
            del arg23_1
            buf33 = reinterpret_tensor(buf19, (16384, 768), (768, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:11
            extern_kernels.addmm(arg25_1, reinterpret_tensor(buf31, (16384, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
            del arg24_1
            del arg25_1
            buf34 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(arg27_1, reinterpret_tensor(buf31, (16384, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf34)
            del arg26_1
            del arg27_1
            buf35 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, linear_6, view_5, , linear_7, view_6, linear_8, view_7], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf35, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attention_mask, linear_6, view_5, , linear_7, view_6, linear_8, view_7], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:13
            buf36 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf32, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf33, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf34, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf35, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf32
            buf37 = buf36[0]
            assert_alignment(buf37, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf36
            buf43 = reinterpret_tensor(buf34, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [transpose_9, contiguous_1], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf37, buf43, 12582912, stream=raw_stream0)
            buf44 = reinterpret_tensor(buf37, (16384, 768), (768, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [transpose_9, contiguous_1, context_4, context_5, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf43, (16384, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), out=buf44)
            del arg28_1
            buf48 = reinterpret_tensor(buf44, (128, 128, 768), (98304, 768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [, context_5, add_3, sa_output_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf48, arg29_1, buf31, arg30_1, arg31_1, 16384, 768, stream=raw_stream0)
            del arg29_1
            del arg30_1
            del arg31_1
            buf49 = reinterpret_tensor(buf26, (16384, 3072), (3072, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [x_4, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(reinterpret_tensor(buf48, (16384, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 3072), (1, 768), 0), out=buf49)
            del arg32_1
            buf50 = reinterpret_tensor(buf49, (128, 128, 3072), (393216, 3072, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [, x_4, x_5], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf50, arg33_1, 50331648, stream=raw_stream0)
            del arg33_1
            buf51 = reinterpret_tensor(buf31, (16384, 768), (768, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [, x_4, x_5, x_6], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf50, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf51)
            del arg34_1
            buf55 = reinterpret_tensor(buf51, (128, 128, 768), (98304, 768, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [, x_6, add_4, ffn_output_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf55, arg35_1, buf48, arg36_1, arg37_1, 16384, 768, stream=raw_stream0)
            del arg35_1
            del arg36_1
            del arg37_1
            buf56 = reinterpret_tensor(buf48, (16384, 768), (768, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:17
            extern_kernels.addmm(arg39_1, reinterpret_tensor(buf55, (16384, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf56)
            del arg38_1
            del arg39_1
            buf57 = reinterpret_tensor(buf43, (16384, 768), (768, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:18
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf55, (16384, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf57)
            del arg40_1
            del arg41_1
            buf58 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:19
            extern_kernels.addmm(arg43_1, reinterpret_tensor(buf55, (16384, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf58)
            del arg42_1
            del arg43_1
            buf59 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, linear_12, view_10, , linear_13, view_11, linear_14, view_12], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf59, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attention_mask, linear_12, view_10, , linear_13, view_11, linear_14, view_12], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:20
            buf60 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf56, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf57, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf58, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf59, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf56
            buf61 = buf60[0]
            assert_alignment(buf61, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf60
            buf67 = reinterpret_tensor(buf58, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [transpose_14, contiguous_2], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf61, buf67, 12582912, stream=raw_stream0)
            buf68 = reinterpret_tensor(buf61, (16384, 768), (768, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [transpose_14, contiguous_2, context_7, context_8, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf67, (16384, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf68)
            del arg44_1
            buf72 = reinterpret_tensor(buf68, (128, 128, 768), (98304, 768, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [, context_8, add_5, sa_output_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf72, arg45_1, buf55, arg46_1, arg47_1, 16384, 768, stream=raw_stream0)
            del arg45_1
            del arg46_1
            del arg47_1
            buf73 = reinterpret_tensor(buf50, (16384, 3072), (3072, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [x_8, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(reinterpret_tensor(buf72, (16384, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf73)
            del arg48_1
            buf74 = reinterpret_tensor(buf73, (128, 128, 3072), (393216, 3072, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [, x_8, x_9], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf74, arg49_1, 50331648, stream=raw_stream0)
            del arg49_1
            buf75 = reinterpret_tensor(buf55, (16384, 768), (768, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [, x_8, x_9, x_10], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:23
            extern_kernels.mm(reinterpret_tensor(buf74, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf75)
            del arg50_1
            buf79 = reinterpret_tensor(buf75, (128, 128, 768), (98304, 768, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [, x_10, add_6, ffn_output_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf79, arg51_1, buf72, arg52_1, arg53_1, 16384, 768, stream=raw_stream0)
            del arg51_1
            del arg52_1
            del arg53_1
            buf80 = reinterpret_tensor(buf72, (16384, 768), (768, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:24
            extern_kernels.addmm(arg55_1, reinterpret_tensor(buf79, (16384, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf80)
            del arg54_1
            del arg55_1
            buf81 = reinterpret_tensor(buf67, (16384, 768), (768, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(arg57_1, reinterpret_tensor(buf79, (16384, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf81)
            del arg56_1
            del arg57_1
            buf82 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:26
            extern_kernels.addmm(arg59_1, reinterpret_tensor(buf79, (16384, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf82)
            del arg58_1
            del arg59_1
            buf83 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, linear_18, view_15, , linear_19, view_16, linear_20, view_17], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf83, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attention_mask, linear_18, view_15, , linear_19, view_16, linear_20, view_17], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:27
            buf84 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf80, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf81, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf82, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf83, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf80
            buf85 = buf84[0]
            assert_alignment(buf85, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf84
            buf91 = reinterpret_tensor(buf82, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, contiguous_3], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf85, buf91, 12582912, stream=raw_stream0)
            buf92 = reinterpret_tensor(buf85, (16384, 768), (768, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, contiguous_3, context_10, context_11, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:28
            extern_kernels.mm(reinterpret_tensor(buf91, (16384, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), out=buf92)
            del arg60_1
            buf96 = reinterpret_tensor(buf92, (128, 128, 768), (98304, 768, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [, context_11, add_7, sa_output_3], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf96, arg61_1, buf79, arg62_1, arg63_1, 16384, 768, stream=raw_stream0)
            del arg61_1
            del arg62_1
            del arg63_1
            buf97 = reinterpret_tensor(buf74, (16384, 3072), (3072, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [x_12, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:29
            extern_kernels.mm(reinterpret_tensor(buf96, (16384, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 3072), (1, 768), 0), out=buf97)
            del arg64_1
            buf98 = reinterpret_tensor(buf97, (128, 128, 3072), (393216, 3072, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [, x_12, x_13], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf98, arg65_1, 50331648, stream=raw_stream0)
            del arg65_1
            buf99 = reinterpret_tensor(buf79, (16384, 768), (768, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [, x_12, x_13, x_14], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf98, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf99)
            del arg66_1
            buf103 = reinterpret_tensor(buf99, (128, 128, 768), (98304, 768, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [, x_14, add_8, ffn_output_3], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf103, arg67_1, buf96, arg68_1, arg69_1, 16384, 768, stream=raw_stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            buf104 = reinterpret_tensor(buf96, (16384, 768), (768, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:31
            extern_kernels.addmm(arg71_1, reinterpret_tensor(buf103, (16384, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf104)
            del arg70_1
            del arg71_1
            buf105 = reinterpret_tensor(buf91, (16384, 768), (768, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:32
            extern_kernels.addmm(arg73_1, reinterpret_tensor(buf103, (16384, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
            del arg72_1
            del arg73_1
            buf106 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:33
            extern_kernels.addmm(arg75_1, reinterpret_tensor(buf103, (16384, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf106)
            del arg74_1
            del arg75_1
            buf107 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, linear_24, view_20, , linear_25, view_21, linear_26, view_22], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf107, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attention_mask, linear_24, view_20, , linear_25, view_21, linear_26, view_22], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:34
            buf108 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf104, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf105, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf106, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf107, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf104
            buf109 = buf108[0]
            assert_alignment(buf109, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf108
            buf115 = reinterpret_tensor(buf106, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, contiguous_4], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf109, buf115, 12582912, stream=raw_stream0)
            buf116 = reinterpret_tensor(buf109, (16384, 768), (768, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, contiguous_4, context_13, context_14, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:35
            extern_kernels.mm(reinterpret_tensor(buf115, (16384, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf116)
            del arg76_1
            buf120 = reinterpret_tensor(buf116, (128, 128, 768), (98304, 768, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [, context_14, add_9, sa_output_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf120, arg77_1, buf103, arg78_1, arg79_1, 16384, 768, stream=raw_stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            buf121 = reinterpret_tensor(buf98, (16384, 3072), (3072, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_16, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:36
            extern_kernels.mm(reinterpret_tensor(buf120, (16384, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf121)
            del arg80_1
            buf122 = reinterpret_tensor(buf121, (128, 128, 3072), (393216, 3072, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [, x_16, x_17], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf122, arg81_1, 50331648, stream=raw_stream0)
            del arg81_1
            buf123 = reinterpret_tensor(buf103, (16384, 768), (768, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [, x_16, x_17, x_18], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf122, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf123)
            del arg82_1
            buf127 = reinterpret_tensor(buf123, (128, 128, 768), (98304, 768, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [, x_18, add_10, ffn_output_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf127, arg83_1, buf120, arg84_1, arg85_1, 16384, 768, stream=raw_stream0)
            del arg83_1
            del arg84_1
            del arg85_1
            buf128 = reinterpret_tensor(buf120, (16384, 768), (768, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:38
            extern_kernels.addmm(arg87_1, reinterpret_tensor(buf127, (16384, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf128)
            del arg86_1
            del arg87_1
            buf129 = reinterpret_tensor(buf115, (16384, 768), (768, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:39
            extern_kernels.addmm(arg89_1, reinterpret_tensor(buf127, (16384, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf129)
            del arg88_1
            del arg89_1
            buf130 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:40
            extern_kernels.addmm(arg91_1, reinterpret_tensor(buf127, (16384, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf130)
            del arg90_1
            del arg91_1
            buf131 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, linear_30, view_25, , linear_31, view_26, linear_32, view_27], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_eq_expand_logical_not_npu_fusion_attention_v3_ones_transpose_view_1.run(buf131, 25165824, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attention_mask, linear_30, view_25, , linear_31, view_26, linear_32, view_27], Original ATen: [aten.ones, aten.view, aten.transpose, aten.eq, aten.expand, aten.logical_not, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:41
            buf132 = torch.ops.npu.npu_fusion_attention_v3.default(reinterpret_tensor(buf128, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf129, (128, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf130, (128, 12, 128, 64), (98304, 64, 768, 1), 0), 12, 'BNSD', None, None, buf131, 0.125, 1.0, 2147483647, 2147483647, 2, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf128
            del buf129
            del buf131
            buf133 = buf132[0]
            assert_alignment(buf133, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf132
            buf139 = reinterpret_tensor(buf130, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [transpose_29, contiguous_5], Original ATen: [aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_transpose_2.run(buf133, buf139, 12582912, stream=raw_stream0)
            buf140 = reinterpret_tensor(buf133, (16384, 768), (768, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [transpose_29, contiguous_5, context_16, context_17, ], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf139, (16384, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf140)
            del arg92_1
            del buf139
            buf144 = reinterpret_tensor(buf140, (128, 128, 768), (98304, 768, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [, context_17, add_11, sa_output_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf144, arg93_1, buf127, arg94_1, arg95_1, 16384, 768, stream=raw_stream0)
            del arg93_1
            del arg94_1
            del arg95_1
            buf145 = reinterpret_tensor(buf122, (16384, 3072), (3072, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [x_20, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:43
            extern_kernels.mm(reinterpret_tensor(buf144, (16384, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf145)
            del arg96_1
            buf146 = reinterpret_tensor(buf145, (128, 128, 3072), (393216, 3072, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [, x_20, x_21], Original ATen: [aten.addmm, aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_view_4.run(buf146, arg97_1, 50331648, stream=raw_stream0)
            del arg97_1
            buf147 = reinterpret_tensor(buf127, (16384, 768), (768, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [, x_20, x_21, x_22], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:44
            extern_kernels.mm(reinterpret_tensor(buf146, (16384, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf147)
            del arg98_1
            del buf146
            buf151 = reinterpret_tensor(buf147, (128, 128, 768), (98304, 768, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [, x_22, add_12, ffn_output_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_3.run(buf151, arg99_1, buf144, arg100_1, arg101_1, 16384, 768, stream=raw_stream0)
            del arg100_1
            del arg101_1
            del arg99_1
            buf152 = reinterpret_tensor(buf144, (16384, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [, x_22, add_12, ffn_output_5, prediction_logits], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:45
            extern_kernels.mm(reinterpret_tensor(buf151, (16384, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf152)
            del arg102_1
            del buf151
            buf156 = reinterpret_tensor(buf152, (128, 128, 768), (98304, 768, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_gelu_native_layer_norm_view_5.run(buf156, arg103_1, arg104_1, arg105_1, 16384, 768, stream=raw_stream0)
            del arg103_1
            del arg104_1
            del arg105_1
            buf157 = empty_strided_npu((16384, 30522), (30522, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, prediction_logits, prediction_logits_1, prediction_logits_2, prediction_logits_3], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:46
            extern_kernels.addmm(arg106_1, reinterpret_tensor(buf156, (16384, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf157)
            del arg106_1
            del arg1_1
            del buf156
            # Topologically Sorted Source Nodes: [prediction_logits_3, view_30, mlm_loss], Original ATen: [aten.view, aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:47
            buf158 = torch.ops.aten._log_softmax.default(buf157, 1, False)
            buf159 = buf158
            assert_alignment(buf159, 16, 'torch.ops.aten._log_softmax.default')
            del buf158
            arg107_1 = copy_if_misaligned(arg107_1)
            buf160 = empty_strided_npu((16384, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_6.run(arg107_1, buf160, 16384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:48
            buf161 = torch.ops.aten.gather.default(buf159, 1, reinterpret_tensor(buf160, (16384, 1), (1, 0), 0))
            del buf159
            del buf160
            buf162 = buf161
            assert_alignment(buf162, 16, 'torch.ops.aten.gather.default')
            del buf161
            buf163 = empty_strided_npu((), (), torch.float32)
            buf165 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_7.run(buf165, arg107_1, buf162, 1, 16384, stream=raw_stream0)
            del arg107_1
            del buf162
        return (buf165, reinterpret_tensor(buf157, (128, 128, 30522), (3906816, 30522, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, 128), (128, 1), device='npu:0', dtype=torch.int64)
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
    arg107_1 = rand_strided((128, 128), (128, 1), device='npu:0', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('DistilBertForMaskedLM', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
