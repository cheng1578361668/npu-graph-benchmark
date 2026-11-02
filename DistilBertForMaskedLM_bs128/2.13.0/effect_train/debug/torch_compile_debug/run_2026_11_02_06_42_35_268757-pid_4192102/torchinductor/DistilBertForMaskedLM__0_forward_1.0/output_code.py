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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/yu/cyulmvxwjtdakqeuscxs6f25osd2ibyfujulj6cvzrf44yx567lo.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => full_default_6, ne, where_6
#   view_31 => view_143
# Graph fragment:
#   %primals_108 : Tensor "i64[128, 128][128, 1]npu:0" = PlaceHolder[target=primals_108]
#   %view_143 : Tensor "i64[16384][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %ne : Tensor "b8[16384][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %full_default_6 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_6 : Tensor "i64[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_143, %full_default_6), kwargs = {})
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/ln/clntwci7fwj2qltptyaiq6g3eopiy2pacbdnxd3fmkx6pf6lp4df.py
# Topologically Sorted Source Nodes: [embeddings, embeddings_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1, add_2, getitem_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf4 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf5]
#   %buf6 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf6]
#   %rsqrt : Tensor "f32[128, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=rsqrt]
#   %primals_5 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_5]
#   %primals_6 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_6]
#   %add : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %getitem_1 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=operator.getitem](args = (%var_mean, 1), kwargs = {})
#   %add_1 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_5), kwargs = {})
#   %add_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   return %buf4,%buf5,%buf6,%rsqrt,%add_2
triton_unk_fused_add_native_layer_norm_1 = async_compile.triton('triton_unk_fused_add_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [128, 128, {'divisors': [1, 128]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 128, 'seed': 2}, {'name': 'r0_2', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 2)[:, :, None]
        tmp6 = tl.full([1, 1, 1], 768.0, tl.float32)
        tmp7 = (tmp4 / tmp6)
        tl.store(in_out_ptr0 + (x3), tmp7, x3mask)
        _tmp14 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp8 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10 - tmp7
            tmp12 = tmp11 * tmp11
            tmp13 = tl.broadcast_to(tmp12, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp15 = _tmp14 + tmp13
            _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp14 = tl.sum(_tmp14, 2)[:, :, None]
        tmp16 = tl.full([1, 1, 1], 768.0, tl.float32)
        tmp17 = (tmp14 / tmp16)
        tmp18 = tl.full([1, 1, 1], 1e-12, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.rsqrt(tmp19)
        tl.store(in_out_ptr1 + (x3), tmp20, x3mask)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp21 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 98304*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp22 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp26 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp23 = tmp21 + tmp22
            tmp24 = tmp23 - tmp7
            tmp25 = tmp24 * tmp20
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(out_ptr0 + (r0_2 + 768*x0 + 98304*x1), tmp29, r0_mask & x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/dx/cdxg3rzzvxveu4poft4og2ieuz6xc66ehdgn7vqrt4y226augzmb.py
# Topologically Sorted Source Nodes: [, linear, view, q, q_1, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
# Source node to ATen node mapping:
#    => add_tensor_23
#   linear => view_1
#   q => permute_1
#   q_1 => div
#   scores => clone
#   view => view_2
# Graph fragment:
#   %primals_8 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_8]
#   %mm_default_23 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_23]
#   %add_tensor_23 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_8, %mm_default_23), kwargs = {})
#   %view_1 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [128, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %div : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_1, 8.0), kwargs = {})
#   %clone : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_unk_fused_addmm_div_matmul_transpose_view_2 = async_compile.triton('triton_unk_fused_addmm_div_matmul_transpose_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 128, 12, 128, {'divisors': [1, 64, 8192, 98304]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 8192, 'seed': 2}, {'name': 'x3', 'length': 128, 'divisor': 98304, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_div_matmul_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_div_matmul_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 12582912
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (128) if (128) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (128) if (128) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 128
    x2numel : tl.constexpr = 12
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2), x0mask & x2mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + 64*x2 + 768*x1 + 98304*x3), x0mask & x1mask & x2mask & x3mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1, 1, 1], 0.125, tl.float32)
        tmp4 = tmp2 * tmp3
        tl.store(out_ptr0 + (x0 + 64*x1 + 8192*x2 + 98304*x3), tmp4, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/px/cpx2ykvqwzr4xbbggwzcrpa26ankf67xsub3qvdd7zpr7db557fo.py
# Topologically Sorted Source Nodes: [, linear, view, q, q_1, permute_257, clone_68], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
# Source node to ATen node mapping:
#    => add_tensor_23
#   clone_68 => clone_68
#   linear => view_1
#   permute_257 => permute_257
#   q => permute_1
#   q_1 => div
#   view => view_2
# Graph fragment:
#   %primals_8 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_8]
#   %mm_default_23 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_23]
#   %add_tensor_23 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_8, %mm_default_23), kwargs = {})
#   %view_1 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [128, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %div : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_1, 8.0), kwargs = {})
#   %permute_257 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 1, 3, 2]), kwargs = {})
#   %clone_68 : Tensor "f32[128, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_76,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_68
triton_unk_fused_addmm_div_matmul_backward_transpose_view_3 = async_compile.triton('triton_unk_fused_addmm_div_matmul_backward_transpose_view_3', '''
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
    size_hints={'y': 98304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 128, {'divisors': [1, 768]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 128, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_div_matmul_backward_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_div_matmul_backward_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 98304
    xnumel = 128
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (128) if (128) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 768
    y1numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0), y0mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (y0 + 768*x2 + 98304*y1), x2mask & y0mask & y1mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1, 1], 0.125, tl.float32)
        tmp4 = tmp2 * tmp3
        tl.store(out_ptr0 + (x2 + 128*y0 + 98304*y1), tmp4, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/vc/cvctzwe56j6ninx23jixfyim24ezbwxu5kvsw437wmns2izr7ydl.py
# Topologically Sorted Source Nodes: [, linear_1, view_1, k, transpose_3, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#    => add_tensor_22
#   k => permute_3
#   linear_1 => view_4
#   scores => clone_1
#   transpose_3 => permute_6
#   view_1 => view_5
# Graph fragment:
#   %primals_10 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_10]
#   %mm_default_22 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_22]
#   %add_tensor_22 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_10, %mm_default_22), kwargs = {})
#   %view_4 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [128, 128, 768]), kwargs = {})
#   %view_5 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [128, -1, 12, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %permute_6 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_3, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[128, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_addmm_matmul_transpose_view_4 = async_compile.triton('triton_unk_fused_addmm_matmul_transpose_view_4', '''
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
    size_hints={'y': 98304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 128, {'divisors': [1, 768]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 128, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_matmul_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_matmul_transpose_view_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 98304
    xnumel = 128
    y_g_tile0 : tl.constexpr = (768) if (768) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (128) if (128) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 768
    y1numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0), y0mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (y0 + 768*x2 + 98304*y1), x2mask & y0mask & y1mask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x2 + 128*y0 + 98304*y1), tmp2, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/3n/c3nr3xnuyeozalk5g5di5rylpvsed5vc3y6zbsl2lnfajdcvauva.py
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
#   %bmm : Tensor "f32[1536, 128, 128][16384, 128, 1]npu:0" = PlaceHolder[target=bmm]
#   %amax : Tensor "f32[128, 12, 128, 1][1536, 128, 1, 196608]npu:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[128, 12, 128, 1][1536, 128, 1, 196608]npu:0" = PlaceHolder[target=sum_1]
#   %full : Tensor "f32[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_11 : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 12, 128, 128]), kwargs = {})
#   %eq : Tensor "b8[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%full, 0), kwargs = {})
#   %view_12 : Tensor "b8[128, 1, 1, 128][128, 128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [128, 1, 1, 128]), kwargs = {})
#   %expand_2 : Tensor "b8[128, 12, 128, 128][128, 0, 0, 1]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view_12, [128, 12, 128, 128]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %view_11), kwargs = {})
#   %amax : Tensor "f32[128, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[128, 12, 128, 1][1536, 128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
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
    size_hints={'x': 196608, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [196608, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 196608, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 196608
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (196608) if (196608) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 196608
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
            _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
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
            _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/ep/cepze2ows72fuglc4yl5a2qfvaoyvbt7gwozkkmkv3axeiyfdy7t.py
# Topologically Sorted Source Nodes: [, linear_1, view_1, k, transpose_3, permute_256, clone_67], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
# Source node to ATen node mapping:
#    => add_tensor_22
#   clone_67 => clone_67
#   k => permute_3
#   linear_1 => view_4
#   permute_256 => permute_256
#   transpose_3 => permute_6
#   view_1 => view_5
# Graph fragment:
#   %primals_10 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_10]
#   %mm_default_22 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_22]
#   %add_tensor_22 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_10, %mm_default_22), kwargs = {})
#   %view_4 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [128, 128, 768]), kwargs = {})
#   %view_5 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [128, -1, 12, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %permute_6 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_3, [0, 1, 3, 2]), kwargs = {})
#   %permute_256 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_6, [0, 1, 3, 2]), kwargs = {})
#   %clone_67 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_75,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_67
triton_unk_fused_addmm_matmul_backward_transpose_view_6 = async_compile.triton('triton_unk_fused_addmm_matmul_backward_transpose_view_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 128, 12, 128, {'divisors': [1, 64, 8192, 98304]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 128, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 8192, 'seed': 2}, {'name': 'x3', 'length': 128, 'divisor': 98304, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_matmul_backward_transpose_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_matmul_backward_transpose_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 12582912
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (128) if (128) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (12) if (12) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (128) if (128) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 128
    x2numel : tl.constexpr = 12
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x2), x0mask & x2mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + 64*x2 + 768*x1 + 98304*x3), x0mask & x1mask & x2mask & x3mask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0 + 64*x1 + 8192*x2 + 98304*x3), tmp2, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/5m/c5mb3eoubnhpcyq37w7prpyvf4zx5u4e6bjwuzpw2oirys7b22qx.py
# Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   context => view_15
#   contiguous => clone_3
#   transpose_4 => permute_7
# Graph fragment:
#   %bmm_1 : Tensor "f32[1536, 128, 64][8192, 64, 1]npu:0" = PlaceHolder[target=bmm_1]
#   %view_15 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [128, 12, 128, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[128, 128, 12, 64][98304, 64, 8192, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_unk_fused_clone_matmul_transpose_7 = async_compile.triton('triton_unk_fused_clone_matmul_transpose_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_transpose_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_transpose_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/pd/cpdkdc5aceveg2zbqpq23es26ctlerss6532ptl6s6spilwm37jt.py
# Topologically Sorted Source Nodes: [permute_255], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   permute_255 => permute_255
# Graph fragment:
#   %expand_3 : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=expand_3]
#   %permute_255 : Tensor "f32[128, 12, 128, 128][196608, 16384, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_4, [0, 1, 3, 2]), kwargs = {})
#   return %expand_72
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
    size_hints={'y': 196608, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [128, 1536, {'divisors': [1, 128]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 1536, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 196608
    xnumel = 128
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (1536) if (1536) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 1536
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/yc/cyc3l54v7al627jgoxftc4p7srcg5gou4ssgoruxzhdwnricvj6d.py
# Topologically Sorted Source Nodes: [view_334], Original ATen: [aten.matmul_backward]
# Source node to ATen node mapping:
#   view_334 => view_334
# Graph fragment:
#   %expand_72 : Tensor "f32[128, 12, 128, 128][196608, 16384, 128, 1]npu:0" = PlaceHolder[target=expand_72]
#   %view_334 : Tensor "f32[1536, 128, 128][16384, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_72, [1536, 128, 128]), kwargs = {})
#   return %view_334
triton_unk_fused_matmul_backward_9 = async_compile.triton('triton_unk_fused_matmul_backward_9', '''
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
    size_hints={'y': 196608, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [128, 1536, {'divisors': [1, 128]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 1536, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_backward_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 196608
    xnumel = 128
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (1536) if (1536) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 1536
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/ef/cefiemuo2aykcyahjzjo7boseqwlixuqtkjqmpvb5ymxrbaiym42.py
# Topologically Sorted Source Nodes: [, context_2, add_1, sa_output, div_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_20
#   add_1 => add_3
#   context_2 => view_18
#   div_38 => div_38
#   sa_output => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %primals_14 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_14]
#   %mm_default_20 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_20]
#   %getitem_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=getitem_2]
#   %buf28 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf28]
#   %getitem_7 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_7]
#   %buf30 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf30]
#   %mul_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_2]
#   %primals_15 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_15]
#   %primals_16 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_16]
#   %add_tensor_20 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_14, %mm_default_20), kwargs = {})
#   %view_18 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [128, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %getitem_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_7), kwargs = {})
#   %mul_2 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_15), kwargs = {})
#   %add_5 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_16), kwargs = {})
#   %div_38 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 768), kwargs = {})
#   return %buf28,%getitem_7,%buf30,%mul_2,%add_5,%div_38
triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/5z/c5z6mb3q7k2pwqlfgh5d5nk2n7plyljlgdknfqwntblfahl3czow.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x => view_20
#   x_1 => add_6, add_7, div_2, exp_1, mul_4, mul_5, mul_6, mul_7
# Graph fragment:
#   %addmm_4 : Tensor "f32[16384, 3072][3072, 1]npu:0" = PlaceHolder[target=addmm_4]
#   %view_20 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [128, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %view_20), kwargs = {})
#   %mul_5 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %view_20), kwargs = {})
#   %mul_6 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, -1.5957691216057308), kwargs = {})
#   %exp_1 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1.0), kwargs = {})
#   %div_2 : Tensor "f32[128, 128, 3072][393216, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, %add_7), kwargs = {})
#   return %div_2
triton_unk_fused_gelu_view_11 = async_compile.triton('triton_unk_fused_gelu_view_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [50331648, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 50331648, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_view_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_gelu_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 50331648
    x_g_tile0 : tl.constexpr = (50331648) if (50331648) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 50331648
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/xa/cxayg4g7px2nof2gzpr2jqlsy6vcbmrwrconu3zg5etug7gyuil4.py
# Topologically Sorted Source Nodes: [add_2, ffn_output, div_37], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_2 => add_8
#   div_37 => div_37
#   ffn_output => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %getitem_8 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=getitem_8]
#   %add_5 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=add_5]
#   %buf39 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf39]
#   %getitem_11 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_11]
#   %buf41 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf41]
#   %mul_8 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_8]
#   %primals_21 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_21]
#   %primals_22 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_22]
#   %add_8 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, %add_5), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-12), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_11), kwargs = {})
#   %mul_8 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_21), kwargs = {})
#   %add_10 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_22), kwargs = {})
#   %div_37 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
#   return %buf39,%getitem_11,%buf41,%mul_8,%add_10,%div_37
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/c6/cc6xprud25nzak3sm2e64aozhe564vmqq4nmkl4wtaft3ylsal5k.py
# Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, q_11, scores_10, permute_92, clone_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#    => add_tensor_3
#   clone_28 => clone_28
#   linear_30 => view_116
#   permute_92 => permute_92
#   q_10 => permute_56
#   q_11 => div_15
#   scores_10 => clone_20
#   view_25 => view_117
# Graph fragment:
#   %primals_88 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_88]
#   %mm_default_3 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_3]
#   %add_tensor_3 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_88, %mm_default_3), kwargs = {})
#   %view_116 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [128, 128, 768]), kwargs = {})
#   %view_117 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_116, [128, -1, 12, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_117, [0, 2, 1, 3]), kwargs = {})
#   %div_15 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_56, 8.0), kwargs = {})
#   %clone_20 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_25,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_92 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div_15, [0, 1, 3, 2]), kwargs = {})
#   %clone_28 : Tensor "f32[128, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_36,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_20,%clone_28
triton_unk_fused_addmm_div_matmul_matmul_backward_transpose_view_13 = async_compile.triton('triton_unk_fused_addmm_div_matmul_matmul_backward_transpose_view_13', '''
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
    size_hints={'y': 98304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 128, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 128, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_div_matmul_matmul_backward_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_div_matmul_matmul_backward_transpose_view_13(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 98304
    xnumel = 128
    y_g_tile0 : tl.constexpr = (64) if (64) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (128) if (128) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 64
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 128
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0 + 64*y1), y0mask & y1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (y0 + 64*y1 + 768*x3 + 98304*y2), x3mask & y0mask & y1mask & y2mask)
        tmp5 = tl.load(in_ptr0 + (y4), y4mask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr1 + (y4 + 768*x3 + 98304*y2), x3mask & y2mask & y4mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1, 1, 1], 0.125, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7 * tmp3
        tl.store(out_ptr0 + (y0 + 64*x3 + 8192*y1 + 98304*y2), tmp4, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 128*y4 + 98304*y2), tmp8, x3mask & y2mask & y4mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/if/cifxtmcfu3jibl6f7bg3crmdc452iacovxegil4edlthyqdylvxw.py
# Topologically Sorted Source Nodes: [, linear_31, view_26, k_5, transpose_28, scores_10, permute_91, clone_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#    => add_tensor_2
#   clone_27 => clone_27
#   k_5 => permute_58
#   linear_31 => view_119
#   permute_91 => permute_91
#   scores_10 => clone_21
#   transpose_28 => permute_61
#   view_26 => view_120
# Graph fragment:
#   %primals_90 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_90]
#   %mm_default_2 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_2]
#   %add_tensor_2 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_90, %mm_default_2), kwargs = {})
#   %view_119 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [128, 128, 768]), kwargs = {})
#   %view_120 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_119, [128, -1, 12, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1, 3]), kwargs = {})
#   %permute_61 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_58, [0, 1, 3, 2]), kwargs = {})
#   %clone_21 : Tensor "f32[128, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_26,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_91 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_61, [0, 1, 3, 2]), kwargs = {})
#   %clone_27 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_35,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_21,%clone_27
triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_14 = async_compile.triton('triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_14', '''
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
    size_hints={'y': 196608, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [128, 12, 128, {'divisors': [1, 128, 1536]}], 'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 128, 'seed': 2}, {'name': 'y2', 'length': 128, 'divisor': 1536, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 196608
    xnumel = 64
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (128) if (128) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x3 + 64*y1), x3mask & y1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x3 + 64*y1 + 768*y0 + 98304*y2), x3mask & y0mask & y1mask & y2mask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (y0 + 128*x3 + 8192*y1 + 98304*y2), tmp2, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 64*y0 + 8192*y1 + 98304*y2), tmp2, x3mask & y0mask & y1mask & y2mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/nf/cnf5ur7n43rrnuo3i7vrtj53cb4dcmdgbxpdmlacvn2ybsnjxkjt.py
# Topologically Sorted Source Nodes: [, linear_32, view_27, v_5, context_15, permute_89, clone_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul, aten.matmul_backward]
# Source node to ATen node mapping:
#    => add_tensor_1
#   clone_25 => clone_25
#   context_15 => clone_22
#   linear_32 => view_122
#   permute_89 => permute_89
#   v_5 => permute_60
#   view_27 => view_123
# Graph fragment:
#   %primals_92 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_92]
#   %mm_default_1 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[16384, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_92, %mm_default_1), kwargs = {})
#   %view_122 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [128, 128, 768]), kwargs = {})
#   %view_123 : Tensor "f32[128, 128, 12, 64][98304, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_122, [128, -1, 12, 64]), kwargs = {})
#   %permute_60 : Tensor "f32[128, 12, 128, 64][98304, 64, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_123, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[128, 12, 128, 64][98304, 8192, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_29,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_89 : Tensor "f32[128, 12, 64, 128][98304, 64, 1, 768]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_60, [0, 1, 3, 2]), kwargs = {})
#   %clone_25 : Tensor "f32[128, 12, 64, 128][98304, 8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_31,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_22,%clone_25
triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_15 = async_compile.triton('triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_15', '''
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
    size_hints={'y': 98304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 128, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 128, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    ynumel = 98304
    xnumel = 128
    y_g_tile0 : tl.constexpr = (64) if (64) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (12) if (12) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (128) if (128) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 64
    y1numel : tl.constexpr = 12
    y2numel : tl.constexpr = 128
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (y0 + 64*y1), y0mask & y1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (y0 + 64*y1 + 768*x3 + 98304*y2), x3mask & y0mask & y1mask & y2mask)
        tmp3 = tl.load(in_ptr0 + (y4), y4mask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr1 + (y4 + 768*x3 + 98304*y2), x3mask & y2mask & y4mask)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tl.store(out_ptr0 + (y0 + 64*x3 + 8192*y1 + 98304*y2), tmp2, x3mask & y0mask & y1mask & y2mask)
        tl.store(out_ptr1 + (x3 + 128*y4 + 98304*y2), tmp5, x3mask & y2mask & y4mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/yf/cyfojm6vnpokzq55rmcgm2c3gwm45q56etgdkfkekwci5oeyniba.py
# Topologically Sorted Source Nodes: [add_12, ffn_output_5, div_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_12 => add_48
#   div_22 => div_22
#   ffn_output_5 => add_49, add_50, mul_48, mul_49, rsqrt_12, sub_18, var_mean_12
# Graph fragment:
#   %getitem_48 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=getitem_48]
#   %add_45 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=add_45]
#   %buf199 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf199]
#   %getitem_51 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=getitem_51]
#   %buf201 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf201]
#   %mul_48 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0" = PlaceHolder[target=mul_48]
#   %primals_101 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_101]
#   %primals_102 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_102]
#   %add_48 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, %add_45), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_48, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_49 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-12), kwargs = {})
#   %rsqrt_12 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %sub_18 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %getitem_51), kwargs = {})
#   %mul_48 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_12), kwargs = {})
#   %mul_49 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_101), kwargs = {})
#   %add_50 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %primals_102), kwargs = {})
#   %div_22 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_12, 768), kwargs = {})
#   return %buf199,%getitem_51,%buf201,%mul_48,%add_50,%div_22
triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_16 = async_compile.triton('triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_16', '''
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_16(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/f2/cf2yqsvjhied7tlrlqyff2a3yfnrbkrikjqca5ugvtg4ktentvc3.py
# Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   prediction_logits => view_139
#   prediction_logits_1 => add_51, add_52, div_18, exp_12, mul_50, mul_51, mul_52, mul_53
#   prediction_logits_2 => add_53, add_54, getitem_53, mul_54, mul_55, rsqrt_13, sub_19, var_mean_13
# Graph fragment:
#   %addmm_36 : Tensor "f32[16384, 768][768, 1]npu:0" = PlaceHolder[target=addmm_36]
#   %buf205 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf205]
#   %buf206 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf206]
#   %buf207 : Tensor "f32[128, 128, 1][128, 1, 16384]npu:0" = PlaceHolder[target=buf207]
#   %rsqrt_13 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0" = PlaceHolder[target=rsqrt_13]
#   %primals_105 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_105]
#   %primals_106 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=primals_106]
#   %view_139 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_36, [128, 128, 768]), kwargs = {})
#   %mul_50 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_139, %view_139), kwargs = {})
#   %mul_51 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %view_139), kwargs = {})
#   %mul_52 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, 0.044715), kwargs = {})
#   %add_51 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_139, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, -1.5957691216057308), kwargs = {})
#   %exp_12 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_53,), kwargs = {})
#   %add_52 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1.0), kwargs = {})
#   %div_18 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_139, %add_52), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%div_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %getitem_53 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=operator.getitem](args = (%var_mean_13, 1), kwargs = {})
#   %add_53 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-12), kwargs = {})
#   %rsqrt_13 : Tensor "f32[128, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %sub_19 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %getitem_53), kwargs = {})
#   %mul_54 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_13), kwargs = {})
#   %mul_55 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %primals_105), kwargs = {})
#   %add_54 : Tensor "f32[128, 128, 768][98304, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %primals_106), kwargs = {})
#   return %buf205,%buf206,%buf207,%rsqrt_13,%add_54
triton_unk_fused_gelu_native_layer_norm_view_17 = async_compile.triton('triton_unk_fused_gelu_native_layer_norm_view_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [16384, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16384, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_gelu_native_layer_norm_view_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_gelu_native_layer_norm_view_17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
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
            _tmp32 = tl.where(r0_mask, tmp33, _tmp32)
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/DistilBertForMaskedLM_bs128/2.13.0/effect_train/cache/pp/cppu5edep5u3rn7h4yps5gfhbtadbmrehklwzqvriqxdnnbnwe3h.py
# Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   mlm_loss => convert_element_type, div_19, full_default_7, ne, neg, squeeze, sum_7, sum_8, where_7
#   view_31 => view_143
# Graph fragment:
#   %primals_108 : Tensor "i64[128, 128][128, 1]npu:0" = PlaceHolder[target=primals_108]
#   %gather : Tensor "f32[16384, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %sum_7 : Tensor "i64[][]npu:0" = PlaceHolder[target=sum_7]
#   %sum_8 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_8]
#   %convert_element_type : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type]
#   %view_143 : Tensor "i64[16384][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_108, [-1]), kwargs = {})
#   %ne : Tensor "b8[16384][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_143, -100), kwargs = {})
#   %squeeze : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_7 : Tensor "f32[16384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_7), kwargs = {})
#   %sum_7 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_7, torch.float32), kwargs = {})
#   %sum_8 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %div_19 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_8, %convert_element_type), kwargs = {})
#   return %sum_7,%sum_8,%convert_element_type,%div_19
triton_unk_fused_nll_loss_forward_view_18 = async_compile.triton('triton_unk_fused_nll_loss_forward_view_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [16384, {'divisors': [1]}]}, 'axis_hints': [{'name': 'r0_0', 'length': 16384, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_nll_loss_forward_view_18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            buf213 = empty_strided_npu((16384, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_0.run(primals_108, buf213, 16384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [position_ids, position_embeddings], Original ATen: [aten.slice, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:1
            buf2 = torch.ops.aten.embedding.default(primals_4, reinterpret_tensor(primals_3, (1, 128), (512, 1), 0))
            del primals_4
            buf3 = buf2
            assert_alignment(buf3, 16, 'torch.ops.aten.embedding.default')
            del buf2
            primals_1 = copy_if_misaligned(primals_1)
            # Topologically Sorted Source Nodes: [input_embeds], Original ATen: [aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf0 = torch.ops.aten.embedding.default(primals_2, primals_1, 0)
            buf1 = buf0
            assert_alignment(buf1, 16, 'torch.ops.aten.embedding.default')
            del buf0
            buf4 = empty_strided_npu((128, 128, 1), (128, 1, 16384), torch.float32)
            buf5 = buf4; del buf4  # reuse
            buf6 = empty_strided_npu((128, 128, 1), (128, 1, 16384), torch.float32)
            buf7 = reinterpret_tensor(buf6, (128, 128, 1), (128, 1, 1), 0); del buf6  # reuse
            buf8 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embeddings, embeddings_1], Original ATen: [aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_1.run(buf5, buf7, buf1, buf3, primals_5, primals_6, buf8, 16384, 768, stream=raw_stream0)
            del primals_6
            # Topologically Sorted Source Nodes: [embeddings, embeddings_1, embeddings_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:3
            buf9 = torch.ops.npu._npu_dropout.default(buf8, 0.1)
            buf10 = buf9[0]
            assert_alignment(buf10, 16, 'torch.ops.npu._npu_dropout.default')
            buf11 = buf9[1]
            assert_alignment(buf11, 16, 'torch.ops.npu._npu_dropout.default')
            del buf9
            buf12 = reinterpret_tensor(buf8, (16384, 768), (768, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf10, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), out=buf12)
            buf13 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, linear_1, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf10, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), out=buf13)
            buf14 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, linear_2, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(reinterpret_tensor(buf10, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_11, (768, 768), (1, 768), 0), out=buf14)
            buf15 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, q_1, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_transpose_view_2.run(primals_8, buf12, buf15, 12582912, stream=raw_stream0)
            buf260 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, q_1, permute_257, clone_68], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_backward_transpose_view_3.run(primals_8, buf12, buf260, 98304, 128, stream=raw_stream0)
            del primals_8
            buf16 = reinterpret_tensor(buf12, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [, linear_1, view_1, k, transpose_3, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_10, buf13, buf16, 98304, 128, stream=raw_stream0)
            buf17 = empty_strided_npu((1536, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, view, q, linear_1, view_1, k, q_1, transpose_3, scores], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:7
            extern_kernels.bmm(reinterpret_tensor(buf15, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf16, (1536, 64, 128), (8192, 128, 1), 0), out=buf17)
            buf20 = reinterpret_tensor(buf17, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, scores, eq, view_3, mask, tensor, scores_1, weights], Original ATen: [aten.ones, aten.matmul, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf20, 196608, 128, stream=raw_stream0)
            buf259 = reinterpret_tensor(buf16, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [, linear_1, view_1, k, transpose_3, permute_256, clone_67], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_10, buf13, buf259, 12582912, stream=raw_stream0)
            del primals_10
            buf24 = reinterpret_tensor(buf13, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [, linear_2, view_2, v, context], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_12, buf14, buf24, 12582912, stream=raw_stream0)
            buf256 = reinterpret_tensor(buf15, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [, linear_2, view_2, v, permute_254, clone_65], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_12, buf14, buf256, 98304, 128, stream=raw_stream0)
            del primals_12
            # Unsorted Source Nodes: [weights_1], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:8
            buf21 = torch.ops.npu._npu_dropout.default(buf20, 0.1)
            buf22 = buf21[0]
            assert_alignment(buf22, 16, 'torch.ops.npu._npu_dropout.default')
            buf23 = buf21[1]
            assert_alignment(buf23, 16, 'torch.ops.npu._npu_dropout.default')
            del buf21
            buf25 = reinterpret_tensor(buf14, (1536, 128, 64), (8192, 64, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [, linear_2, view_2, v, context], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:9
            extern_kernels.bmm(reinterpret_tensor(buf22, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf24, (1536, 128, 64), (8192, 64, 1), 0), out=buf25)
            buf26 = reinterpret_tensor(buf24, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf25, buf26, 12582912, stream=raw_stream0)
            buf257 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_255], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf22, buf257, 196608, 128, stream=raw_stream0)
            buf258 = reinterpret_tensor(buf22, (1536, 128, 128), (16384, 1, 128), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [view_334], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf257, buf258, 196608, 128, stream=raw_stream0)
            buf27 = reinterpret_tensor(buf25, (16384, 768), (768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [context, transpose_4, contiguous, context_1, context_2, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf26, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 768), (1, 768), 0), out=buf27)
            buf31 = reinterpret_tensor(buf27, (128, 128, 768), (98304, 768, 1), 0); del buf27  # reuse
            buf32 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf255 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_2, add_1, sa_output, div_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf31, primals_14, buf10, primals_15, primals_16, buf32, buf255, 16384, 768, stream=raw_stream0)
            del primals_14
            del primals_16
            buf33 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output, x], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:11
            extern_kernels.addmm(primals_18, reinterpret_tensor(buf32, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf33)
            del primals_18
            buf34 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf33, buf34, 50331648, stream=raw_stream0)
            buf35 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(primals_20, reinterpret_tensor(buf34, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf35)
            del primals_20
            # Unsorted Source Nodes: [x_3, x_2], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:13
            buf36 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf35, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf37 = buf36[0]
            assert_alignment(buf37, 16, 'torch.ops.npu._npu_dropout.default')
            buf38 = buf36[1]
            assert_alignment(buf38, 16, 'torch.ops.npu._npu_dropout.default')
            del buf36
            buf42 = reinterpret_tensor(buf35, (128, 128, 768), (98304, 768, 1), 0); del buf35  # reuse
            buf43 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf254 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_2, ffn_output, div_37], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf37, buf32, primals_21, primals_22, buf42, buf43, buf254, 16384, 768, stream=raw_stream0)
            del primals_22
            buf44 = reinterpret_tensor(buf37, (16384, 768), (768, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf43, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), out=buf44)
            buf45 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, linear_7, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(reinterpret_tensor(buf43, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), out=buf45)
            buf46 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output, linear_6, linear_8, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf43, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_27, (768, 768), (1, 768), 0), out=buf46)
            buf47 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, q_3, scores_2], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_transpose_view_2.run(primals_24, buf44, buf47, 12582912, stream=raw_stream0)
            buf253 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, q_3, permute_224, clone_60], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_backward_transpose_view_3.run(primals_24, buf44, buf253, 98304, 128, stream=raw_stream0)
            del primals_24
            buf48 = reinterpret_tensor(buf44, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [, linear_7, view_6, k_1, transpose_8, scores_2], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_26, buf45, buf48, 98304, 128, stream=raw_stream0)
            buf49 = reinterpret_tensor(buf257, (1536, 128, 128), (16384, 128, 1), 0); del buf257  # reuse
            # Topologically Sorted Source Nodes: [, linear_6, view_5, q_2, linear_7, view_6, k_1, q_3, transpose_8, scores_2], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:17
            extern_kernels.bmm(reinterpret_tensor(buf47, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf48, (1536, 64, 128), (8192, 128, 1), 0), out=buf49)
            buf52 = reinterpret_tensor(buf49, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_2, scores_3, weights_2], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf52, 196608, 128, stream=raw_stream0)
            buf252 = reinterpret_tensor(buf48, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [, linear_7, view_6, k_1, transpose_8, permute_223, clone_59], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_26, buf45, buf252, 12582912, stream=raw_stream0)
            del primals_26
            buf56 = reinterpret_tensor(buf45, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [, linear_8, view_7, v_1, context_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_28, buf46, buf56, 12582912, stream=raw_stream0)
            buf249 = reinterpret_tensor(buf47, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [, linear_8, view_7, v_1, permute_221, clone_57], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_28, buf46, buf249, 98304, 128, stream=raw_stream0)
            del primals_28
            # Unsorted Source Nodes: [weights_3], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:18
            buf53 = torch.ops.npu._npu_dropout.default(buf52, 0.1)
            buf54 = buf53[0]
            assert_alignment(buf54, 16, 'torch.ops.npu._npu_dropout.default')
            buf55 = buf53[1]
            assert_alignment(buf55, 16, 'torch.ops.npu._npu_dropout.default')
            del buf53
            buf57 = reinterpret_tensor(buf46, (1536, 128, 64), (8192, 64, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [, linear_8, view_7, v_1, context_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:19
            extern_kernels.bmm(reinterpret_tensor(buf54, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf56, (1536, 128, 64), (8192, 64, 1), 0), out=buf57)
            buf58 = reinterpret_tensor(buf56, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf57, buf58, 12582912, stream=raw_stream0)
            buf250 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_222], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf54, buf250, 196608, 128, stream=raw_stream0)
            buf251 = reinterpret_tensor(buf54, (1536, 128, 128), (16384, 1, 128), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [view_300], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf250, buf251, 196608, 128, stream=raw_stream0)
            buf59 = reinterpret_tensor(buf57, (16384, 768), (768, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [context_3, transpose_9, contiguous_1, context_4, context_5, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(reinterpret_tensor(buf58, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 768), (1, 768), 0), out=buf59)
            buf63 = reinterpret_tensor(buf59, (128, 128, 768), (98304, 768, 1), 0); del buf59  # reuse
            buf64 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf248 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_5, add_3, sa_output_1, div_35], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf63, primals_30, buf43, primals_31, primals_32, buf64, buf248, 16384, 768, stream=raw_stream0)
            del primals_30
            del primals_32
            buf65 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_1, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:21
            extern_kernels.addmm(primals_34, reinterpret_tensor(buf64, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_33, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf65)
            del primals_34
            buf66 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf65, buf66, 50331648, stream=raw_stream0)
            buf67 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(primals_36, reinterpret_tensor(buf66, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf67)
            del primals_36
            # Unsorted Source Nodes: [x_7, x_6], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:23
            buf68 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf67, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf69 = buf68[0]
            assert_alignment(buf69, 16, 'torch.ops.npu._npu_dropout.default')
            buf70 = buf68[1]
            assert_alignment(buf70, 16, 'torch.ops.npu._npu_dropout.default')
            del buf68
            buf74 = reinterpret_tensor(buf67, (128, 128, 768), (98304, 768, 1), 0); del buf67  # reuse
            buf75 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf247 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_4, ffn_output_1, div_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf69, buf64, primals_37, primals_38, buf74, buf75, buf247, 16384, 768, stream=raw_stream0)
            del primals_38
            buf76 = reinterpret_tensor(buf69, (16384, 768), (768, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:24
            extern_kernels.mm(reinterpret_tensor(buf75, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), out=buf76)
            buf77 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, linear_13, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(reinterpret_tensor(buf75, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 768), (1, 768), 0), out=buf77)
            buf78 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_1, linear_12, linear_14, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:26
            extern_kernels.mm(reinterpret_tensor(buf75, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_43, (768, 768), (1, 768), 0), out=buf78)
            buf79 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, q_5, scores_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_transpose_view_2.run(primals_40, buf76, buf79, 12582912, stream=raw_stream0)
            buf246 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, q_5, permute_191, clone_52], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_backward_transpose_view_3.run(primals_40, buf76, buf246, 98304, 128, stream=raw_stream0)
            del primals_40
            buf80 = reinterpret_tensor(buf76, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [, linear_13, view_11, k_2, transpose_13, scores_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_42, buf77, buf80, 98304, 128, stream=raw_stream0)
            buf81 = reinterpret_tensor(buf250, (1536, 128, 128), (16384, 128, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [, linear_12, view_10, q_4, linear_13, view_11, k_2, q_5, transpose_13, scores_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:27
            extern_kernels.bmm(reinterpret_tensor(buf79, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf80, (1536, 64, 128), (8192, 128, 1), 0), out=buf81)
            buf84 = reinterpret_tensor(buf81, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_4, scores_5, weights_4], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf84, 196608, 128, stream=raw_stream0)
            buf245 = reinterpret_tensor(buf80, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [, linear_13, view_11, k_2, transpose_13, permute_190, clone_51], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_42, buf77, buf245, 12582912, stream=raw_stream0)
            del primals_42
            buf88 = reinterpret_tensor(buf77, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [, linear_14, view_12, v_2, context_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_44, buf78, buf88, 12582912, stream=raw_stream0)
            buf242 = reinterpret_tensor(buf79, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [, linear_14, view_12, v_2, permute_188, clone_49], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_44, buf78, buf242, 98304, 128, stream=raw_stream0)
            del primals_44
            # Unsorted Source Nodes: [weights_5], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:28
            buf85 = torch.ops.npu._npu_dropout.default(buf84, 0.1)
            buf86 = buf85[0]
            assert_alignment(buf86, 16, 'torch.ops.npu._npu_dropout.default')
            buf87 = buf85[1]
            assert_alignment(buf87, 16, 'torch.ops.npu._npu_dropout.default')
            del buf85
            buf89 = reinterpret_tensor(buf78, (1536, 128, 64), (8192, 64, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [, linear_14, view_12, v_2, context_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:29
            extern_kernels.bmm(reinterpret_tensor(buf86, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf88, (1536, 128, 64), (8192, 64, 1), 0), out=buf89)
            buf90 = reinterpret_tensor(buf88, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf89, buf90, 12582912, stream=raw_stream0)
            buf243 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_189], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf86, buf243, 196608, 128, stream=raw_stream0)
            buf244 = reinterpret_tensor(buf86, (1536, 128, 128), (16384, 1, 128), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [view_266], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf243, buf244, 196608, 128, stream=raw_stream0)
            buf91 = reinterpret_tensor(buf89, (16384, 768), (768, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [context_6, transpose_14, contiguous_2, context_7, context_8, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf90, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_45, (768, 768), (1, 768), 0), out=buf91)
            buf95 = reinterpret_tensor(buf91, (128, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
            buf96 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf241 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_8, add_5, sa_output_2, div_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf95, primals_46, buf75, primals_47, primals_48, buf96, buf241, 16384, 768, stream=raw_stream0)
            del primals_46
            del primals_48
            buf97 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_2, x_8], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:31
            extern_kernels.addmm(primals_50, reinterpret_tensor(buf96, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf97)
            del primals_50
            buf98 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf97, buf98, 50331648, stream=raw_stream0)
            buf99 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:32
            extern_kernels.addmm(primals_52, reinterpret_tensor(buf98, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_51, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf99)
            del primals_52
            # Unsorted Source Nodes: [x_11, x_10], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:33
            buf100 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf99, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf101 = buf100[0]
            assert_alignment(buf101, 16, 'torch.ops.npu._npu_dropout.default')
            buf102 = buf100[1]
            assert_alignment(buf102, 16, 'torch.ops.npu._npu_dropout.default')
            del buf100
            buf106 = reinterpret_tensor(buf99, (128, 128, 768), (98304, 768, 1), 0); del buf99  # reuse
            buf107 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf240 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_6, ffn_output_2, div_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf101, buf96, primals_53, primals_54, buf106, buf107, buf240, 16384, 768, stream=raw_stream0)
            del primals_54
            buf108 = reinterpret_tensor(buf101, (16384, 768), (768, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(reinterpret_tensor(buf107, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), out=buf108)
            buf109 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, linear_19, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:35
            extern_kernels.mm(reinterpret_tensor(buf107, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), out=buf109)
            buf110 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_2, linear_18, linear_20, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:36
            extern_kernels.mm(reinterpret_tensor(buf107, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_59, (768, 768), (1, 768), 0), out=buf110)
            buf111 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, q_7, scores_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_transpose_view_2.run(primals_56, buf108, buf111, 12582912, stream=raw_stream0)
            buf239 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, q_7, permute_158, clone_44], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_backward_transpose_view_3.run(primals_56, buf108, buf239, 98304, 128, stream=raw_stream0)
            del primals_56
            buf112 = reinterpret_tensor(buf108, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [, linear_19, view_16, k_3, transpose_18, scores_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_58, buf109, buf112, 98304, 128, stream=raw_stream0)
            buf113 = reinterpret_tensor(buf243, (1536, 128, 128), (16384, 128, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [, linear_18, view_15, q_6, linear_19, view_16, k_3, q_7, transpose_18, scores_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:37
            extern_kernels.bmm(reinterpret_tensor(buf111, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf112, (1536, 64, 128), (8192, 128, 1), 0), out=buf113)
            buf116 = reinterpret_tensor(buf113, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_6, scores_7, weights_6], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf116, 196608, 128, stream=raw_stream0)
            buf238 = reinterpret_tensor(buf112, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [, linear_19, view_16, k_3, transpose_18, permute_157, clone_43], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_58, buf109, buf238, 12582912, stream=raw_stream0)
            del primals_58
            buf120 = reinterpret_tensor(buf109, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [, linear_20, view_17, v_3, context_9], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_60, buf110, buf120, 12582912, stream=raw_stream0)
            buf235 = reinterpret_tensor(buf111, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [, linear_20, view_17, v_3, permute_155, clone_41], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_60, buf110, buf235, 98304, 128, stream=raw_stream0)
            del primals_60
            # Unsorted Source Nodes: [weights_7], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:38
            buf117 = torch.ops.npu._npu_dropout.default(buf116, 0.1)
            buf118 = buf117[0]
            assert_alignment(buf118, 16, 'torch.ops.npu._npu_dropout.default')
            buf119 = buf117[1]
            assert_alignment(buf119, 16, 'torch.ops.npu._npu_dropout.default')
            del buf117
            buf121 = reinterpret_tensor(buf110, (1536, 128, 64), (8192, 64, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [, linear_20, view_17, v_3, context_9], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf118, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf120, (1536, 128, 64), (8192, 64, 1), 0), out=buf121)
            buf122 = reinterpret_tensor(buf120, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf121, buf122, 12582912, stream=raw_stream0)
            buf236 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_156], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf118, buf236, 196608, 128, stream=raw_stream0)
            buf237 = reinterpret_tensor(buf118, (1536, 128, 128), (16384, 1, 128), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [view_232], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf236, buf237, 196608, 128, stream=raw_stream0)
            buf123 = reinterpret_tensor(buf121, (16384, 768), (768, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [context_9, transpose_19, contiguous_3, context_10, context_11, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:40
            extern_kernels.mm(reinterpret_tensor(buf122, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 768), (1, 768), 0), out=buf123)
            buf127 = reinterpret_tensor(buf123, (128, 128, 768), (98304, 768, 1), 0); del buf123  # reuse
            buf128 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf234 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_11, add_7, sa_output_3, div_29], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf127, primals_62, buf107, primals_63, primals_64, buf128, buf234, 16384, 768, stream=raw_stream0)
            del primals_62
            del primals_64
            buf129 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_3, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:41
            extern_kernels.addmm(primals_66, reinterpret_tensor(buf128, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf129)
            del primals_66
            buf130 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf129, buf130, 50331648, stream=raw_stream0)
            buf131 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, x_14], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:42
            extern_kernels.addmm(primals_68, reinterpret_tensor(buf130, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_67, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf131)
            del primals_68
            # Unsorted Source Nodes: [x_15, x_14], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:43
            buf132 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf131, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf133 = buf132[0]
            assert_alignment(buf133, 16, 'torch.ops.npu._npu_dropout.default')
            buf134 = buf132[1]
            assert_alignment(buf134, 16, 'torch.ops.npu._npu_dropout.default')
            del buf132
            buf138 = reinterpret_tensor(buf131, (128, 128, 768), (98304, 768, 1), 0); del buf131  # reuse
            buf139 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf233 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_8, ffn_output_3, div_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf133, buf128, primals_69, primals_70, buf138, buf139, buf233, 16384, 768, stream=raw_stream0)
            del primals_70
            buf140 = reinterpret_tensor(buf133, (16384, 768), (768, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:44
            extern_kernels.mm(reinterpret_tensor(buf139, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), out=buf140)
            buf141 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, linear_25, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:45
            extern_kernels.mm(reinterpret_tensor(buf139, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), out=buf141)
            buf142 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_3, linear_24, linear_26, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:46
            extern_kernels.mm(reinterpret_tensor(buf139, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), out=buf142)
            buf143 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, q_9, scores_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_transpose_view_2.run(primals_72, buf140, buf143, 12582912, stream=raw_stream0)
            buf232 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, q_9, permute_125, clone_36], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_backward_transpose_view_3.run(primals_72, buf140, buf232, 98304, 128, stream=raw_stream0)
            del primals_72
            buf144 = reinterpret_tensor(buf140, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [, linear_25, view_21, k_4, transpose_23, scores_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_74, buf141, buf144, 98304, 128, stream=raw_stream0)
            buf145 = reinterpret_tensor(buf236, (1536, 128, 128), (16384, 128, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [, linear_24, view_20, q_8, linear_25, view_21, k_4, q_9, transpose_23, scores_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:47
            extern_kernels.bmm(reinterpret_tensor(buf143, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf144, (1536, 64, 128), (8192, 128, 1), 0), out=buf145)
            buf148 = reinterpret_tensor(buf145, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_8, scores_9, weights_8], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf148, 196608, 128, stream=raw_stream0)
            buf231 = reinterpret_tensor(buf144, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [, linear_25, view_21, k_4, transpose_23, permute_124, clone_35], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_74, buf141, buf231, 12582912, stream=raw_stream0)
            del primals_74
            buf152 = reinterpret_tensor(buf141, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [, linear_26, view_22, v_4, context_12], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_backward_transpose_view_6.run(primals_76, buf142, buf152, 12582912, stream=raw_stream0)
            buf228 = reinterpret_tensor(buf143, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [, linear_26, view_22, v_4, permute_122, clone_33], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_transpose_view_4.run(primals_76, buf142, buf228, 98304, 128, stream=raw_stream0)
            del primals_76
            # Unsorted Source Nodes: [weights_9], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:48
            buf149 = torch.ops.npu._npu_dropout.default(buf148, 0.1)
            buf150 = buf149[0]
            assert_alignment(buf150, 16, 'torch.ops.npu._npu_dropout.default')
            buf151 = buf149[1]
            assert_alignment(buf151, 16, 'torch.ops.npu._npu_dropout.default')
            del buf149
            buf153 = reinterpret_tensor(buf142, (1536, 128, 64), (8192, 64, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [, linear_26, view_22, v_4, context_12], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:49
            extern_kernels.bmm(reinterpret_tensor(buf150, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf152, (1536, 128, 64), (8192, 64, 1), 0), out=buf153)
            buf154 = reinterpret_tensor(buf152, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf153, buf154, 12582912, stream=raw_stream0)
            buf229 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_123], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf150, buf229, 196608, 128, stream=raw_stream0)
            buf230 = reinterpret_tensor(buf150, (1536, 128, 128), (16384, 1, 128), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [view_198], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf229, buf230, 196608, 128, stream=raw_stream0)
            buf155 = reinterpret_tensor(buf153, (16384, 768), (768, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [context_12, transpose_24, contiguous_4, context_13, context_14, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf154, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 768), (1, 768), 0), out=buf155)
            buf159 = reinterpret_tensor(buf155, (128, 128, 768), (98304, 768, 1), 0); del buf155  # reuse
            buf160 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf227 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_14, add_9, sa_output_4, div_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf159, primals_78, buf139, primals_79, primals_80, buf160, buf227, 16384, 768, stream=raw_stream0)
            del primals_78
            del primals_80
            buf161 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_4, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:51
            extern_kernels.addmm(primals_82, reinterpret_tensor(buf160, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf161)
            del primals_82
            buf162 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf161, buf162, 50331648, stream=raw_stream0)
            buf163 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17, x_18], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(primals_84, reinterpret_tensor(buf162, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf163)
            del primals_84
            # Unsorted Source Nodes: [x_19, x_18], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:53
            buf164 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf163, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf165 = buf164[0]
            assert_alignment(buf165, 16, 'torch.ops.npu._npu_dropout.default')
            buf166 = buf164[1]
            assert_alignment(buf166, 16, 'torch.ops.npu._npu_dropout.default')
            del buf164
            buf170 = reinterpret_tensor(buf163, (128, 128, 768), (98304, 768, 1), 0); del buf163  # reuse
            buf171 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            buf226 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_10, ffn_output_4, div_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf165, buf160, primals_85, primals_86, buf170, buf171, buf226, 16384, 768, stream=raw_stream0)
            del primals_86
            buf172 = reinterpret_tensor(buf165, (16384, 768), (768, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:54
            extern_kernels.mm(reinterpret_tensor(buf171, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), out=buf172)
            buf173 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, linear_31, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:55
            extern_kernels.mm(reinterpret_tensor(buf171, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), out=buf173)
            buf174 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_4, linear_30, linear_32, ], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:56
            extern_kernels.mm(reinterpret_tensor(buf171, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), out=buf174)
            buf175 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            buf225 = empty_strided_npu((128, 12, 64, 128), (98304, 8192, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, q_11, scores_10, permute_92, clone_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_div_matmul_matmul_backward_transpose_view_13.run(primals_88, buf172, buf175, buf225, 98304, 128, stream=raw_stream0)
            del primals_88
            buf176 = reinterpret_tensor(buf172, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf172  # reuse
            buf224 = empty_strided_npu((128, 12, 128, 64), (98304, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_31, view_26, k_5, transpose_28, scores_10, permute_91, clone_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_14.run(primals_90, buf173, buf176, buf224, 196608, 64, stream=raw_stream0)
            del primals_90
            buf177 = reinterpret_tensor(buf229, (1536, 128, 128), (16384, 128, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [, linear_30, view_25, q_10, linear_31, view_26, k_5, q_11, transpose_28, scores_10], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.div, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:57
            extern_kernels.bmm(reinterpret_tensor(buf175, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf176, (1536, 64, 128), (8192, 128, 1), 0), out=buf177)
            buf180 = reinterpret_tensor(buf177, (128, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [attention_mask, eq, view_3, mask, tensor, scores_10, scores_11, weights_10], Original ATen: [aten.ones, aten.eq, aten.view, aten.expand, aten.lift_fresh, aten.matmul, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_eq_expand_lift_fresh_masked_fill_matmul_ones_view_5.run(buf180, 196608, 128, stream=raw_stream0)
            buf184 = reinterpret_tensor(buf176, (128, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf176  # reuse
            buf221 = reinterpret_tensor(buf175, (128, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [, linear_32, view_27, v_5, context_15, permute_89, clone_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_addmm_matmul_matmul_backward_transpose_view_15.run(primals_92, buf174, buf184, buf221, 98304, 128, stream=raw_stream0)
            del primals_92
            # Unsorted Source Nodes: [weights_11], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:58
            buf181 = torch.ops.npu._npu_dropout.default(buf180, 0.1)
            buf182 = buf181[0]
            assert_alignment(buf182, 16, 'torch.ops.npu._npu_dropout.default')
            buf183 = buf181[1]
            assert_alignment(buf183, 16, 'torch.ops.npu._npu_dropout.default')
            del buf181
            buf185 = reinterpret_tensor(buf174, (1536, 128, 64), (8192, 64, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [, linear_32, view_27, v_5, context_15], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:59
            extern_kernels.bmm(reinterpret_tensor(buf182, (1536, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf184, (1536, 128, 64), (8192, 64, 1), 0), out=buf185)
            buf186 = reinterpret_tensor(buf184, (128, 128, 12, 64), (98304, 768, 64, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5], Original ATen: [aten.matmul, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_transpose_7.run(buf185, buf186, 12582912, stream=raw_stream0)
            buf222 = empty_strided_npu((128, 12, 128, 128), (196608, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_90], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_8.run(buf182, buf222, 196608, 128, stream=raw_stream0)
            buf223 = reinterpret_tensor(buf182, (1536, 128, 128), (16384, 1, 128), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [view_164], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_9.run(buf222, buf223, 196608, 128, stream=raw_stream0)
            del buf222
            buf187 = reinterpret_tensor(buf185, (16384, 768), (768, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [context_15, transpose_29, contiguous_5, context_16, context_17, ], Original ATen: [aten.matmul, aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:60
            extern_kernels.mm(reinterpret_tensor(buf186, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_93, (768, 768), (1, 768), 0), out=buf187)
            buf191 = reinterpret_tensor(buf187, (128, 128, 768), (98304, 768, 1), 0); del buf187  # reuse
            buf192 = reinterpret_tensor(buf173, (128, 128, 768), (98304, 768, 1), 0); del buf173  # reuse
            buf220 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, context_17, add_11, sa_output_5, div_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_10.run(buf191, primals_94, buf171, primals_95, primals_96, buf192, buf220, 16384, 768, stream=raw_stream0)
            del primals_94
            del primals_96
            buf193 = empty_strided_npu((16384, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sa_output_5, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:61
            extern_kernels.addmm(primals_98, reinterpret_tensor(buf192, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf193)
            del primals_98
            buf194 = empty_strided_npu((128, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_view_11.run(buf193, buf194, 50331648, stream=raw_stream0)
            buf195 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, x_22], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:62
            extern_kernels.addmm(primals_100, reinterpret_tensor(buf194, (16384, 3072), (3072, 1), 0), reinterpret_tensor(primals_99, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf195)
            del primals_100
            # Unsorted Source Nodes: [x_23, x_22], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:63
            buf196 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf195, (128, 128, 768), (98304, 768, 1), 0), 0.1)
            buf197 = buf196[0]
            assert_alignment(buf197, 16, 'torch.ops.npu._npu_dropout.default')
            buf198 = buf196[1]
            assert_alignment(buf198, 16, 'torch.ops.npu._npu_dropout.default')
            del buf196
            buf202 = buf197; del buf197  # reuse
            buf203 = reinterpret_tensor(buf195, (128, 128, 768), (98304, 768, 1), 0); del buf195  # reuse
            buf219 = empty_strided_npu((128, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_12, ffn_output_5, div_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf202, buf192, primals_101, primals_102, buf203, buf219, 16384, 768, stream=raw_stream0)
            del primals_102
            buf204 = empty_strided_npu((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_output_5, prediction_logits], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:64
            extern_kernels.addmm(primals_104, reinterpret_tensor(buf203, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf204)
            del primals_104
            buf205 = empty_strided_npu((128, 128, 1), (128, 1, 16384), torch.float32)
            buf206 = buf205; del buf205  # reuse
            buf207 = empty_strided_npu((128, 128, 1), (128, 1, 16384), torch.float32)
            buf208 = reinterpret_tensor(buf207, (128, 128, 1), (128, 1, 1), 0); del buf207  # reuse
            buf209 = empty_strided_npu((128, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_gelu_native_layer_norm_view_17.run(buf206, buf208, buf204, primals_105, primals_106, buf209, 16384, 768, stream=raw_stream0)
            del primals_106
            buf210 = empty_strided_npu((16384, 30522), (30522, 1), torch.float32)
            # Topologically Sorted Source Nodes: [prediction_logits, prediction_logits_1, prediction_logits_2, prediction_logits_3], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(primals_107, reinterpret_tensor(buf209, (16384, 768), (768, 1), 0), reinterpret_tensor(primals_2, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf210)
            del primals_107
            # Topologically Sorted Source Nodes: [prediction_logits_3, view_30, mlm_loss], Original ATen: [aten.view, aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:66
            buf211 = torch.ops.aten._log_softmax.default(buf210, 1, False)
            buf212 = buf211
            assert_alignment(buf212, 16, 'torch.ops.aten._log_softmax.default')
            del buf211
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:67
            buf214 = torch.ops.aten.gather.default(buf212, 1, reinterpret_tensor(buf213, (16384, 1), (1, 0), 0))
            del buf213
            buf215 = buf214
            assert_alignment(buf215, 16, 'torch.ops.aten.gather.default')
            del buf214
            buf218 = empty_strided_npu((), (), torch.float32)
            buf217 = empty_strided_npu((), (), torch.float32)
            buf261 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [view_31, mlm_loss], Original ATen: [aten.view, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_view_18.run(buf261, primals_108, buf215, buf217, 1, 16384, stream=raw_stream0)
            del buf215
        return (buf261, reinterpret_tensor(buf210, (128, 128, 30522), (3906816, 30522, 1), 0), primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, buf1, buf3, reinterpret_tensor(buf5, (128, 128, 1), (128, 1, 1), 0), buf7, buf11, reinterpret_tensor(buf10, (16384, 768), (768, 1), 0), buf20, buf23, reinterpret_tensor(buf26, (16384, 768), (768, 1), 0), buf31, reinterpret_tensor(buf32, (16384, 768), (768, 1), 0), buf33, reinterpret_tensor(buf34, (16384, 3072), (3072, 1), 0), buf38, buf42, reinterpret_tensor(buf43, (16384, 768), (768, 1), 0), buf52, buf55, reinterpret_tensor(buf58, (16384, 768), (768, 1), 0), buf63, reinterpret_tensor(buf64, (16384, 768), (768, 1), 0), buf65, reinterpret_tensor(buf66, (16384, 3072), (3072, 1), 0), buf70, buf74, reinterpret_tensor(buf75, (16384, 768), (768, 1), 0), buf84, buf87, reinterpret_tensor(buf90, (16384, 768), (768, 1), 0), buf95, reinterpret_tensor(buf96, (16384, 768), (768, 1), 0), buf97, reinterpret_tensor(buf98, (16384, 3072), (3072, 1), 0), buf102, buf106, reinterpret_tensor(buf107, (16384, 768), (768, 1), 0), buf116, buf119, reinterpret_tensor(buf122, (16384, 768), (768, 1), 0), buf127, reinterpret_tensor(buf128, (16384, 768), (768, 1), 0), buf129, reinterpret_tensor(buf130, (16384, 3072), (3072, 1), 0), buf134, buf138, reinterpret_tensor(buf139, (16384, 768), (768, 1), 0), buf148, buf151, reinterpret_tensor(buf154, (16384, 768), (768, 1), 0), buf159, reinterpret_tensor(buf160, (16384, 768), (768, 1), 0), buf161, reinterpret_tensor(buf162, (16384, 3072), (3072, 1), 0), buf166, buf170, reinterpret_tensor(buf171, (16384, 768), (768, 1), 0), buf180, buf183, reinterpret_tensor(buf186, (16384, 768), (768, 1), 0), buf191, reinterpret_tensor(buf192, (16384, 768), (768, 1), 0), buf193, reinterpret_tensor(buf194, (16384, 3072), (3072, 1), 0), buf198, buf202, reinterpret_tensor(buf203, (16384, 768), (768, 1), 0), buf204, reinterpret_tensor(buf206, (128, 128, 1), (128, 1, 1), 0), buf208, reinterpret_tensor(buf209, (16384, 768), (768, 1), 0), buf212, buf217, buf219, buf220, reinterpret_tensor(buf221, (1536, 64, 128), (8192, 128, 1), 0), buf223, reinterpret_tensor(buf224, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf225, (1536, 64, 128), (8192, 128, 1), 0), buf226, buf227, reinterpret_tensor(buf228, (1536, 64, 128), (8192, 128, 1), 0), buf230, reinterpret_tensor(buf231, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf232, (1536, 64, 128), (8192, 128, 1), 0), buf233, buf234, reinterpret_tensor(buf235, (1536, 64, 128), (8192, 128, 1), 0), buf237, reinterpret_tensor(buf238, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf239, (1536, 64, 128), (8192, 128, 1), 0), buf240, buf241, reinterpret_tensor(buf242, (1536, 64, 128), (8192, 128, 1), 0), buf244, reinterpret_tensor(buf245, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf246, (1536, 64, 128), (8192, 128, 1), 0), buf247, buf248, reinterpret_tensor(buf249, (1536, 64, 128), (8192, 128, 1), 0), buf251, reinterpret_tensor(buf252, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf253, (1536, 64, 128), (8192, 128, 1), 0), buf254, buf255, reinterpret_tensor(buf256, (1536, 64, 128), (8192, 128, 1), 0), buf258, reinterpret_tensor(buf259, (1536, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf260, (1536, 64, 128), (8192, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((128, 128), (128, 1), device='npu:0', dtype=torch.int64)
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
    primals_108 = rand_strided((128, 128), (128, 1), device='npu:0', dtype=torch.int64)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('DistilBertForMaskedLM', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
