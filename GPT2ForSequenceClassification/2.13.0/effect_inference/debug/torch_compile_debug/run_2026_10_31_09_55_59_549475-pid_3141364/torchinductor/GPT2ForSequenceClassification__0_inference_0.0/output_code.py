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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/w6/cw67cgrodtq74st7mzwd7qiadsxgkohpleddzuujgofxwr6dqfpb.py
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/st/cstubhdg575sj47rhrtuoqacnq73gfxgvs5du672p6ws7vu3b4ao.py
# Topologically Sorted Source Nodes: [add, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf5 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf5]
#   %getitem_1 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf7]
#   %arg3_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg4_1]
#   %add : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   return %buf5,%getitem_1,%buf7,%add_2
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
    size_hints={'x': 4096, 'r0_': 768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, 4, {'divisors': [1, 1024]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 1024, 'seed': 2}, {'name': 'r0_2', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 1024
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
        x3 = x0 + 1024*x1
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
            tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
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
            tmp8 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp9 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
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
            tmp16 = tl.load(in_ptr0 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr1 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp26 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp7
            tmp20 = tl.full([1, 1, 1], 768.0, tl.float32)
            tmp21 = (tmp14 / tmp20)
            tmp22 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp25 * tmp26
            tmp29 = tmp27 + tmp28
            tl.store(out_ptr1 + (r0_2 + 768*x0 + 786432*x1), tmp29, r0_mask & x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/td/ctdawajcasab24igmirxmvwlnbclgx5lw2cdtz5hzkm56bhwiqv2.py
# Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   attn_weights => clone_1
#   query_1 => permute
#   split => split
#   tensor => view_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_3 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 1024, 12, 64]), kwargs = {})
#   %permute : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_matmul_permute_split_view_2 = async_compile.triton('triton_unk_fused_matmul_permute_split_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/76/c76ewttlomq7afwt4hrq2sy6j3n677svx4jzaky57jh4yesnqxoy.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#   attn_weights => clone_2
#   key_1 => permute_1
#   split => split
#   tensor_1 => view_4
#   transpose => permute_3
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_4 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [4, 1024, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f32[4, 12, 64, 1024][2359296, 64, 1, 2304]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[4, 12, 64, 1024][786432, 65536, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_unk_fused_matmul_permute_split_transpose_view_3 = async_compile.triton('triton_unk_fused_matmul_permute_split_transpose_view_3', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_transpose_view_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/ra/cras52jsic5pixjjg6fqiamtsh7numnvznuocf2bgnpspcsqojjt.py
# Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
# Source node to ATen node mapping:
#   attn_weights => view_8
#   attn_weights_1 => div
#   attn_weights_2 => where
#   full => full_default
#   mask_value => full_default_1
# Graph fragment:
#   %arg7_1 : Tensor "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]npu:0" = PlaceHolder[target=arg7_1]
#   %bmm : Tensor "f32[48, 1024, 1024][1048576, 1024, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_8 : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 12, 1024, 1024]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 8.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %div : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_8, %full_default), kwargs = {})
#   %full_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[4, 12, 1024, 1024][12582912, 1048576, 1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg7_1, %div, %full_default_1), kwargs = {})
#   return %where
triton_unk_fused__to_copy_div_full_matmul_where_4 = async_compile.triton('triton_unk_fused__to_copy_div_full_matmul_where_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_div_full_matmul_where_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_div_full_matmul_where_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/lk/clkbq7ey6wjgdln7ifsj5gorvrzh7gdjdblctbkcx2offy33x7hz.py
# Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   attn_output => clone_4
#   split => split
#   tensor_2 => view_5
#   value_1 => permute_2
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[4096, 2304][2304, 1]npu:0" = PlaceHolder[target=addmm]
#   %view_2 : Tensor "f32[4, 1024, 2304][2359296, 2304, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 1024, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 768, 2), kwargs = {})
#   %view_5 : Tensor "f32[4, 1024, 12, 64][2359296, 2304, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 1024, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[4, 12, 1024, 64][2359296, 64, 2304, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_unk_fused_matmul_permute_split_view_5 = async_compile.triton('triton_unk_fused_matmul_permute_split_view_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_split_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_split_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/gy/cgy67jnsdtso63jxkulxglxf63ns234lrb5g3hny3ghretxam6sw.py
# Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3], Original ATen: [aten.matmul, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   attn_output => view_11
#   permute_3 => permute_4
#   tensor_3 => clone_5
# Graph fragment:
#   %bmm_1 : Tensor "f32[48, 1024, 64][65536, 64, 1]npu:0" = PlaceHolder[target=bmm_1]
#   %view_11 : Tensor "f32[4, 12, 1024, 64][786432, 65536, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [4, 12, 1024, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[4, 1024, 12, 64][786432, 64, 65536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "f32[4, 1024, 12, 64][786432, 768, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_unk_fused_clone_matmul_permute_6 = async_compile.triton('triton_unk_fused_clone_matmul_permute_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_matmul_permute_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_matmul_permute_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/eg/cegu4u6ilj3gybyr5z5amjxnoet5rw3l4px6v7drxuqupr3t2epg.py
# Topologically Sorted Source Nodes: [add, , x_3, hidden_states_1, hidden_states_2], Original ATen: [aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_35
#   add => add
#   hidden_states_1 => add_3
#   hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_3 => view_14
# Graph fragment:
#   %arg8_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg8_1]
#   %mm_default_35 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_35]
#   %embedding : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %buf20 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf20]
#   %getitem_6 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_6]
#   %buf22 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf22]
#   %arg10_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg11_1]
#   %add : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default_35), kwargs = {})
#   %view_14 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [4, 1024, 768]), kwargs = {})
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_6), kwargs = {})
#   %add_4 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_5, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg10_1), kwargs = {})
#   %add_5 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg11_1), kwargs = {})
#   return %buf20,%getitem_6,%buf22,%add_5
triton_unk_fused_add_addmm_native_layer_norm_view_7 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, 4, {'divisors': [1, 1024]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 1024, 'seed': 2}, {'name': 'r0_2', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 1024
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
        x3 = x0 + 1024*x1
        x3mask = x0mask & x1mask
        r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp8 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tl.load(in_ptr2 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp4 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp5 = tmp3 + tmp4
            tmp6 = tmp2 + tmp5
            tmp7 = tl.broadcast_to(tmp6, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tmp9
        tmp8 = tl.sum(_tmp8, 2)[:, :, None]
        tmp10 = tl.full([1, 1, 1], 768.0, tl.float32)
        tmp11 = (tmp8 / tmp10)
        _tmp22 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp12 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp13 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tl.load(in_ptr2 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp16 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp14 = tmp12 + tmp13
            tmp17 = tmp15 + tmp16
            tmp18 = tmp14 + tmp17
            tmp19 = tmp18 - tmp11
            tmp20 = tmp19 * tmp19
            tmp21 = tl.broadcast_to(tmp20, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp23 = _tmp22 + tmp21
            _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
        tmp22 = tl.sum(_tmp22, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp24 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp25 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp27 = tl.load(in_ptr2 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp28 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp38 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp40 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp26 = tmp24 + tmp25
            tmp29 = tmp27 + tmp28
            tmp30 = tmp26 + tmp29
            tmp31 = tmp30 - tmp11
            tmp32 = tl.full([1, 1, 1], 768.0, tl.float32)
            tmp33 = (tmp22 / tmp32)
            tmp34 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp35 = tmp33 + tmp34
            tmp36 = tl.rsqrt(tmp35)
            tmp37 = tmp31 * tmp36
            tmp39 = tmp37 * tmp38
            tmp41 = tmp39 + tmp40
            tl.store(out_ptr1 + (r0_2 + 768*x0 + 786432*x1), tmp41, r0_mask & x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/yb/cybssx6shyvl2ws7f2ftyeswlgoyl4nnpcoremcgnceco4i6i6j4.py
# Topologically Sorted Source Nodes: [, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#    => add_tensor_34
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
#   %arg12_1 : Tensor "f32[3072][1]npu:0" = PlaceHolder[target=arg12_1]
#   %mm_default_34 : Tensor "f32[4096, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_default_34]
#   %add_tensor_34 : Tensor "f32[4096, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_34), kwargs = {})
#   %view_16 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [4, 1024, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_16, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_7 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[4, 1024, 3072][3145728, 3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_7
triton_unk_fused_add_addmm_mul_pow_tanh_view_8 = async_compile.triton('triton_unk_fused_add_addmm_mul_pow_tanh_view_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [3072, 4096, {'divisors': [1, 3072]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 3072, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4096, 'divisor': 3072, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_mul_pow_tanh_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_addmm_mul_pow_tanh_view_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 12582912
    x_g_tile0 : tl.constexpr = (3072) if (3072) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4096) if (4096) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 3072
    x1numel : tl.constexpr = 4096
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
        tmp3 = tl.full([1, 1], 0.5, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp2 * tmp2
        tmp6 = tmp5 * tmp2
        tmp7 = tl.full([1, 1], 0.044715, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 + tmp8
        tmp10 = tl.full([1, 1], 0.7978845608028654, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = libdevice.tanh(tmp11)
        tmp13 = tl.full([1, 1], 1.0, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp4 * tmp14
        tl.store(in_out_ptr0 + (x0 + 3072*x1), tmp15, x0mask & x1mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/kg/ckg7rpoe2oi66ct5pqoumujpjumbrbdh35gvvef4j4shu5dilx4a.py
# Topologically Sorted Source Nodes: [add, , x_3, hidden_states_1, x_7, hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_33, add_tensor_35
#   add => add
#   hidden_states_1 => add_3
#   hidden_states_5 => add_8
#   hidden_states_6 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
#   x_3 => view_14
#   x_7 => view_18
# Graph fragment:
#   %arg8_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg8_1]
#   %mm_default_35 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_35]
#   %embedding : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %arg14_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg14_1]
#   %mm_default_33 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_33]
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_8]
#   %buf28 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf28]
#   %getitem_8 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_8]
#   %buf30 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf30]
#   %arg16_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg16_1]
#   %arg17_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg17_1]
#   %add : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default_35), kwargs = {})
#   %view_14 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [4, 1024, 768]), kwargs = {})
#   %add_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add), kwargs = {})
#   %add_tensor_33 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg14_1, %mm_default_33), kwargs = {})
#   %view_18 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [4, 1024, 768]), kwargs = {})
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_18), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_8), kwargs = {})
#   %add_9 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_7, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg16_1), kwargs = {})
#   %add_10 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg17_1), kwargs = {})
#   return %add_8,%buf28,%getitem_8,%buf30,%add_10
triton_unk_fused_add_addmm_native_layer_norm_view_9 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, 4, {'divisors': [1, 1024]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 1024, 'seed': 2}, {'name': 'r0_2', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 4096
    r0_numel = 768
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 1024
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
        x3 = x0 + 1024*x1
        x3mask = x0mask & x1mask
        r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp12 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp4 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp7 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp8 = tl.load(in_ptr4 + (r0_2 + 768*x0 + 786432*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp5 = tmp3 + tmp4
            tmp6 = tmp2 + tmp5
            tmp9 = tmp7 + tmp8
            tmp10 = tmp6 + tmp9
            tmp11 = tl.broadcast_to(tmp10, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tmp13
            tl.store(in_out_ptr0 + (r0_2 + 768*x0 + 786432*x1), tmp10, r0_mask & x0mask & x1mask)
        tmp12 = tl.sum(_tmp12, 2)[:, :, None]
        tmp14 = tl.full([1, 1, 1], 768.0, tl.float32)
        tmp15 = (tmp12 / tmp14)
        _tmp20 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp16 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & x3mask, eviction_policy='evict_last', other=0.0)
            tmp17 = tmp16 - tmp15
            tmp18 = tmp17 * tmp17
            tmp19 = tl.broadcast_to(tmp18, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp21 = _tmp20 + tmp19
            _tmp20 = tl.where(r0_mask, tmp21, _tmp20)
        tmp20 = tl.sum(_tmp20, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp22 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & x3mask, eviction_policy='evict_first', other=0.0)
            tmp30 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp32 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp23 = tmp22 - tmp15
            tmp24 = tl.full([1, 1, 1], 768.0, tl.float32)
            tmp25 = (tmp20 / tmp24)
            tmp26 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp27 = tmp25 + tmp26
            tmp28 = tl.rsqrt(tmp27)
            tmp29 = tmp23 * tmp28
            tmp31 = tmp29 * tmp30
            tmp33 = tmp31 + tmp32
            tl.store(out_ptr1 + (r0_2 + 768*x3), tmp33, r0_mask & x3mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/wz/cwzfyv27pl7vt77tl5xji4hi3ca6le3d75xtuex735biuqo6kaij.py
# Topologically Sorted Source Nodes: [, x_11, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_32
#   hidden_states_7 => add_11
#   hidden_states_8 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
#   x_11 => view_32
# Graph fragment:
#   %arg21_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg21_1]
#   %mm_default_32 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_32]
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_8]
#   %buf43 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf43]
#   %getitem_13 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_13]
#   %buf45 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf45]
#   %arg23_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg24_1]
#   %add_tensor_32 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg21_1, %mm_default_32), kwargs = {})
#   %view_32 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [4, 1024, 768]), kwargs = {})
#   %add_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_32, %add_8), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_13), kwargs = {})
#   %add_12 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_10 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg23_1), kwargs = {})
#   %add_13 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg24_1), kwargs = {})
#   return %buf43,%getitem_13,%buf45,%add_13
triton_unk_fused_add_addmm_native_layer_norm_view_10 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
            tmp10 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
            tmp20 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/yw/cywiqc7ttuypdhy7bjtkktsdxmy3b2grhjvx6lvt2yatgujwa4h3.py
# Topologically Sorted Source Nodes: [, x_11, hidden_states_7, x_15, hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_30, add_tensor_32
#   hidden_states_11 => add_16
#   hidden_states_12 => add_17, add_18, mul_16, mul_17, rsqrt_4, sub_4, var_mean_4
#   hidden_states_7 => add_11
#   x_11 => view_32
#   x_15 => view_36
# Graph fragment:
#   %arg21_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg21_1]
#   %mm_default_32 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_32]
#   %add_8 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_8]
#   %arg27_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg27_1]
#   %mm_default_30 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_30]
#   %add_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_16]
#   %buf51 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf51]
#   %getitem_15 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_15]
#   %buf53 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf53]
#   %arg29_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg30_1]
#   %add_tensor_32 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg21_1, %mm_default_32), kwargs = {})
#   %view_32 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [4, 1024, 768]), kwargs = {})
#   %add_11 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_32, %add_8), kwargs = {})
#   %add_tensor_30 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg27_1, %mm_default_30), kwargs = {})
#   %view_36 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [4, 1024, 768]), kwargs = {})
#   %add_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_36), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_15), kwargs = {})
#   %add_17 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_17 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg29_1), kwargs = {})
#   %add_18 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg30_1), kwargs = {})
#   return %add_16,%buf51,%getitem_15,%buf53,%add_18
triton_unk_fused_add_addmm_native_layer_norm_view_11 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp6 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            tmp7 = tmp5 + tmp6
            tmp8 = tmp4 + tmp7
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
            tl.store(out_ptr1 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/wh/cwhtxwu67ao6rdoz7ysnsyuewtcofoevgoewotsqfjyl6cwwym3v.py
# Topologically Sorted Source Nodes: [, x_91, hidden_states_67, x_95, hidden_states_71, layer_norm_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor, add_tensor_2
#   hidden_states_67 => add_91
#   hidden_states_71 => add_96
#   layer_norm_24 => add_97, add_98, mul_96, mul_97, rsqrt_24, sub_24, var_mean_24
#   x_91 => view_212
#   x_95 => view_216
# Graph fragment:
#   %arg151_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg151_1]
#   %mm_default_2 : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default_2]
#   %add_88 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_88]
#   %arg157_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg157_1]
#   %mm_default : Tensor "f32[4096, 768][768, 1]npu:0" = PlaceHolder[target=mm_default]
#   %add_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=add_96]
#   %buf281 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf281]
#   %getitem_85 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_85]
#   %buf283 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf283]
#   %arg159_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg159_1]
#   %arg160_1 : Tensor "f32[768][1]npu:0" = PlaceHolder[target=arg160_1]
#   %add_tensor_2 : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg151_1, %mm_default_2), kwargs = {})
#   %view_212 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [4, 1024, 768]), kwargs = {})
#   %add_91 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_212, %add_88), kwargs = {})
#   %add_tensor : Tensor "f32[4096, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg157_1, %mm_default), kwargs = {})
#   %view_216 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [4, 1024, 768]), kwargs = {})
#   %add_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_216), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_96, %getitem_85), kwargs = {})
#   %add_97 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_84, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[4, 1024, 1][1024, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_97,), kwargs = {})
#   %mul_96 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %arg159_1), kwargs = {})
#   %add_98 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg160_1), kwargs = {})
#   return %add_96,%buf281,%getitem_85,%buf283,%add_98
triton_unk_fused_add_addmm_native_layer_norm_view_12 = async_compile.triton('triton_unk_fused_add_addmm_native_layer_norm_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_addmm_native_layer_norm_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True}
)
@triton.jit
def triton_unk_fused_add_addmm_native_layer_norm_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp6 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            tmp7 = tmp5 + tmp6
            tmp8 = tmp4 + tmp7
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
            tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/4i/c4iwdwkdkiuviyw2mbwitbypcildhcqcpazpu3bmbm75owtjzsqq.py
# Topologically Sorted Source Nodes: [eq, int_1], Original ATen: [aten.eq, aten._to_copy]
# Source node to ATen node mapping:
#   eq => eq
#   int_1 => convert_element_type_12
# Graph fragment:
#   %arg0_1 : Tensor "i64[4, 1024][1024, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %eq : Tensor "b8[4, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %convert_element_type_12 : Tensor "i32[4, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.int32), kwargs = {})
#   return %convert_element_type_12
triton_unk_fused__to_copy_eq_13 = async_compile.triton('triton_unk_fused__to_copy_eq_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_eq_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_eq_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/x6/cx6m33ph332egpqyzl2r27sgg3qdtbpzsdux3dvvevif5kfziqjk.py
# Topologically Sorted Source Nodes: [arange_1], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   arange_1 => iota_1
# Graph fragment:
#   %iota_1 : Tensor "i64[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   return %iota_1
triton_unk_fused_arange_14 = async_compile.triton('triton_unk_fused_arange_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/72/c72wh4apltau4pmyu62z6cbbbokpxxfupxnds7hjwpfsgdxuovsc.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub => sub_25
# Graph fragment:
#   %argmax : Tensor "i64[4][1]npu:0" = PlaceHolder[target=argmax]
#   %sub_25 : Tensor "i64[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%argmax, 1), kwargs = {})
#   return %sub_25
triton_unk_fused_sub_15 = async_compile.triton('triton_unk_fused_sub_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sub_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_sub_15(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/ry/cryndjgws7xufu2yrcznl7was655z76ihtwpowaoscpytezh5wrf.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => full_default_24, ne, where_12
# Graph fragment:
#   %arg162_1 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=arg162_1]
#   %ne : Tensor "b8[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg162_1, -100), kwargs = {})
#   %full_default_24 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_12 : Tensor "i64[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %arg162_1, %full_default_24), kwargs = {})
#   return %where_12
triton_unk_fused_nll_loss_forward_16 = async_compile.triton('triton_unk_fused_nll_loss_forward_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='npu')


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.13.0/effect_inference/cache/4r/c4rktem3eq2g44iux3nhepocisaexqhonu5bybjvr6cl6xwbrqpi.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_13, div_12, full_default_25, ne_1, ne_2, neg, squeeze, sum_1, sum_2, where_13
# Graph fragment:
#   %arg162_1 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=arg162_1]
#   %gather : Tensor "f32[4, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %ne_1 : Tensor "b8[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg162_1, -100), kwargs = {})
#   %squeeze : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_13 : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_25), kwargs = {})
#   %sum_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_13,), kwargs = {})
#   %ne_2 : Tensor "b8[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg162_1, -100), kwargs = {})
#   %sum_1 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %div_12 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %convert_element_type_13), kwargs = {})
#   return %div_12
triton_unk_fused_nll_loss_forward_17 = async_compile.triton('triton_unk_fused_nll_loss_forward_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': []},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_nll_loss_forward_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'npu_num_x_nodes': 0, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_nll_loss_forward_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    xnumel = 1
    group_size = 1 if group_id < 1 else 0
    group_base = group_id
    for i in range(group_size):
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tmp0
        tmp4 = tl.load(in_ptr1 + (0))
        tmp5 = tmp4
        tmp9 = tl.load(in_ptr0 + (1))
        tmp10 = tmp9
        tmp12 = tl.load(in_ptr1 + (1))
        tmp13 = tmp12
        tmp17 = tl.load(in_ptr0 + (2))
        tmp18 = tmp17
        tmp20 = tl.load(in_ptr1 + (2))
        tmp21 = tmp20
        tmp25 = tl.load(in_ptr0 + (3))
        tmp26 = tmp25
        tmp28 = tl.load(in_ptr1 + (3))
        tmp29 = tmp28
        tmp2 = tl.full([], -100, tl.int32)
        tmp3 = tmp1 != tmp2
        tmp6 = -tmp5
        tmp7 = tl.full([], 0.0, tl.float32)
        tmp8 = tl.where(tmp3, tmp6, tmp7)
        tmp11 = tmp10 != tmp2
        tmp14 = -tmp13
        tmp15 = tl.where(tmp11, tmp14, tmp7)
        tmp16 = tmp8 + tmp15
        tmp19 = tmp18 != tmp2
        tmp22 = -tmp21
        tmp23 = tl.where(tmp19, tmp22, tmp7)
        tmp24 = tmp16 + tmp23
        tmp27 = tmp26 != tmp2
        tmp30 = -tmp29
        tmp31 = tl.where(tmp27, tmp30, tmp7)
        tmp32 = tmp24 + tmp31
        tmp33 = tmp3.to(tl.int32)
        tmp34 = tmp11.to(tl.int32)
        tmp35 = tmp33 + tmp34
        tmp36 = tmp19.to(tl.int32)
        tmp37 = tmp35 + tmp36
        tmp38 = tmp27.to(tl.int32)
        tmp39 = tmp37 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp41 = (tmp32 / tmp40)
        tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp41, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [inputs_embeds], Original ATen: [aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:1
            buf0 = torch.ops.aten.embedding.default(arg1_1, arg0_1)
            del arg1_1
            buf1 = buf0
            assert_alignment(buf1, 16, 'torch.ops.aten.embedding.default')
            del buf0
            buf2 = empty_strided_npu((1024, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_0.run(buf2, 1024, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [position_ids, position_ids_1, position_embeds], Original ATen: [aten.arange, aten.unsqueeze, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf3 = torch.ops.aten.embedding.default(arg2_1, reinterpret_tensor(buf2, (1, 1024), (0, 1), 0))
            del arg2_1
            del buf2
            buf4 = buf3
            assert_alignment(buf4, 16, 'torch.ops.aten.embedding.default')
            del buf3
            buf8 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_1.run(buf1, buf4, arg3_1, arg4_1, buf8, 4096, 768, stream=raw_stream0)
            del arg3_1
            del arg4_1
            buf9 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, hidden_states, view_1, x], Original ATen: [aten.add, aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:3
            extern_kernels.addmm(arg5_1, reinterpret_tensor(buf8, (4096, 768), (768, 1), 0), arg6_1, alpha=1, beta=1, out=buf9)
            del arg5_1
            del arg6_1
            buf10 = reinterpret_tensor(buf8, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf9, buf10, 3145728, stream=raw_stream0)
            buf11 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf9, buf11, 3072, 1024, stream=raw_stream0)
            buf12 = empty_strided_npu((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights, tensor_1, key_1, transpose], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:4
            extern_kernels.bmm(reinterpret_tensor(buf10, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf11, (48, 64, 1024), (65536, 1024, 1), 0), out=buf12)
            buf13 = reinterpret_tensor(buf12, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf13, arg7_1, 50331648, stream=raw_stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2, attn_weights_3], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:5
            buf14 = torch.ops.aten._softmax.default(buf13, -1, False)
            del buf13
            buf15 = buf14
            assert_alignment(buf15, 16, 'torch.ops.aten._softmax.default')
            del buf14
            buf16 = reinterpret_tensor(buf11, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf9, buf16, 3145728, stream=raw_stream0)
            buf17 = reinterpret_tensor(buf10, (48, 1024, 64), (65536, 64, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [x_1, split, attn_output, tensor_2, value_1], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:6
            extern_kernels.bmm(reinterpret_tensor(buf15, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf16, (48, 1024, 64), (65536, 64, 1), 0), out=buf17)
            buf18 = reinterpret_tensor(buf16, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf17, buf18, 3145728, stream=raw_stream0)
            buf19 = reinterpret_tensor(buf17, (4096, 768), (768, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3, attn_output_1, view_7, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf18, (4096, 768), (768, 1), 0), arg9_1, out=buf19)
            del arg9_1
            buf23 = reinterpret_tensor(buf18, (4, 1024, 768), (786432, 768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [add, , x_3, hidden_states_1, hidden_states_2], Original ATen: [aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_7.run(arg8_1, buf19, buf1, buf4, arg10_1, arg11_1, buf23, 4096, 768, stream=raw_stream0)
            del arg10_1
            del arg11_1
            buf24 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, , x_3, hidden_states_1, hidden_states_2, view_9], Original ATen: [aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf23, (4096, 768), (768, 1), 0), arg13_1, out=buf24)
            del arg13_1
            buf25 = reinterpret_tensor(buf24, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf25, arg12_1, 12582912, stream=raw_stream0)
            del arg12_1
            buf26 = reinterpret_tensor(buf23, (4096, 768), (768, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3, view_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(reinterpret_tensor(buf25, (4096, 3072), (3072, 1), 0), arg15_1, out=buf26)
            del arg15_1
            buf27 = reinterpret_tensor(buf19, (4, 1024, 768), (786432, 768, 1), 0); del buf19  # reuse
            buf31 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, , x_3, hidden_states_1, x_7, hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_9.run(buf27, arg8_1, buf1, buf4, arg14_1, buf26, arg16_1, arg17_1, buf31, 4096, 768, stream=raw_stream0)
            del arg14_1
            del arg16_1
            del arg17_1
            del arg8_1
            del buf4
            buf32 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_6, view_13, x_8], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:10
            extern_kernels.addmm(arg18_1, reinterpret_tensor(buf31, (4096, 768), (768, 1), 0), arg19_1, alpha=1, beta=1, out=buf32)
            del arg18_1
            del arg19_1
            buf33 = reinterpret_tensor(buf31, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf32, buf33, 3145728, stream=raw_stream0)
            buf34 = reinterpret_tensor(buf26, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_5, key_3, transpose_1, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf32, buf34, 3072, 1024, stream=raw_stream0)
            buf35 = reinterpret_tensor(buf15, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, attn_weights_6, tensor_5, key_3, transpose_1], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:11
            extern_kernels.bmm(reinterpret_tensor(buf33, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf34, (48, 64, 1024), (65536, 1024, 1), 0), out=buf35)
            buf36 = reinterpret_tensor(buf35, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_6, full_2, attn_weights_7, mask_value_1, attn_weights_8], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf36, arg20_1, 50331648, stream=raw_stream0)
            del arg20_1
            # Topologically Sorted Source Nodes: [attn_weights_6, full_2, attn_weights_7, mask_value_1, attn_weights_8, attn_weights_9], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:12
            buf37 = torch.ops.aten._softmax.default(buf36, -1, False)
            del buf36
            buf38 = buf37
            assert_alignment(buf38, 16, 'torch.ops.aten._softmax.default')
            del buf37
            buf39 = reinterpret_tensor(buf34, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, attn_output_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf32, buf39, 3145728, stream=raw_stream0)
            buf40 = reinterpret_tensor(buf33, (48, 1024, 64), (65536, 64, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, attn_output_3, tensor_6, value_3], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:13
            extern_kernels.bmm(reinterpret_tensor(buf38, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf39, (48, 1024, 64), (65536, 64, 1), 0), out=buf40)
            buf41 = reinterpret_tensor(buf39, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf40, buf41, 3145728, stream=raw_stream0)
            buf42 = reinterpret_tensor(buf40, (4096, 768), (768, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7, attn_output_4, view_19, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf41, (4096, 768), (768, 1), 0), arg22_1, out=buf42)
            del arg22_1
            buf46 = reinterpret_tensor(buf41, (4, 1024, 768), (786432, 768, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [, x_11, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg21_1, buf42, buf27, arg23_1, arg24_1, buf46, 4096, 768, stream=raw_stream0)
            del arg23_1
            del arg24_1
            buf47 = reinterpret_tensor(buf25, (4096, 3072), (3072, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [, x_11, hidden_states_7, hidden_states_8, view_21], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(reinterpret_tensor(buf46, (4096, 768), (768, 1), 0), arg26_1, out=buf47)
            del arg26_1
            buf48 = reinterpret_tensor(buf47, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf48, arg25_1, 12582912, stream=raw_stream0)
            del arg25_1
            buf49 = reinterpret_tensor(buf46, (4096, 768), (768, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9, view_23], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf48, (4096, 3072), (3072, 1), 0), arg28_1, out=buf49)
            del arg28_1
            buf50 = reinterpret_tensor(buf42, (4, 1024, 768), (786432, 768, 1), 0); del buf42  # reuse
            buf54 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [, x_11, hidden_states_7, x_15, hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf50, arg21_1, buf27, arg27_1, buf49, arg29_1, arg30_1, buf54, 4096, 768, stream=raw_stream0)
            del arg21_1
            del arg27_1
            del arg29_1
            del arg30_1
            buf55 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_12, view_25, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:17
            extern_kernels.addmm(arg31_1, reinterpret_tensor(buf54, (4096, 768), (768, 1), 0), arg32_1, alpha=1, beta=1, out=buf55)
            del arg31_1
            del arg32_1
            buf56 = reinterpret_tensor(buf54, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf55, buf56, 3145728, stream=raw_stream0)
            buf57 = reinterpret_tensor(buf49, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_9, key_5, transpose_2, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf55, buf57, 3072, 1024, stream=raw_stream0)
            buf58 = reinterpret_tensor(buf38, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, attn_weights_12, tensor_9, key_5, transpose_2], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:18
            extern_kernels.bmm(reinterpret_tensor(buf56, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf57, (48, 64, 1024), (65536, 1024, 1), 0), out=buf58)
            buf59 = reinterpret_tensor(buf58, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_12, full_4, attn_weights_13, mask_value_2, attn_weights_14], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf59, arg33_1, 50331648, stream=raw_stream0)
            del arg33_1
            # Topologically Sorted Source Nodes: [attn_weights_12, full_4, attn_weights_13, mask_value_2, attn_weights_14, attn_weights_15], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:19
            buf60 = torch.ops.aten._softmax.default(buf59, -1, False)
            del buf59
            buf61 = buf60
            assert_alignment(buf61, 16, 'torch.ops.aten._softmax.default')
            del buf60
            buf62 = reinterpret_tensor(buf57, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, attn_output_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf55, buf62, 3145728, stream=raw_stream0)
            buf63 = reinterpret_tensor(buf56, (48, 1024, 64), (65536, 64, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, attn_output_6, tensor_10, value_5], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:20
            extern_kernels.bmm(reinterpret_tensor(buf61, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf62, (48, 1024, 64), (65536, 64, 1), 0), out=buf63)
            buf64 = reinterpret_tensor(buf62, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf63, buf64, 3145728, stream=raw_stream0)
            buf65 = reinterpret_tensor(buf63, (4096, 768), (768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11, attn_output_7, view_31, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf64, (4096, 768), (768, 1), 0), arg35_1, out=buf65)
            del arg35_1
            buf69 = reinterpret_tensor(buf64, (4, 1024, 768), (786432, 768, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [, x_19, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg34_1, buf65, buf50, arg36_1, arg37_1, buf69, 4096, 768, stream=raw_stream0)
            del arg36_1
            del arg37_1
            buf70 = reinterpret_tensor(buf48, (4096, 3072), (3072, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [, x_19, hidden_states_13, hidden_states_14, view_33], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(reinterpret_tensor(buf69, (4096, 768), (768, 1), 0), arg39_1, out=buf70)
            del arg39_1
            buf71 = reinterpret_tensor(buf70, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf71, arg38_1, 12582912, stream=raw_stream0)
            del arg38_1
            buf72 = reinterpret_tensor(buf69, (4096, 768), (768, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15, view_35], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:23
            extern_kernels.mm(reinterpret_tensor(buf71, (4096, 3072), (3072, 1), 0), arg41_1, out=buf72)
            del arg41_1
            buf73 = reinterpret_tensor(buf65, (4, 1024, 768), (786432, 768, 1), 0); del buf65  # reuse
            buf77 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [, x_19, hidden_states_13, x_23, hidden_states_17, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf73, arg34_1, buf50, arg40_1, buf72, arg42_1, arg43_1, buf77, 4096, 768, stream=raw_stream0)
            del arg34_1
            del arg40_1
            del arg42_1
            del arg43_1
            buf78 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_18, view_37, x_24], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:24
            extern_kernels.addmm(arg44_1, reinterpret_tensor(buf77, (4096, 768), (768, 1), 0), arg45_1, alpha=1, beta=1, out=buf78)
            del arg44_1
            del arg45_1
            buf79 = reinterpret_tensor(buf77, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf78, buf79, 3145728, stream=raw_stream0)
            buf80 = reinterpret_tensor(buf72, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_13, key_7, transpose_3, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf78, buf80, 3072, 1024, stream=raw_stream0)
            buf81 = reinterpret_tensor(buf61, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, attn_weights_18, tensor_13, key_7, transpose_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:25
            extern_kernels.bmm(reinterpret_tensor(buf79, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf80, (48, 64, 1024), (65536, 1024, 1), 0), out=buf81)
            buf82 = reinterpret_tensor(buf81, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_18, full_6, attn_weights_19, mask_value_3, attn_weights_20], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf82, arg46_1, 50331648, stream=raw_stream0)
            del arg46_1
            # Topologically Sorted Source Nodes: [attn_weights_18, full_6, attn_weights_19, mask_value_3, attn_weights_20, attn_weights_21], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:26
            buf83 = torch.ops.aten._softmax.default(buf82, -1, False)
            del buf82
            buf84 = buf83
            assert_alignment(buf84, 16, 'torch.ops.aten._softmax.default')
            del buf83
            buf85 = reinterpret_tensor(buf80, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, attn_output_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf78, buf85, 3145728, stream=raw_stream0)
            buf86 = reinterpret_tensor(buf79, (48, 1024, 64), (65536, 64, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, attn_output_9, tensor_14, value_7], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:27
            extern_kernels.bmm(reinterpret_tensor(buf84, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf85, (48, 1024, 64), (65536, 64, 1), 0), out=buf86)
            buf87 = reinterpret_tensor(buf85, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf86, buf87, 3145728, stream=raw_stream0)
            buf88 = reinterpret_tensor(buf86, (4096, 768), (768, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15, attn_output_10, view_43, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:28
            extern_kernels.mm(reinterpret_tensor(buf87, (4096, 768), (768, 1), 0), arg48_1, out=buf88)
            del arg48_1
            buf92 = reinterpret_tensor(buf87, (4, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [, x_27, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg47_1, buf88, buf73, arg49_1, arg50_1, buf92, 4096, 768, stream=raw_stream0)
            del arg49_1
            del arg50_1
            buf93 = reinterpret_tensor(buf71, (4096, 3072), (3072, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [, x_27, hidden_states_19, hidden_states_20, view_45], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:29
            extern_kernels.mm(reinterpret_tensor(buf92, (4096, 768), (768, 1), 0), arg52_1, out=buf93)
            del arg52_1
            buf94 = reinterpret_tensor(buf93, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf94, arg51_1, 12582912, stream=raw_stream0)
            del arg51_1
            buf95 = reinterpret_tensor(buf92, (4096, 768), (768, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21, view_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf94, (4096, 3072), (3072, 1), 0), arg54_1, out=buf95)
            del arg54_1
            buf96 = reinterpret_tensor(buf88, (4, 1024, 768), (786432, 768, 1), 0); del buf88  # reuse
            buf100 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [, x_27, hidden_states_19, x_31, hidden_states_23, hidden_states_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf96, arg47_1, buf73, arg53_1, buf95, arg55_1, arg56_1, buf100, 4096, 768, stream=raw_stream0)
            del arg47_1
            del arg53_1
            del arg55_1
            del arg56_1
            buf101 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_24, view_49, x_32], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:31
            extern_kernels.addmm(arg57_1, reinterpret_tensor(buf100, (4096, 768), (768, 1), 0), arg58_1, alpha=1, beta=1, out=buf101)
            del arg57_1
            del arg58_1
            buf102 = reinterpret_tensor(buf100, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf101, buf102, 3145728, stream=raw_stream0)
            buf103 = reinterpret_tensor(buf95, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_17, key_9, transpose_4, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf101, buf103, 3072, 1024, stream=raw_stream0)
            buf104 = reinterpret_tensor(buf84, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, attn_weights_24, tensor_17, key_9, transpose_4], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:32
            extern_kernels.bmm(reinterpret_tensor(buf102, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf103, (48, 64, 1024), (65536, 1024, 1), 0), out=buf104)
            buf105 = reinterpret_tensor(buf104, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_24, full_8, attn_weights_25, mask_value_4, attn_weights_26], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf105, arg59_1, 50331648, stream=raw_stream0)
            del arg59_1
            # Topologically Sorted Source Nodes: [attn_weights_24, full_8, attn_weights_25, mask_value_4, attn_weights_26, attn_weights_27], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:33
            buf106 = torch.ops.aten._softmax.default(buf105, -1, False)
            del buf105
            buf107 = buf106
            assert_alignment(buf107, 16, 'torch.ops.aten._softmax.default')
            del buf106
            buf108 = reinterpret_tensor(buf103, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, attn_output_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf101, buf108, 3145728, stream=raw_stream0)
            buf109 = reinterpret_tensor(buf102, (48, 1024, 64), (65536, 64, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, attn_output_12, tensor_18, value_9], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:34
            extern_kernels.bmm(reinterpret_tensor(buf107, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf108, (48, 1024, 64), (65536, 64, 1), 0), out=buf109)
            buf110 = reinterpret_tensor(buf108, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf109, buf110, 3145728, stream=raw_stream0)
            buf111 = reinterpret_tensor(buf109, (4096, 768), (768, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19, attn_output_13, view_55, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:35
            extern_kernels.mm(reinterpret_tensor(buf110, (4096, 768), (768, 1), 0), arg61_1, out=buf111)
            del arg61_1
            buf115 = reinterpret_tensor(buf110, (4, 1024, 768), (786432, 768, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [, x_35, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg60_1, buf111, buf96, arg62_1, arg63_1, buf115, 4096, 768, stream=raw_stream0)
            del arg62_1
            del arg63_1
            buf116 = reinterpret_tensor(buf94, (4096, 3072), (3072, 1), 0); del buf94  # reuse
            # Topologically Sorted Source Nodes: [, x_35, hidden_states_25, hidden_states_26, view_57], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:36
            extern_kernels.mm(reinterpret_tensor(buf115, (4096, 768), (768, 1), 0), arg65_1, out=buf116)
            del arg65_1
            buf117 = reinterpret_tensor(buf116, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf117, arg64_1, 12582912, stream=raw_stream0)
            del arg64_1
            buf118 = reinterpret_tensor(buf115, (4096, 768), (768, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27, view_59], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf117, (4096, 3072), (3072, 1), 0), arg67_1, out=buf118)
            del arg67_1
            buf119 = reinterpret_tensor(buf111, (4, 1024, 768), (786432, 768, 1), 0); del buf111  # reuse
            buf123 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [, x_35, hidden_states_25, x_39, hidden_states_29, hidden_states_30], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf119, arg60_1, buf96, arg66_1, buf118, arg68_1, arg69_1, buf123, 4096, 768, stream=raw_stream0)
            del arg60_1
            del arg66_1
            del arg68_1
            del arg69_1
            buf124 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_30, view_61, x_40], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:38
            extern_kernels.addmm(arg70_1, reinterpret_tensor(buf123, (4096, 768), (768, 1), 0), arg71_1, alpha=1, beta=1, out=buf124)
            del arg70_1
            del arg71_1
            buf125 = reinterpret_tensor(buf123, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf124, buf125, 3145728, stream=raw_stream0)
            buf126 = reinterpret_tensor(buf96, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_21, key_11, transpose_5, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf124, buf126, 3072, 1024, stream=raw_stream0)
            buf127 = reinterpret_tensor(buf107, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, attn_weights_30, tensor_21, key_11, transpose_5], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf125, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf126, (48, 64, 1024), (65536, 1024, 1), 0), out=buf127)
            buf128 = reinterpret_tensor(buf127, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_30, full_10, attn_weights_31, mask_value_5, attn_weights_32], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf128, arg72_1, 50331648, stream=raw_stream0)
            del arg72_1
            # Topologically Sorted Source Nodes: [attn_weights_30, full_10, attn_weights_31, mask_value_5, attn_weights_32, attn_weights_33], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:40
            buf129 = torch.ops.aten._softmax.default(buf128, -1, False)
            del buf128
            buf130 = buf129
            assert_alignment(buf130, 16, 'torch.ops.aten._softmax.default')
            del buf129
            buf131 = reinterpret_tensor(buf126, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, attn_output_15], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf124, buf131, 3145728, stream=raw_stream0)
            buf132 = reinterpret_tensor(buf125, (48, 1024, 64), (65536, 64, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, attn_output_15, tensor_22, value_11], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:41
            extern_kernels.bmm(reinterpret_tensor(buf130, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf131, (48, 1024, 64), (65536, 64, 1), 0), out=buf132)
            buf133 = reinterpret_tensor(buf131, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf132, buf133, 3145728, stream=raw_stream0)
            buf134 = reinterpret_tensor(buf132, (4096, 768), (768, 1), 0); del buf132  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23, attn_output_16, view_67, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf133, (4096, 768), (768, 1), 0), arg74_1, out=buf134)
            del arg74_1
            buf138 = reinterpret_tensor(buf133, (4, 1024, 768), (786432, 768, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [, x_43, hidden_states_31, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg73_1, buf134, buf119, arg75_1, arg76_1, buf138, 4096, 768, stream=raw_stream0)
            del arg75_1
            del arg76_1
            buf139 = reinterpret_tensor(buf117, (4096, 3072), (3072, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [, x_43, hidden_states_31, hidden_states_32, view_69], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:43
            extern_kernels.mm(reinterpret_tensor(buf138, (4096, 768), (768, 1), 0), arg78_1, out=buf139)
            del arg78_1
            buf140 = reinterpret_tensor(buf139, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf140, arg77_1, 12582912, stream=raw_stream0)
            del arg77_1
            buf141 = reinterpret_tensor(buf138, (4096, 768), (768, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33, view_71], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:44
            extern_kernels.mm(reinterpret_tensor(buf140, (4096, 3072), (3072, 1), 0), arg80_1, out=buf141)
            del arg80_1
            buf142 = reinterpret_tensor(buf134, (4, 1024, 768), (786432, 768, 1), 0); del buf134  # reuse
            buf146 = reinterpret_tensor(buf118, (4, 1024, 768), (786432, 768, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [, x_43, hidden_states_31, x_47, hidden_states_35, hidden_states_36], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf142, arg73_1, buf119, arg79_1, buf141, arg81_1, arg82_1, buf146, 4096, 768, stream=raw_stream0)
            del arg73_1
            del arg79_1
            del arg81_1
            del arg82_1
            buf147 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_36, view_73, x_48], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:45
            extern_kernels.addmm(arg83_1, reinterpret_tensor(buf146, (4096, 768), (768, 1), 0), arg84_1, alpha=1, beta=1, out=buf147)
            del arg83_1
            del arg84_1
            buf148 = reinterpret_tensor(buf146, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf147, buf148, 3145728, stream=raw_stream0)
            buf149 = reinterpret_tensor(buf141, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_25, key_13, transpose_6, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf147, buf149, 3072, 1024, stream=raw_stream0)
            buf150 = reinterpret_tensor(buf130, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, attn_weights_36, tensor_25, key_13, transpose_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:46
            extern_kernels.bmm(reinterpret_tensor(buf148, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf149, (48, 64, 1024), (65536, 1024, 1), 0), out=buf150)
            buf151 = reinterpret_tensor(buf150, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_36, full_12, attn_weights_37, mask_value_6, attn_weights_38], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf151, arg85_1, 50331648, stream=raw_stream0)
            del arg85_1
            # Topologically Sorted Source Nodes: [attn_weights_36, full_12, attn_weights_37, mask_value_6, attn_weights_38, attn_weights_39], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:47
            buf152 = torch.ops.aten._softmax.default(buf151, -1, False)
            del buf151
            buf153 = buf152
            assert_alignment(buf153, 16, 'torch.ops.aten._softmax.default')
            del buf152
            buf154 = reinterpret_tensor(buf149, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, attn_output_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf147, buf154, 3145728, stream=raw_stream0)
            buf155 = reinterpret_tensor(buf148, (48, 1024, 64), (65536, 64, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, attn_output_18, tensor_26, value_13], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:48
            extern_kernels.bmm(reinterpret_tensor(buf153, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf154, (48, 1024, 64), (65536, 64, 1), 0), out=buf155)
            buf156 = reinterpret_tensor(buf154, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf155, buf156, 3145728, stream=raw_stream0)
            buf157 = reinterpret_tensor(buf155, (4096, 768), (768, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27, attn_output_19, view_79, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:49
            extern_kernels.mm(reinterpret_tensor(buf156, (4096, 768), (768, 1), 0), arg87_1, out=buf157)
            del arg87_1
            buf161 = reinterpret_tensor(buf156, (4, 1024, 768), (786432, 768, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [, x_51, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg86_1, buf157, buf142, arg88_1, arg89_1, buf161, 4096, 768, stream=raw_stream0)
            del arg88_1
            del arg89_1
            buf162 = reinterpret_tensor(buf140, (4096, 3072), (3072, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [, x_51, hidden_states_37, hidden_states_38, view_81], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf161, (4096, 768), (768, 1), 0), arg91_1, out=buf162)
            del arg91_1
            buf163 = reinterpret_tensor(buf162, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf162  # reuse
            # Topologically Sorted Source Nodes: [, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf163, arg90_1, 12582912, stream=raw_stream0)
            del arg90_1
            buf164 = reinterpret_tensor(buf161, (4096, 768), (768, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39, view_83], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf163, (4096, 3072), (3072, 1), 0), arg93_1, out=buf164)
            del arg93_1
            buf165 = reinterpret_tensor(buf157, (4, 1024, 768), (786432, 768, 1), 0); del buf157  # reuse
            buf169 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [, x_51, hidden_states_37, x_55, hidden_states_41, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf165, arg86_1, buf142, arg92_1, buf164, arg94_1, arg95_1, buf169, 4096, 768, stream=raw_stream0)
            del arg86_1
            del arg92_1
            del arg94_1
            del arg95_1
            buf170 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_42, view_85, x_56], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(arg96_1, reinterpret_tensor(buf169, (4096, 768), (768, 1), 0), arg97_1, alpha=1, beta=1, out=buf170)
            del arg96_1
            del arg97_1
            buf171 = reinterpret_tensor(buf169, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf170, buf171, 3145728, stream=raw_stream0)
            buf172 = reinterpret_tensor(buf164, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_29, key_15, transpose_7, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf170, buf172, 3072, 1024, stream=raw_stream0)
            buf173 = reinterpret_tensor(buf153, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, attn_weights_42, tensor_29, key_15, transpose_7], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:53
            extern_kernels.bmm(reinterpret_tensor(buf171, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf172, (48, 64, 1024), (65536, 1024, 1), 0), out=buf173)
            buf174 = reinterpret_tensor(buf173, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_42, full_14, attn_weights_43, mask_value_7, attn_weights_44], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf174, arg98_1, 50331648, stream=raw_stream0)
            del arg98_1
            # Topologically Sorted Source Nodes: [attn_weights_42, full_14, attn_weights_43, mask_value_7, attn_weights_44, attn_weights_45], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:54
            buf175 = torch.ops.aten._softmax.default(buf174, -1, False)
            del buf174
            buf176 = buf175
            assert_alignment(buf176, 16, 'torch.ops.aten._softmax.default')
            del buf175
            buf177 = reinterpret_tensor(buf172, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, attn_output_21], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf170, buf177, 3145728, stream=raw_stream0)
            buf178 = reinterpret_tensor(buf171, (48, 1024, 64), (65536, 64, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, attn_output_21, tensor_30, value_15], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:55
            extern_kernels.bmm(reinterpret_tensor(buf176, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf177, (48, 1024, 64), (65536, 64, 1), 0), out=buf178)
            buf179 = reinterpret_tensor(buf177, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf178, buf179, 3145728, stream=raw_stream0)
            buf180 = reinterpret_tensor(buf178, (4096, 768), (768, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31, attn_output_22, view_91, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:56
            extern_kernels.mm(reinterpret_tensor(buf179, (4096, 768), (768, 1), 0), arg100_1, out=buf180)
            del arg100_1
            buf184 = reinterpret_tensor(buf179, (4, 1024, 768), (786432, 768, 1), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [, x_59, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg99_1, buf180, buf165, arg101_1, arg102_1, buf184, 4096, 768, stream=raw_stream0)
            del arg101_1
            del arg102_1
            buf185 = reinterpret_tensor(buf163, (4096, 3072), (3072, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [, x_59, hidden_states_43, hidden_states_44, view_93], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(reinterpret_tensor(buf184, (4096, 768), (768, 1), 0), arg104_1, out=buf185)
            del arg104_1
            buf186 = reinterpret_tensor(buf185, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf186, arg103_1, 12582912, stream=raw_stream0)
            del arg103_1
            buf187 = reinterpret_tensor(buf184, (4096, 768), (768, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45, view_95], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:58
            extern_kernels.mm(reinterpret_tensor(buf186, (4096, 3072), (3072, 1), 0), arg106_1, out=buf187)
            del arg106_1
            buf188 = reinterpret_tensor(buf180, (4, 1024, 768), (786432, 768, 1), 0); del buf180  # reuse
            buf192 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [, x_59, hidden_states_43, x_63, hidden_states_47, hidden_states_48], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf188, arg99_1, buf165, arg105_1, buf187, arg107_1, arg108_1, buf192, 4096, 768, stream=raw_stream0)
            del arg105_1
            del arg107_1
            del arg108_1
            del arg99_1
            buf193 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_48, view_97, x_64], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:59
            extern_kernels.addmm(arg109_1, reinterpret_tensor(buf192, (4096, 768), (768, 1), 0), arg110_1, alpha=1, beta=1, out=buf193)
            del arg109_1
            del arg110_1
            buf194 = reinterpret_tensor(buf192, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf193, buf194, 3145728, stream=raw_stream0)
            buf195 = reinterpret_tensor(buf187, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_33, key_17, transpose_8, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf193, buf195, 3072, 1024, stream=raw_stream0)
            buf196 = reinterpret_tensor(buf176, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, attn_weights_48, tensor_33, key_17, transpose_8], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:60
            extern_kernels.bmm(reinterpret_tensor(buf194, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf195, (48, 64, 1024), (65536, 1024, 1), 0), out=buf196)
            buf197 = reinterpret_tensor(buf196, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_48, full_16, attn_weights_49, mask_value_8, attn_weights_50], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf197, arg111_1, 50331648, stream=raw_stream0)
            del arg111_1
            # Topologically Sorted Source Nodes: [attn_weights_48, full_16, attn_weights_49, mask_value_8, attn_weights_50, attn_weights_51], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:61
            buf198 = torch.ops.aten._softmax.default(buf197, -1, False)
            del buf197
            buf199 = buf198
            assert_alignment(buf199, 16, 'torch.ops.aten._softmax.default')
            del buf198
            buf200 = reinterpret_tensor(buf195, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, attn_output_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf193, buf200, 3145728, stream=raw_stream0)
            buf201 = reinterpret_tensor(buf194, (48, 1024, 64), (65536, 64, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, attn_output_24, tensor_34, value_17], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:62
            extern_kernels.bmm(reinterpret_tensor(buf199, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf200, (48, 1024, 64), (65536, 64, 1), 0), out=buf201)
            buf202 = reinterpret_tensor(buf200, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf201, buf202, 3145728, stream=raw_stream0)
            buf203 = reinterpret_tensor(buf201, (4096, 768), (768, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35, attn_output_25, view_103, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:63
            extern_kernels.mm(reinterpret_tensor(buf202, (4096, 768), (768, 1), 0), arg113_1, out=buf203)
            del arg113_1
            buf207 = reinterpret_tensor(buf202, (4, 1024, 768), (786432, 768, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [, x_67, hidden_states_49, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg112_1, buf203, buf188, arg114_1, arg115_1, buf207, 4096, 768, stream=raw_stream0)
            del arg114_1
            del arg115_1
            buf208 = reinterpret_tensor(buf186, (4096, 3072), (3072, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [, x_67, hidden_states_49, hidden_states_50, view_105], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:64
            extern_kernels.mm(reinterpret_tensor(buf207, (4096, 768), (768, 1), 0), arg117_1, out=buf208)
            del arg117_1
            buf209 = reinterpret_tensor(buf208, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf209, arg116_1, 12582912, stream=raw_stream0)
            del arg116_1
            buf210 = reinterpret_tensor(buf207, (4096, 768), (768, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51, view_107], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:65
            extern_kernels.mm(reinterpret_tensor(buf209, (4096, 3072), (3072, 1), 0), arg119_1, out=buf210)
            del arg119_1
            buf211 = reinterpret_tensor(buf203, (4, 1024, 768), (786432, 768, 1), 0); del buf203  # reuse
            buf215 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [, x_67, hidden_states_49, x_71, hidden_states_53, hidden_states_54], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf211, arg112_1, buf188, arg118_1, buf210, arg120_1, arg121_1, buf215, 4096, 768, stream=raw_stream0)
            del arg112_1
            del arg118_1
            del arg120_1
            del arg121_1
            del buf188
            buf216 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_54, view_109, x_72], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:66
            extern_kernels.addmm(arg122_1, reinterpret_tensor(buf215, (4096, 768), (768, 1), 0), arg123_1, alpha=1, beta=1, out=buf216)
            del arg122_1
            del arg123_1
            buf217 = reinterpret_tensor(buf215, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf216, buf217, 3145728, stream=raw_stream0)
            buf218 = reinterpret_tensor(buf210, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_37, key_19, transpose_9, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf216, buf218, 3072, 1024, stream=raw_stream0)
            buf219 = reinterpret_tensor(buf199, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, attn_weights_54, tensor_37, key_19, transpose_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:67
            extern_kernels.bmm(reinterpret_tensor(buf217, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf218, (48, 64, 1024), (65536, 1024, 1), 0), out=buf219)
            buf220 = reinterpret_tensor(buf219, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_54, full_18, attn_weights_55, mask_value_9, attn_weights_56], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf220, arg124_1, 50331648, stream=raw_stream0)
            del arg124_1
            # Topologically Sorted Source Nodes: [attn_weights_54, full_18, attn_weights_55, mask_value_9, attn_weights_56, attn_weights_57], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:68
            buf221 = torch.ops.aten._softmax.default(buf220, -1, False)
            del buf220
            buf222 = buf221
            assert_alignment(buf222, 16, 'torch.ops.aten._softmax.default')
            del buf221
            buf223 = reinterpret_tensor(buf218, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, attn_output_27], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf216, buf223, 3145728, stream=raw_stream0)
            buf224 = reinterpret_tensor(buf217, (48, 1024, 64), (65536, 64, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, attn_output_27, tensor_38, value_19], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:69
            extern_kernels.bmm(reinterpret_tensor(buf222, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf223, (48, 1024, 64), (65536, 64, 1), 0), out=buf224)
            buf225 = reinterpret_tensor(buf223, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf224, buf225, 3145728, stream=raw_stream0)
            buf226 = reinterpret_tensor(buf224, (4096, 768), (768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39, attn_output_28, view_115, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(reinterpret_tensor(buf225, (4096, 768), (768, 1), 0), arg126_1, out=buf226)
            del arg126_1
            buf230 = reinterpret_tensor(buf225, (4, 1024, 768), (786432, 768, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [, x_75, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg125_1, buf226, buf211, arg127_1, arg128_1, buf230, 4096, 768, stream=raw_stream0)
            del arg127_1
            del arg128_1
            buf231 = reinterpret_tensor(buf209, (4096, 3072), (3072, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [, x_75, hidden_states_55, hidden_states_56, view_117], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:71
            extern_kernels.mm(reinterpret_tensor(buf230, (4096, 768), (768, 1), 0), arg130_1, out=buf231)
            del arg130_1
            buf232 = reinterpret_tensor(buf231, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf232, arg129_1, 12582912, stream=raw_stream0)
            del arg129_1
            buf233 = reinterpret_tensor(buf230, (4096, 768), (768, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57, view_119], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:72
            extern_kernels.mm(reinterpret_tensor(buf232, (4096, 3072), (3072, 1), 0), arg132_1, out=buf233)
            del arg132_1
            del buf232
            buf234 = reinterpret_tensor(buf226, (4, 1024, 768), (786432, 768, 1), 0); del buf226  # reuse
            buf238 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_75, hidden_states_55, x_79, hidden_states_59, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf234, arg125_1, buf211, arg131_1, buf233, arg133_1, arg134_1, buf238, 4096, 768, stream=raw_stream0)
            del arg125_1
            del arg131_1
            del arg133_1
            del arg134_1
            buf239 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_60, view_121, x_80], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:73
            extern_kernels.addmm(arg135_1, reinterpret_tensor(buf238, (4096, 768), (768, 1), 0), arg136_1, alpha=1, beta=1, out=buf239)
            del arg135_1
            del arg136_1
            buf240 = reinterpret_tensor(buf238, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf239, buf240, 3145728, stream=raw_stream0)
            buf241 = reinterpret_tensor(buf233, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_41, key_21, transpose_10, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf239, buf241, 3072, 1024, stream=raw_stream0)
            buf242 = reinterpret_tensor(buf222, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf222  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, attn_weights_60, tensor_41, key_21, transpose_10], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:74
            extern_kernels.bmm(reinterpret_tensor(buf240, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf241, (48, 64, 1024), (65536, 1024, 1), 0), out=buf242)
            buf243 = reinterpret_tensor(buf242, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_60, full_20, attn_weights_61, mask_value_10, attn_weights_62], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf243, arg137_1, 50331648, stream=raw_stream0)
            del arg137_1
            # Topologically Sorted Source Nodes: [attn_weights_60, full_20, attn_weights_61, mask_value_10, attn_weights_62, attn_weights_63], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:75
            buf244 = torch.ops.aten._softmax.default(buf243, -1, False)
            del buf243
            buf245 = buf244
            assert_alignment(buf245, 16, 'torch.ops.aten._softmax.default')
            del buf244
            buf246 = reinterpret_tensor(buf241, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, attn_output_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf239, buf246, 3145728, stream=raw_stream0)
            buf247 = reinterpret_tensor(buf240, (48, 1024, 64), (65536, 64, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, attn_output_30, tensor_42, value_21], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:76
            extern_kernels.bmm(reinterpret_tensor(buf245, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf246, (48, 1024, 64), (65536, 64, 1), 0), out=buf247)
            buf248 = reinterpret_tensor(buf246, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf247, buf248, 3145728, stream=raw_stream0)
            buf249 = reinterpret_tensor(buf247, (4096, 768), (768, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43, attn_output_31, view_127, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:77
            extern_kernels.mm(reinterpret_tensor(buf248, (4096, 768), (768, 1), 0), arg139_1, out=buf249)
            del arg139_1
            buf253 = reinterpret_tensor(buf248, (4, 1024, 768), (786432, 768, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [, x_83, hidden_states_61, hidden_states_62], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg138_1, buf249, buf234, arg140_1, arg141_1, buf253, 4096, 768, stream=raw_stream0)
            del arg140_1
            del arg141_1
            buf254 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_83, hidden_states_61, hidden_states_62, view_129], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:78
            extern_kernels.mm(reinterpret_tensor(buf253, (4096, 768), (768, 1), 0), arg143_1, out=buf254)
            del arg143_1
            buf255 = reinterpret_tensor(buf254, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf255, arg142_1, 12582912, stream=raw_stream0)
            del arg142_1
            buf256 = reinterpret_tensor(buf253, (4096, 768), (768, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63, view_131], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:79
            extern_kernels.mm(reinterpret_tensor(buf255, (4096, 3072), (3072, 1), 0), arg145_1, out=buf256)
            del arg145_1
            del buf255
            buf257 = reinterpret_tensor(buf249, (4, 1024, 768), (786432, 768, 1), 0); del buf249  # reuse
            buf261 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [, x_83, hidden_states_61, x_87, hidden_states_65, hidden_states_66], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_11.run(buf257, arg138_1, buf234, arg144_1, buf256, arg146_1, arg147_1, buf261, 4096, 768, stream=raw_stream0)
            del arg138_1
            del arg144_1
            del arg146_1
            del arg147_1
            del buf234
            buf262 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_66, view_133, x_88], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:80
            extern_kernels.addmm(arg148_1, reinterpret_tensor(buf261, (4096, 768), (768, 1), 0), arg149_1, alpha=1, beta=1, out=buf262)
            del arg148_1
            del arg149_1
            buf263 = reinterpret_tensor(buf261, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, attn_weights_66], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_2.run(buf262, buf263, 3145728, stream=raw_stream0)
            buf264 = reinterpret_tensor(buf256, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf256  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_45, key_23, transpose_11, attn_weights_66], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_3.run(buf262, buf264, 3072, 1024, stream=raw_stream0)
            buf265 = reinterpret_tensor(buf245, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, attn_weights_66, tensor_45, key_23, transpose_11], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:81
            extern_kernels.bmm(reinterpret_tensor(buf263, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf264, (48, 64, 1024), (65536, 1024, 1), 0), out=buf265)
            del buf263
            del buf264
            buf266 = reinterpret_tensor(buf265, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [attn_weights_66, full_22, attn_weights_67, mask_value_11, attn_weights_68], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_4.run(buf266, arg150_1, 50331648, stream=raw_stream0)
            del arg150_1
            # Topologically Sorted Source Nodes: [attn_weights_66, full_22, attn_weights_67, mask_value_11, attn_weights_68, attn_weights_69], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:82
            buf267 = torch.ops.aten._softmax.default(buf266, -1, False)
            del buf266
            buf268 = buf267
            assert_alignment(buf268, 16, 'torch.ops.aten._softmax.default')
            del buf267
            buf269 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_5.run(buf262, buf269, 3145728, stream=raw_stream0)
            buf270 = empty_strided_npu((48, 1024, 64), (65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, attn_output_33, tensor_46, value_23], Original ATen: [aten.view, aten.split, aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:83
            extern_kernels.bmm(reinterpret_tensor(buf268, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf269, (48, 1024, 64), (65536, 64, 1), 0), out=buf270)
            del buf268
            buf271 = reinterpret_tensor(buf269, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_6.run(buf270, buf271, 3145728, stream=raw_stream0)
            buf272 = reinterpret_tensor(buf270, (4096, 768), (768, 1), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47, attn_output_34, view_139, ], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:84
            extern_kernels.mm(reinterpret_tensor(buf271, (4096, 768), (768, 1), 0), arg152_1, out=buf272)
            del arg152_1
            buf276 = reinterpret_tensor(buf271, (4, 1024, 768), (786432, 768, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [, x_91, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_10.run(arg151_1, buf272, buf257, arg153_1, arg154_1, buf276, 4096, 768, stream=raw_stream0)
            del arg153_1
            del arg154_1
            buf277 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_91, hidden_states_67, hidden_states_68, view_141], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] extern_kernels.mm:85
            extern_kernels.mm(reinterpret_tensor(buf276, (4096, 768), (768, 1), 0), arg156_1, out=buf277)
            del arg156_1
            buf278 = reinterpret_tensor(buf277, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_mul_pow_tanh_view_8.run(buf278, arg155_1, 12582912, stream=raw_stream0)
            del arg155_1
            buf279 = reinterpret_tensor(buf276, (4096, 768), (768, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69, view_143], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(reinterpret_tensor(buf278, (4096, 3072), (3072, 1), 0), arg158_1, out=buf279)
            del arg158_1
            del buf278
            buf280 = reinterpret_tensor(buf272, (4, 1024, 768), (786432, 768, 1), 0); del buf272  # reuse
            buf284 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [, x_91, hidden_states_67, x_95, hidden_states_71, layer_norm_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_addmm_native_layer_norm_view_12.run(buf284, arg151_1, buf257, arg157_1, buf279, arg159_1, arg160_1, 4096, 768, stream=raw_stream0)
            del arg151_1
            del arg157_1
            del arg159_1
            del arg160_1
            del buf257
            del buf279
            buf285 = empty_strided_npu((4096, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_24, logits], Original ATen: [aten.native_layer_norm, aten.matmul, aten.t]
            # [Provenance debug handles] extern_kernels.mm:87
            extern_kernels.mm(reinterpret_tensor(buf284, (4096, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 2), (1, 768), 0), out=buf285)
            del arg161_1
            del buf284
            buf286 = empty_strided_npu((4, 1024), (1024, 1), torch.int32)
            # Topologically Sorted Source Nodes: [eq, int_1], Original ATen: [aten.eq, aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_eq_13.run(arg0_1, buf286, 4096, stream=raw_stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [eq, int_1, argmax], Original ATen: [aten.eq, aten._to_copy, aten.argmax]
            # [Provenance debug handles] torch.ops.aten.argmax.default:88
            buf287 = torch.ops.aten.argmax.default(buf286, -1)
            del buf286
            buf288 = buf287
            assert_alignment(buf288, 16, 'torch.ops.aten.argmax.default')
            del buf287
            buf289 = empty_strided_npu((4, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [arange_1], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_14.run(buf289, 4, stream=raw_stream0)
            buf290 = buf288; del buf288  # reuse
            # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sub_15.run(buf290, 4, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [logits, arange_1, sub, pooled_logits], Original ATen: [aten.matmul, aten.arange, aten.sub, aten.index]
            # [Provenance debug handles] torch.ops.aten.index.Tensor:89
            buf291 = torch.ops.aten.index.Tensor(reinterpret_tensor(buf285, (4, 1024, 2), (2048, 2, 1), 0), [buf289, buf290])
            del buf285
            del buf289
            buf292 = buf291
            assert_alignment(buf292, 16, 'torch.ops.aten.index.Tensor')
            del buf291
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:90
            buf293 = torch.ops.aten._log_softmax.default(buf292, 1, False)
            buf294 = buf293
            assert_alignment(buf294, 16, 'torch.ops.aten._log_softmax.default')
            del buf293
            arg162_1 = copy_if_misaligned(arg162_1)
            buf295 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_16.run(arg162_1, buf295, 4, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:91
            buf296 = torch.ops.aten.gather.default(buf294, 1, reinterpret_tensor(buf295, (4, 1), (1, 0), 0))
            del buf294
            del buf295
            buf297 = buf296
            assert_alignment(buf297, 16, 'torch.ops.aten.gather.default')
            del buf296
            buf298 = empty_strided_npu((), (), torch.float32)
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_17.run(arg162_1, buf297, buf298, 1, stream=raw_stream0)
            del arg162_1
            del buf297
        return (buf298, buf292, reinterpret_tensor(buf9, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf9, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf32, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf32, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf55, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf55, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf78, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf78, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf101, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf101, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf124, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf124, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf147, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf147, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf170, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf170, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf193, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf193, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf216, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf216, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf239, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf239, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf262, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf262, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 1024), (1024, 1), device='npu:0', dtype=torch.int64)
    arg1_1 = rand_strided((50257, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg8_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg21_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg31_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg33_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg34_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg41_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg44_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg46_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg47_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg54_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg57_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg59_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg60_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg70_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg72_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg73_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg77_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg83_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg85_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg86_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg90_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg93_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg96_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg98_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg99_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg103_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg109_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg111_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg112_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg122_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg124_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg125_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg129_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg135_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg137_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg138_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg148_1 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    arg150_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='npu:0', dtype=torch.bool)
    arg151_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg155_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='npu:0', dtype=torch.float32)
    arg161_1 = rand_strided((2, 768), (768, 1), device='npu:0', dtype=torch.float32)
    arg162_1 = rand_strided((4, ), (1, ), device='npu:0', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('GPT2ForSequenceClassification', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
