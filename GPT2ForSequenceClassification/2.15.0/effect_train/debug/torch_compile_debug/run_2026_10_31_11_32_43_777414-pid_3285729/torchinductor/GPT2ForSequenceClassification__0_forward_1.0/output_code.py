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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/qn/cqn6fvliwgxa2ucdqork5dm4nkwxzdvwdzl73yx63xsnqqio5van.py
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
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/fr/cfrzb6af2xyqnk45qa6qi3tsckdql5yqcjwf3gn52da24gmsa4pa.py
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/5d/c5dvrelxszjtat4nn7vlhr74plimiwxwdgwe64nhb2xkslp37bbb.py
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
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/hn/chn2mayjqbcz2wn5jqdw3j5ot2cze3dfznxodf4rzerhhdfnihtp.py
# Topologically Sorted Source Nodes: [loss, unsqueeze_2, ne_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   loss => full_default_24, ne, where_12
#   ne_3 => ne_3
#   unsqueeze_2 => unsqueeze_2
# Graph fragment:
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %ne : Tensor "b8[4][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_163, -100), kwargs = {})
#   %full_default_24 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/7n/c7nkpjl2moda6xhkd7wlo4zrb3b5fw3arwzgifnrk33v336egeep.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %embedding : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=embedding_1]
#   %add : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   return %add
triton_unk_fused_add_4 = async_compile.triton('triton_unk_fused_add_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [786432, 4, {'divisors': [1, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 786432, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 786432, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/4b/c4bdtfpldcr6zyzns2numzsthovlmxvfzxa7pzma5pzimr53purd.py
# Topologically Sorted Source Nodes: [unsqueeze_2, scalar_tensor_2, where_14, ], Original ATen: [aten.nll_loss_backward, aten.expand]
# Source node to ATen node mapping:
#    => expand_default
#   scalar_tensor_2 => full_default_26
#   unsqueeze_2 => unsqueeze_2
#   where_14 => where_14
# Graph fragment:
#   %ne_3 : Tensor "b8[4, 1][1, 1]npu:0" = PlaceHolder[target=ne_3]
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %unsqueeze_2 : Tensor "i64[4, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_163, 1), kwargs = {})
#   %full_default_26 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_14 : Tensor "i64[4, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_2, %full_default_26), kwargs = {})
#   %expand_default : Tensor "i64[4, 2][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_14, [4, 2]), kwargs = {})
#   return %expand_default
triton_unk_fused_expand_nll_loss_backward_5 = async_compile.triton('triton_unk_fused_expand_nll_loss_backward_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr1': '*i64', 'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_expand_nll_loss_backward_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_expand_nll_loss_backward_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/e4/ce45clptljearwp4t2nfqeuknsrjvbx7rlneu2mf5c5mghwkcnfs.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub => sub_25
# Graph fragment:
#   %argmax : Tensor "i64[4][1]npu:0" = PlaceHolder[target=argmax]
#   %sub_25 : Tensor "i64[4][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%argmax, 1), kwargs = {})
#   return %sub_25
triton_unk_fused_sub_6 = async_compile.triton('triton_unk_fused_sub_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'in_out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sub_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_sub_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/k4/ck4mktc5f2eqniznsmmgcvwkqocxxnh656kplbca3gvuqf7ej7at.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.arange, aten.view, aten.eq]
# Source node to ATen node mapping:
#    => eq_tensor, iota_default, view_default
# Graph fragment:
#   %expand_default : Tensor "i64[4, 2][2, 1]npu:0" = PlaceHolder[target=expand_default]
#   %iota_default : Tensor "i64[2][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 2]), kwargs = {})
#   %eq_tensor : Tensor "b8[4, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   return %eq_tensor
triton_unk_fused_arange_eq_view_7 = async_compile.triton('triton_unk_fused_arange_eq_view_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [2, 4, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_eq_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '9F681E53ED2A3D3571AE6D150EB6CFCC50660564ABA6681E2B34CC5DF0391ADB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'npu_linearize': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_eq_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/bn/cbnzzbl6swva4cpnvz4jusogyyyomyj3tp2msp7yv6xe2wnwhwln.py
# Topologically Sorted Source Nodes: [hidden_states, div_50], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_50 => div_50
#   hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %buf7 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf7]
#   %getitem_3 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_3]
#   %buf9 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf9]
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
#   return %buf7,%getitem_3,%buf9,%mul,%add_2,%div_50
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/ju/cjuqovopmnsaiatcvohmhriohwkh3c43x2okxef35hxjbjf4azwi.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/ug/cug442i7iegf5htlco3ufvdcthekya4fih32vuholx5bdt6hzpyq.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/hf/chf6rlorbikvhcdxvfafyg3iqesvjv74lpzil6bgbdenzxeejofg.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/sb/csbxevitzil7jrsi75vvrx66msv7wb3weodtzuf4lcwynmrvehhe.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/56/c56wc6dgdxmy5zfauzviyogqafaleyoqlajrsml74ofmrcpnrycs.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 1024, 12, 4, {'divisors': [1, 64, 65536, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 1024, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 12, 'divisor': 65536, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/mh/cmhfte7e2glw6em2bhembfhhdkyuz4ed2kf274cfkbnlcf6237xk.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [768, 4, {'divisors': [1, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 768, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/vn/cvnoxnynu3detiuwrd5r27hlwphgyhc7jlq6m3vvxvp2c3h4asno.py
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [1048576, 48, {'divisors': [1, 1048576]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1048576, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 48, 'divisor': 1048576, 'seed': 2}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/b3/cb3nadz63ai7ds7mvmc6d76aj67onaicadazp22takm5tq6vf4cu.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 48, {'divisors': [1, 1024]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 48, 'divisor': 1024, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/p2/cp2wgiiax2llrn2p6f3bprrpmdbsv2kwwh7e2lgctkm5x35nnhje.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [64, 12, 1024, 4, {'divisors': [1, 64, 768, 786432]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 786432, 'seed': 2}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/iq/ciq637jc5p2ave33ds3m44v6otklo2oe4hopiq36xk22rqh2gyx5.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 48, {'divisors': [1, 1024]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 48, 'divisor': 1024, 'seed': 2}, {'name': 'x2', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/t5/ct57zq7i6vbuy5mxuynb5q4rrhgxbebhm3kqkilopsozk6evef3t.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, div_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_48 => div_48
#   hidden_states_1 => add_3
#   hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %getitem_9 : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem_9]
#   %getitem : Tensor "f32[4, 1024, 768][786432, 768, 1]npu:0" = PlaceHolder[target=getitem]
#   %buf28 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf28]
#   %getitem_12 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_12]
#   %buf30 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf30]
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
#   return %buf28,%getitem_12,%buf30,%mul_2,%add_5,%div_48
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/f7/cf7kpnkvlgodviavgm3i3ejsiua3ekc6762qpoubqcf73cdeq4mf.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [12582912, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 12582912, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/hs/chse4svk6aa4y5cbtib2jshgjo77s3gigo7mtycvg6e5cor2f2yl.py
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
#   %buf39 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf39]
#   %getitem_16 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_16]
#   %buf41 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf41]
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
#   return %buf39,%getitem_16,%buf41,%mul_8,%add_10,%div_47
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/2e/c2e3x5omz4xuhhs2d7u433hlaxpj6nle254cbovnlmgu3kjhqj62.py
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
#   %buf60 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf60]
#   %getitem_25 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_25]
#   %buf62 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf62]
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
#   return %buf60,%getitem_25,%buf62,%mul_10,%add_13,%div_45
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/kr/ckrjigxrtluya3kkywqohk3bin565sikoaaxeh25pclkvmutwll6.py
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
#   %buf72 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf72]
#   %getitem_29 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_29]
#   %buf74 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf74]
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
#   return %add_16,%buf72,%getitem_29,%buf74,%mul_16,%add_18,%div_44
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/2n/c2nw35nerpjxjn52pfvglkptcfms6zonkip2pq7gprif4u5iicu5.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 4, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/77/c775zczfqtjlkq223hsraovfrgjp25cpbvpfvlwgs5i23cub5isx.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [1024, 12, 4, {'divisors': [1, 1024, 12288]}], 'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 1024, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 1024, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 12288, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/ee/ceeiq7hu6lnyif7wmmpqbnqooxqjr5gx2tvu5hjx3adrkcp4f3ky.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'block_hints': {'YBLOCK_HINT': [64, 12, 4, {'divisors': [1, 64, 768]}], 'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 12, 'divisor': 64, 'seed': 2}, {'name': 'y2', 'length': 4, 'divisor': 768, 'seed': 2}, {'name': 'x3', 'length': 1024, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/m7/cm7frv5skupbmvuuexgmyqsxdr6ewl4vobirqmkygykyz77y7hkz.py
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
#   %buf397 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf397]
#   %getitem_159 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=getitem_159]
#   %buf399 : Tensor "f32[4, 1024, 1][1024, 1, 4096]npu:0" = PlaceHolder[target=buf399]
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
#   return %add_96,%buf397,%getitem_159,%buf399,%mul_96,%add_98,%div_14
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [768, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 768, 'divisor': 1, 'seed': 1}]},
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


# kernel path: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/GPT2ForSequenceClassification/2.15.0/effect_train/cache/pv/cpvjewitu2pmy4664abhvajth77p43nqpceejwow5wirfwcy6rr5.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_13, div_12, full_default_25, ne, ne_2, neg, squeeze, sum_1, sum_2, where_13
# Graph fragment:
#   %primals_163 : Tensor "i64[4][1]npu:0" = PlaceHolder[target=primals_163]
#   %gather : Tensor "f32[4, 1][1, 1]npu:0" = PlaceHolder[target=gather]
#   %convert_element_type_13 : Tensor "f32[][]npu:0" = PlaceHolder[target=convert_element_type_13]
#   %ne : Tensor "b8[4][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_163, -100), kwargs = {})
#   %squeeze : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_25 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_13 : Tensor "f32[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_25), kwargs = {})
#   %ne_2 : Tensor "b8[4][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_163, -100), kwargs = {})
#   %sum_1 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr'}, 'downcast_args': {'in_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'xnumel': 1}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'block_hints': {'XBLOCK_HINT': [{'divisors': []}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': []},
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
            assert_alignment(buf1, 16, 'torch.ops.aten.embedding.default')
            del primals_2
            buf403 = empty_strided_npu((4, 1024), (1024, 1), torch.int32)
            # Topologically Sorted Source Nodes: [eq, int_1], Original ATen: [aten.eq, aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_eq_1.run(primals_1, buf403, 4096, stream=raw_stream0)
            buf406 = empty_strided_npu((4, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [arange_1], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_2.run(buf406, 4, stream=raw_stream0)
            primals_163 = copy_if_misaligned(primals_163)
            buf409 = empty_strided_npu((4, ), (1, ), torch.int64)
            buf412 = empty_strided_npu((4, 1), (1, 1), torch.bool)
            # Topologically Sorted Source Nodes: [loss, unsqueeze_2, ne_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_backward_nll_loss_forward_3.run(primals_163, buf409, buf412, 4, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [position_ids, position_ids_1, position_embeds], Original ATen: [aten.arange, aten.unsqueeze, aten.embedding]
            # [Provenance debug handles] torch.ops.aten.embedding.default:2
            buf2 = torch.ops.aten.embedding.default(primals_3, reinterpret_tensor(buf0, (1, 1024), (0, 1), 0))
            assert_alignment(buf2, 16, 'torch.ops.aten.embedding.default')
            del primals_3
            buf3 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_4.run(buf3, buf2, 3145728, stream=raw_stream0)
            del buf2
            # Topologically Sorted Source Nodes: [eq, int_1, argmax], Original ATen: [aten.eq, aten._to_copy, aten.argmax]
            # [Provenance debug handles] torch.ops.aten.argmax.default:3
            buf404 = torch.ops.aten.argmax.default(buf403, -1)
            assert_alignment(buf404, 16, 'torch.ops.aten.argmax.default')
            del buf403
            buf413 = empty_strided_npu((4, 2), (2, 1), torch.int64)
            # Topologically Sorted Source Nodes: [unsqueeze_2, scalar_tensor_2, where_14, ], Original ATen: [aten.nll_loss_backward, aten.expand]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_expand_nll_loss_backward_5.run(buf412, primals_163, buf413, 8, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [add, residual], Original ATen: [aten.add, aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:4
            buf4 = torch.ops.npu._npu_dropout.default(buf3, 0.1)
            buf405 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_sub_6.run(buf405, 4, stream=raw_stream0)
            buf414 = empty_strided_npu((4, 2), (2, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.arange, aten.view, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_arange_eq_view_7.run(buf413, buf414, 8, stream=raw_stream0)
            del buf413
            buf5 = buf4[0]
            assert_alignment(buf5, 16, 'torch.ops.npu._npu_dropout.default')
            buf6 = buf4[1]
            assert_alignment(buf6, 16, 'torch.ops.npu._npu_dropout.default')
            del buf4
            buf10 = buf3; del buf3  # reuse
            buf11 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf499 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states, div_50], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_native_layer_norm_native_layer_norm_backward_8.run(buf5, primals_4, primals_5, buf10, buf11, buf499, 4096, 768, stream=raw_stream0)
            del primals_5
            buf12 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states, view_1, x], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(primals_6, reinterpret_tensor(buf11, (4096, 768), (768, 1), 0), primals_7, alpha=1, beta=1, out=buf12)
            del primals_6
            buf13 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf12, buf13, 3145728, stream=raw_stream0)
            buf14 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf12, buf14, 3072, 1024, stream=raw_stream0)
            buf21 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf12, buf21, 3145728, stream=raw_stream0)
            buf494 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, permute_258, clone_115], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf12, buf494, 3072, 1024, stream=raw_stream0)
            buf497 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor_1, key_1, transpose, permute_260, clone_117], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf12, buf497, 3145728, stream=raw_stream0)
            buf498 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, permute_261, clone_118], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf12, buf498, 3072, 1024, stream=raw_stream0)
            buf15 = empty_strided_npu((48, 1024, 1024), (1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, split, tensor, query_1, tensor_1, key_1, transpose, attn_weights], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:6
            extern_kernels.bmm(reinterpret_tensor(buf13, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf14, (48, 64, 1024), (65536, 1024, 1), 0), out=buf15)
            buf16 = reinterpret_tensor(buf15, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf16, primals_8, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [attn_weights, full, attn_weights_1, mask_value, attn_weights_2, attn_weights_3], Original ATen: [aten.matmul, aten.full, aten.div, aten._to_copy, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:7
            buf17 = torch.ops.aten._softmax.default(buf16, -1, False)
            assert_alignment(buf17, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_5], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:8
            buf18 = torch.ops.npu._npu_dropout.default(buf17, 0.1)
            buf19 = buf18[0]
            assert_alignment(buf19, 16, 'torch.ops.npu._npu_dropout.default')
            buf20 = buf18[1]
            assert_alignment(buf20, 16, 'torch.ops.npu._npu_dropout.default')
            del buf18
            buf22 = reinterpret_tensor(buf14, (48, 1024, 64), (65536, 64, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [x_1, split, tensor_2, value_1, attn_output], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:9
            extern_kernels.bmm(reinterpret_tensor(buf19, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf21, (48, 1024, 64), (65536, 64, 1), 0), out=buf22)
            buf495 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [permute_259], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf19, buf495, 49152, 1024, stream=raw_stream0)
            buf23 = reinterpret_tensor(buf21, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf22, buf23, 3145728, stream=raw_stream0)
            buf496 = reinterpret_tensor(buf19, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [view_549], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf495, buf496, 49152, 1024, stream=raw_stream0)
            buf24 = reinterpret_tensor(buf22, (4096, 768), (768, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [attn_output, permute_3, tensor_3, attn_output_1, view_7, x_2], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:10
            extern_kernels.addmm(primals_9, reinterpret_tensor(buf23, (4096, 768), (768, 1), 0), primals_10, alpha=1, beta=1, out=buf24)
            del primals_9
            # Unsorted Source Nodes: [attn_output_2, x_3], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:11
            buf25 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf24, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf26 = buf25[0]
            assert_alignment(buf26, 16, 'torch.ops.npu._npu_dropout.default')
            buf31 = reinterpret_tensor(buf24, (4, 1024, 768), (786432, 768, 1), 0); del buf24  # reuse
            buf32 = reinterpret_tensor(buf13, (4, 1024, 768), (786432, 768, 1), 0); del buf13  # reuse
            buf493 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, div_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf26, buf5, primals_11, primals_12, buf31, buf32, buf493, 4096, 768, stream=raw_stream0)
            del primals_12
            buf27 = buf25[1]
            assert_alignment(buf27, 16, 'torch.ops.npu._npu_dropout.default')
            del buf25
            buf33 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_2, view_9, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:12
            extern_kernels.addmm(primals_13, reinterpret_tensor(buf32, (4096, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf33)
            del primals_13
            buf34 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf33, buf34, 12582912, stream=raw_stream0)
            buf35 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_3, view_11, x_6], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:13
            extern_kernels.addmm(primals_15, reinterpret_tensor(buf34, (4096, 3072), (3072, 1), 0), primals_16, alpha=1, beta=1, out=buf35)
            del primals_15
            # Unsorted Source Nodes: [hidden_states_4, x_7], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:14
            buf36 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf35, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf37 = buf36[0]
            assert_alignment(buf37, 16, 'torch.ops.npu._npu_dropout.default')
            buf42 = reinterpret_tensor(buf35, (4, 1024, 768), (786432, 768, 1), 0); del buf35  # reuse
            buf43 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf492 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_6, div_47], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf26, buf5, buf37, primals_17, primals_18, buf42, buf43, buf492, 4096, 768, stream=raw_stream0)
            del primals_18
            buf38 = buf36[1]
            assert_alignment(buf38, 16, 'torch.ops.npu._npu_dropout.default')
            del buf36
            buf44 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_6, view_13, x_8], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:15
            extern_kernels.addmm(primals_19, reinterpret_tensor(buf43, (4096, 768), (768, 1), 0), primals_20, alpha=1, beta=1, out=buf44)
            del primals_19
            buf45 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf44, buf45, 3145728, stream=raw_stream0)
            buf46 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_5, key_3, transpose_1, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf44, buf46, 3072, 1024, stream=raw_stream0)
            buf53 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, attn_output_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf44, buf53, 3145728, stream=raw_stream0)
            buf487 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, permute_241, clone_109], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf44, buf487, 3072, 1024, stream=raw_stream0)
            buf490 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_5, key_3, transpose_1, permute_243, clone_111], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf44, buf490, 3145728, stream=raw_stream0)
            buf491 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, permute_244, clone_112], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf44, buf491, 3072, 1024, stream=raw_stream0)
            buf47 = reinterpret_tensor(buf495, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf495  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_4, query_3, tensor_5, key_3, transpose_1, attn_weights_6], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:16
            extern_kernels.bmm(reinterpret_tensor(buf45, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf46, (48, 64, 1024), (65536, 1024, 1), 0), out=buf47)
            buf48 = reinterpret_tensor(buf47, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_6, attn_weights_7, attn_weights_8], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf48, primals_21, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_6, attn_weights_7, attn_weights_8, attn_weights_9], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:17
            buf49 = torch.ops.aten._softmax.default(buf48, -1, False)
            assert_alignment(buf49, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_11], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:18
            buf50 = torch.ops.npu._npu_dropout.default(buf49, 0.1)
            buf51 = buf50[0]
            assert_alignment(buf51, 16, 'torch.ops.npu._npu_dropout.default')
            buf52 = buf50[1]
            assert_alignment(buf52, 16, 'torch.ops.npu._npu_dropout.default')
            del buf50
            buf54 = reinterpret_tensor(buf46, (48, 1024, 64), (65536, 64, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [x_9, split_1, tensor_6, value_3, attn_output_3], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:19
            extern_kernels.bmm(reinterpret_tensor(buf51, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf53, (48, 1024, 64), (65536, 64, 1), 0), out=buf54)
            buf488 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [permute_242], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf51, buf488, 49152, 1024, stream=raw_stream0)
            buf55 = reinterpret_tensor(buf53, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf54, buf55, 3145728, stream=raw_stream0)
            buf489 = reinterpret_tensor(buf51, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [view_521], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf488, buf489, 49152, 1024, stream=raw_stream0)
            buf56 = reinterpret_tensor(buf54, (4096, 768), (768, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, permute_7, tensor_7, attn_output_4, view_19, x_10], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:20
            extern_kernels.addmm(primals_22, reinterpret_tensor(buf55, (4096, 768), (768, 1), 0), primals_23, alpha=1, beta=1, out=buf56)
            del primals_22
            # Unsorted Source Nodes: [attn_output_5, x_11], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:21
            buf57 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf56, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf58 = buf57[0]
            assert_alignment(buf58, 16, 'torch.ops.npu._npu_dropout.default')
            buf63 = reinterpret_tensor(buf56, (4, 1024, 768), (786432, 768, 1), 0); del buf56  # reuse
            buf64 = reinterpret_tensor(buf45, (4, 1024, 768), (786432, 768, 1), 0); del buf45  # reuse
            buf486 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_8, div_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf58, buf26, buf5, buf37, primals_24, primals_25, buf63, buf64, buf486, 4096, 768, stream=raw_stream0)
            del primals_25
            buf59 = buf57[1]
            assert_alignment(buf59, 16, 'torch.ops.npu._npu_dropout.default')
            del buf57
            buf65 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_8, view_21, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(primals_26, reinterpret_tensor(buf64, (4096, 768), (768, 1), 0), primals_27, alpha=1, beta=1, out=buf65)
            del primals_26
            buf66 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf65, buf66, 12582912, stream=raw_stream0)
            buf67 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_9, view_23, x_14], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:23
            extern_kernels.addmm(primals_28, reinterpret_tensor(buf66, (4096, 3072), (3072, 1), 0), primals_29, alpha=1, beta=1, out=buf67)
            del primals_28
            # Unsorted Source Nodes: [hidden_states_10, x_15], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:24
            buf68 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf67, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf69 = buf68[0]
            assert_alignment(buf69, 16, 'torch.ops.npu._npu_dropout.default')
            buf71 = buf26; del buf26  # reuse
            buf75 = reinterpret_tensor(buf67, (4, 1024, 768), (786432, 768, 1), 0); del buf67  # reuse
            buf76 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf485 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_5, hidden_states_7, hidden_states_11, hidden_states_12, div_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf71, buf58, buf5, buf37, buf69, primals_30, primals_31, buf75, buf76, buf485, 4096, 768, stream=raw_stream0)
            del primals_31
            buf70 = buf68[1]
            assert_alignment(buf70, 16, 'torch.ops.npu._npu_dropout.default')
            del buf68
            buf77 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_12, view_25, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(primals_32, reinterpret_tensor(buf76, (4096, 768), (768, 1), 0), primals_33, alpha=1, beta=1, out=buf77)
            del primals_32
            buf78 = reinterpret_tensor(buf69, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf77, buf78, 3145728, stream=raw_stream0)
            buf79 = reinterpret_tensor(buf58, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_9, key_5, transpose_2, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf77, buf79, 3072, 1024, stream=raw_stream0)
            buf86 = reinterpret_tensor(buf5, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, attn_output_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf77, buf86, 3145728, stream=raw_stream0)
            buf480 = reinterpret_tensor(buf37, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, permute_224, clone_103], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf77, buf480, 3072, 1024, stream=raw_stream0)
            buf483 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_9, key_5, transpose_2, permute_226, clone_105], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf77, buf483, 3145728, stream=raw_stream0)
            buf484 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, permute_227, clone_106], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf77, buf484, 3072, 1024, stream=raw_stream0)
            buf80 = reinterpret_tensor(buf488, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_8, query_5, tensor_9, key_5, transpose_2, attn_weights_12], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:26
            extern_kernels.bmm(reinterpret_tensor(buf78, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf79, (48, 64, 1024), (65536, 1024, 1), 0), out=buf80)
            buf81 = reinterpret_tensor(buf80, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_12, attn_weights_13, attn_weights_14], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf81, primals_34, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_12, attn_weights_13, attn_weights_14, attn_weights_15], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:27
            buf82 = torch.ops.aten._softmax.default(buf81, -1, False)
            assert_alignment(buf82, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_17], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:28
            buf83 = torch.ops.npu._npu_dropout.default(buf82, 0.1)
            buf84 = buf83[0]
            assert_alignment(buf84, 16, 'torch.ops.npu._npu_dropout.default')
            buf85 = buf83[1]
            assert_alignment(buf85, 16, 'torch.ops.npu._npu_dropout.default')
            del buf83
            buf87 = reinterpret_tensor(buf79, (48, 1024, 64), (65536, 64, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [x_17, split_2, tensor_10, value_5, attn_output_6], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:29
            extern_kernels.bmm(reinterpret_tensor(buf84, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf86, (48, 1024, 64), (65536, 64, 1), 0), out=buf87)
            buf481 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [permute_225], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf84, buf481, 49152, 1024, stream=raw_stream0)
            buf88 = reinterpret_tensor(buf86, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf87, buf88, 3145728, stream=raw_stream0)
            buf482 = reinterpret_tensor(buf84, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [view_493], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf481, buf482, 49152, 1024, stream=raw_stream0)
            buf89 = reinterpret_tensor(buf87, (4096, 768), (768, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [attn_output_6, permute_11, tensor_11, attn_output_7, view_31, x_18], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:30
            extern_kernels.addmm(primals_35, reinterpret_tensor(buf88, (4096, 768), (768, 1), 0), primals_36, alpha=1, beta=1, out=buf89)
            del primals_35
            # Unsorted Source Nodes: [attn_output_8, x_19], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:31
            buf90 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf89, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf91 = buf90[0]
            assert_alignment(buf91, 16, 'torch.ops.npu._npu_dropout.default')
            buf96 = reinterpret_tensor(buf89, (4, 1024, 768), (786432, 768, 1), 0); del buf89  # reuse
            buf97 = reinterpret_tensor(buf78, (4, 1024, 768), (786432, 768, 1), 0); del buf78  # reuse
            buf479 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_14, div_42], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf91, buf71, primals_37, primals_38, buf96, buf97, buf479, 4096, 768, stream=raw_stream0)
            del primals_38
            buf92 = buf90[1]
            assert_alignment(buf92, 16, 'torch.ops.npu._npu_dropout.default')
            del buf90
            buf98 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_14, view_33, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:32
            extern_kernels.addmm(primals_39, reinterpret_tensor(buf97, (4096, 768), (768, 1), 0), primals_40, alpha=1, beta=1, out=buf98)
            del primals_39
            buf99 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf98, buf99, 12582912, stream=raw_stream0)
            buf100 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_15, view_35, x_22], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:33
            extern_kernels.addmm(primals_41, reinterpret_tensor(buf99, (4096, 3072), (3072, 1), 0), primals_42, alpha=1, beta=1, out=buf100)
            del primals_41
            # Unsorted Source Nodes: [hidden_states_16, x_23], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:34
            buf101 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf100, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf102 = buf101[0]
            assert_alignment(buf102, 16, 'torch.ops.npu._npu_dropout.default')
            buf107 = reinterpret_tensor(buf100, (4, 1024, 768), (786432, 768, 1), 0); del buf100  # reuse
            buf108 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf478 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_18, div_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf91, buf71, buf102, primals_43, primals_44, buf107, buf108, buf478, 4096, 768, stream=raw_stream0)
            del primals_44
            buf103 = buf101[1]
            assert_alignment(buf103, 16, 'torch.ops.npu._npu_dropout.default')
            del buf101
            buf109 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_18, view_37, x_24], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:35
            extern_kernels.addmm(primals_45, reinterpret_tensor(buf108, (4096, 768), (768, 1), 0), primals_46, alpha=1, beta=1, out=buf109)
            del primals_45
            buf110 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf109, buf110, 3145728, stream=raw_stream0)
            buf111 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_13, key_7, transpose_3, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf109, buf111, 3072, 1024, stream=raw_stream0)
            buf118 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, attn_output_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf109, buf118, 3145728, stream=raw_stream0)
            buf473 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, permute_207, clone_97], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf109, buf473, 3072, 1024, stream=raw_stream0)
            buf476 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_13, key_7, transpose_3, permute_209, clone_99], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf109, buf476, 3145728, stream=raw_stream0)
            buf477 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, permute_210, clone_100], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf109, buf477, 3072, 1024, stream=raw_stream0)
            buf112 = reinterpret_tensor(buf481, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf481  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_12, query_7, tensor_13, key_7, transpose_3, attn_weights_18], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:36
            extern_kernels.bmm(reinterpret_tensor(buf110, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf111, (48, 64, 1024), (65536, 1024, 1), 0), out=buf112)
            buf113 = reinterpret_tensor(buf112, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_18, attn_weights_19, attn_weights_20], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf113, primals_47, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_18, attn_weights_19, attn_weights_20, attn_weights_21], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:37
            buf114 = torch.ops.aten._softmax.default(buf113, -1, False)
            assert_alignment(buf114, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_23], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:38
            buf115 = torch.ops.npu._npu_dropout.default(buf114, 0.1)
            buf116 = buf115[0]
            assert_alignment(buf116, 16, 'torch.ops.npu._npu_dropout.default')
            buf117 = buf115[1]
            assert_alignment(buf117, 16, 'torch.ops.npu._npu_dropout.default')
            del buf115
            buf119 = reinterpret_tensor(buf111, (48, 1024, 64), (65536, 64, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [x_25, split_3, tensor_14, value_7, attn_output_9], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:39
            extern_kernels.bmm(reinterpret_tensor(buf116, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf118, (48, 1024, 64), (65536, 64, 1), 0), out=buf119)
            buf474 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [permute_208], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf116, buf474, 49152, 1024, stream=raw_stream0)
            buf120 = reinterpret_tensor(buf118, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf119, buf120, 3145728, stream=raw_stream0)
            buf475 = reinterpret_tensor(buf116, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [view_465], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf474, buf475, 49152, 1024, stream=raw_stream0)
            buf121 = reinterpret_tensor(buf119, (4096, 768), (768, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [attn_output_9, permute_15, tensor_15, attn_output_10, view_43, x_26], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:40
            extern_kernels.addmm(primals_48, reinterpret_tensor(buf120, (4096, 768), (768, 1), 0), primals_49, alpha=1, beta=1, out=buf121)
            del primals_48
            # Unsorted Source Nodes: [attn_output_11, x_27], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:41
            buf122 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf121, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf123 = buf122[0]
            assert_alignment(buf123, 16, 'torch.ops.npu._npu_dropout.default')
            buf128 = reinterpret_tensor(buf121, (4, 1024, 768), (786432, 768, 1), 0); del buf121  # reuse
            buf129 = reinterpret_tensor(buf110, (4, 1024, 768), (786432, 768, 1), 0); del buf110  # reuse
            buf472 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_19, hidden_states_20, div_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf123, buf91, buf71, buf102, primals_50, primals_51, buf128, buf129, buf472, 4096, 768, stream=raw_stream0)
            del primals_51
            buf124 = buf122[1]
            assert_alignment(buf124, 16, 'torch.ops.npu._npu_dropout.default')
            del buf122
            buf130 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_20, view_45, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:42
            extern_kernels.addmm(primals_52, reinterpret_tensor(buf129, (4096, 768), (768, 1), 0), primals_53, alpha=1, beta=1, out=buf130)
            del primals_52
            buf131 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf130, buf131, 12582912, stream=raw_stream0)
            buf132 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_21, view_47, x_30], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:43
            extern_kernels.addmm(primals_54, reinterpret_tensor(buf131, (4096, 3072), (3072, 1), 0), primals_55, alpha=1, beta=1, out=buf132)
            del primals_54
            # Unsorted Source Nodes: [hidden_states_22, x_31], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:44
            buf133 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf132, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf134 = buf133[0]
            assert_alignment(buf134, 16, 'torch.ops.npu._npu_dropout.default')
            buf136 = buf91; del buf91  # reuse
            buf140 = reinterpret_tensor(buf132, (4, 1024, 768), (786432, 768, 1), 0); del buf132  # reuse
            buf141 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf471 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_13, hidden_states_17, hidden_states_19, hidden_states_23, hidden_states_24, div_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf136, buf123, buf71, buf102, buf134, primals_56, primals_57, buf140, buf141, buf471, 4096, 768, stream=raw_stream0)
            del primals_57
            buf135 = buf133[1]
            assert_alignment(buf135, 16, 'torch.ops.npu._npu_dropout.default')
            del buf133
            buf142 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_24, view_49, x_32], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:45
            extern_kernels.addmm(primals_58, reinterpret_tensor(buf141, (4096, 768), (768, 1), 0), primals_59, alpha=1, beta=1, out=buf142)
            del primals_58
            buf143 = reinterpret_tensor(buf71, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf142, buf143, 3145728, stream=raw_stream0)
            buf144 = reinterpret_tensor(buf134, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_17, key_9, transpose_4, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf142, buf144, 3072, 1024, stream=raw_stream0)
            buf151 = reinterpret_tensor(buf123, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, attn_output_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf142, buf151, 3145728, stream=raw_stream0)
            buf466 = reinterpret_tensor(buf102, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, permute_190, clone_91], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf142, buf466, 3072, 1024, stream=raw_stream0)
            buf469 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_17, key_9, transpose_4, permute_192, clone_93], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf142, buf469, 3145728, stream=raw_stream0)
            buf470 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, permute_193, clone_94], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf142, buf470, 3072, 1024, stream=raw_stream0)
            buf145 = reinterpret_tensor(buf474, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_16, query_9, tensor_17, key_9, transpose_4, attn_weights_24], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:46
            extern_kernels.bmm(reinterpret_tensor(buf143, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf144, (48, 64, 1024), (65536, 1024, 1), 0), out=buf145)
            buf146 = reinterpret_tensor(buf145, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_24, attn_weights_25, attn_weights_26], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf146, primals_60, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_24, attn_weights_25, attn_weights_26, attn_weights_27], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:47
            buf147 = torch.ops.aten._softmax.default(buf146, -1, False)
            assert_alignment(buf147, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_29], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:48
            buf148 = torch.ops.npu._npu_dropout.default(buf147, 0.1)
            buf149 = buf148[0]
            assert_alignment(buf149, 16, 'torch.ops.npu._npu_dropout.default')
            buf150 = buf148[1]
            assert_alignment(buf150, 16, 'torch.ops.npu._npu_dropout.default')
            del buf148
            buf152 = reinterpret_tensor(buf144, (48, 1024, 64), (65536, 64, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [x_33, split_4, tensor_18, value_9, attn_output_12], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:49
            extern_kernels.bmm(reinterpret_tensor(buf149, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf151, (48, 1024, 64), (65536, 64, 1), 0), out=buf152)
            buf467 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [permute_191], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf149, buf467, 49152, 1024, stream=raw_stream0)
            buf153 = reinterpret_tensor(buf151, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf152, buf153, 3145728, stream=raw_stream0)
            buf468 = reinterpret_tensor(buf149, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [view_437], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf467, buf468, 49152, 1024, stream=raw_stream0)
            buf154 = reinterpret_tensor(buf152, (4096, 768), (768, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, permute_19, tensor_19, attn_output_13, view_55, x_34], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:50
            extern_kernels.addmm(primals_61, reinterpret_tensor(buf153, (4096, 768), (768, 1), 0), primals_62, alpha=1, beta=1, out=buf154)
            del primals_61
            # Unsorted Source Nodes: [attn_output_14, x_35], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:51
            buf155 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf154, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf156 = buf155[0]
            assert_alignment(buf156, 16, 'torch.ops.npu._npu_dropout.default')
            buf161 = reinterpret_tensor(buf154, (4, 1024, 768), (786432, 768, 1), 0); del buf154  # reuse
            buf162 = reinterpret_tensor(buf143, (4, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
            buf465 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, div_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf156, buf136, primals_63, primals_64, buf161, buf162, buf465, 4096, 768, stream=raw_stream0)
            del primals_64
            buf157 = buf155[1]
            assert_alignment(buf157, 16, 'torch.ops.npu._npu_dropout.default')
            del buf155
            buf163 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_26, view_57, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(primals_65, reinterpret_tensor(buf162, (4096, 768), (768, 1), 0), primals_66, alpha=1, beta=1, out=buf163)
            del primals_65
            buf164 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf163, buf164, 12582912, stream=raw_stream0)
            buf165 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_27, view_59, x_38], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:53
            extern_kernels.addmm(primals_67, reinterpret_tensor(buf164, (4096, 3072), (3072, 1), 0), primals_68, alpha=1, beta=1, out=buf165)
            del primals_67
            # Unsorted Source Nodes: [hidden_states_28, x_39], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:54
            buf166 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf165, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf167 = buf166[0]
            assert_alignment(buf167, 16, 'torch.ops.npu._npu_dropout.default')
            buf172 = reinterpret_tensor(buf165, (4, 1024, 768), (786432, 768, 1), 0); del buf165  # reuse
            buf173 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf464 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_30, div_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf156, buf136, buf167, primals_69, primals_70, buf172, buf173, buf464, 4096, 768, stream=raw_stream0)
            del primals_70
            buf168 = buf166[1]
            assert_alignment(buf168, 16, 'torch.ops.npu._npu_dropout.default')
            del buf166
            buf174 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_30, view_61, x_40], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:55
            extern_kernels.addmm(primals_71, reinterpret_tensor(buf173, (4096, 768), (768, 1), 0), primals_72, alpha=1, beta=1, out=buf174)
            del primals_71
            buf175 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf174, buf175, 3145728, stream=raw_stream0)
            buf176 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_21, key_11, transpose_5, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf174, buf176, 3072, 1024, stream=raw_stream0)
            buf183 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, attn_output_15], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf174, buf183, 3145728, stream=raw_stream0)
            buf459 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, permute_173, clone_85], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf174, buf459, 3072, 1024, stream=raw_stream0)
            buf462 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_21, key_11, transpose_5, permute_175, clone_87], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf174, buf462, 3145728, stream=raw_stream0)
            buf463 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, permute_176, clone_88], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf174, buf463, 3072, 1024, stream=raw_stream0)
            buf177 = reinterpret_tensor(buf467, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf467  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_20, query_11, tensor_21, key_11, transpose_5, attn_weights_30], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:56
            extern_kernels.bmm(reinterpret_tensor(buf175, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf176, (48, 64, 1024), (65536, 1024, 1), 0), out=buf177)
            buf178 = reinterpret_tensor(buf177, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_30, attn_weights_31, attn_weights_32], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf178, primals_73, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_30, attn_weights_31, attn_weights_32, attn_weights_33], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:57
            buf179 = torch.ops.aten._softmax.default(buf178, -1, False)
            assert_alignment(buf179, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_35], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:58
            buf180 = torch.ops.npu._npu_dropout.default(buf179, 0.1)
            buf181 = buf180[0]
            assert_alignment(buf181, 16, 'torch.ops.npu._npu_dropout.default')
            buf182 = buf180[1]
            assert_alignment(buf182, 16, 'torch.ops.npu._npu_dropout.default')
            del buf180
            buf184 = reinterpret_tensor(buf176, (48, 1024, 64), (65536, 64, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [x_41, split_5, tensor_22, value_11, attn_output_15], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:59
            extern_kernels.bmm(reinterpret_tensor(buf181, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf183, (48, 1024, 64), (65536, 64, 1), 0), out=buf184)
            buf460 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [permute_174], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf181, buf460, 49152, 1024, stream=raw_stream0)
            buf185 = reinterpret_tensor(buf183, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf184, buf185, 3145728, stream=raw_stream0)
            buf461 = reinterpret_tensor(buf181, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [view_409], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf460, buf461, 49152, 1024, stream=raw_stream0)
            buf186 = reinterpret_tensor(buf184, (4096, 768), (768, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [attn_output_15, permute_23, tensor_23, attn_output_16, view_67, x_42], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:60
            extern_kernels.addmm(primals_74, reinterpret_tensor(buf185, (4096, 768), (768, 1), 0), primals_75, alpha=1, beta=1, out=buf186)
            del primals_74
            # Unsorted Source Nodes: [attn_output_17, x_43], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:61
            buf187 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf186, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf188 = buf187[0]
            assert_alignment(buf188, 16, 'torch.ops.npu._npu_dropout.default')
            buf193 = reinterpret_tensor(buf186, (4, 1024, 768), (786432, 768, 1), 0); del buf186  # reuse
            buf194 = reinterpret_tensor(buf175, (4, 1024, 768), (786432, 768, 1), 0); del buf175  # reuse
            buf458 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_31, hidden_states_32, div_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf188, buf156, buf136, buf167, primals_76, primals_77, buf193, buf194, buf458, 4096, 768, stream=raw_stream0)
            del primals_77
            buf189 = buf187[1]
            assert_alignment(buf189, 16, 'torch.ops.npu._npu_dropout.default')
            del buf187
            buf195 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_32, view_69, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:62
            extern_kernels.addmm(primals_78, reinterpret_tensor(buf194, (4096, 768), (768, 1), 0), primals_79, alpha=1, beta=1, out=buf195)
            del primals_78
            buf196 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf195, buf196, 12582912, stream=raw_stream0)
            buf197 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_33, view_71, x_46], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:63
            extern_kernels.addmm(primals_80, reinterpret_tensor(buf196, (4096, 3072), (3072, 1), 0), primals_81, alpha=1, beta=1, out=buf197)
            del primals_80
            # Unsorted Source Nodes: [hidden_states_34, x_47], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:64
            buf198 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf197, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf199 = buf198[0]
            assert_alignment(buf199, 16, 'torch.ops.npu._npu_dropout.default')
            buf201 = buf156; del buf156  # reuse
            buf205 = reinterpret_tensor(buf197, (4, 1024, 768), (786432, 768, 1), 0); del buf197  # reuse
            buf206 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf457 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_29, hidden_states_31, hidden_states_35, hidden_states_36, div_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf201, buf188, buf136, buf167, buf199, primals_82, primals_83, buf205, buf206, buf457, 4096, 768, stream=raw_stream0)
            del primals_83
            buf200 = buf198[1]
            assert_alignment(buf200, 16, 'torch.ops.npu._npu_dropout.default')
            del buf198
            buf207 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_36, view_73, x_48], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(primals_84, reinterpret_tensor(buf206, (4096, 768), (768, 1), 0), primals_85, alpha=1, beta=1, out=buf207)
            del primals_84
            buf208 = reinterpret_tensor(buf199, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf207, buf208, 3145728, stream=raw_stream0)
            buf209 = reinterpret_tensor(buf188, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_25, key_13, transpose_6, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf207, buf209, 3072, 1024, stream=raw_stream0)
            buf216 = reinterpret_tensor(buf167, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, attn_output_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf207, buf216, 3145728, stream=raw_stream0)
            buf452 = reinterpret_tensor(buf136, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, permute_156, clone_79], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf207, buf452, 3072, 1024, stream=raw_stream0)
            buf455 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_25, key_13, transpose_6, permute_158, clone_81], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf207, buf455, 3145728, stream=raw_stream0)
            buf456 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, permute_159, clone_82], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf207, buf456, 3072, 1024, stream=raw_stream0)
            buf210 = reinterpret_tensor(buf460, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf460  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_24, query_13, tensor_25, key_13, transpose_6, attn_weights_36], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:66
            extern_kernels.bmm(reinterpret_tensor(buf208, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf209, (48, 64, 1024), (65536, 1024, 1), 0), out=buf210)
            buf211 = reinterpret_tensor(buf210, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_36, attn_weights_37, attn_weights_38], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf211, primals_86, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_36, attn_weights_37, attn_weights_38, attn_weights_39], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:67
            buf212 = torch.ops.aten._softmax.default(buf211, -1, False)
            assert_alignment(buf212, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_41], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:68
            buf213 = torch.ops.npu._npu_dropout.default(buf212, 0.1)
            buf214 = buf213[0]
            assert_alignment(buf214, 16, 'torch.ops.npu._npu_dropout.default')
            buf215 = buf213[1]
            assert_alignment(buf215, 16, 'torch.ops.npu._npu_dropout.default')
            del buf213
            buf217 = reinterpret_tensor(buf209, (48, 1024, 64), (65536, 64, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [x_49, split_6, tensor_26, value_13, attn_output_18], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:69
            extern_kernels.bmm(reinterpret_tensor(buf214, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf216, (48, 1024, 64), (65536, 64, 1), 0), out=buf217)
            buf453 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [permute_157], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf214, buf453, 49152, 1024, stream=raw_stream0)
            buf218 = reinterpret_tensor(buf216, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf217, buf218, 3145728, stream=raw_stream0)
            buf454 = reinterpret_tensor(buf214, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf214  # reuse
            # Topologically Sorted Source Nodes: [view_381], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf453, buf454, 49152, 1024, stream=raw_stream0)
            buf219 = reinterpret_tensor(buf217, (4096, 768), (768, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [attn_output_18, permute_27, tensor_27, attn_output_19, view_79, x_50], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:70
            extern_kernels.addmm(primals_87, reinterpret_tensor(buf218, (4096, 768), (768, 1), 0), primals_88, alpha=1, beta=1, out=buf219)
            del primals_87
            # Unsorted Source Nodes: [attn_output_20, x_51], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:71
            buf220 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf219, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf221 = buf220[0]
            assert_alignment(buf221, 16, 'torch.ops.npu._npu_dropout.default')
            buf226 = reinterpret_tensor(buf219, (4, 1024, 768), (786432, 768, 1), 0); del buf219  # reuse
            buf227 = reinterpret_tensor(buf208, (4, 1024, 768), (786432, 768, 1), 0); del buf208  # reuse
            buf451 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38, div_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf221, buf201, primals_89, primals_90, buf226, buf227, buf451, 4096, 768, stream=raw_stream0)
            del primals_90
            buf222 = buf220[1]
            assert_alignment(buf222, 16, 'torch.ops.npu._npu_dropout.default')
            del buf220
            buf228 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_38, view_81, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:72
            extern_kernels.addmm(primals_91, reinterpret_tensor(buf227, (4096, 768), (768, 1), 0), primals_92, alpha=1, beta=1, out=buf228)
            del primals_91
            buf229 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf228, buf229, 12582912, stream=raw_stream0)
            buf230 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_39, view_83, x_54], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:73
            extern_kernels.addmm(primals_93, reinterpret_tensor(buf229, (4096, 3072), (3072, 1), 0), primals_94, alpha=1, beta=1, out=buf230)
            del primals_93
            # Unsorted Source Nodes: [hidden_states_40, x_55], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:74
            buf231 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf230, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf232 = buf231[0]
            assert_alignment(buf232, 16, 'torch.ops.npu._npu_dropout.default')
            buf237 = reinterpret_tensor(buf230, (4, 1024, 768), (786432, 768, 1), 0); del buf230  # reuse
            buf238 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf450 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_42, div_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf221, buf201, buf232, primals_95, primals_96, buf237, buf238, buf450, 4096, 768, stream=raw_stream0)
            del primals_96
            buf233 = buf231[1]
            assert_alignment(buf233, 16, 'torch.ops.npu._npu_dropout.default')
            del buf231
            buf239 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_42, view_85, x_56], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:75
            extern_kernels.addmm(primals_97, reinterpret_tensor(buf238, (4096, 768), (768, 1), 0), primals_98, alpha=1, beta=1, out=buf239)
            del primals_97
            buf240 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf239, buf240, 3145728, stream=raw_stream0)
            buf241 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_29, key_15, transpose_7, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf239, buf241, 3072, 1024, stream=raw_stream0)
            buf248 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, attn_output_21], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf239, buf248, 3145728, stream=raw_stream0)
            buf445 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, permute_139, clone_73], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf239, buf445, 3072, 1024, stream=raw_stream0)
            buf448 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_29, key_15, transpose_7, permute_141, clone_75], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf239, buf448, 3145728, stream=raw_stream0)
            buf449 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, permute_142, clone_76], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf239, buf449, 3072, 1024, stream=raw_stream0)
            buf242 = reinterpret_tensor(buf453, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf453  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_28, query_15, tensor_29, key_15, transpose_7, attn_weights_42], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:76
            extern_kernels.bmm(reinterpret_tensor(buf240, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf241, (48, 64, 1024), (65536, 1024, 1), 0), out=buf242)
            buf243 = reinterpret_tensor(buf242, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_42, attn_weights_43, attn_weights_44], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf243, primals_99, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_42, attn_weights_43, attn_weights_44, attn_weights_45], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:77
            buf244 = torch.ops.aten._softmax.default(buf243, -1, False)
            assert_alignment(buf244, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_47], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:78
            buf245 = torch.ops.npu._npu_dropout.default(buf244, 0.1)
            buf246 = buf245[0]
            assert_alignment(buf246, 16, 'torch.ops.npu._npu_dropout.default')
            buf247 = buf245[1]
            assert_alignment(buf247, 16, 'torch.ops.npu._npu_dropout.default')
            del buf245
            buf249 = reinterpret_tensor(buf241, (48, 1024, 64), (65536, 64, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [x_57, split_7, tensor_30, value_15, attn_output_21], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:79
            extern_kernels.bmm(reinterpret_tensor(buf246, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf248, (48, 1024, 64), (65536, 64, 1), 0), out=buf249)
            buf446 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [permute_140], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf246, buf446, 49152, 1024, stream=raw_stream0)
            buf250 = reinterpret_tensor(buf248, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf249, buf250, 3145728, stream=raw_stream0)
            buf447 = reinterpret_tensor(buf246, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [view_353], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf446, buf447, 49152, 1024, stream=raw_stream0)
            buf251 = reinterpret_tensor(buf249, (4096, 768), (768, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [attn_output_21, permute_31, tensor_31, attn_output_22, view_91, x_58], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:80
            extern_kernels.addmm(primals_100, reinterpret_tensor(buf250, (4096, 768), (768, 1), 0), primals_101, alpha=1, beta=1, out=buf251)
            del primals_100
            # Unsorted Source Nodes: [attn_output_23, x_59], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:81
            buf252 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf251, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf253 = buf252[0]
            assert_alignment(buf253, 16, 'torch.ops.npu._npu_dropout.default')
            buf258 = reinterpret_tensor(buf251, (4, 1024, 768), (786432, 768, 1), 0); del buf251  # reuse
            buf259 = reinterpret_tensor(buf240, (4, 1024, 768), (786432, 768, 1), 0); del buf240  # reuse
            buf444 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_43, hidden_states_44, div_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf253, buf221, buf201, buf232, primals_102, primals_103, buf258, buf259, buf444, 4096, 768, stream=raw_stream0)
            del primals_103
            buf254 = buf252[1]
            assert_alignment(buf254, 16, 'torch.ops.npu._npu_dropout.default')
            del buf252
            buf260 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_44, view_93, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:82
            extern_kernels.addmm(primals_104, reinterpret_tensor(buf259, (4096, 768), (768, 1), 0), primals_105, alpha=1, beta=1, out=buf260)
            del primals_104
            buf261 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf260, buf261, 12582912, stream=raw_stream0)
            buf262 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_45, view_95, x_62], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:83
            extern_kernels.addmm(primals_106, reinterpret_tensor(buf261, (4096, 3072), (3072, 1), 0), primals_107, alpha=1, beta=1, out=buf262)
            del primals_106
            # Unsorted Source Nodes: [hidden_states_46, x_63], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:84
            buf263 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf262, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf264 = buf263[0]
            assert_alignment(buf264, 16, 'torch.ops.npu._npu_dropout.default')
            buf266 = buf221; del buf221  # reuse
            buf270 = reinterpret_tensor(buf262, (4, 1024, 768), (786432, 768, 1), 0); del buf262  # reuse
            buf271 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf443 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_41, hidden_states_43, hidden_states_47, hidden_states_48, div_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf266, buf253, buf201, buf232, buf264, primals_108, primals_109, buf270, buf271, buf443, 4096, 768, stream=raw_stream0)
            del primals_109
            buf265 = buf263[1]
            assert_alignment(buf265, 16, 'torch.ops.npu._npu_dropout.default')
            del buf263
            buf272 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_48, view_97, x_64], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:85
            extern_kernels.addmm(primals_110, reinterpret_tensor(buf271, (4096, 768), (768, 1), 0), primals_111, alpha=1, beta=1, out=buf272)
            del primals_110
            buf273 = reinterpret_tensor(buf264, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf264  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf272, buf273, 3145728, stream=raw_stream0)
            buf274 = reinterpret_tensor(buf253, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_33, key_17, transpose_8, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf272, buf274, 3072, 1024, stream=raw_stream0)
            buf281 = reinterpret_tensor(buf232, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, attn_output_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf272, buf281, 3145728, stream=raw_stream0)
            buf438 = reinterpret_tensor(buf201, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, permute_122, clone_67], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf272, buf438, 3072, 1024, stream=raw_stream0)
            buf441 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_33, key_17, transpose_8, permute_124, clone_69], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf272, buf441, 3145728, stream=raw_stream0)
            buf442 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, permute_125, clone_70], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf272, buf442, 3072, 1024, stream=raw_stream0)
            buf275 = reinterpret_tensor(buf446, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_32, query_17, tensor_33, key_17, transpose_8, attn_weights_48], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:86
            extern_kernels.bmm(reinterpret_tensor(buf273, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf274, (48, 64, 1024), (65536, 1024, 1), 0), out=buf275)
            buf276 = reinterpret_tensor(buf275, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf275  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_48, attn_weights_49, attn_weights_50], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf276, primals_112, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_48, attn_weights_49, attn_weights_50, attn_weights_51], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:87
            buf277 = torch.ops.aten._softmax.default(buf276, -1, False)
            assert_alignment(buf277, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_53], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:88
            buf278 = torch.ops.npu._npu_dropout.default(buf277, 0.1)
            buf279 = buf278[0]
            assert_alignment(buf279, 16, 'torch.ops.npu._npu_dropout.default')
            buf280 = buf278[1]
            assert_alignment(buf280, 16, 'torch.ops.npu._npu_dropout.default')
            del buf278
            buf282 = reinterpret_tensor(buf274, (48, 1024, 64), (65536, 64, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [x_65, split_8, tensor_34, value_17, attn_output_24], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:89
            extern_kernels.bmm(reinterpret_tensor(buf279, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf281, (48, 1024, 64), (65536, 64, 1), 0), out=buf282)
            buf439 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [permute_123], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf279, buf439, 49152, 1024, stream=raw_stream0)
            buf283 = reinterpret_tensor(buf281, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf282, buf283, 3145728, stream=raw_stream0)
            buf440 = reinterpret_tensor(buf279, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf279  # reuse
            # Topologically Sorted Source Nodes: [view_325], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf439, buf440, 49152, 1024, stream=raw_stream0)
            buf284 = reinterpret_tensor(buf282, (4096, 768), (768, 1), 0); del buf282  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, permute_35, tensor_35, attn_output_25, view_103, x_66], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:90
            extern_kernels.addmm(primals_113, reinterpret_tensor(buf283, (4096, 768), (768, 1), 0), primals_114, alpha=1, beta=1, out=buf284)
            del primals_113
            # Unsorted Source Nodes: [attn_output_26, x_67], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:91
            buf285 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf284, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf286 = buf285[0]
            assert_alignment(buf286, 16, 'torch.ops.npu._npu_dropout.default')
            buf291 = reinterpret_tensor(buf284, (4, 1024, 768), (786432, 768, 1), 0); del buf284  # reuse
            buf292 = reinterpret_tensor(buf273, (4, 1024, 768), (786432, 768, 1), 0); del buf273  # reuse
            buf437 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50, div_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf286, buf266, primals_115, primals_116, buf291, buf292, buf437, 4096, 768, stream=raw_stream0)
            del primals_116
            buf287 = buf285[1]
            assert_alignment(buf287, 16, 'torch.ops.npu._npu_dropout.default')
            del buf285
            buf293 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_50, view_105, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:92
            extern_kernels.addmm(primals_117, reinterpret_tensor(buf292, (4096, 768), (768, 1), 0), primals_118, alpha=1, beta=1, out=buf293)
            del primals_117
            buf294 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf293, buf294, 12582912, stream=raw_stream0)
            buf295 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_51, view_107, x_70], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:93
            extern_kernels.addmm(primals_119, reinterpret_tensor(buf294, (4096, 3072), (3072, 1), 0), primals_120, alpha=1, beta=1, out=buf295)
            del primals_119
            # Unsorted Source Nodes: [hidden_states_52, x_71], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:94
            buf296 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf295, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf297 = buf296[0]
            assert_alignment(buf297, 16, 'torch.ops.npu._npu_dropout.default')
            buf302 = reinterpret_tensor(buf295, (4, 1024, 768), (786432, 768, 1), 0); del buf295  # reuse
            buf303 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf436 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_54, div_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf286, buf266, buf297, primals_121, primals_122, buf302, buf303, buf436, 4096, 768, stream=raw_stream0)
            del primals_122
            buf298 = buf296[1]
            assert_alignment(buf298, 16, 'torch.ops.npu._npu_dropout.default')
            del buf296
            buf304 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_54, view_109, x_72], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:95
            extern_kernels.addmm(primals_123, reinterpret_tensor(buf303, (4096, 768), (768, 1), 0), primals_124, alpha=1, beta=1, out=buf304)
            del primals_123
            buf305 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf304, buf305, 3145728, stream=raw_stream0)
            buf306 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_37, key_19, transpose_9, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf304, buf306, 3072, 1024, stream=raw_stream0)
            buf313 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, attn_output_27], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf304, buf313, 3145728, stream=raw_stream0)
            buf431 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, permute_105, clone_61], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf304, buf431, 3072, 1024, stream=raw_stream0)
            buf434 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_37, key_19, transpose_9, permute_107, clone_63], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf304, buf434, 3145728, stream=raw_stream0)
            buf435 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, permute_108, clone_64], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf304, buf435, 3072, 1024, stream=raw_stream0)
            buf307 = reinterpret_tensor(buf439, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf439  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_36, query_19, tensor_37, key_19, transpose_9, attn_weights_54], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:96
            extern_kernels.bmm(reinterpret_tensor(buf305, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf306, (48, 64, 1024), (65536, 1024, 1), 0), out=buf307)
            buf308 = reinterpret_tensor(buf307, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf307  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_54, attn_weights_55, attn_weights_56], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf308, primals_125, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_54, attn_weights_55, attn_weights_56, attn_weights_57], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:97
            buf309 = torch.ops.aten._softmax.default(buf308, -1, False)
            assert_alignment(buf309, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_59], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:98
            buf310 = torch.ops.npu._npu_dropout.default(buf309, 0.1)
            buf311 = buf310[0]
            assert_alignment(buf311, 16, 'torch.ops.npu._npu_dropout.default')
            buf312 = buf310[1]
            assert_alignment(buf312, 16, 'torch.ops.npu._npu_dropout.default')
            del buf310
            buf314 = reinterpret_tensor(buf306, (48, 1024, 64), (65536, 64, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [x_73, split_9, tensor_38, value_19, attn_output_27], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:99
            extern_kernels.bmm(reinterpret_tensor(buf311, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf313, (48, 1024, 64), (65536, 64, 1), 0), out=buf314)
            buf432 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [permute_106], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf311, buf432, 49152, 1024, stream=raw_stream0)
            buf315 = reinterpret_tensor(buf313, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf314, buf315, 3145728, stream=raw_stream0)
            buf433 = reinterpret_tensor(buf311, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [view_297], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf432, buf433, 49152, 1024, stream=raw_stream0)
            buf316 = reinterpret_tensor(buf314, (4096, 768), (768, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, permute_39, tensor_39, attn_output_28, view_115, x_74], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:100
            extern_kernels.addmm(primals_126, reinterpret_tensor(buf315, (4096, 768), (768, 1), 0), primals_127, alpha=1, beta=1, out=buf316)
            del primals_126
            # Unsorted Source Nodes: [attn_output_29, x_75], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:101
            buf317 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf316, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf318 = buf317[0]
            assert_alignment(buf318, 16, 'torch.ops.npu._npu_dropout.default')
            buf323 = reinterpret_tensor(buf316, (4, 1024, 768), (786432, 768, 1), 0); del buf316  # reuse
            buf324 = reinterpret_tensor(buf305, (4, 1024, 768), (786432, 768, 1), 0); del buf305  # reuse
            buf430 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_55, hidden_states_56, div_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf318, buf286, buf266, buf297, primals_128, primals_129, buf323, buf324, buf430, 4096, 768, stream=raw_stream0)
            del primals_129
            buf319 = buf317[1]
            assert_alignment(buf319, 16, 'torch.ops.npu._npu_dropout.default')
            del buf317
            buf325 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_56, view_117, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:102
            extern_kernels.addmm(primals_130, reinterpret_tensor(buf324, (4096, 768), (768, 1), 0), primals_131, alpha=1, beta=1, out=buf325)
            del primals_130
            buf326 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf325, buf326, 12582912, stream=raw_stream0)
            buf327 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_57, view_119, x_78], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:103
            extern_kernels.addmm(primals_132, reinterpret_tensor(buf326, (4096, 3072), (3072, 1), 0), primals_133, alpha=1, beta=1, out=buf327)
            del primals_132
            # Unsorted Source Nodes: [hidden_states_58, x_79], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:104
            buf328 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf327, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf329 = buf328[0]
            assert_alignment(buf329, 16, 'torch.ops.npu._npu_dropout.default')
            buf331 = buf286; del buf286  # reuse
            buf335 = reinterpret_tensor(buf327, (4, 1024, 768), (786432, 768, 1), 0); del buf327  # reuse
            buf336 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf429 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_53, hidden_states_55, hidden_states_59, hidden_states_60, div_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_23.run(buf331, buf318, buf266, buf297, buf329, primals_134, primals_135, buf335, buf336, buf429, 4096, 768, stream=raw_stream0)
            del primals_135
            buf330 = buf328[1]
            assert_alignment(buf330, 16, 'torch.ops.npu._npu_dropout.default')
            del buf328
            buf337 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_60, view_121, x_80], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:105
            extern_kernels.addmm(primals_136, reinterpret_tensor(buf336, (4096, 768), (768, 1), 0), primals_137, alpha=1, beta=1, out=buf337)
            del primals_136
            buf338 = reinterpret_tensor(buf329, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf329  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_9.run(buf337, buf338, 3145728, stream=raw_stream0)
            buf339 = reinterpret_tensor(buf318, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_41, key_21, transpose_10, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_transpose_view_10.run(buf337, buf339, 3072, 1024, stream=raw_stream0)
            buf346 = reinterpret_tensor(buf297, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, attn_output_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_split_view_11.run(buf337, buf346, 3145728, stream=raw_stream0)
            buf424 = reinterpret_tensor(buf266, (4, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, permute_88, clone_55], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_12.run(buf337, buf424, 3072, 1024, stream=raw_stream0)
            buf427 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_41, key_21, transpose_10, permute_90, clone_57], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_transpose_view_13.run(buf337, buf427, 3145728, stream=raw_stream0)
            buf428 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, permute_91, clone_58], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_permute_split_view_14.run(buf337, buf428, 3072, 1024, stream=raw_stream0)
            buf340 = reinterpret_tensor(buf432, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf432  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_40, query_21, tensor_41, key_21, transpose_10, attn_weights_60], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:106
            extern_kernels.bmm(reinterpret_tensor(buf338, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf339, (48, 64, 1024), (65536, 1024, 1), 0), out=buf340)
            buf341 = reinterpret_tensor(buf340, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_60, attn_weights_61, attn_weights_62], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf341, primals_138, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_60, attn_weights_61, attn_weights_62, attn_weights_63], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:107
            buf342 = torch.ops.aten._softmax.default(buf341, -1, False)
            assert_alignment(buf342, 16, 'torch.ops.aten._softmax.default')
            # Unsorted Source Nodes: [attn_weights_65], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:108
            buf343 = torch.ops.npu._npu_dropout.default(buf342, 0.1)
            buf344 = buf343[0]
            assert_alignment(buf344, 16, 'torch.ops.npu._npu_dropout.default')
            buf345 = buf343[1]
            assert_alignment(buf345, 16, 'torch.ops.npu._npu_dropout.default')
            del buf343
            buf347 = reinterpret_tensor(buf339, (48, 1024, 64), (65536, 64, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [x_81, split_10, tensor_42, value_21, attn_output_30], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:109
            extern_kernels.bmm(reinterpret_tensor(buf344, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf346, (48, 1024, 64), (65536, 64, 1), 0), out=buf347)
            buf425 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [permute_89], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf344, buf425, 49152, 1024, stream=raw_stream0)
            buf348 = reinterpret_tensor(buf346, (4, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf347, buf348, 3145728, stream=raw_stream0)
            buf426 = reinterpret_tensor(buf344, (48, 1024, 1024), (1048576, 1, 1024), 0); del buf344  # reuse
            # Topologically Sorted Source Nodes: [view_269], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf425, buf426, 49152, 1024, stream=raw_stream0)
            buf349 = reinterpret_tensor(buf347, (4096, 768), (768, 1), 0); del buf347  # reuse
            # Topologically Sorted Source Nodes: [attn_output_30, permute_43, tensor_43, attn_output_31, view_127, x_82], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:110
            extern_kernels.addmm(primals_139, reinterpret_tensor(buf348, (4096, 768), (768, 1), 0), primals_140, alpha=1, beta=1, out=buf349)
            del primals_139
            # Unsorted Source Nodes: [attn_output_32, x_83], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:111
            buf350 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf349, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf351 = buf350[0]
            assert_alignment(buf351, 16, 'torch.ops.npu._npu_dropout.default')
            buf356 = reinterpret_tensor(buf349, (4, 1024, 768), (786432, 768, 1), 0); del buf349  # reuse
            buf357 = reinterpret_tensor(buf338, (4, 1024, 768), (786432, 768, 1), 0); del buf338  # reuse
            buf423 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, div_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf351, buf331, primals_141, primals_142, buf356, buf357, buf423, 4096, 768, stream=raw_stream0)
            del primals_142
            buf352 = buf350[1]
            assert_alignment(buf352, 16, 'torch.ops.npu._npu_dropout.default')
            del buf350
            buf358 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_62, view_129, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:112
            extern_kernels.addmm(primals_143, reinterpret_tensor(buf357, (4096, 768), (768, 1), 0), primals_144, alpha=1, beta=1, out=buf358)
            del primals_143
            buf359 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf358, buf359, 12582912, stream=raw_stream0)
            buf360 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_63, view_131, x_86], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:113
            extern_kernels.addmm(primals_145, reinterpret_tensor(buf359, (4096, 3072), (3072, 1), 0), primals_146, alpha=1, beta=1, out=buf360)
            del primals_145
            # Unsorted Source Nodes: [hidden_states_64, x_87], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:114
            buf361 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf360, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf362 = buf361[0]
            assert_alignment(buf362, 16, 'torch.ops.npu._npu_dropout.default')
            buf367 = reinterpret_tensor(buf360, (4, 1024, 768), (786432, 768, 1), 0); del buf360  # reuse
            buf368 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf422 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_66, div_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf351, buf331, buf362, primals_147, primals_148, buf367, buf368, buf422, 4096, 768, stream=raw_stream0)
            del primals_148
            buf363 = buf361[1]
            assert_alignment(buf363, 16, 'torch.ops.npu._npu_dropout.default')
            del buf361
            buf369 = empty_strided_npu((4096, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_66, view_133, x_88], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:115
            extern_kernels.addmm(primals_149, reinterpret_tensor(buf368, (4096, 768), (768, 1), 0), primals_150, alpha=1, beta=1, out=buf369)
            del primals_149
            buf370 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            buf421 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, attn_weights_66, permute_74, clone_52], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_view_24.run(buf369, buf370, buf421, 3072, 1024, stream=raw_stream0)
            buf371 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            buf420 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_45, key_23, transpose_11, attn_weights_66, permute_73, clone_51], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_transpose_view_25.run(buf369, buf371, buf420, 49152, 64, stream=raw_stream0)
            buf378 = empty_strided_npu((4, 12, 1024, 64), (786432, 65536, 64, 1), torch.float32)
            buf417 = empty_strided_npu((4, 12, 64, 1024), (786432, 65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33, permute_71, clone_49], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_matmul_backward_permute_split_view_26.run(buf369, buf378, buf417, 3072, 1024, stream=raw_stream0)
            buf372 = reinterpret_tensor(buf425, (48, 1024, 1024), (1048576, 1024, 1), 0); del buf425  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_44, query_23, tensor_45, key_23, transpose_11, attn_weights_66], Original ATen: [aten.view, aten.split, aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:116
            extern_kernels.bmm(reinterpret_tensor(buf370, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf371, (48, 64, 1024), (65536, 1024, 1), 0), out=buf372)
            del buf370
            buf373 = reinterpret_tensor(buf372, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_66, attn_weights_67, attn_weights_68], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_div_full_matmul_where_15.run(buf373, primals_151, 50331648, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [full, mask_value, attn_weights_66, attn_weights_67, attn_weights_68, attn_weights_69], Original ATen: [aten.full, aten._to_copy, aten.matmul, aten.div, aten.where, aten._softmax]
            # [Provenance debug handles] torch.ops.aten._softmax.default:117
            buf374 = torch.ops.aten._softmax.default(buf373, -1, False)
            assert_alignment(buf374, 16, 'torch.ops.aten._softmax.default')
            del buf373
            # Unsorted Source Nodes: [attn_weights_71], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:118
            buf375 = torch.ops.npu._npu_dropout.default(buf374, 0.1)
            buf376 = buf375[0]
            assert_alignment(buf376, 16, 'torch.ops.npu._npu_dropout.default')
            buf377 = buf375[1]
            assert_alignment(buf377, 16, 'torch.ops.npu._npu_dropout.default')
            del buf375
            buf379 = reinterpret_tensor(buf371, (48, 1024, 64), (65536, 64, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [x_89, split_11, tensor_46, value_23, attn_output_33], Original ATen: [aten.view, aten.split, aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:119
            extern_kernels.bmm(reinterpret_tensor(buf376, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf378, (48, 1024, 64), (65536, 64, 1), 0), out=buf379)
            del buf378
            buf418 = empty_strided_npu((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_72], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_16.run(buf376, buf418, 49152, 1024, stream=raw_stream0)
            del buf376
            buf380 = empty_strided_npu((4, 1024, 12, 64), (786432, 768, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47], Original ATen: [aten.matmul, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_matmul_permute_17.run(buf379, buf380, 3145728, stream=raw_stream0)
            del buf379
            buf419 = empty_strided_npu((48, 1024, 1024), (1048576, 1, 1024), torch.float32)
            # Topologically Sorted Source Nodes: [view_241], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_backward_18.run(buf418, buf419, 49152, 1024, stream=raw_stream0)
            del buf418
            buf381 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_output_33, permute_47, tensor_47, attn_output_34, view_139, x_90], Original ATen: [aten.matmul, aten.permute, aten.clone, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:120
            extern_kernels.addmm(primals_152, reinterpret_tensor(buf380, (4096, 768), (768, 1), 0), primals_153, alpha=1, beta=1, out=buf381)
            del primals_152
            # Unsorted Source Nodes: [attn_output_35, x_91], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:121
            buf382 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf381, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf383 = buf382[0]
            assert_alignment(buf383, 16, 'torch.ops.npu._npu_dropout.default')
            buf388 = reinterpret_tensor(buf381, (4, 1024, 768), (786432, 768, 1), 0); del buf381  # reuse
            buf389 = empty_strided_npu((4, 1024, 768), (786432, 768, 1), torch.float32)
            buf416 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_67, hidden_states_68, div_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf383, buf351, buf331, buf362, primals_154, primals_155, buf388, buf389, buf416, 4096, 768, stream=raw_stream0)
            del primals_155
            buf384 = buf382[1]
            assert_alignment(buf384, 16, 'torch.ops.npu._npu_dropout.default')
            del buf382
            buf390 = empty_strided_npu((4096, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_68, view_141, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:122
            extern_kernels.addmm(primals_156, reinterpret_tensor(buf389, (4096, 768), (768, 1), 0), primals_157, alpha=1, beta=1, out=buf390)
            del primals_156
            buf391 = empty_strided_npu((4, 1024, 3072), (3145728, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_pow_tanh_view_20.run(buf390, buf391, 12582912, stream=raw_stream0)
            buf392 = empty_strided_npu((4096, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_69, view_143, x_94], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:123
            extern_kernels.addmm(primals_158, reinterpret_tensor(buf391, (4096, 3072), (3072, 1), 0), primals_159, alpha=1, beta=1, out=buf392)
            del primals_158
            # Unsorted Source Nodes: [hidden_states_70, x_95], Original ATen: [aten.native_dropout, aten.view]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:124
            buf393 = torch.ops.npu._npu_dropout.default(reinterpret_tensor(buf392, (4, 1024, 768), (786432, 768, 1), 0), 0.1)
            buf394 = buf393[0]
            assert_alignment(buf394, 16, 'torch.ops.npu._npu_dropout.default')
            buf396 = buf351; del buf351  # reuse
            buf400 = buf396; del buf396  # reuse
            buf401 = reinterpret_tensor(buf392, (4, 1024, 768), (786432, 768, 1), 0); del buf392  # reuse
            buf415 = empty_strided_npu((4, 1024, 1), (1024, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_65, hidden_states_67, hidden_states_71, layer_norm_24, div_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf400, buf383, buf331, buf362, buf394, primals_160, primals_161, buf401, buf415, 4096, 768, stream=raw_stream0)
            del buf331
            del buf362
            del buf383
            del buf394
            del primals_161
            buf395 = buf393[1]
            assert_alignment(buf395, 16, 'torch.ops.npu._npu_dropout.default')
            del buf393
            buf402 = empty_strided_npu((4096, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_24, logits], Original ATen: [aten.native_layer_norm, aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:125
            extern_kernels.mm(reinterpret_tensor(buf401, (4096, 768), (768, 1), 0), reinterpret_tensor(primals_162, (768, 2), (1, 768), 0), out=buf402)
            # Topologically Sorted Source Nodes: [logits, pooled_logits], Original ATen: [aten.matmul, aten.index]
            # [Provenance debug handles] torch.ops.aten.index.Tensor:126
            buf407 = torch.ops.aten.index.Tensor(reinterpret_tensor(buf402, (4, 1024, 2), (2048, 2, 1), 0), [buf406, buf405])
            assert_alignment(buf407, 16, 'torch.ops.aten.index.Tensor')
            del buf402
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
            # [Provenance debug handles] torch.ops.aten._log_softmax.default:127
            buf408 = torch.ops.aten._log_softmax.default(buf407, 1, False)
            assert_alignment(buf408, 16, 'torch.ops.aten._log_softmax.default')
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            # [Provenance debug handles] torch.ops.aten.gather.default:128
            buf410 = torch.ops.aten.gather.default(buf408, 1, reinterpret_tensor(buf409, (4, 1), (1, 0), 0))
            assert_alignment(buf410, 16, 'torch.ops.aten.gather.default')
            del buf409
            buf411 = empty_strided_npu((), (), torch.float32)
            buf500 = empty_strided_npu((), (), torch.float32)
            # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_nll_loss_forward_28.run(primals_163, buf410, buf411, buf500, 1, stream=raw_stream0)
            del buf410
            del primals_163
        return (buf500, buf407, reinterpret_tensor(buf12, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf12, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf44, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf44, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf77, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf77, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf109, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf109, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf142, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf142, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf174, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf174, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf207, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf207, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf239, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf239, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf272, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf272, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf304, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf304, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf337, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf337, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf369, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf369, (4, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), primals_1, primals_4, primals_7, primals_8, primals_10, primals_11, primals_14, primals_16, primals_17, primals_20, primals_21, primals_23, primals_24, primals_27, primals_29, primals_30, primals_33, primals_34, primals_36, primals_37, primals_40, primals_42, primals_43, primals_46, primals_47, primals_49, primals_50, primals_53, primals_55, primals_56, primals_59, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_86, primals_88, primals_89, primals_92, primals_94, primals_95, primals_98, primals_99, primals_101, primals_102, primals_105, primals_107, primals_108, primals_111, primals_112, primals_114, primals_115, primals_118, primals_120, primals_121, primals_124, primals_125, primals_127, primals_128, primals_131, primals_133, primals_134, primals_137, primals_138, primals_140, primals_141, primals_144, primals_146, primals_147, primals_150, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_162, reinterpret_tensor(buf0, (1, 1024), (1024, 1), 0), buf6, buf10, buf17, buf20, buf27, buf31, buf33, buf38, buf42, buf49, buf52, buf59, buf63, buf65, buf70, buf75, buf82, buf85, buf92, buf96, buf98, buf103, buf107, buf114, buf117, buf124, buf128, buf130, buf135, buf140, buf147, buf150, buf157, buf161, buf163, buf168, buf172, buf179, buf182, buf189, buf193, buf195, buf200, buf205, buf212, buf215, buf222, buf226, buf228, buf233, buf237, buf244, buf247, buf254, buf258, buf260, buf265, buf270, buf277, buf280, buf287, buf291, buf293, buf298, buf302, buf309, buf312, buf319, buf323, buf325, buf330, buf335, buf342, buf345, buf352, buf356, buf358, buf363, buf367, buf374, buf377, buf384, buf388, buf390, buf395, buf400, buf405, buf406, buf408, buf411, buf412, buf414, reinterpret_tensor(buf401, (768, 4096), (1, 768), 0), buf415, reinterpret_tensor(buf391, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf389, (768, 4096), (1, 768), 0), buf416, reinterpret_tensor(buf380, (768, 4096), (1, 768), 0), reinterpret_tensor(buf417, (48, 64, 1024), (65536, 1024, 1), 0), buf419, reinterpret_tensor(buf420, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf421, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf368, (768, 4096), (1, 768), 0), buf422, reinterpret_tensor(buf359, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf357, (768, 4096), (1, 768), 0), buf423, reinterpret_tensor(buf348, (768, 4096), (1, 768), 0), reinterpret_tensor(buf424, (48, 64, 1024), (65536, 1024, 1), 0), buf426, reinterpret_tensor(buf427, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf428, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf336, (768, 4096), (1, 768), 0), buf429, reinterpret_tensor(buf326, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf324, (768, 4096), (1, 768), 0), buf430, reinterpret_tensor(buf315, (768, 4096), (1, 768), 0), reinterpret_tensor(buf431, (48, 64, 1024), (65536, 1024, 1), 0), buf433, reinterpret_tensor(buf434, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf435, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf303, (768, 4096), (1, 768), 0), buf436, reinterpret_tensor(buf294, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf292, (768, 4096), (1, 768), 0), buf437, reinterpret_tensor(buf283, (768, 4096), (1, 768), 0), reinterpret_tensor(buf438, (48, 64, 1024), (65536, 1024, 1), 0), buf440, reinterpret_tensor(buf441, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf442, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf271, (768, 4096), (1, 768), 0), buf443, reinterpret_tensor(buf261, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf259, (768, 4096), (1, 768), 0), buf444, reinterpret_tensor(buf250, (768, 4096), (1, 768), 0), reinterpret_tensor(buf445, (48, 64, 1024), (65536, 1024, 1), 0), buf447, reinterpret_tensor(buf448, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf449, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf238, (768, 4096), (1, 768), 0), buf450, reinterpret_tensor(buf229, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf227, (768, 4096), (1, 768), 0), buf451, reinterpret_tensor(buf218, (768, 4096), (1, 768), 0), reinterpret_tensor(buf452, (48, 64, 1024), (65536, 1024, 1), 0), buf454, reinterpret_tensor(buf455, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf456, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf206, (768, 4096), (1, 768), 0), buf457, reinterpret_tensor(buf196, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf194, (768, 4096), (1, 768), 0), buf458, reinterpret_tensor(buf185, (768, 4096), (1, 768), 0), reinterpret_tensor(buf459, (48, 64, 1024), (65536, 1024, 1), 0), buf461, reinterpret_tensor(buf462, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf463, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf173, (768, 4096), (1, 768), 0), buf464, reinterpret_tensor(buf164, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf162, (768, 4096), (1, 768), 0), buf465, reinterpret_tensor(buf153, (768, 4096), (1, 768), 0), reinterpret_tensor(buf466, (48, 64, 1024), (65536, 1024, 1), 0), buf468, reinterpret_tensor(buf469, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf470, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf141, (768, 4096), (1, 768), 0), buf471, reinterpret_tensor(buf131, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf129, (768, 4096), (1, 768), 0), buf472, reinterpret_tensor(buf120, (768, 4096), (1, 768), 0), reinterpret_tensor(buf473, (48, 64, 1024), (65536, 1024, 1), 0), buf475, reinterpret_tensor(buf476, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf477, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf108, (768, 4096), (1, 768), 0), buf478, reinterpret_tensor(buf99, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf97, (768, 4096), (1, 768), 0), buf479, reinterpret_tensor(buf88, (768, 4096), (1, 768), 0), reinterpret_tensor(buf480, (48, 64, 1024), (65536, 1024, 1), 0), buf482, reinterpret_tensor(buf483, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf484, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf76, (768, 4096), (1, 768), 0), buf485, reinterpret_tensor(buf66, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf64, (768, 4096), (1, 768), 0), buf486, reinterpret_tensor(buf55, (768, 4096), (1, 768), 0), reinterpret_tensor(buf487, (48, 64, 1024), (65536, 1024, 1), 0), buf489, reinterpret_tensor(buf490, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf491, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf43, (768, 4096), (1, 768), 0), buf492, reinterpret_tensor(buf34, (3072, 4096), (1, 3072), 0), reinterpret_tensor(buf32, (768, 4096), (1, 768), 0), buf493, reinterpret_tensor(buf23, (768, 4096), (1, 768), 0), reinterpret_tensor(buf494, (48, 64, 1024), (65536, 1024, 1), 0), buf496, reinterpret_tensor(buf497, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf498, (48, 64, 1024), (65536, 1024, 1), 0), reinterpret_tensor(buf11, (768, 4096), (1, 768), 0), buf499, )

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
