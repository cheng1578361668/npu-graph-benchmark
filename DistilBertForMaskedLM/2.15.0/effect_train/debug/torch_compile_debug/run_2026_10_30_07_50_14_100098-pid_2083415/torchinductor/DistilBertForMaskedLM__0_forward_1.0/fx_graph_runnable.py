
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.15.0/effect_train/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/opt/ascend_pytorch/pytorch/benchmarks/torchbench/collect/dmb_v2_retry/2.15.0/effect_train/debug'
os.environ['TORCHBENCH_DATA_PATH'] = '/opt/ascend_pytorch/pytorch/benchmarks/torchbench/dataset/'
os.environ['TRITON_ASCEND_HOME'] = '/opt/npu-envs/triton-home'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'triton_experimental'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.use_recursive_dict_tags_for_guards = False
torch._inductor.config.size_asserts = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.layout_optimization = False
torch._inductor.config.fallback_random = True
torch._inductor.config.loop_reindexing_after_fusion = False
torch._inductor.config.split_reductions = False
torch._inductor.config.deterministic = True
torch._inductor.config.aten_distributed_optimizations.collective_estimator = 'benchmark'
torch._inductor.config.aten_distributed_optimizations.compute_estimator = 'benchmark'
torch._inductor.config.aten_distributed_optimizations.enable_fusion_regions = False
torch._inductor.config.aten_distributed_optimizations.pre_bucketing_fsdp_collectives = False
torch._inductor.config.shape_padding = False
torch._inductor.config.cpp.march = ''
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False
torch._inductor.config.triton.coalesce_tiling_analysis = False
torch._inductor.config.triton.mix_order_reduction = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.15.0.dev20261005+cpu
# torch cuda version: None
# torch git version: 7bf52b3a9782d3440a17eb3f00983795f3b12af0


# torch.cuda.is_available()==False, no GPU info collected
torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108):
        embedding = torch.ops.aten.embedding.default(primals_2, primals_1, 0)
        slice_1 = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128)
        embedding_1 = torch.ops.aten.embedding.default(primals_4, slice_1);  primals_4 = slice_1 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_5)
        add_2 = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        _npu_dropout = torch.ops.npu._npu_dropout.default(add_2, 0.1);  add_2 = None
        getitem_2 = _npu_dropout[0]
        getitem_3 = _npu_dropout[1];  _npu_dropout = None
        full = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        view = torch.ops.aten.view.default(getitem_2, [128, 768])
        permute = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_8, view, permute);  primals_8 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 128, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = permute_2 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_4 = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_12, view, permute_4);  primals_12 = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [1, 128, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        div = torch.ops.aten.div.Tensor(permute_1, 8.0);  permute_1 = None
        permute_6 = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand = torch.ops.aten.expand.default(div, [1, 12, 128, 64])
        view_9 = torch.ops.aten.view.default(expand, [12, 128, 64]);  expand = None
        expand_1 = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 128])
        view_10 = torch.ops.aten.view.default(expand_1, [12, 64, 128]);  expand_1 = None
        bmm = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11 = torch.ops.aten.view.default(bmm, [1, 12, 128, 128]);  bmm = None
        eq = torch.ops.aten.eq.Scalar(full, 0);  full = None
        view_12 = torch.ops.aten.view.default(eq, [1, 1, 1, 128]);  eq = None
        expand_2 = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None
        full_default = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(expand_2, full_default, view_11);  view_11 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        _npu_dropout_1 = torch.ops.npu._npu_dropout.default(div_1, 0.1)
        getitem_4 = _npu_dropout_1[0]
        getitem_5 = _npu_dropout_1[1];  _npu_dropout_1 = None
        expand_3 = torch.ops.aten.expand.default(getitem_4, [1, 12, 128, 128])
        view_13 = torch.ops.aten.view.default(expand_3, [12, 128, 128]);  expand_3 = None
        expand_4 = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64])
        view_14 = torch.ops.aten.view.default(expand_4, [12, 128, 64]);  expand_4 = None
        bmm_1 = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15 = torch.ops.aten.view.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16 = torch.ops.aten.view.default(clone, [1, -1, 768]);  clone = None
        view_17 = torch.ops.aten.view.default(view_16, [128, 768]);  view_16 = None
        permute_8 = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_14, view_17, permute_8);  primals_14 = permute_8 = None
        view_18 = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
        add_3 = torch.ops.aten.add.Tensor(view_18, getitem_2);  view_18 = getitem_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_15)
        add_5 = torch.ops.aten.add.Tensor(mul_3, primals_16);  mul_3 = primals_16 = None
        view_19 = torch.ops.aten.view.default(add_5, [128, 768])
        permute_9 = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(primals_18, view_19, permute_9);  primals_18 = permute_9 = None
        view_20 = torch.ops.aten.view.default(addmm_4, [1, 128, 3072])
        mul_4 = torch.ops.aten.mul.Tensor(view_20, view_20)
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, view_20);  mul_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, 0.044715);  mul_5 = None
        add_6 = torch.ops.aten.add.Tensor(view_20, mul_6);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_6, -1.5957691216057308);  add_6 = None
        exp_1 = torch.ops.aten.exp.default(mul_7);  mul_7 = None
        add_7 = torch.ops.aten.add.Tensor(exp_1, 1.0);  exp_1 = None
        div_2 = torch.ops.aten.div.Tensor(view_20, add_7);  view_20 = add_7 = None
        view_21 = torch.ops.aten.view.default(div_2, [128, 3072]);  div_2 = None
        permute_10 = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_5 = torch.ops.aten.addmm.default(primals_20, view_21, permute_10);  primals_20 = permute_10 = None
        view_22 = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
        _npu_dropout_2 = torch.ops.npu._npu_dropout.default(view_22, 0.1);  view_22 = None
        getitem_8 = _npu_dropout_2[0]
        getitem_9 = _npu_dropout_2[1];  _npu_dropout_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_8, add_5);  getitem_8 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_2[0]
        getitem_11 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_11);  add_8 = getitem_11 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, primals_21)
        add_10 = torch.ops.aten.add.Tensor(mul_9, primals_22);  mul_9 = primals_22 = None
        view_23 = torch.ops.aten.view.default(add_10, [128, 768])
        permute_11 = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_6 = torch.ops.aten.addmm.default(primals_24, view_23, permute_11);  primals_24 = permute_11 = None
        view_24 = torch.ops.aten.view.default(addmm_6, [1, 128, 768]);  addmm_6 = None
        view_25 = torch.ops.aten.view.default(view_24, [1, -1, 12, 64]);  view_24 = None
        permute_12 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        permute_13 = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_7 = torch.ops.aten.addmm.default(primals_26, view_23, permute_13);  primals_26 = permute_13 = None
        view_27 = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
        view_28 = torch.ops.aten.view.default(view_27, [1, -1, 12, 64]);  view_27 = None
        permute_14 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_15 = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_8 = torch.ops.aten.addmm.default(primals_28, view_23, permute_15);  primals_28 = permute_15 = None
        view_30 = torch.ops.aten.view.default(addmm_8, [1, 128, 768]);  addmm_8 = None
        view_31 = torch.ops.aten.view.default(view_30, [1, -1, 12, 64]);  view_30 = None
        permute_16 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        div_3 = torch.ops.aten.div.Tensor(permute_12, 8.0);  permute_12 = None
        permute_17 = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_5 = torch.ops.aten.expand.default(div_3, [1, 12, 128, 64])
        view_32 = torch.ops.aten.view.default(expand_5, [12, 128, 64]);  expand_5 = None
        expand_6 = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 128])
        view_33 = torch.ops.aten.view.default(expand_6, [12, 64, 128]);  expand_6 = None
        bmm_2 = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
        view_34 = torch.ops.aten.view.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None
        where_1 = torch.ops.aten.where.self(expand_2, full_default, view_34);  view_34 = None
        amax_1 = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_2 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        _npu_dropout_3 = torch.ops.npu._npu_dropout.default(div_4, 0.1)
        getitem_12 = _npu_dropout_3[0]
        getitem_13 = _npu_dropout_3[1];  _npu_dropout_3 = None
        expand_8 = torch.ops.aten.expand.default(getitem_12, [1, 12, 128, 128])
        view_36 = torch.ops.aten.view.default(expand_8, [12, 128, 128]);  expand_8 = None
        expand_9 = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64])
        view_37 = torch.ops.aten.view.default(expand_9, [12, 128, 64]);  expand_9 = None
        bmm_3 = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_1 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39 = torch.ops.aten.view.default(clone_1, [1, -1, 768]);  clone_1 = None
        view_40 = torch.ops.aten.view.default(view_39, [128, 768]);  view_39 = None
        permute_19 = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_9 = torch.ops.aten.addmm.default(primals_30, view_40, permute_19);  primals_30 = permute_19 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
        add_11 = torch.ops.aten.add.Tensor(view_41, add_10);  view_41 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, primals_31)
        add_13 = torch.ops.aten.add.Tensor(mul_11, primals_32);  mul_11 = primals_32 = None
        view_42 = torch.ops.aten.view.default(add_13, [128, 768])
        permute_20 = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_10 = torch.ops.aten.addmm.default(primals_34, view_42, permute_20);  primals_34 = permute_20 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [1, 128, 3072])
        mul_12 = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, view_43);  mul_12 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, 0.044715);  mul_13 = None
        add_14 = torch.ops.aten.add.Tensor(view_43, mul_14);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_14, -1.5957691216057308);  add_14 = None
        exp_3 = torch.ops.aten.exp.default(mul_15);  mul_15 = None
        add_15 = torch.ops.aten.add.Tensor(exp_3, 1.0);  exp_3 = None
        div_5 = torch.ops.aten.div.Tensor(view_43, add_15);  view_43 = add_15 = None
        view_44 = torch.ops.aten.view.default(div_5, [128, 3072]);  div_5 = None
        permute_21 = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_11 = torch.ops.aten.addmm.default(primals_36, view_44, permute_21);  primals_36 = permute_21 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
        _npu_dropout_4 = torch.ops.npu._npu_dropout.default(view_45, 0.1);  view_45 = None
        getitem_16 = _npu_dropout_4[0]
        getitem_17 = _npu_dropout_4[1];  _npu_dropout_4 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_16, add_13);  getitem_16 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_4[0]
        getitem_19 = var_mean_4[1];  var_mean_4 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_16, getitem_19);  add_16 = getitem_19 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_37)
        add_18 = torch.ops.aten.add.Tensor(mul_17, primals_38);  mul_17 = primals_38 = None
        view_46 = torch.ops.aten.view.default(add_18, [128, 768])
        permute_22 = torch.ops.aten.permute.default(primals_39, [1, 0])
        addmm_12 = torch.ops.aten.addmm.default(primals_40, view_46, permute_22);  primals_40 = permute_22 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [1, 128, 768]);  addmm_12 = None
        view_48 = torch.ops.aten.view.default(view_47, [1, -1, 12, 64]);  view_47 = None
        permute_23 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_24 = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_13 = torch.ops.aten.addmm.default(primals_42, view_46, permute_24);  primals_42 = permute_24 = None
        view_50 = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
        view_51 = torch.ops.aten.view.default(view_50, [1, -1, 12, 64]);  view_50 = None
        permute_25 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        permute_26 = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_14 = torch.ops.aten.addmm.default(primals_44, view_46, permute_26);  primals_44 = permute_26 = None
        view_53 = torch.ops.aten.view.default(addmm_14, [1, 128, 768]);  addmm_14 = None
        view_54 = torch.ops.aten.view.default(view_53, [1, -1, 12, 64]);  view_53 = None
        permute_27 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        div_6 = torch.ops.aten.div.Tensor(permute_23, 8.0);  permute_23 = None
        permute_28 = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_10 = torch.ops.aten.expand.default(div_6, [1, 12, 128, 64])
        view_55 = torch.ops.aten.view.default(expand_10, [12, 128, 64]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 128])
        view_56 = torch.ops.aten.view.default(expand_11, [12, 64, 128]);  expand_11 = None
        bmm_4 = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57 = torch.ops.aten.view.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None
        where_2 = torch.ops.aten.where.self(expand_2, full_default, view_57);  view_57 = None
        amax_2 = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7 = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_4 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        _npu_dropout_5 = torch.ops.npu._npu_dropout.default(div_7, 0.1)
        getitem_20 = _npu_dropout_5[0]
        getitem_21 = _npu_dropout_5[1];  _npu_dropout_5 = None
        expand_13 = torch.ops.aten.expand.default(getitem_20, [1, 12, 128, 128])
        view_59 = torch.ops.aten.view.default(expand_13, [12, 128, 128]);  expand_13 = None
        expand_14 = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64])
        view_60 = torch.ops.aten.view.default(expand_14, [12, 128, 64]);  expand_14 = None
        bmm_5 = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = view_60 = None
        view_61 = torch.ops.aten.view.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_2 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_62 = torch.ops.aten.view.default(clone_2, [1, -1, 768]);  clone_2 = None
        view_63 = torch.ops.aten.view.default(view_62, [128, 768]);  view_62 = None
        permute_30 = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_15 = torch.ops.aten.addmm.default(primals_46, view_63, permute_30);  primals_46 = permute_30 = None
        view_64 = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
        add_19 = torch.ops.aten.add.Tensor(view_64, add_18);  view_64 = add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_19, getitem_23);  add_19 = getitem_23 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, primals_47)
        add_21 = torch.ops.aten.add.Tensor(mul_19, primals_48);  mul_19 = primals_48 = None
        view_65 = torch.ops.aten.view.default(add_21, [128, 768])
        permute_31 = torch.ops.aten.permute.default(primals_49, [1, 0])
        addmm_16 = torch.ops.aten.addmm.default(primals_50, view_65, permute_31);  primals_50 = permute_31 = None
        view_66 = torch.ops.aten.view.default(addmm_16, [1, 128, 3072])
        mul_20 = torch.ops.aten.mul.Tensor(view_66, view_66)
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, view_66);  mul_20 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, 0.044715);  mul_21 = None
        add_22 = torch.ops.aten.add.Tensor(view_66, mul_22);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_22, -1.5957691216057308);  add_22 = None
        exp_5 = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        add_23 = torch.ops.aten.add.Tensor(exp_5, 1.0);  exp_5 = None
        div_8 = torch.ops.aten.div.Tensor(view_66, add_23);  view_66 = add_23 = None
        view_67 = torch.ops.aten.view.default(div_8, [128, 3072]);  div_8 = None
        permute_32 = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_17 = torch.ops.aten.addmm.default(primals_52, view_67, permute_32);  primals_52 = permute_32 = None
        view_68 = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
        _npu_dropout_6 = torch.ops.npu._npu_dropout.default(view_68, 0.1);  view_68 = None
        getitem_24 = _npu_dropout_6[0]
        getitem_25 = _npu_dropout_6[1];  _npu_dropout_6 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_24, add_21);  getitem_24 = add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_6[0]
        getitem_27 = var_mean_6[1];  var_mean_6 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_24, getitem_27);  add_24 = getitem_27 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, primals_53)
        add_26 = torch.ops.aten.add.Tensor(mul_25, primals_54);  mul_25 = primals_54 = None
        view_69 = torch.ops.aten.view.default(add_26, [128, 768])
        permute_33 = torch.ops.aten.permute.default(primals_55, [1, 0])
        addmm_18 = torch.ops.aten.addmm.default(primals_56, view_69, permute_33);  primals_56 = permute_33 = None
        view_70 = torch.ops.aten.view.default(addmm_18, [1, 128, 768]);  addmm_18 = None
        view_71 = torch.ops.aten.view.default(view_70, [1, -1, 12, 64]);  view_70 = None
        permute_34 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_35 = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_19 = torch.ops.aten.addmm.default(primals_58, view_69, permute_35);  primals_58 = permute_35 = None
        view_73 = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
        view_74 = torch.ops.aten.view.default(view_73, [1, -1, 12, 64]);  view_73 = None
        permute_36 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_37 = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_20 = torch.ops.aten.addmm.default(primals_60, view_69, permute_37);  primals_60 = permute_37 = None
        view_76 = torch.ops.aten.view.default(addmm_20, [1, 128, 768]);  addmm_20 = None
        view_77 = torch.ops.aten.view.default(view_76, [1, -1, 12, 64]);  view_76 = None
        permute_38 = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        div_9 = torch.ops.aten.div.Tensor(permute_34, 8.0);  permute_34 = None
        permute_39 = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_15 = torch.ops.aten.expand.default(div_9, [1, 12, 128, 64])
        view_78 = torch.ops.aten.view.default(expand_15, [12, 128, 64]);  expand_15 = None
        expand_16 = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 128])
        view_79 = torch.ops.aten.view.default(expand_16, [12, 64, 128]);  expand_16 = None
        bmm_6 = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80 = torch.ops.aten.view.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None
        where_3 = torch.ops.aten.where.self(expand_2, full_default, view_80);  view_80 = None
        amax_3 = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_6 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        _npu_dropout_7 = torch.ops.npu._npu_dropout.default(div_10, 0.1)
        getitem_28 = _npu_dropout_7[0]
        getitem_29 = _npu_dropout_7[1];  _npu_dropout_7 = None
        expand_18 = torch.ops.aten.expand.default(getitem_28, [1, 12, 128, 128])
        view_82 = torch.ops.aten.view.default(expand_18, [12, 128, 128]);  expand_18 = None
        expand_19 = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64])
        view_83 = torch.ops.aten.view.default(expand_19, [12, 128, 64]);  expand_19 = None
        bmm_7 = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84 = torch.ops.aten.view.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_3 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85 = torch.ops.aten.view.default(clone_3, [1, -1, 768]);  clone_3 = None
        view_86 = torch.ops.aten.view.default(view_85, [128, 768]);  view_85 = None
        permute_41 = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_21 = torch.ops.aten.addmm.default(primals_62, view_86, permute_41);  primals_62 = permute_41 = None
        view_87 = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
        add_27 = torch.ops.aten.add.Tensor(view_87, add_26);  view_87 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_27, getitem_31);  add_27 = getitem_31 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, primals_63)
        add_29 = torch.ops.aten.add.Tensor(mul_27, primals_64);  mul_27 = primals_64 = None
        view_88 = torch.ops.aten.view.default(add_29, [128, 768])
        permute_42 = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_22 = torch.ops.aten.addmm.default(primals_66, view_88, permute_42);  primals_66 = permute_42 = None
        view_89 = torch.ops.aten.view.default(addmm_22, [1, 128, 3072])
        mul_28 = torch.ops.aten.mul.Tensor(view_89, view_89)
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, view_89);  mul_28 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, 0.044715);  mul_29 = None
        add_30 = torch.ops.aten.add.Tensor(view_89, mul_30);  mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(add_30, -1.5957691216057308);  add_30 = None
        exp_7 = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        add_31 = torch.ops.aten.add.Tensor(exp_7, 1.0);  exp_7 = None
        div_11 = torch.ops.aten.div.Tensor(view_89, add_31);  view_89 = add_31 = None
        view_90 = torch.ops.aten.view.default(div_11, [128, 3072]);  div_11 = None
        permute_43 = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_23 = torch.ops.aten.addmm.default(primals_68, view_90, permute_43);  primals_68 = permute_43 = None
        view_91 = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
        _npu_dropout_8 = torch.ops.npu._npu_dropout.default(view_91, 0.1);  view_91 = None
        getitem_32 = _npu_dropout_8[0]
        getitem_33 = _npu_dropout_8[1];  _npu_dropout_8 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_32, add_29);  getitem_32 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_8[0]
        getitem_35 = var_mean_8[1];  var_mean_8 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_32, getitem_35);  add_32 = getitem_35 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, primals_69)
        add_34 = torch.ops.aten.add.Tensor(mul_33, primals_70);  mul_33 = primals_70 = None
        view_92 = torch.ops.aten.view.default(add_34, [128, 768])
        permute_44 = torch.ops.aten.permute.default(primals_71, [1, 0])
        addmm_24 = torch.ops.aten.addmm.default(primals_72, view_92, permute_44);  primals_72 = permute_44 = None
        view_93 = torch.ops.aten.view.default(addmm_24, [1, 128, 768]);  addmm_24 = None
        view_94 = torch.ops.aten.view.default(view_93, [1, -1, 12, 64]);  view_93 = None
        permute_45 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_46 = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_25 = torch.ops.aten.addmm.default(primals_74, view_92, permute_46);  primals_74 = permute_46 = None
        view_96 = torch.ops.aten.view.default(addmm_25, [1, 128, 768]);  addmm_25 = None
        view_97 = torch.ops.aten.view.default(view_96, [1, -1, 12, 64]);  view_96 = None
        permute_47 = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        permute_48 = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_26 = torch.ops.aten.addmm.default(primals_76, view_92, permute_48);  primals_76 = permute_48 = None
        view_99 = torch.ops.aten.view.default(addmm_26, [1, 128, 768]);  addmm_26 = None
        view_100 = torch.ops.aten.view.default(view_99, [1, -1, 12, 64]);  view_99 = None
        permute_49 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        div_12 = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None
        permute_50 = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_20 = torch.ops.aten.expand.default(div_12, [1, 12, 128, 64])
        view_101 = torch.ops.aten.view.default(expand_20, [12, 128, 64]);  expand_20 = None
        expand_21 = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 128])
        view_102 = torch.ops.aten.view.default(expand_21, [12, 64, 128]);  expand_21 = None
        bmm_8 = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103 = torch.ops.aten.view.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None
        where_4 = torch.ops.aten.where.self(expand_2, full_default, view_103);  view_103 = None
        amax_4 = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_13 = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_8 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        _npu_dropout_9 = torch.ops.npu._npu_dropout.default(div_13, 0.1)
        getitem_36 = _npu_dropout_9[0]
        getitem_37 = _npu_dropout_9[1];  _npu_dropout_9 = None
        expand_23 = torch.ops.aten.expand.default(getitem_36, [1, 12, 128, 128])
        view_105 = torch.ops.aten.view.default(expand_23, [12, 128, 128]);  expand_23 = None
        expand_24 = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64])
        view_106 = torch.ops.aten.view.default(expand_24, [12, 128, 64]);  expand_24 = None
        bmm_9 = torch.ops.aten.bmm.default(view_105, view_106);  view_105 = view_106 = None
        view_107 = torch.ops.aten.view.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        clone_4 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_108 = torch.ops.aten.view.default(clone_4, [1, -1, 768]);  clone_4 = None
        view_109 = torch.ops.aten.view.default(view_108, [128, 768]);  view_108 = None
        permute_52 = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_27 = torch.ops.aten.addmm.default(primals_78, view_109, permute_52);  primals_78 = permute_52 = None
        view_110 = torch.ops.aten.view.default(addmm_27, [1, 128, 768]);  addmm_27 = None
        add_35 = torch.ops.aten.add.Tensor(view_110, add_34);  view_110 = add_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_35, getitem_39);  add_35 = getitem_39 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, primals_79)
        add_37 = torch.ops.aten.add.Tensor(mul_35, primals_80);  mul_35 = primals_80 = None
        view_111 = torch.ops.aten.view.default(add_37, [128, 768])
        permute_53 = torch.ops.aten.permute.default(primals_81, [1, 0])
        addmm_28 = torch.ops.aten.addmm.default(primals_82, view_111, permute_53);  primals_82 = permute_53 = None
        view_112 = torch.ops.aten.view.default(addmm_28, [1, 128, 3072])
        mul_36 = torch.ops.aten.mul.Tensor(view_112, view_112)
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, view_112);  mul_36 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, 0.044715);  mul_37 = None
        add_38 = torch.ops.aten.add.Tensor(view_112, mul_38);  mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_38, -1.5957691216057308);  add_38 = None
        exp_9 = torch.ops.aten.exp.default(mul_39);  mul_39 = None
        add_39 = torch.ops.aten.add.Tensor(exp_9, 1.0);  exp_9 = None
        div_14 = torch.ops.aten.div.Tensor(view_112, add_39);  view_112 = add_39 = None
        view_113 = torch.ops.aten.view.default(div_14, [128, 3072]);  div_14 = None
        permute_54 = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_29 = torch.ops.aten.addmm.default(primals_84, view_113, permute_54);  primals_84 = permute_54 = None
        view_114 = torch.ops.aten.view.default(addmm_29, [1, 128, 768]);  addmm_29 = None
        _npu_dropout_10 = torch.ops.npu._npu_dropout.default(view_114, 0.1);  view_114 = None
        getitem_40 = _npu_dropout_10[0]
        getitem_41 = _npu_dropout_10[1];  _npu_dropout_10 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_40, add_37);  getitem_40 = add_37 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_10[0]
        getitem_43 = var_mean_10[1];  var_mean_10 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_40, getitem_43);  add_40 = getitem_43 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, primals_85)
        add_42 = torch.ops.aten.add.Tensor(mul_41, primals_86);  mul_41 = primals_86 = None
        view_115 = torch.ops.aten.view.default(add_42, [128, 768])
        permute_55 = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_30 = torch.ops.aten.addmm.default(primals_88, view_115, permute_55);  primals_88 = permute_55 = None
        view_116 = torch.ops.aten.view.default(addmm_30, [1, 128, 768]);  addmm_30 = None
        view_117 = torch.ops.aten.view.default(view_116, [1, -1, 12, 64]);  view_116 = None
        permute_56 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        permute_57 = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_31 = torch.ops.aten.addmm.default(primals_90, view_115, permute_57);  primals_90 = permute_57 = None
        view_119 = torch.ops.aten.view.default(addmm_31, [1, 128, 768]);  addmm_31 = None
        view_120 = torch.ops.aten.view.default(view_119, [1, -1, 12, 64]);  view_119 = None
        permute_58 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        permute_59 = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_32 = torch.ops.aten.addmm.default(primals_92, view_115, permute_59);  primals_92 = permute_59 = None
        view_122 = torch.ops.aten.view.default(addmm_32, [1, 128, 768]);  addmm_32 = None
        view_123 = torch.ops.aten.view.default(view_122, [1, -1, 12, 64]);  view_122 = None
        permute_60 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        div_15 = torch.ops.aten.div.Tensor(permute_56, 8.0);  permute_56 = None
        permute_61 = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_25 = torch.ops.aten.expand.default(div_15, [1, 12, 128, 64])
        view_124 = torch.ops.aten.view.default(expand_25, [12, 128, 64]);  expand_25 = None
        expand_26 = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 128])
        view_125 = torch.ops.aten.view.default(expand_26, [12, 64, 128]);  expand_26 = None
        bmm_10 = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126 = torch.ops.aten.view.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None
        where_5 = torch.ops.aten.where.self(expand_2, full_default, view_126);  expand_2 = full_default = view_126 = None
        amax_5 = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_10 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        _npu_dropout_11 = torch.ops.npu._npu_dropout.default(div_16, 0.1)
        getitem_44 = _npu_dropout_11[0]
        getitem_45 = _npu_dropout_11[1];  _npu_dropout_11 = None
        expand_28 = torch.ops.aten.expand.default(getitem_44, [1, 12, 128, 128])
        view_128 = torch.ops.aten.view.default(expand_28, [12, 128, 128]);  expand_28 = None
        expand_29 = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64])
        view_129 = torch.ops.aten.view.default(expand_29, [12, 128, 64]);  expand_29 = None
        bmm_11 = torch.ops.aten.bmm.default(view_128, view_129);  view_128 = view_129 = None
        view_130 = torch.ops.aten.view.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_5 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_131 = torch.ops.aten.view.default(clone_5, [1, -1, 768]);  clone_5 = None
        view_132 = torch.ops.aten.view.default(view_131, [128, 768]);  view_131 = None
        permute_63 = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_33 = torch.ops.aten.addmm.default(primals_94, view_132, permute_63);  primals_94 = permute_63 = None
        view_133 = torch.ops.aten.view.default(addmm_33, [1, 128, 768]);  addmm_33 = None
        add_43 = torch.ops.aten.add.Tensor(view_133, add_42);  view_133 = add_42 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_43, getitem_47);  add_43 = getitem_47 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_95)
        add_45 = torch.ops.aten.add.Tensor(mul_43, primals_96);  mul_43 = primals_96 = None
        view_134 = torch.ops.aten.view.default(add_45, [128, 768])
        permute_64 = torch.ops.aten.permute.default(primals_97, [1, 0])
        addmm_34 = torch.ops.aten.addmm.default(primals_98, view_134, permute_64);  primals_98 = permute_64 = None
        view_135 = torch.ops.aten.view.default(addmm_34, [1, 128, 3072])
        mul_44 = torch.ops.aten.mul.Tensor(view_135, view_135)
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, view_135);  mul_44 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.044715);  mul_45 = None
        add_46 = torch.ops.aten.add.Tensor(view_135, mul_46);  mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_46, -1.5957691216057308);  add_46 = None
        exp_11 = torch.ops.aten.exp.default(mul_47);  mul_47 = None
        add_47 = torch.ops.aten.add.Tensor(exp_11, 1.0);  exp_11 = None
        div_17 = torch.ops.aten.div.Tensor(view_135, add_47);  view_135 = add_47 = None
        view_136 = torch.ops.aten.view.default(div_17, [128, 3072]);  div_17 = None
        permute_65 = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_35 = torch.ops.aten.addmm.default(primals_100, view_136, permute_65);  primals_100 = permute_65 = None
        view_137 = torch.ops.aten.view.default(addmm_35, [1, 128, 768]);  addmm_35 = None
        _npu_dropout_12 = torch.ops.npu._npu_dropout.default(view_137, 0.1);  view_137 = None
        getitem_48 = _npu_dropout_12[0]
        getitem_49 = _npu_dropout_12[1];  _npu_dropout_12 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_48, add_45);  getitem_48 = add_45 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_12[0]
        getitem_51 = var_mean_12[1];  var_mean_12 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_48, getitem_51);  add_48 = getitem_51 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, primals_101)
        add_50 = torch.ops.aten.add.Tensor(mul_49, primals_102);  mul_49 = primals_102 = None
        view_138 = torch.ops.aten.view.default(add_50, [128, 768]);  add_50 = None
        permute_66 = torch.ops.aten.permute.default(primals_103, [1, 0])
        addmm_36 = torch.ops.aten.addmm.default(primals_104, view_138, permute_66);  primals_104 = permute_66 = None
        view_139 = torch.ops.aten.view.default(addmm_36, [1, 128, 768])
        mul_50 = torch.ops.aten.mul.Tensor(view_139, view_139)
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, view_139);  mul_50 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, 0.044715);  mul_51 = None
        add_51 = torch.ops.aten.add.Tensor(view_139, mul_52);  mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_51, -1.5957691216057308);  add_51 = None
        exp_12 = torch.ops.aten.exp.default(mul_53);  mul_53 = None
        add_52 = torch.ops.aten.add.Tensor(exp_12, 1.0);  exp_12 = None
        div_18 = torch.ops.aten.div.Tensor(view_139, add_52);  view_139 = add_52 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(div_18, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_13[0]
        getitem_53 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_19 = torch.ops.aten.sub.Tensor(div_18, getitem_53);  div_18 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, primals_105);  mul_54 = None
        add_54 = torch.ops.aten.add.Tensor(mul_55, primals_106);  mul_55 = primals_106 = None
        view_140 = torch.ops.aten.view.default(add_54, [128, 768]);  add_54 = None
        permute_67 = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm_37 = torch.ops.aten.addmm.default(primals_107, view_140, permute_67);  primals_107 = permute_67 = None
        view_141 = torch.ops.aten.view.default(addmm_37, [1, 128, 30522]);  addmm_37 = None
        view_142 = torch.ops.aten.view.default(view_141, [-1, 30522])
        view_143 = torch.ops.aten.view.default(primals_108, [-1])
        _log_softmax = torch.ops.aten._log_softmax.default(view_142, 1, False);  view_142 = None
        ne = torch.ops.aten.ne.Scalar(view_143, -100)
        full_default_6 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_6 = torch.ops.aten.where.self(ne, view_143, full_default_6);  full_default_6 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather = torch.ops.aten.gather.default(_log_softmax, 1, unsqueeze);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_7 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(ne, neg, full_default_7);  ne = neg = full_default_7 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_143, -100);  view_143 = None
        sum_7 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        sum_8 = torch.ops.aten.sum.default(where_7);  where_7 = None
        div_19 = torch.ops.aten.div.Tensor(sum_8, convert_element_type);  sum_8 = None
        div_22 = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_23 = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_89 = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
        expand_31 = torch.ops.aten.expand.default(permute_89, [1, 12, 64, 128]);  permute_89 = None
        view_162 = torch.ops.aten.view.default(expand_31, [12, 64, 128]);  expand_31 = None
        permute_90 = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_32 = torch.ops.aten.expand.default(permute_90, [1, 12, 128, 128]);  permute_90 = None
        view_164 = torch.ops.aten.view.default(expand_32, [12, 128, 128]);  expand_32 = None
        permute_91 = torch.ops.aten.permute.default(permute_61, [0, 1, 3, 2]);  permute_61 = None
        expand_35 = torch.ops.aten.expand.default(permute_91, [1, 12, 128, 64]);  permute_91 = None
        view_168 = torch.ops.aten.view.default(expand_35, [12, 128, 64]);  expand_35 = None
        permute_92 = torch.ops.aten.permute.default(div_15, [0, 1, 3, 2]);  div_15 = None
        expand_36 = torch.ops.aten.expand.default(permute_92, [1, 12, 64, 128]);  permute_92 = None
        view_170 = torch.ops.aten.view.default(expand_36, [12, 64, 128]);  expand_36 = None
        div_25 = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_26 = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_122 = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
        expand_39 = torch.ops.aten.expand.default(permute_122, [1, 12, 64, 128]);  permute_122 = None
        view_196 = torch.ops.aten.view.default(expand_39, [12, 64, 128]);  expand_39 = None
        permute_123 = torch.ops.aten.permute.default(getitem_36, [0, 1, 3, 2]);  getitem_36 = None
        expand_40 = torch.ops.aten.expand.default(permute_123, [1, 12, 128, 128]);  permute_123 = None
        view_198 = torch.ops.aten.view.default(expand_40, [12, 128, 128]);  expand_40 = None
        permute_124 = torch.ops.aten.permute.default(permute_50, [0, 1, 3, 2]);  permute_50 = None
        expand_43 = torch.ops.aten.expand.default(permute_124, [1, 12, 128, 64]);  permute_124 = None
        view_202 = torch.ops.aten.view.default(expand_43, [12, 128, 64]);  expand_43 = None
        permute_125 = torch.ops.aten.permute.default(div_12, [0, 1, 3, 2]);  div_12 = None
        expand_44 = torch.ops.aten.expand.default(permute_125, [1, 12, 64, 128]);  permute_125 = None
        view_204 = torch.ops.aten.view.default(expand_44, [12, 64, 128]);  expand_44 = None
        div_28 = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_29 = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_155 = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        expand_47 = torch.ops.aten.expand.default(permute_155, [1, 12, 64, 128]);  permute_155 = None
        view_230 = torch.ops.aten.view.default(expand_47, [12, 64, 128]);  expand_47 = None
        permute_156 = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_48 = torch.ops.aten.expand.default(permute_156, [1, 12, 128, 128]);  permute_156 = None
        view_232 = torch.ops.aten.view.default(expand_48, [12, 128, 128]);  expand_48 = None
        permute_157 = torch.ops.aten.permute.default(permute_39, [0, 1, 3, 2]);  permute_39 = None
        expand_51 = torch.ops.aten.expand.default(permute_157, [1, 12, 128, 64]);  permute_157 = None
        view_236 = torch.ops.aten.view.default(expand_51, [12, 128, 64]);  expand_51 = None
        permute_158 = torch.ops.aten.permute.default(div_9, [0, 1, 3, 2]);  div_9 = None
        expand_52 = torch.ops.aten.expand.default(permute_158, [1, 12, 64, 128]);  permute_158 = None
        view_238 = torch.ops.aten.view.default(expand_52, [12, 64, 128]);  expand_52 = None
        div_31 = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_32 = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_188 = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
        expand_55 = torch.ops.aten.expand.default(permute_188, [1, 12, 64, 128]);  permute_188 = None
        view_264 = torch.ops.aten.view.default(expand_55, [12, 64, 128]);  expand_55 = None
        permute_189 = torch.ops.aten.permute.default(getitem_20, [0, 1, 3, 2]);  getitem_20 = None
        expand_56 = torch.ops.aten.expand.default(permute_189, [1, 12, 128, 128]);  permute_189 = None
        view_266 = torch.ops.aten.view.default(expand_56, [12, 128, 128]);  expand_56 = None
        permute_190 = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_59 = torch.ops.aten.expand.default(permute_190, [1, 12, 128, 64]);  permute_190 = None
        view_270 = torch.ops.aten.view.default(expand_59, [12, 128, 64]);  expand_59 = None
        permute_191 = torch.ops.aten.permute.default(div_6, [0, 1, 3, 2]);  div_6 = None
        expand_60 = torch.ops.aten.expand.default(permute_191, [1, 12, 64, 128]);  permute_191 = None
        view_272 = torch.ops.aten.view.default(expand_60, [12, 64, 128]);  expand_60 = None
        div_34 = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_35 = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_221 = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_63 = torch.ops.aten.expand.default(permute_221, [1, 12, 64, 128]);  permute_221 = None
        view_298 = torch.ops.aten.view.default(expand_63, [12, 64, 128]);  expand_63 = None
        permute_222 = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_64 = torch.ops.aten.expand.default(permute_222, [1, 12, 128, 128]);  permute_222 = None
        view_300 = torch.ops.aten.view.default(expand_64, [12, 128, 128]);  expand_64 = None
        permute_223 = torch.ops.aten.permute.default(permute_17, [0, 1, 3, 2]);  permute_17 = None
        expand_67 = torch.ops.aten.expand.default(permute_223, [1, 12, 128, 64]);  permute_223 = None
        view_304 = torch.ops.aten.view.default(expand_67, [12, 128, 64]);  expand_67 = None
        permute_224 = torch.ops.aten.permute.default(div_3, [0, 1, 3, 2]);  div_3 = None
        expand_68 = torch.ops.aten.expand.default(permute_224, [1, 12, 64, 128]);  permute_224 = None
        view_306 = torch.ops.aten.view.default(expand_68, [12, 64, 128]);  expand_68 = None
        div_37 = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        div_38 = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_254 = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        expand_71 = torch.ops.aten.expand.default(permute_254, [1, 12, 64, 128]);  permute_254 = None
        view_332 = torch.ops.aten.view.default(expand_71, [12, 64, 128]);  expand_71 = None
        permute_255 = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
        expand_72 = torch.ops.aten.expand.default(permute_255, [1, 12, 128, 128]);  permute_255 = None
        view_334 = torch.ops.aten.view.default(expand_72, [12, 128, 128]);  expand_72 = None
        permute_256 = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        expand_75 = torch.ops.aten.expand.default(permute_256, [1, 12, 128, 64]);  permute_256 = None
        view_338 = torch.ops.aten.view.default(expand_75, [12, 128, 64]);  expand_75 = None
        permute_257 = torch.ops.aten.permute.default(div, [0, 1, 3, 2]);  div = None
        expand_76 = torch.ops.aten.expand.default(permute_257, [1, 12, 64, 128]);  permute_257 = None
        view_340 = torch.ops.aten.view.default(expand_76, [12, 64, 128]);  expand_76 = None
        div_40 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_19, view_141, primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, mul, getitem_3, view, div_1, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_8, view_23, div_4, getitem_13, view_40, mul_10, view_42, addmm_10, view_44, getitem_17, mul_16, view_46, div_7, getitem_21, view_63, mul_18, view_65, addmm_16, view_67, getitem_25, mul_24, view_69, div_10, getitem_29, view_86, mul_26, view_88, addmm_22, view_90, getitem_33, mul_32, view_92, div_13, getitem_37, view_109, mul_34, view_111, addmm_28, view_113, getitem_41, mul_40, view_115, div_16, getitem_45, view_132, mul_42, view_134, addmm_34, view_136, getitem_49, mul_48, view_138, addmm_36, getitem_53, rsqrt_13, view_140, _log_softmax, convert_element_type, div_22, div_23, view_162, view_164, view_168, view_170, div_25, div_26, view_196, view_198, view_202, view_204, div_28, div_29, view_230, view_232, view_236, view_238, div_31, div_32, view_264, view_266, view_270, view_272, div_34, div_35, view_298, view_300, view_304, view_306, div_37, div_38, view_332, view_334, view_338, view_340, div_40)

def load_args(reader):
    buf0 = reader.storage(None, 1272, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 93763584, device=device(type='npu', index=0))
    reader.tensor(buf1, (30522, 768), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 1572864, device=device(type='npu', index=0))
    reader.tensor(buf3, (512, 768), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf6, (768, 768), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf10, (768, 768), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf12, (768, 768), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf16, (3072, 768), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf17, (3072,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf18, (768, 3072), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf22, (768, 768), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf24, (768, 768), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf26, (768, 768), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf28, (768, 768), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf32, (3072, 768), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf33, (3072,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf34, (768, 3072), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf36, (768,), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf38, (768, 768), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf40, (768, 768), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf42, (768, 768), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf44, (768, 768), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf48, (3072, 768), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf49, (3072,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf50, (768, 3072), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf54, (768, 768), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf58, (768, 768), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf60, (768, 768), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf64, (3072, 768), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf65, (3072,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf66, (768, 3072), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf70, (768, 768), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf72, (768, 768), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf74, (768, 768), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf76, (768, 768), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf80, (3072, 768), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf81, (3072,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf82, (768, 3072), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf84, (768,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf86, (768, 768), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf88, (768, 768), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf90, (768, 768), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf92, (768, 768), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf96, (3072, 768), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 12288, device=device(type='npu', index=0))
    reader.tensor(buf97, (3072,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 9437184, device=device(type='npu', index=0))
    reader.tensor(buf98, (768, 3072), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 2359296, device=device(type='npu', index=0))
    reader.tensor(buf102, (768, 768), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 3072, device=device(type='npu', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 122088, device=device(type='npu', index=0))
    reader.tensor(buf106, (30522,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 1272, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf107, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_108
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None, is_inference=False)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None, is_inference=False)
        # mod(*args)