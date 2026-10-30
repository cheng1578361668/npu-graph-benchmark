class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_4: "f32[512, 768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[3072, 768]", primals_18: "f32[3072]", primals_19: "f32[768, 3072]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768, 768]", primals_26: "f32[768]", primals_27: "f32[768, 768]", primals_28: "f32[768]", primals_29: "f32[768, 768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[3072, 768]", primals_34: "f32[3072]", primals_35: "f32[768, 3072]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768, 768]", primals_40: "f32[768]", primals_41: "f32[768, 768]", primals_42: "f32[768]", primals_43: "f32[768, 768]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[3072, 768]", primals_50: "f32[3072]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768, 768]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768, 768]", primals_60: "f32[768]", primals_61: "f32[768, 768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[3072, 768]", primals_66: "f32[3072]", primals_67: "f32[768, 3072]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768, 768]", primals_72: "f32[768]", primals_73: "f32[768, 768]", primals_74: "f32[768]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_77: "f32[768, 768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[3072, 768]", primals_82: "f32[3072]", primals_83: "f32[768, 3072]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768, 768]", primals_88: "f32[768]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[768, 768]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[3072, 768]", primals_98: "f32[3072]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768, 768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[30522]", primals_108: "i64[1, 128]"):
        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:141 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:149 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:154 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(primals_4, slice_1);  primals_4 = slice_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:156 in forward, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5)
        add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:158 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        _npu_dropout = torch.ops.npu._npu_dropout.default(add_2, 0.1);  add_2 = None
        getitem_2: "f32[1, 128, 768]" = _npu_dropout[0]
        getitem_3: "u8[12288]" = _npu_dropout[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:818 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view: "f32[128, 768]" = torch.ops.aten.reshape.default(getitem_2, [128, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        mm_default_11: "f32[128, 768]" = torch.ops.aten.mm.default(view, permute);  permute = None
        add_tensor_11: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_8, mm_default_11);  primals_8 = mm_default_11 = None
        view_1: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 128, 768]);  add_tensor_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_2: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = permute_2 = None
        view_4: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 768]);  addmm_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_5: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_12, view, permute_4);  primals_12 = permute_4 = None
        view_7: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 768]);  addmm_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_8: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_1, 8.0);  permute_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_6: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div, [1, 12, 128, 64])
        view_9: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand, [12, 128, 64]);  expand = None
        expand_1: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 128])
        view_10: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 128]);  expand_1 = None
        bmm: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 12, 128, 128]);  bmm = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
        view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq, [1, 1, 1, 128]);  eq = None
        expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_11);  view_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        sub_1: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_1 = self.subgraph__npu_dropout_1

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_1 = torch.ops.higher_order.control_deps((_npu_dropout,), subgraph__npu_dropout_1, div_1);  _npu_dropout = subgraph__npu_dropout_1 = None
        getitem_4: "f32[1, 12, 128, 128]" = _npu_dropout_1[0]
        getitem_5: "u8[24576]" = _npu_dropout_1[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_3: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_4, [1, 12, 128, 128])
        view_13: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [12, 128, 128]);  expand_3 = None
        expand_4: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64])
        view_14: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 128, 64]);  expand_4 = None
        bmm_1: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_7: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone, [1, -1, 768]);  clone = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_17: "f32[128, 768]" = torch.ops.aten.reshape.default(view_16, [128, 768]);  view_16 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        mm_default_10: "f32[128, 768]" = torch.ops.aten.mm.default(view_17, permute_8);  permute_8 = None
        add_tensor_10: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_14, mm_default_10);  primals_14 = mm_default_10 = None
        view_18: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 128, 768]);  add_tensor_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_18, getitem_2);  view_18 = getitem_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
        mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_15)
        add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_16);  mul_3 = primals_16 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_19: "f32[128, 768]" = torch.ops.aten.reshape.default(add_5, [128, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_4: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_18, view_19, permute_9);  primals_18 = permute_9 = None
        view_20: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, view_20)
        mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, view_20);  mul_4 = None
        mul_6: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_5, 0.044715);  mul_5 = None
        add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_20, mul_6);  mul_6 = None
        mul_7: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_6, -1.5957691216057308);  add_6 = None
        exp_1: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_7);  mul_7 = None
        add_7: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_1, 1.0);  exp_1 = None
        div_2: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_20, add_7);  view_20 = add_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_21: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_2, [128, 3072]);  div_2 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_5: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_20, view_21, permute_10);  primals_20 = permute_10 = None
        view_22: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 768]);  addmm_5 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_2 = self.subgraph__npu_dropout_2

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_2 = torch.ops.higher_order.control_deps((_npu_dropout_1,), subgraph__npu_dropout_2, view_22);  _npu_dropout_1 = subgraph__npu_dropout_2 = view_22 = None
        getitem_8: "f32[1, 128, 768]" = _npu_dropout_2[0]
        getitem_9: "u8[12288]" = _npu_dropout_2[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_8: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_8, add_5);  getitem_8 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_11: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_11);  add_8 = getitem_11 = None
        mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_21)
        add_10: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_22);  mul_9 = primals_22 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_23: "f32[128, 768]" = torch.ops.aten.reshape.default(add_10, [128, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        mm_default_9: "f32[128, 768]" = torch.ops.aten.mm.default(view_23, permute_11);  permute_11 = None
        add_tensor_9: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_24, mm_default_9);  primals_24 = mm_default_9 = None
        view_24: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 128, 768]);  add_tensor_9 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_25: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_24, [1, -1, 12, 64]);  view_24 = None
        permute_12: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_7: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_26, view_23, permute_13);  primals_26 = permute_13 = None
        view_27: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 768]);  addmm_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_28: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 12, 64]);  view_27 = None
        permute_14: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_8: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_28, view_23, permute_15);  primals_28 = permute_15 = None
        view_30: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 768]);  addmm_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_31: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_30, [1, -1, 12, 64]);  view_30 = None
        permute_16: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_3: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_12, 8.0);  permute_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_17: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_5: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_3, [1, 12, 128, 64])
        view_32: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 128, 64]);  expand_5 = None
        expand_6: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 128])
        view_33: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_6, [12, 64, 128]);  expand_6 = None
        bmm_2: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
        view_34: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_1: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_34);  view_34 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_1: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_4: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_2: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_3 = self.subgraph__npu_dropout_3

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_3 = torch.ops.higher_order.control_deps((_npu_dropout_2,), subgraph__npu_dropout_3, div_4);  _npu_dropout_2 = subgraph__npu_dropout_3 = None
        getitem_12: "f32[1, 12, 128, 128]" = _npu_dropout_3[0]
        getitem_13: "u8[24576]" = _npu_dropout_3[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_8: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_12, [1, 12, 128, 128])
        view_36: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [12, 128, 128]);  expand_8 = None
        expand_9: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64])
        view_37: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_9, [12, 128, 64]);  expand_9 = None
        bmm_3: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_18: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_1: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_1, [1, -1, 768]);  clone_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_40: "f32[128, 768]" = torch.ops.aten.reshape.default(view_39, [128, 768]);  view_39 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        mm_default_8: "f32[128, 768]" = torch.ops.aten.mm.default(view_40, permute_19);  permute_19 = None
        add_tensor_8: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_30, mm_default_8);  primals_30 = mm_default_8 = None
        view_41: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 128, 768]);  add_tensor_8 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_11: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_10);  view_41 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_15: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_11: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_31)
        add_13: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_32);  mul_11 = primals_32 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_42: "f32[128, 768]" = torch.ops.aten.reshape.default(add_13, [128, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_10: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_34, view_42, permute_20);  primals_34 = permute_20 = None
        view_43: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_13: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, view_43);  mul_12 = None
        mul_14: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_13, 0.044715);  mul_13 = None
        add_14: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_14);  mul_14 = None
        mul_15: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_14, -1.5957691216057308);  add_14 = None
        exp_3: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_15);  mul_15 = None
        add_15: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_3, 1.0);  exp_3 = None
        div_5: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_43, add_15);  view_43 = add_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_44: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_5, [128, 3072]);  div_5 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_11: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_36, view_44, permute_21);  primals_36 = permute_21 = None
        view_45: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 768]);  addmm_11 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_4 = self.subgraph__npu_dropout_4

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_4 = torch.ops.higher_order.control_deps((_npu_dropout_3,), subgraph__npu_dropout_4, view_45);  _npu_dropout_3 = subgraph__npu_dropout_4 = view_45 = None
        getitem_16: "f32[1, 128, 768]" = _npu_dropout_4[0]
        getitem_17: "u8[12288]" = _npu_dropout_4[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_13);  getitem_16 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_19: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_19);  add_16 = getitem_19 = None
        mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_37)
        add_18: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_38);  mul_17 = primals_38 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_46: "f32[128, 768]" = torch.ops.aten.reshape.default(add_18, [128, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        mm_default_7: "f32[128, 768]" = torch.ops.aten.mm.default(view_46, permute_22);  permute_22 = None
        add_tensor_7: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_40, mm_default_7);  primals_40 = mm_default_7 = None
        view_47: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 128, 768]);  add_tensor_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_48: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, -1, 12, 64]);  view_47 = None
        permute_23: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_13: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_42, view_46, permute_24);  primals_42 = permute_24 = None
        view_50: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 768]);  addmm_13 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_51: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, -1, 12, 64]);  view_50 = None
        permute_25: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_14: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_44, view_46, permute_26);  primals_44 = permute_26 = None
        view_53: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 768]);  addmm_14 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_54: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_53, [1, -1, 12, 64]);  view_53 = None
        permute_27: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_6: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_23, 8.0);  permute_23 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_28: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_10: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_6, [1, 12, 128, 64])
        view_55: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 128, 64]);  expand_10 = None
        expand_11: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 128])
        view_56: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_11, [12, 64, 128]);  expand_11 = None
        bmm_4: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_2: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_57);  view_57 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_2: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_4: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_7: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_5 = self.subgraph__npu_dropout_5

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_5 = torch.ops.higher_order.control_deps((_npu_dropout_4,), subgraph__npu_dropout_5, div_7);  _npu_dropout_4 = subgraph__npu_dropout_5 = None
        getitem_20: "f32[1, 12, 128, 128]" = _npu_dropout_5[0]
        getitem_21: "u8[24576]" = _npu_dropout_5[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_13: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_20, [1, 12, 128, 128])
        view_59: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_13, [12, 128, 128]);  expand_13 = None
        expand_14: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64])
        view_60: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 128, 64]);  expand_14 = None
        bmm_5: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = view_60 = None
        view_61: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_29: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_2: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_62: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_2, [1, -1, 768]);  clone_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_63: "f32[128, 768]" = torch.ops.aten.reshape.default(view_62, [128, 768]);  view_62 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        mm_default_6: "f32[128, 768]" = torch.ops.aten.mm.default(view_63, permute_30);  permute_30 = None
        add_tensor_6: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_46, mm_default_6);  primals_46 = mm_default_6 = None
        view_64: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 128, 768]);  add_tensor_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_18);  view_64 = add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_23: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  add_19 = getitem_23 = None
        mul_18: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_19: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_47)
        add_21: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_48);  mul_19 = primals_48 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_65: "f32[128, 768]" = torch.ops.aten.reshape.default(add_21, [128, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0])
        addmm_16: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_50, view_65, permute_31);  primals_50 = permute_31 = None
        view_66: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, view_66)
        mul_21: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, view_66);  mul_20 = None
        mul_22: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_21, 0.044715);  mul_21 = None
        add_22: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_66, mul_22);  mul_22 = None
        mul_23: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_22, -1.5957691216057308);  add_22 = None
        exp_5: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        add_23: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_5, 1.0);  exp_5 = None
        div_8: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_66, add_23);  view_66 = add_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_67: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_8, [128, 3072]);  div_8 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_17: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_52, view_67, permute_32);  primals_52 = permute_32 = None
        view_68: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 768]);  addmm_17 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_6 = self.subgraph__npu_dropout_6

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_6 = torch.ops.higher_order.control_deps((_npu_dropout_5,), subgraph__npu_dropout_6, view_68);  _npu_dropout_5 = subgraph__npu_dropout_6 = view_68 = None
        getitem_24: "f32[1, 128, 768]" = _npu_dropout_6[0]
        getitem_25: "u8[12288]" = _npu_dropout_6[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_24: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_24, add_21);  getitem_24 = add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_27: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_27);  add_24 = getitem_27 = None
        mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_25: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_53)
        add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_54);  mul_25 = primals_54 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_69: "f32[128, 768]" = torch.ops.aten.reshape.default(add_26, [128, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        mm_default_5: "f32[128, 768]" = torch.ops.aten.mm.default(view_69, permute_33);  permute_33 = None
        add_tensor_5: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_56, mm_default_5);  primals_56 = mm_default_5 = None
        view_70: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 768]);  add_tensor_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_71: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_70, [1, -1, 12, 64]);  view_70 = None
        permute_34: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_19: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_58, view_69, permute_35);  primals_58 = permute_35 = None
        view_73: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 768]);  addmm_19 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_74: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_73, [1, -1, 12, 64]);  view_73 = None
        permute_36: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_20: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_60, view_69, permute_37);  primals_60 = permute_37 = None
        view_76: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 768]);  addmm_20 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_77: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_76, [1, -1, 12, 64]);  view_76 = None
        permute_38: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_9: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_34, 8.0);  permute_34 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_39: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_15: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_9, [1, 12, 128, 64])
        view_78: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 128, 64]);  expand_15 = None
        expand_16: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 128])
        view_79: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 128]);  expand_16 = None
        bmm_6: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_3: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_80);  view_80 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_3: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_10: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_6: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_7 = self.subgraph__npu_dropout_7

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_7 = torch.ops.higher_order.control_deps((_npu_dropout_6,), subgraph__npu_dropout_7, div_10);  _npu_dropout_6 = subgraph__npu_dropout_7 = None
        getitem_28: "f32[1, 12, 128, 128]" = _npu_dropout_7[0]
        getitem_29: "u8[24576]" = _npu_dropout_7[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_18: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_28, [1, 12, 128, 128])
        view_82: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [12, 128, 128]);  expand_18 = None
        expand_19: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64])
        view_83: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 128, 64]);  expand_19 = None
        bmm_7: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_40: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_3: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_3, [1, -1, 768]);  clone_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_86: "f32[128, 768]" = torch.ops.aten.reshape.default(view_85, [128, 768]);  view_85 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        mm_default_4: "f32[128, 768]" = torch.ops.aten.mm.default(view_86, permute_41);  permute_41 = None
        add_tensor_4: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_62, mm_default_4);  primals_62 = mm_default_4 = None
        view_87: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 768]);  add_tensor_4 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_27: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_26);  view_87 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_31: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_31);  add_27 = getitem_31 = None
        mul_26: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_27: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_63)
        add_29: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_64);  mul_27 = primals_64 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_88: "f32[128, 768]" = torch.ops.aten.reshape.default(add_29, [128, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_22: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_66, view_88, permute_42);  primals_66 = permute_42 = None
        view_89: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_28: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, view_89)
        mul_29: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, view_89);  mul_28 = None
        mul_30: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_29, 0.044715);  mul_29 = None
        add_30: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_30);  mul_30 = None
        mul_31: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_30, -1.5957691216057308);  add_30 = None
        exp_7: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        add_31: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_7, 1.0);  exp_7 = None
        div_11: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_89, add_31);  view_89 = add_31 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_90: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_11, [128, 3072]);  div_11 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_23: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_68, view_90, permute_43);  primals_68 = permute_43 = None
        view_91: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 768]);  addmm_23 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_8 = self.subgraph__npu_dropout_8

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_8 = torch.ops.higher_order.control_deps((_npu_dropout_7,), subgraph__npu_dropout_8, view_91);  _npu_dropout_7 = subgraph__npu_dropout_8 = view_91 = None
        getitem_32: "f32[1, 128, 768]" = _npu_dropout_8[0]
        getitem_33: "u8[12288]" = _npu_dropout_8[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_32: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_32, add_29);  getitem_32 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_35: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_35);  add_32 = getitem_35 = None
        mul_32: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_33: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_69)
        add_34: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_70);  mul_33 = primals_70 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_92: "f32[128, 768]" = torch.ops.aten.reshape.default(add_34, [128, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        mm_default_3: "f32[128, 768]" = torch.ops.aten.mm.default(view_92, permute_44);  permute_44 = None
        add_tensor_3: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_72, mm_default_3);  primals_72 = mm_default_3 = None
        view_93: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 768]);  add_tensor_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_94: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 12, 64]);  view_93 = None
        permute_45: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_25: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_74, view_92, permute_46);  primals_74 = permute_46 = None
        view_96: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 768]);  addmm_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_97: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_96, [1, -1, 12, 64]);  view_96 = None
        permute_47: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_26: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_76, view_92, permute_48);  primals_76 = permute_48 = None
        view_99: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 768]);  addmm_26 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_100: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_99, [1, -1, 12, 64]);  view_99 = None
        permute_49: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_12: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_50: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_20: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_12, [1, 12, 128, 64])
        view_101: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 128, 64]);  expand_20 = None
        expand_21: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 128])
        view_102: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 128]);  expand_21 = None
        bmm_8: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_4: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_103);  view_103 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_4: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_13: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_8: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_13: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_9 = self.subgraph__npu_dropout_9

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_9 = torch.ops.higher_order.control_deps((_npu_dropout_8,), subgraph__npu_dropout_9, div_13);  _npu_dropout_8 = subgraph__npu_dropout_9 = None
        getitem_36: "f32[1, 12, 128, 128]" = _npu_dropout_9[0]
        getitem_37: "u8[24576]" = _npu_dropout_9[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_23: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_36, [1, 12, 128, 128])
        view_105: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [12, 128, 128]);  expand_23 = None
        expand_24: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64])
        view_106: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 128, 64]);  expand_24 = None
        bmm_9: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_105, view_106);  view_105 = view_106 = None
        view_107: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_51: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        clone_4: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_108: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_4, [1, -1, 768]);  clone_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_109: "f32[128, 768]" = torch.ops.aten.reshape.default(view_108, [128, 768]);  view_108 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        mm_default_2: "f32[128, 768]" = torch.ops.aten.mm.default(view_109, permute_52);  permute_52 = None
        add_tensor_2: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_78, mm_default_2);  primals_78 = mm_default_2 = None
        view_110: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 768]);  add_tensor_2 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_35: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_34);  view_110 = add_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_39: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_39);  add_35 = getitem_39 = None
        mul_34: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_79)
        add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_80);  mul_35 = primals_80 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_111: "f32[128, 768]" = torch.ops.aten.reshape.default(add_37, [128, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0])
        addmm_28: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_82, view_111, permute_53);  primals_82 = permute_53 = None
        view_112: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_36: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
        mul_37: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, view_112);  mul_36 = None
        mul_38: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_37, 0.044715);  mul_37 = None
        add_38: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_112, mul_38);  mul_38 = None
        mul_39: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_38, -1.5957691216057308);  add_38 = None
        exp_9: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_39);  mul_39 = None
        add_39: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_9, 1.0);  exp_9 = None
        div_14: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_112, add_39);  view_112 = add_39 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_113: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_14, [128, 3072]);  div_14 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_29: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_84, view_113, permute_54);  primals_84 = permute_54 = None
        view_114: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 768]);  addmm_29 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_10 = self.subgraph__npu_dropout_10

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_10 = torch.ops.higher_order.control_deps((_npu_dropout_9,), subgraph__npu_dropout_10, view_114);  _npu_dropout_9 = subgraph__npu_dropout_10 = view_114 = None
        getitem_40: "f32[1, 128, 768]" = _npu_dropout_10[0]
        getitem_41: "u8[12288]" = _npu_dropout_10[1]

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_37);  getitem_40 = add_37 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_43: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_15: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_43);  add_40 = getitem_43 = None
        mul_40: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_41: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_85)
        add_42: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_86);  mul_41 = primals_86 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_115: "f32[128, 768]" = torch.ops.aten.reshape.default(add_42, [128, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        mm_default_1: "f32[128, 768]" = torch.ops.aten.mm.default(view_115, permute_55);  permute_55 = None
        add_tensor_1: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_88, mm_default_1);  primals_88 = mm_default_1 = None
        view_116: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 768]);  add_tensor_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_117: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, -1, 12, 64]);  view_116 = None
        permute_56: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_31: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_90, view_115, permute_57);  primals_90 = permute_57 = None
        view_119: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 768]);  addmm_31 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_120: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_119, [1, -1, 12, 64]);  view_119 = None
        permute_58: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_32: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_92, view_115, permute_59);  primals_92 = permute_59 = None
        view_122: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 768]);  addmm_32 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_123: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_122, [1, -1, 12, 64]);  view_122 = None
        permute_60: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_15: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_56, 8.0);  permute_56 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_61: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_25: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_15, [1, 12, 128, 64])
        view_124: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 128, 64]);  expand_25 = None
        expand_26: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 128])
        view_125: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_26, [12, 64, 128]);  expand_26 = None
        bmm_10: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_5: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_126);  expand_2 = full_default = view_126 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_5: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_16: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_10: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_16: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_11 = self.subgraph__npu_dropout_11

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        _npu_dropout_11 = torch.ops.higher_order.control_deps((_npu_dropout_10,), subgraph__npu_dropout_11, div_16);  _npu_dropout_10 = subgraph__npu_dropout_11 = None
        getitem_44: "f32[1, 12, 128, 128]" = _npu_dropout_11[0]
        getitem_45: "u8[24576]" = _npu_dropout_11[1]

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_28: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_44, [1, 12, 128, 128])
        view_128: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [12, 128, 128]);  expand_28 = None
        expand_29: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64])
        view_129: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_29, [12, 128, 64]);  expand_29 = None
        bmm_11: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_129);  view_128 = view_129 = None
        view_130: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_62: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_5: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_131: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_5, [1, -1, 768]);  clone_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_132: "f32[128, 768]" = torch.ops.aten.reshape.default(view_131, [128, 768]);  view_131 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        mm_default: "f32[128, 768]" = torch.ops.aten.mm.default(view_132, permute_63);  permute_63 = None
        add_tensor: "f32[128, 768]" = torch.ops.aten.add.Tensor(primals_94, mm_default);  primals_94 = mm_default = None
        view_133: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 768]);  add_tensor = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_43: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_42);  view_133 = add_42 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_47: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_47);  add_43 = getitem_47 = None
        mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_95)
        add_45: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_96);  mul_43 = primals_96 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_134: "f32[128, 768]" = torch.ops.aten.reshape.default(add_45, [128, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0])
        addmm_34: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_98, view_134, permute_64);  primals_98 = permute_64 = None
        view_135: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 3072])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_44: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, view_135)
        mul_45: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, view_135);  mul_44 = None
        mul_46: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_45, 0.044715);  mul_45 = None
        add_46: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_135, mul_46);  mul_46 = None
        mul_47: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_46, -1.5957691216057308);  add_46 = None
        exp_11: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_47);  mul_47 = None
        add_47: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_11, 1.0);  exp_11 = None
        div_17: "f32[1, 128, 3072]" = torch.ops.aten.div.Tensor(view_135, add_47);  view_135 = add_47 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_136: "f32[128, 3072]" = torch.ops.aten.reshape.default(div_17, [128, 3072]);  div_17 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_35: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_100, view_136, permute_65);  primals_100 = permute_65 = None
        view_137: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 768]);  addmm_35 = None

        # No stacktrace found for following nodes
        subgraph__npu_dropout_12 = self.subgraph__npu_dropout_12

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        _npu_dropout_12 = torch.ops.higher_order.control_deps((_npu_dropout_11,), subgraph__npu_dropout_12, view_137);  _npu_dropout_11 = subgraph__npu_dropout_12 = view_137 = None
        getitem_48: "f32[1, 128, 768]" = _npu_dropout_12[0]
        getitem_49: "u8[12288]" = _npu_dropout_12[1];  _npu_dropout_12 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_48: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_45);  getitem_48 = add_45 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_51: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_18: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_51);  add_48 = getitem_51 = None
        mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_49: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_101)
        add_50: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_102);  mul_49 = primals_102 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:913 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_138: "f32[128, 768]" = torch.ops.aten.reshape.default(add_50, [128, 768]);  add_50 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        addmm_36: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_104, view_138, permute_66);  primals_104 = permute_66 = None
        view_139: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 768])

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_50: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, view_139)
        mul_51: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, view_139);  mul_50 = None
        mul_52: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, 0.044715);  mul_51 = None
        add_51: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_139, mul_52);  mul_52 = None
        mul_53: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_51, -1.5957691216057308);  add_51 = None
        exp_12: "f32[1, 128, 768]" = torch.ops.aten.exp.default(mul_53);  mul_53 = None
        add_52: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(exp_12, 1.0);  exp_12 = None
        div_18: "f32[1, 128, 768]" = torch.ops.aten.div.Tensor(view_139, add_52);  view_139 = add_52 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:915 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        var_mean_13 = torch.ops.aten.var_mean.correction(div_18, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
        getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_19: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(div_18, getitem_53);  div_18 = None
        mul_54: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_55: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_105);  mul_54 = None
        add_54: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_106);  mul_55 = primals_106 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:916 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_140: "f32[128, 768]" = torch.ops.aten.reshape.default(add_54, [128, 768]);  add_54 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm_37: "f32[128, 30522]" = torch.ops.aten.addmm.default(primals_107, view_140, permute_67);  primals_107 = permute_67 = None
        view_141: "f32[1, 128, 30522]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 30522]);  addmm_37 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_142: "f32[128, 30522]" = torch.ops.aten.reshape.default(view_141, [-1, 30522])
        view_143: "i64[128]" = torch.ops.aten.reshape.default(primals_108, [-1])
        _log_softmax: "f32[128, 30522]" = torch.ops.aten._log_softmax.default(view_142, 1, False);  view_142 = None
        ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)
        full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_6: "i64[128]" = torch.ops.aten.where.self(ne, view_143, full_default_6);  view_143 = full_default_6 = None
        unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "f32[128, 1]" = torch.ops.aten.gather.default(_log_softmax, 1, unsqueeze);  unsqueeze = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_7: "f32[128]" = torch.ops.aten.where.self(ne, neg, full_default_7);  neg = full_default_7 = None
        sum_7: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        sum_8: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        div_19: "f32[]" = torch.ops.aten.div.Tensor(sum_8, convert_element_type);  sum_8 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_22: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_23: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_89: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
        expand_31: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_89, [1, 12, 64, 128]);  permute_89 = None
        view_162: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_31, [12, 64, 128]);  expand_31 = None
        permute_90: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_32: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_90, [1, 12, 128, 128]);  permute_90 = None
        view_164: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_32, [12, 128, 128]);  expand_32 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_91: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_61, [0, 1, 3, 2]);  permute_61 = None
        expand_35: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_91, [1, 12, 128, 64]);  permute_91 = None
        view_168: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 128, 64]);  expand_35 = None
        permute_92: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div_15, [0, 1, 3, 2]);  div_15 = None
        expand_36: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_92, [1, 12, 64, 128]);  permute_92 = None
        view_170: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_36, [12, 64, 128]);  expand_36 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_26: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_122: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
        expand_39: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_122, [1, 12, 64, 128]);  permute_122 = None
        view_196: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_39, [12, 64, 128]);  expand_39 = None
        permute_123: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 1, 3, 2]);  getitem_36 = None
        expand_40: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_123, [1, 12, 128, 128]);  permute_123 = None
        view_198: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_40, [12, 128, 128]);  expand_40 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_124: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_50, [0, 1, 3, 2]);  permute_50 = None
        expand_43: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_124, [1, 12, 128, 64]);  permute_124 = None
        view_202: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 128, 64]);  expand_43 = None
        permute_125: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div_12, [0, 1, 3, 2]);  div_12 = None
        expand_44: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_125, [1, 12, 64, 128]);  permute_125 = None
        view_204: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_44, [12, 64, 128]);  expand_44 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_29: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_155: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        expand_47: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_155, [1, 12, 64, 128]);  permute_155 = None
        view_230: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_47, [12, 64, 128]);  expand_47 = None
        permute_156: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_48: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_156, [1, 12, 128, 128]);  permute_156 = None
        view_232: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_48, [12, 128, 128]);  expand_48 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_157: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_39, [0, 1, 3, 2]);  permute_39 = None
        expand_51: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_157, [1, 12, 128, 64]);  permute_157 = None
        view_236: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_51, [12, 128, 64]);  expand_51 = None
        permute_158: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div_9, [0, 1, 3, 2]);  div_9 = None
        expand_52: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_158, [1, 12, 64, 128]);  permute_158 = None
        view_238: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_52, [12, 64, 128]);  expand_52 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_31: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_32: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_188: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
        expand_55: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_188, [1, 12, 64, 128]);  permute_188 = None
        view_264: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_55, [12, 64, 128]);  expand_55 = None
        permute_189: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_20, [0, 1, 3, 2]);  getitem_20 = None
        expand_56: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_189, [1, 12, 128, 128]);  permute_189 = None
        view_266: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_56, [12, 128, 128]);  expand_56 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_190: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_59: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_190, [1, 12, 128, 64]);  permute_190 = None
        view_270: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_59, [12, 128, 64]);  expand_59 = None
        permute_191: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div_6, [0, 1, 3, 2]);  div_6 = None
        expand_60: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_191, [1, 12, 64, 128]);  permute_191 = None
        view_272: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_60, [12, 64, 128]);  expand_60 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_34: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_35: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_221: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_63: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_221, [1, 12, 64, 128]);  permute_221 = None
        view_298: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_63, [12, 64, 128]);  expand_63 = None
        permute_222: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_64: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_222, [1, 12, 128, 128]);  permute_222 = None
        view_300: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [12, 128, 128]);  expand_64 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_223: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_17, [0, 1, 3, 2]);  permute_17 = None
        expand_67: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_223, [1, 12, 128, 64]);  permute_223 = None
        view_304: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_67, [12, 128, 64]);  expand_67 = None
        permute_224: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div_3, [0, 1, 3, 2]);  div_3 = None
        expand_68: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_224, [1, 12, 64, 128]);  permute_224 = None
        view_306: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_68, [12, 64, 128]);  expand_68 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        div_37: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        div_38: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        permute_254: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        expand_71: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_254, [1, 12, 64, 128]);  permute_254 = None
        view_332: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_71, [12, 64, 128]);  expand_71 = None
        permute_255: "f32[1, 12, 128, 128]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
        expand_72: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(permute_255, [1, 12, 128, 128]);  permute_255 = None
        view_334: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_72, [12, 128, 128]);  expand_72 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_256: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        expand_75: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_256, [1, 12, 128, 64]);  permute_256 = None
        view_338: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_75, [12, 128, 64]);  expand_75 = None
        permute_257: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(div, [0, 1, 3, 2]);  div = None
        expand_76: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_257, [1, 12, 64, 128]);  permute_257 = None
        view_340: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_76, [12, 64, 128]);  expand_76 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        div_40: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_19, view_141, primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_108, mul, getitem_3, view, div_1, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_8, view_23, div_4, getitem_13, view_40, mul_10, view_42, addmm_10, view_44, getitem_17, mul_16, view_46, div_7, getitem_21, view_63, mul_18, view_65, addmm_16, view_67, getitem_25, mul_24, view_69, div_10, getitem_29, view_86, mul_26, view_88, addmm_22, view_90, getitem_33, mul_32, view_92, div_13, getitem_37, view_109, mul_34, view_111, addmm_28, view_113, getitem_41, mul_40, view_115, div_16, getitem_45, view_132, mul_42, view_134, addmm_34, view_136, getitem_49, mul_48, view_138, addmm_36, getitem_53, rsqrt_13, view_140, _log_softmax, convert_element_type, div_22, div_23, view_162, view_164, view_168, view_170, div_25, div_26, view_196, view_198, view_202, view_204, div_28, div_29, view_230, view_232, view_236, view_238, div_31, div_32, view_264, view_266, view_270, view_272, div_34, div_35, view_298, view_300, view_304, view_306, div_37, div_38, view_332, view_334, view_338, view_340, div_40)

    class subgraph__npu_dropout_1(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_2(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_3(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_4(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_5(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_6(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_7(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_8(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_9(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_10(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_11(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 12, 128, 128]"):
            # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default

    class subgraph__npu_dropout_12(torch.nn.Module):
        def forward(self, arg_0: "f32[1, 128, 768]"):
            # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
            _npu_dropout_default = torch.ops.npu._npu_dropout.default(arg_0, 0.1);  arg_0 = None
            return _npu_dropout_default
