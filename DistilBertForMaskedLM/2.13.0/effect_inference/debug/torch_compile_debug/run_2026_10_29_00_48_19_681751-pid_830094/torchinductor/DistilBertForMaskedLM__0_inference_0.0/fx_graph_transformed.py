class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[30522, 768]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[3072, 768]", arg17_1: "f32[3072]", arg18_1: "f32[768, 3072]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768, 768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[3072, 768]", arg33_1: "f32[3072]", arg34_1: "f32[768, 3072]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[3072, 768]", arg49_1: "f32[3072]", arg50_1: "f32[768, 3072]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768, 768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[3072, 768]", arg65_1: "f32[3072]", arg66_1: "f32[768, 3072]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[3072, 768]", arg81_1: "f32[3072]", arg82_1: "f32[768, 3072]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[3072, 768]", arg97_1: "f32[3072]", arg98_1: "f32[768, 3072]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[30522]", arg107_1: "i64[1, 128]"):
        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:141 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:149 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:154 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_1);  arg3_1 = slice_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:156 in forward, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:818 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
        view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq, [1, 1, 1, 128]);  eq = None
        expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view: "f32[128, 768]" = torch.ops.aten.reshape.default(add_2, [128, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_default_24: "f32[128, 768]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        add_tensor_24: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg7_1, mm_default_24);  arg7_1 = mm_default_24 = None
        view_1: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 128, 768]);  add_tensor_24 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_2: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_1, 8.0);  permute_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div, [1, 12, 128, 64]);  div = None
        view_9: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand, [12, 128, 64]);  expand = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_3: "f32[128, 768]" = torch.ops.aten.reshape.default(add_2, [128, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f32[128, 768]" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 768]);  addmm_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_5: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_6: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 128]);  permute_6 = None
        view_10: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 128]);  expand_1 = None
        bmm: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 12, 128, 128]);  bmm = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_11);  expand_2 = full_default = view_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        sub_1: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_3: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_1, [1, 12, 128, 128]);  div_1 = None
        view_13: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [12, 128, 128]);  expand_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_6: "f32[128, 768]" = torch.ops.aten.reshape.default(add_2, [128, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "f32[128, 768]" = torch.ops.aten.addmm.default(arg11_1, view_6, permute_4);  arg11_1 = view_6 = permute_4 = None
        view_7: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 768]);  addmm_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_8: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_4: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64]);  permute_5 = None
        view_14: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 128, 64]);  expand_4 = None
        bmm_1: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_7: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone_2: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_2, [1, -1, 768]);  clone_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_17: "f32[128, 768]" = torch.ops.aten.reshape.default(view_16, [128, 768]);  view_16 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_default_23: "f32[128, 768]" = torch.ops.aten.mm.default(view_17, permute_8);  view_17 = permute_8 = None
        add_tensor_23: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg13_1, mm_default_23);  arg13_1 = mm_default_23 = None
        view_18: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 128, 768]);  add_tensor_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_18, add_2);  view_18 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg14_1);  mul_2 = arg14_1 = None
        add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_19: "f32[128, 768]" = torch.ops.aten.reshape.default(add_5, [128, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_default_22: "f32[128, 3072]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        add_tensor_22: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg17_1, mm_default_22);  arg17_1 = mm_default_22 = None
        view_20: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 128, 3072]);  add_tensor_22 = None

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
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_default_21: "f32[128, 768]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        add_tensor_21: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg19_1, mm_default_21);  arg19_1 = mm_default_21 = None
        view_22: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 128, 768]);  add_tensor_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_8: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_22, add_5);  view_22 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
        view_35: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq_1, [1, 1, 1, 128]);  eq_1 = None
        expand_7: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_35, [1, 12, 128, 128]);  view_35 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
        add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg20_1);  mul_8 = arg20_1 = None
        add_10: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, arg21_1);  mul_9 = arg21_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_23: "f32[128, 768]" = torch.ops.aten.reshape.default(add_10, [128, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_default_20: "f32[128, 768]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        add_tensor_20: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg23_1, mm_default_20);  arg23_1 = mm_default_20 = None
        view_24: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 128, 768]);  add_tensor_20 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_25: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_24, [1, -1, 12, 64]);  view_24 = None
        permute_12: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_3: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_12, 8.0);  permute_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_5: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_3, [1, 12, 128, 64]);  div_3 = None
        view_32: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 128, 64]);  expand_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_26: "f32[128, 768]" = torch.ops.aten.reshape.default(add_10, [128, 768])
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "f32[128, 768]" = torch.ops.aten.addmm.default(arg25_1, view_26, permute_13);  arg25_1 = view_26 = permute_13 = None
        view_27: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 768]);  addmm_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_28: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 12, 64]);  view_27 = None
        permute_14: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_17: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_6: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 128]);  permute_17 = None
        view_33: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_6, [12, 64, 128]);  expand_6 = None
        bmm_2: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
        view_34: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_1: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_7, full_default_1, view_34);  expand_7 = full_default_1 = view_34 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_1: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_4: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_2: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_8: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_4, [1, 12, 128, 128]);  div_4 = None
        view_36: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [12, 128, 128]);  expand_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_29: "f32[128, 768]" = torch.ops.aten.reshape.default(add_10, [128, 768])
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "f32[128, 768]" = torch.ops.aten.addmm.default(arg27_1, view_29, permute_15);  arg27_1 = view_29 = permute_15 = None
        view_30: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 768]);  addmm_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_31: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_30, [1, -1, 12, 64]);  view_30 = None
        permute_16: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_9: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64]);  permute_16 = None
        view_37: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_9, [12, 128, 64]);  expand_9 = None
        bmm_3: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_18: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_5: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_5, [1, -1, 768]);  clone_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_40: "f32[128, 768]" = torch.ops.aten.reshape.default(view_39, [128, 768]);  view_39 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_default_19: "f32[128, 768]" = torch.ops.aten.mm.default(view_40, permute_19);  view_40 = permute_19 = None
        add_tensor_19: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg29_1, mm_default_19);  arg29_1 = mm_default_19 = None
        view_41: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 128, 768]);  add_tensor_19 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_11: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_10);  view_41 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
        add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        add_13: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, arg31_1);  mul_11 = arg31_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_42: "f32[128, 768]" = torch.ops.aten.reshape.default(add_13, [128, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_default_18: "f32[128, 3072]" = torch.ops.aten.mm.default(view_42, permute_20);  view_42 = permute_20 = None
        add_tensor_18: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg33_1, mm_default_18);  arg33_1 = mm_default_18 = None
        view_43: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 128, 3072]);  add_tensor_18 = None

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
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_default_17: "f32[128, 768]" = torch.ops.aten.mm.default(view_44, permute_21);  view_44 = permute_21 = None
        add_tensor_17: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg35_1, mm_default_17);  arg35_1 = mm_default_17 = None
        view_45: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 128, 768]);  add_tensor_17 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_45, add_13);  view_45 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq_2: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
        view_58: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq_2, [1, 1, 1, 128]);  eq_2 = None
        expand_12: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_58, [1, 12, 128, 128]);  view_58 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  add_16 = getitem_9 = None
        add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        add_18: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg37_1);  mul_17 = arg37_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_46: "f32[128, 768]" = torch.ops.aten.reshape.default(add_18, [128, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_default_16: "f32[128, 768]" = torch.ops.aten.mm.default(view_46, permute_22);  view_46 = permute_22 = None
        add_tensor_16: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg39_1, mm_default_16);  arg39_1 = mm_default_16 = None
        view_47: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 128, 768]);  add_tensor_16 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_48: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, -1, 12, 64]);  view_47 = None
        permute_23: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_6: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_23, 8.0);  permute_23 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_10: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_6, [1, 12, 128, 64]);  div_6 = None
        view_55: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 128, 64]);  expand_10 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_49: "f32[128, 768]" = torch.ops.aten.reshape.default(add_18, [128, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "f32[128, 768]" = torch.ops.aten.addmm.default(arg41_1, view_49, permute_24);  arg41_1 = view_49 = permute_24 = None
        view_50: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 768]);  addmm_13 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_51: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, -1, 12, 64]);  view_50 = None
        permute_25: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_28: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_11: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 128]);  permute_28 = None
        view_56: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_11, [12, 64, 128]);  expand_11 = None
        bmm_4: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_2: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_12, full_default_2, view_57);  expand_12 = full_default_2 = view_57 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_2: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_4: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_7: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_13: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_7, [1, 12, 128, 128]);  div_7 = None
        view_59: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_13, [12, 128, 128]);  expand_13 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_52: "f32[128, 768]" = torch.ops.aten.reshape.default(add_18, [128, 768])
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "f32[128, 768]" = torch.ops.aten.addmm.default(arg43_1, view_52, permute_26);  arg43_1 = view_52 = permute_26 = None
        view_53: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 768]);  addmm_14 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_54: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_53, [1, -1, 12, 64]);  view_53 = None
        permute_27: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_14: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64]);  permute_27 = None
        view_60: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 128, 64]);  expand_14 = None
        bmm_5: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = view_60 = None
        view_61: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_29: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_8: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_62: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_8, [1, -1, 768]);  clone_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_63: "f32[128, 768]" = torch.ops.aten.reshape.default(view_62, [128, 768]);  view_62 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_default_15: "f32[128, 768]" = torch.ops.aten.mm.default(view_63, permute_30);  view_63 = permute_30 = None
        add_tensor_15: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg45_1, mm_default_15);  arg45_1 = mm_default_15 = None
        view_64: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 128, 768]);  add_tensor_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_18);  view_64 = add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_11);  add_19 = getitem_11 = None
        add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_19: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        add_21: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, arg47_1);  mul_19 = arg47_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_65: "f32[128, 768]" = torch.ops.aten.reshape.default(add_21, [128, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_default_14: "f32[128, 3072]" = torch.ops.aten.mm.default(view_65, permute_31);  view_65 = permute_31 = None
        add_tensor_14: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg49_1, mm_default_14);  arg49_1 = mm_default_14 = None
        view_66: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 128, 3072]);  add_tensor_14 = None

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
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_default_13: "f32[128, 768]" = torch.ops.aten.mm.default(view_67, permute_32);  view_67 = permute_32 = None
        add_tensor_13: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg51_1, mm_default_13);  arg51_1 = mm_default_13 = None
        view_68: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 128, 768]);  add_tensor_13 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_24: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_68, add_21);  view_68 = add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq_3: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
        view_81: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq_3, [1, 1, 1, 128]);  eq_3 = None
        expand_17: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_81, [1, 12, 128, 128]);  view_81 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  add_24 = getitem_13 = None
        add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_25: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg52_1);  mul_24 = arg52_1 = None
        add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, arg53_1);  mul_25 = arg53_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_69: "f32[128, 768]" = torch.ops.aten.reshape.default(add_26, [128, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_default_12: "f32[128, 768]" = torch.ops.aten.mm.default(view_69, permute_33);  view_69 = permute_33 = None
        add_tensor_12: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg55_1, mm_default_12);  arg55_1 = mm_default_12 = None
        view_70: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 128, 768]);  add_tensor_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_71: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_70, [1, -1, 12, 64]);  view_70 = None
        permute_34: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_9: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_34, 8.0);  permute_34 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_15: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_9, [1, 12, 128, 64]);  div_9 = None
        view_78: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 128, 64]);  expand_15 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_72: "f32[128, 768]" = torch.ops.aten.reshape.default(add_26, [128, 768])
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "f32[128, 768]" = torch.ops.aten.addmm.default(arg57_1, view_72, permute_35);  arg57_1 = view_72 = permute_35 = None
        view_73: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 768]);  addmm_19 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_74: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_73, [1, -1, 12, 64]);  view_73 = None
        permute_36: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_39: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_16: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 128]);  permute_39 = None
        view_79: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 128]);  expand_16 = None
        bmm_6: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_3: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_17, full_default_3, view_80);  expand_17 = full_default_3 = view_80 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_3: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_10: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_6: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_18: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_10, [1, 12, 128, 128]);  div_10 = None
        view_82: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [12, 128, 128]);  expand_18 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_75: "f32[128, 768]" = torch.ops.aten.reshape.default(add_26, [128, 768])
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "f32[128, 768]" = torch.ops.aten.addmm.default(arg59_1, view_75, permute_37);  arg59_1 = view_75 = permute_37 = None
        view_76: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 768]);  addmm_20 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_77: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_76, [1, -1, 12, 64]);  view_76 = None
        permute_38: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_19: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64]);  permute_38 = None
        view_83: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 128, 64]);  expand_19 = None
        bmm_7: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_40: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_11: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_11, [1, -1, 768]);  clone_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_86: "f32[128, 768]" = torch.ops.aten.reshape.default(view_85, [128, 768]);  view_85 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_default_11: "f32[128, 768]" = torch.ops.aten.mm.default(view_86, permute_41);  view_86 = permute_41 = None
        add_tensor_11: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg61_1, mm_default_11);  arg61_1 = mm_default_11 = None
        view_87: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 128, 768]);  add_tensor_11 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_27: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_26);  view_87 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  add_27 = getitem_15 = None
        add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_26: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_27: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg62_1);  mul_26 = arg62_1 = None
        add_29: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, arg63_1);  mul_27 = arg63_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_88: "f32[128, 768]" = torch.ops.aten.reshape.default(add_29, [128, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_default_10: "f32[128, 3072]" = torch.ops.aten.mm.default(view_88, permute_42);  view_88 = permute_42 = None
        add_tensor_10: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg65_1, mm_default_10);  arg65_1 = mm_default_10 = None
        view_89: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 128, 3072]);  add_tensor_10 = None

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
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_default_9: "f32[128, 768]" = torch.ops.aten.mm.default(view_90, permute_43);  view_90 = permute_43 = None
        add_tensor_9: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg67_1, mm_default_9);  arg67_1 = mm_default_9 = None
        view_91: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 128, 768]);  add_tensor_9 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_32: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_91, add_29);  view_91 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq_4: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
        view_104: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq_4, [1, 1, 1, 128]);  eq_4 = None
        expand_22: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_104, [1, 12, 128, 128]);  view_104 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  add_32 = getitem_17 = None
        add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_32: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_33: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg68_1);  mul_32 = arg68_1 = None
        add_34: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, arg69_1);  mul_33 = arg69_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_92: "f32[128, 768]" = torch.ops.aten.reshape.default(add_34, [128, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_default_8: "f32[128, 768]" = torch.ops.aten.mm.default(view_92, permute_44);  view_92 = permute_44 = None
        add_tensor_8: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg71_1, mm_default_8);  arg71_1 = mm_default_8 = None
        view_93: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 128, 768]);  add_tensor_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_94: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 12, 64]);  view_93 = None
        permute_45: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_12: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_20: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_12, [1, 12, 128, 64]);  div_12 = None
        view_101: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 128, 64]);  expand_20 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_95: "f32[128, 768]" = torch.ops.aten.reshape.default(add_34, [128, 768])
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "f32[128, 768]" = torch.ops.aten.addmm.default(arg73_1, view_95, permute_46);  arg73_1 = view_95 = permute_46 = None
        view_96: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 768]);  addmm_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_97: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_96, [1, -1, 12, 64]);  view_96 = None
        permute_47: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_50: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_21: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 128]);  permute_50 = None
        view_102: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 128]);  expand_21 = None
        bmm_8: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_4: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_22, full_default_4, view_103);  expand_22 = full_default_4 = view_103 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_4: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_13: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_8: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_13: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_23: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_13, [1, 12, 128, 128]);  div_13 = None
        view_105: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [12, 128, 128]);  expand_23 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_98: "f32[128, 768]" = torch.ops.aten.reshape.default(add_34, [128, 768])
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "f32[128, 768]" = torch.ops.aten.addmm.default(arg75_1, view_98, permute_48);  arg75_1 = view_98 = permute_48 = None
        view_99: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 768]);  addmm_26 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_100: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_99, [1, -1, 12, 64]);  view_99 = None
        permute_49: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_24: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64]);  permute_49 = None
        view_106: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 128, 64]);  expand_24 = None
        bmm_9: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_105, view_106);  view_105 = view_106 = None
        view_107: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_51: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        clone_14: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_108: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_14, [1, -1, 768]);  clone_14 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_109: "f32[128, 768]" = torch.ops.aten.reshape.default(view_108, [128, 768]);  view_108 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_default_7: "f32[128, 768]" = torch.ops.aten.mm.default(view_109, permute_52);  view_109 = permute_52 = None
        add_tensor_7: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg77_1, mm_default_7);  arg77_1 = mm_default_7 = None
        view_110: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 128, 768]);  add_tensor_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_35: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_34);  view_110 = add_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_19);  add_35 = getitem_19 = None
        add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_34: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg78_1);  mul_34 = arg78_1 = None
        add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, arg79_1);  mul_35 = arg79_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_111: "f32[128, 768]" = torch.ops.aten.reshape.default(add_37, [128, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_default_6: "f32[128, 3072]" = torch.ops.aten.mm.default(view_111, permute_53);  view_111 = permute_53 = None
        add_tensor_6: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg81_1, mm_default_6);  arg81_1 = mm_default_6 = None
        view_112: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 128, 3072]);  add_tensor_6 = None

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
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_default_5: "f32[128, 768]" = torch.ops.aten.mm.default(view_113, permute_54);  view_113 = permute_54 = None
        add_tensor_5: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg83_1, mm_default_5);  arg83_1 = mm_default_5 = None
        view_114: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 768]);  add_tensor_5 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_114, add_37);  view_114 = add_37 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq_5: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
        view_127: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq_5, [1, 1, 1, 128]);  eq_5 = None
        expand_27: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_127, [1, 12, 128, 128]);  view_127 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        sub_15: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  add_40 = getitem_21 = None
        add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_40: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_41: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg84_1);  mul_40 = arg84_1 = None
        add_42: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, arg85_1);  mul_41 = arg85_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_115: "f32[128, 768]" = torch.ops.aten.reshape.default(add_42, [128, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_default_4: "f32[128, 768]" = torch.ops.aten.mm.default(view_115, permute_55);  view_115 = permute_55 = None
        add_tensor_4: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg87_1, mm_default_4);  arg87_1 = mm_default_4 = None
        view_116: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 768]);  add_tensor_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_117: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, -1, 12, 64]);  view_116 = None
        permute_56: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_15: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_56, 8.0);  permute_56 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_25: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_15, [1, 12, 128, 64]);  div_15 = None
        view_124: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 128, 64]);  expand_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_118: "f32[128, 768]" = torch.ops.aten.reshape.default(add_42, [128, 768])
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "f32[128, 768]" = torch.ops.aten.addmm.default(arg89_1, view_118, permute_57);  arg89_1 = view_118 = permute_57 = None
        view_119: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 768]);  addmm_31 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_120: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_119, [1, -1, 12, 64]);  view_119 = None
        permute_58: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        permute_61: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_26: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 128]);  permute_61 = None
        view_125: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_26, [12, 64, 128]);  expand_26 = None
        bmm_10: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_5: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_27, full_default_5, view_126);  expand_27 = full_default_5 = view_126 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        amax_5: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_16: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_10: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_16: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_28: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(div_16, [1, 12, 128, 128]);  div_16 = None
        view_128: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [12, 128, 128]);  expand_28 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_121: "f32[128, 768]" = torch.ops.aten.reshape.default(add_42, [128, 768])
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "f32[128, 768]" = torch.ops.aten.addmm.default(arg91_1, view_121, permute_59);  arg91_1 = view_121 = permute_59 = None
        view_122: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 768]);  addmm_32 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_123: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_122, [1, -1, 12, 64]);  view_122 = None
        permute_60: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_29: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64]);  permute_60 = None
        view_129: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_29, [12, 128, 64]);  expand_29 = None
        bmm_11: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_129);  view_128 = view_129 = None
        view_130: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_62: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_17: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_131: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_17, [1, -1, 768]);  clone_17 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_132: "f32[128, 768]" = torch.ops.aten.reshape.default(view_131, [128, 768]);  view_131 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_default_3: "f32[128, 768]" = torch.ops.aten.mm.default(view_132, permute_63);  view_132 = permute_63 = None
        add_tensor_3: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg93_1, mm_default_3);  arg93_1 = mm_default_3 = None
        view_133: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 768]);  add_tensor_3 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_43: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_42);  view_133 = add_42 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  add_43 = getitem_23 = None
        add_44: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg94_1);  mul_42 = arg94_1 = None
        add_45: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg95_1);  mul_43 = arg95_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_134: "f32[128, 768]" = torch.ops.aten.reshape.default(add_45, [128, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_default_2: "f32[128, 3072]" = torch.ops.aten.mm.default(view_134, permute_64);  view_134 = permute_64 = None
        add_tensor_2: "f32[128, 3072]" = torch.ops.aten.add.Tensor(arg97_1, mm_default_2);  arg97_1 = mm_default_2 = None
        view_135: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 3072]);  add_tensor_2 = None

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
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_default_1: "f32[128, 768]" = torch.ops.aten.mm.default(view_136, permute_65);  view_136 = permute_65 = None
        add_tensor_1: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg99_1, mm_default_1);  arg99_1 = mm_default_1 = None
        view_137: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 768]);  add_tensor_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_48: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_137, add_45);  view_137 = add_45 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  add_48 = getitem_25 = None
        add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_49: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg100_1);  mul_48 = arg100_1 = None
        add_50: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, arg101_1);  mul_49 = arg101_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:913 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_138: "f32[128, 768]" = torch.ops.aten.reshape.default(add_50, [128, 768]);  add_50 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        mm_default: "f32[128, 768]" = torch.ops.aten.mm.default(view_138, permute_66);  view_138 = permute_66 = None
        add_tensor: "f32[128, 768]" = torch.ops.aten.add.Tensor(arg103_1, mm_default);  arg103_1 = mm_default = None
        view_139: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 768]);  add_tensor = None

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
        getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_143: "i64[128]" = torch.ops.aten.reshape.default(arg107_1, [-1]);  arg107_1 = None
        ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:915 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_19: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(div_18, getitem_27);  div_18 = getitem_27 = None
        add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_54: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
        mul_55: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg104_1);  mul_54 = arg104_1 = None
        add_54: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_55, arg105_1);  mul_55 = arg105_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:916 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_140: "f32[128, 768]" = torch.ops.aten.reshape.default(add_54, [128, 768]);  add_54 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm_37: "f32[128, 30522]" = torch.ops.aten.addmm.default(arg106_1, view_140, permute_67);  arg106_1 = view_140 = permute_67 = None
        view_141: "f32[1, 128, 30522]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 30522]);  addmm_37 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_142: "f32[128, 30522]" = torch.ops.aten.reshape.default(view_141, [-1, 30522])
        _log_softmax: "f32[128, 30522]" = torch.ops.aten._log_softmax.default(view_142, 1, False);  view_142 = None
        ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)
        full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_6: "i64[128]" = torch.ops.aten.where.self(ne, view_143, full_default_6);  ne = full_default_6 = None
        unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "f32[128, 1]" = torch.ops.aten.gather.default(_log_softmax, 1, unsqueeze);  _log_softmax = unsqueeze = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_7: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, full_default_7);  ne_1 = neg = full_default_7 = None
        sum_8: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100);  view_143 = None
        sum_7: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        div_19: "f32[]" = torch.ops.aten.div.Tensor(sum_8, convert_element_type);  sum_8 = convert_element_type = None
        return (div_19, view_141)
