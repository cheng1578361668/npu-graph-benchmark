class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[128, 128]", arg1_1: "f32[30522, 768]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[3072, 768]", arg17_1: "f32[3072]", arg18_1: "f32[768, 3072]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768, 768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[3072, 768]", arg33_1: "f32[3072]", arg34_1: "f32[768, 3072]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[3072, 768]", arg49_1: "f32[3072]", arg50_1: "f32[768, 3072]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768, 768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[3072, 768]", arg65_1: "f32[3072]", arg66_1: "f32[768, 3072]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[3072, 768]", arg81_1: "f32[3072]", arg82_1: "f32[768, 3072]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[3072, 768]", arg97_1: "f32[3072]", arg98_1: "f32[768, 3072]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[30522]", arg107_1: "i64[128, 128]"):
        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:141 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:149 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:154 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_1);  arg3_1 = slice_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:156 in forward, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 128, 1]" = var_mean[0]
        getitem_1: "f32[128, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:818 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm, [128, 128, 768]);  addmm = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_2: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [128, -1, 12, 64]);  view_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_3: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_1, [128, 128, 768]);  addmm_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_5: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [128, -1, 12, 64]);  view_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_6: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg11_1, view_6, permute_4);  arg11_1 = view_6 = permute_4 = None
        view_7: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_2, [128, 128, 768]);  addmm_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_8: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [128, -1, 12, 64]);  view_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None

        # No stacktrace found for following nodes
        eq_scalar_5: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1)
        view_default_5: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar_5, [128, 1, 1, 128]);  eq_scalar_5 = None
        expand_default_5: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default_5, [128, 12, 128, 128]);  view_default_5 = None
        permute_default_15: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_default_16: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_default_17: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        logical_not_default_5: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default_5);  expand_default_5 = None
        npu_fusion_attention_v3_default_5 = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_15, permute_default_16, permute_default_17, 12, 'BNSD', None, None, logical_not_default_5, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default_15 = permute_default_16 = permute_default_17 = logical_not_default_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_33: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default_5[0];  npu_fusion_attention_v3_default_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_7: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_33, [0, 2, 1, 3]);  getitem_33 = None
        clone_5: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_16: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_5, [128, -1, 768]);  clone_5 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_17: "f32[16384, 768]" = torch.ops.aten.view.default(view_16, [16384, 768]);  view_16 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg13_1, view_17, permute_8);  arg13_1 = view_17 = permute_8 = None
        view_18: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_3, [128, 128, 768]);  addmm_3 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_18, add_2);  view_18 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[128, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg14_1);  mul_2 = arg14_1 = None
        add_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_19: "f32[16384, 768]" = torch.ops.aten.view.default(add_5, [16384, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg17_1, view_19, permute_9);  arg17_1 = view_19 = permute_9 = None
        view_20: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [128, 128, 3072]);  addmm_4 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_4: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, view_20)
        mul_5: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, view_20);  mul_4 = None
        mul_6: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_5, 0.044715);  mul_5 = None
        add_6: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_20, mul_6);  mul_6 = None
        mul_7: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_6, -1.5957691216057308);  add_6 = None
        exp_1: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_7);  mul_7 = None
        add_7: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_1, 1.0);  exp_1 = None
        div_2: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_20, add_7);  view_20 = add_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_21: "f32[16384, 3072]" = torch.ops.aten.view.default(div_2, [16384, 3072]);  div_2 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg19_1, view_21, permute_10);  arg19_1 = view_21 = permute_10 = None
        view_22: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_5, [128, 128, 768]);  addmm_5 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_8: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_22, add_5);  view_22 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[128, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[128, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
        mul_8: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg20_1);  mul_8 = arg20_1 = None
        add_10: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, arg21_1);  mul_9 = arg21_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_23: "f32[16384, 768]" = torch.ops.aten.view.default(add_10, [16384, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg23_1, view_23, permute_11);  arg23_1 = view_23 = permute_11 = None
        view_24: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_6, [128, 128, 768]);  addmm_6 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_25: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_24, [128, -1, 12, 64]);  view_24 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_26: "f32[16384, 768]" = torch.ops.aten.view.default(add_10, [16384, 768])
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg25_1, view_26, permute_13);  arg25_1 = view_26 = permute_13 = None
        view_27: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_7, [128, 128, 768]);  addmm_7 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_28: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_27, [128, -1, 12, 64]);  view_27 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_29: "f32[16384, 768]" = torch.ops.aten.view.default(add_10, [16384, 768])
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg27_1, view_29, permute_15);  arg27_1 = view_29 = permute_15 = None
        view_30: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_8, [128, 128, 768]);  addmm_8 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_31: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_30, [128, -1, 12, 64]);  view_30 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant1: "f32[]" = self._tensor_constant1;  _tensor_constant1 = None

        # No stacktrace found for following nodes
        eq_scalar_4: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1)
        view_default_4: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar_4, [128, 1, 1, 128]);  eq_scalar_4 = None
        expand_default_4: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default_4, [128, 12, 128, 128]);  view_default_4 = None
        permute_default_12: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        permute_default_13: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_default_14: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        logical_not_default_4: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default_4);  expand_default_4 = None
        npu_fusion_attention_v3_default_4 = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_12, permute_default_13, permute_default_14, 12, 'BNSD', None, None, logical_not_default_4, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default_12 = permute_default_13 = permute_default_14 = logical_not_default_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_32: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default_4[0];  npu_fusion_attention_v3_default_4 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_18: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None
        clone_11: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_11, [128, -1, 768]);  clone_11 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_40: "f32[16384, 768]" = torch.ops.aten.view.default(view_39, [16384, 768]);  view_39 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg29_1, view_40, permute_19);  arg29_1 = view_40 = permute_19 = None
        view_41: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_9, [128, 128, 768]);  addmm_9 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_11: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_10);  view_41 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_6: "f32[128, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[128, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
        mul_10: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg30_1);  mul_10 = arg30_1 = None
        add_13: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, arg31_1);  mul_11 = arg31_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_42: "f32[16384, 768]" = torch.ops.aten.view.default(add_13, [16384, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg33_1, view_42, permute_20);  arg33_1 = view_42 = permute_20 = None
        view_43: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [128, 128, 3072]);  addmm_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_13: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, view_43);  mul_12 = None
        mul_14: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_13, 0.044715);  mul_13 = None
        add_14: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_14);  mul_14 = None
        mul_15: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_14, -1.5957691216057308);  add_14 = None
        exp_3: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_15);  mul_15 = None
        add_15: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_3, 1.0);  exp_3 = None
        div_5: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_43, add_15);  view_43 = add_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_44: "f32[16384, 3072]" = torch.ops.aten.view.default(div_5, [16384, 3072]);  div_5 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg35_1, view_44, permute_21);  arg35_1 = view_44 = permute_21 = None
        view_45: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_11, [128, 128, 768]);  addmm_11 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_16: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_45, add_13);  view_45 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[128, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[128, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_6: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  add_16 = getitem_9 = None
        mul_16: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        add_18: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg37_1);  mul_17 = arg37_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_46: "f32[16384, 768]" = torch.ops.aten.view.default(add_18, [16384, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg39_1, view_46, permute_22);  arg39_1 = view_46 = permute_22 = None
        view_47: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_12, [128, 128, 768]);  addmm_12 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_48: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_47, [128, -1, 12, 64]);  view_47 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_49: "f32[16384, 768]" = torch.ops.aten.view.default(add_18, [16384, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg41_1, view_49, permute_24);  arg41_1 = view_49 = permute_24 = None
        view_50: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_13, [128, 128, 768]);  addmm_13 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_51: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_50, [128, -1, 12, 64]);  view_50 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_52: "f32[16384, 768]" = torch.ops.aten.view.default(add_18, [16384, 768])
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg43_1, view_52, permute_26);  arg43_1 = view_52 = permute_26 = None
        view_53: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_14, [128, 128, 768]);  addmm_14 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_54: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_53, [128, -1, 12, 64]);  view_53 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant2: "f32[]" = self._tensor_constant2;  _tensor_constant2 = None

        # No stacktrace found for following nodes
        eq_scalar_3: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1)
        view_default_3: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar_3, [128, 1, 1, 128]);  eq_scalar_3 = None
        expand_default_3: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default_3, [128, 12, 128, 128]);  view_default_3 = None
        permute_default_9: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_default_10: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        permute_default_11: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        logical_not_default_3: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default_3);  expand_default_3 = None
        npu_fusion_attention_v3_default_3 = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_9, permute_default_10, permute_default_11, 12, 'BNSD', None, None, logical_not_default_3, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default_9 = permute_default_10 = permute_default_11 = logical_not_default_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_31: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default_3[0];  npu_fusion_attention_v3_default_3 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_29: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_31, [0, 2, 1, 3]);  getitem_31 = None
        clone_17: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_62: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_17, [128, -1, 768]);  clone_17 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_63: "f32[16384, 768]" = torch.ops.aten.view.default(view_62, [16384, 768]);  view_62 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg45_1, view_63, permute_30);  arg45_1 = view_63 = permute_30 = None
        view_64: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_15, [128, 128, 768]);  addmm_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_19: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_18);  view_64 = add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_10: "f32[128, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[128, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_11);  add_19 = getitem_11 = None
        mul_18: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_19: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        add_21: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, arg47_1);  mul_19 = arg47_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_65: "f32[16384, 768]" = torch.ops.aten.view.default(add_21, [16384, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg49_1, view_65, permute_31);  arg49_1 = view_65 = permute_31 = None
        view_66: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [128, 128, 3072]);  addmm_16 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_20: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, view_66)
        mul_21: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, view_66);  mul_20 = None
        mul_22: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_21, 0.044715);  mul_21 = None
        add_22: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_66, mul_22);  mul_22 = None
        mul_23: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_22, -1.5957691216057308);  add_22 = None
        exp_5: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        add_23: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_5, 1.0);  exp_5 = None
        div_8: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_66, add_23);  view_66 = add_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_67: "f32[16384, 3072]" = torch.ops.aten.view.default(div_8, [16384, 3072]);  div_8 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg51_1, view_67, permute_32);  arg51_1 = view_67 = permute_32 = None
        view_68: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_17, [128, 128, 768]);  addmm_17 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_24: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_68, add_21);  view_68 = add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[128, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[128, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_9: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  add_24 = getitem_13 = None
        mul_24: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_25: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg52_1);  mul_24 = arg52_1 = None
        add_26: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, arg53_1);  mul_25 = arg53_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_69: "f32[16384, 768]" = torch.ops.aten.view.default(add_26, [16384, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg55_1, view_69, permute_33);  arg55_1 = view_69 = permute_33 = None
        view_70: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_18, [128, 128, 768]);  addmm_18 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_71: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_70, [128, -1, 12, 64]);  view_70 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_72: "f32[16384, 768]" = torch.ops.aten.view.default(add_26, [16384, 768])
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg57_1, view_72, permute_35);  arg57_1 = view_72 = permute_35 = None
        view_73: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_19, [128, 128, 768]);  addmm_19 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_74: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_73, [128, -1, 12, 64]);  view_73 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_75: "f32[16384, 768]" = torch.ops.aten.view.default(add_26, [16384, 768])
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg59_1, view_75, permute_37);  arg59_1 = view_75 = permute_37 = None
        view_76: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_20, [128, 128, 768]);  addmm_20 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_77: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_76, [128, -1, 12, 64]);  view_76 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant3: "f32[]" = self._tensor_constant3;  _tensor_constant3 = None

        # No stacktrace found for following nodes
        eq_scalar_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1)
        view_default_2: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar_2, [128, 1, 1, 128]);  eq_scalar_2 = None
        expand_default_2: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default_2, [128, 12, 128, 128]);  view_default_2 = None
        permute_default_6: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_default_7: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_default_8: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        logical_not_default_2: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default_2);  expand_default_2 = None
        npu_fusion_attention_v3_default_2 = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_6, permute_default_7, permute_default_8, 12, 'BNSD', None, None, logical_not_default_2, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default_6 = permute_default_7 = permute_default_8 = logical_not_default_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_30: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default_2[0];  npu_fusion_attention_v3_default_2 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_40: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3]);  getitem_30 = None
        clone_23: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_23, [128, -1, 768]);  clone_23 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_86: "f32[16384, 768]" = torch.ops.aten.view.default(view_85, [16384, 768]);  view_85 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg61_1, view_86, permute_41);  arg61_1 = view_86 = permute_41 = None
        view_87: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_21, [128, 128, 768]);  addmm_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_27: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_26);  view_87 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[128, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  add_27 = getitem_15 = None
        mul_26: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_27: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg62_1);  mul_26 = arg62_1 = None
        add_29: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, arg63_1);  mul_27 = arg63_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_88: "f32[16384, 768]" = torch.ops.aten.view.default(add_29, [16384, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg65_1, view_88, permute_42);  arg65_1 = view_88 = permute_42 = None
        view_89: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [128, 128, 3072]);  addmm_22 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_28: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, view_89)
        mul_29: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, view_89);  mul_28 = None
        mul_30: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_29, 0.044715);  mul_29 = None
        add_30: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_30);  mul_30 = None
        mul_31: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_30, -1.5957691216057308);  add_30 = None
        exp_7: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        add_31: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_7, 1.0);  exp_7 = None
        div_11: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_89, add_31);  view_89 = add_31 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_90: "f32[16384, 3072]" = torch.ops.aten.view.default(div_11, [16384, 3072]);  div_11 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg67_1, view_90, permute_43);  arg67_1 = view_90 = permute_43 = None
        view_91: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_23, [128, 128, 768]);  addmm_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_32: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_91, add_29);  view_91 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_16: "f32[128, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[128, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_12: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  add_32 = getitem_17 = None
        mul_32: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_33: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg68_1);  mul_32 = arg68_1 = None
        add_34: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, arg69_1);  mul_33 = arg69_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_92: "f32[16384, 768]" = torch.ops.aten.view.default(add_34, [16384, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg71_1, view_92, permute_44);  arg71_1 = view_92 = permute_44 = None
        view_93: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_24, [128, 128, 768]);  addmm_24 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_94: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_93, [128, -1, 12, 64]);  view_93 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_95: "f32[16384, 768]" = torch.ops.aten.view.default(add_34, [16384, 768])
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg73_1, view_95, permute_46);  arg73_1 = view_95 = permute_46 = None
        view_96: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_25, [128, 128, 768]);  addmm_25 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_97: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_96, [128, -1, 12, 64]);  view_96 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_98: "f32[16384, 768]" = torch.ops.aten.view.default(add_34, [16384, 768])
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg75_1, view_98, permute_48);  arg75_1 = view_98 = permute_48 = None
        view_99: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_26, [128, 128, 768]);  addmm_26 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_100: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_99, [128, -1, 12, 64]);  view_99 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant4: "f32[]" = self._tensor_constant4;  _tensor_constant4 = None

        # No stacktrace found for following nodes
        eq_scalar_1: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1)
        view_default_1: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar_1, [128, 1, 1, 128]);  eq_scalar_1 = None
        expand_default_1: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default_1, [128, 12, 128, 128]);  view_default_1 = None
        permute_default_3: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_default_4: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        permute_default_5: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        logical_not_default_1: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default_1);  expand_default_1 = None
        npu_fusion_attention_v3_default_1 = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_3, permute_default_4, permute_default_5, 12, 'BNSD', None, None, logical_not_default_1, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default_3 = permute_default_4 = permute_default_5 = logical_not_default_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_29: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default_1[0];  npu_fusion_attention_v3_default_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_51: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_29, [0, 2, 1, 3]);  getitem_29 = None
        clone_29: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_108: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_29, [128, -1, 768]);  clone_29 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_109: "f32[16384, 768]" = torch.ops.aten.view.default(view_108, [16384, 768]);  view_108 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg77_1, view_109, permute_52);  arg77_1 = view_109 = permute_52 = None
        view_110: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_27, [128, 128, 768]);  addmm_27 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_35: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_34);  view_110 = add_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_18: "f32[128, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[128, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_36: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_19);  add_35 = getitem_19 = None
        mul_34: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_35: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg78_1);  mul_34 = arg78_1 = None
        add_37: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, arg79_1);  mul_35 = arg79_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_111: "f32[16384, 768]" = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg81_1, view_111, permute_53);  arg81_1 = view_111 = permute_53 = None
        view_112: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [128, 128, 3072]);  addmm_28 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_36: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
        mul_37: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, view_112);  mul_36 = None
        mul_38: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_37, 0.044715);  mul_37 = None
        add_38: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_112, mul_38);  mul_38 = None
        mul_39: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_38, -1.5957691216057308);  add_38 = None
        exp_9: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_39);  mul_39 = None
        add_39: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_9, 1.0);  exp_9 = None
        div_14: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_112, add_39);  view_112 = add_39 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_113: "f32[16384, 3072]" = torch.ops.aten.view.default(div_14, [16384, 3072]);  div_14 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg83_1, view_113, permute_54);  arg83_1 = view_113 = permute_54 = None
        view_114: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_29, [128, 128, 768]);  addmm_29 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_40: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_114, add_37);  view_114 = add_37 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[128, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[128, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_15: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  add_40 = getitem_21 = None
        mul_40: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_41: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg84_1);  mul_40 = arg84_1 = None
        add_42: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, arg85_1);  mul_41 = arg85_1 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_115: "f32[16384, 768]" = torch.ops.aten.view.default(add_42, [16384, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg87_1, view_115, permute_55);  arg87_1 = view_115 = permute_55 = None
        view_116: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_30, [128, 128, 768]);  addmm_30 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_117: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_116, [128, -1, 12, 64]);  view_116 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_118: "f32[16384, 768]" = torch.ops.aten.view.default(add_42, [16384, 768])
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg89_1, view_118, permute_57);  arg89_1 = view_118 = permute_57 = None
        view_119: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_31, [128, 128, 768]);  addmm_31 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_120: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_119, [128, -1, 12, 64]);  view_119 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_121: "f32[16384, 768]" = torch.ops.aten.view.default(add_42, [16384, 768])
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg91_1, view_121, permute_59);  arg91_1 = view_121 = permute_59 = None
        view_122: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_32, [128, 128, 768]);  addmm_32 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_123: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_122, [128, -1, 12, 64]);  view_122 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:244 in _hf_distilbert_multiheadselfattention_forward_new, code: mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
        _tensor_constant5: "f32[]" = self._tensor_constant5;  _tensor_constant5 = None

        # No stacktrace found for following nodes
        eq_scalar: "b8[128, 128]" = torch.ops.aten.eq.Scalar(full, 1);  full = None
        view_default: "b8[128, 1, 1, 128]" = torch.ops.aten.view.default(eq_scalar, [128, 1, 1, 128]);  eq_scalar = None
        expand_default: "b8[128, 12, 128, 128]" = torch.ops.aten.expand.default(view_default, [128, 12, 128, 128]);  view_default = None
        permute_default: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        permute_default_1: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        permute_default_2: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        logical_not_default: "b8[128, 12, 128, 128]" = torch.ops.aten.logical_not.default(expand_default);  expand_default = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 12, 'BNSD', None, None, logical_not_default, 0.125, 1.0, 2147483647, 2147483647, 2);  permute_default = permute_default_1 = permute_default_2 = logical_not_default = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        getitem_28: "f32[128, 12, 128, 64]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        permute_62: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
        clone_35: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_131: "f32[128, 128, 768]" = torch.ops.aten.view.default(clone_35, [128, -1, 768]);  clone_35 = None

        # File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_132: "f32[16384, 768]" = torch.ops.aten.view.default(view_131, [16384, 768]);  view_131 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg93_1, view_132, permute_63);  arg93_1 = view_132 = permute_63 = None
        view_133: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_33, [128, 128, 768]);  addmm_33 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_43: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_42);  view_133 = add_42 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_22: "f32[128, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[128, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  add_43 = getitem_23 = None
        mul_42: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_43: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg94_1);  mul_42 = arg94_1 = None
        add_45: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg95_1);  mul_43 = arg95_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_134: "f32[16384, 768]" = torch.ops.aten.view.default(add_45, [16384, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg97_1, view_134, permute_64);  arg97_1 = view_134 = permute_64 = None
        view_135: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [128, 128, 3072]);  addmm_34 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_44: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, view_135)
        mul_45: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, view_135);  mul_44 = None
        mul_46: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_45, 0.044715);  mul_45 = None
        add_46: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(view_135, mul_46);  mul_46 = None
        mul_47: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_46, -1.5957691216057308);  add_46 = None
        exp_11: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_47);  mul_47 = None
        add_47: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(exp_11, 1.0);  exp_11 = None
        div_17: "f32[128, 128, 3072]" = torch.ops.aten.div.Tensor(view_135, add_47);  view_135 = add_47 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_136: "f32[16384, 3072]" = torch.ops.aten.view.default(div_17, [16384, 3072]);  div_17 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg99_1, view_136, permute_65);  arg99_1 = view_136 = permute_65 = None
        view_137: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_35, [128, 128, 768]);  addmm_35 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_48: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_137, add_45);  view_137 = add_45 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_24: "f32[128, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[128, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_49: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_18: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  add_48 = getitem_25 = None
        mul_48: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_49: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg100_1);  mul_48 = arg100_1 = None
        add_50: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, arg101_1);  mul_49 = arg101_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:913 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_138: "f32[16384, 768]" = torch.ops.aten.view.default(add_50, [16384, 768]);  add_50 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg103_1, view_138, permute_66);  arg103_1 = view_138 = permute_66 = None
        view_139: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_36, [128, 128, 768]);  addmm_36 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_50: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, view_139)
        mul_51: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, view_139);  mul_50 = None
        mul_52: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, 0.044715);  mul_51 = None
        add_51: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_139, mul_52);  mul_52 = None
        mul_53: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_51, -1.5957691216057308);  add_51 = None
        exp_12: "f32[128, 128, 768]" = torch.ops.aten.exp.default(mul_53);  mul_53 = None
        add_52: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(exp_12, 1.0);  exp_12 = None
        div_18: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_139, add_52);  view_139 = add_52 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:915 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        var_mean_13 = torch.ops.aten.var_mean.correction(div_18, [2], correction = 0, keepdim = True)
        getitem_26: "f32[128, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[128, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_19: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(div_18, getitem_27);  div_18 = getitem_27 = None
        mul_54: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
        mul_55: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg104_1);  mul_54 = arg104_1 = None
        add_54: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_55, arg105_1);  mul_55 = arg105_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:916 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_140: "f32[16384, 768]" = torch.ops.aten.view.default(add_54, [16384, 768]);  add_54 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm_37: "f32[16384, 30522]" = torch.ops.aten.addmm.default(arg106_1, view_140, permute_67);  arg106_1 = view_140 = permute_67 = None
        view_141: "f32[128, 128, 30522]" = torch.ops.aten.view.default(addmm_37, [128, 128, 30522]);  addmm_37 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_142: "f32[16384, 30522]" = torch.ops.aten.view.default(view_141, [-1, 30522])
        view_143: "i64[16384]" = torch.ops.aten.view.default(arg107_1, [-1]);  arg107_1 = None
        _log_softmax: "f32[16384, 30522]" = torch.ops.aten._log_softmax.default(view_142, 1, False);  view_142 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_143, -100)
        full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_6: "i64[16384]" = torch.ops.aten.where.self(ne, view_143, full_default_6);  ne = full_default_6 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(_log_softmax, 1, unsqueeze);  _log_softmax = unsqueeze = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[16384]" = torch.ops.aten.ne.Scalar(view_143, -100)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_7: "f32[16384]" = torch.ops.aten.where.self(ne_1, neg, full_default_7);  ne_1 = neg = full_default_7 = None
        ne_2: "b8[16384]" = torch.ops.aten.ne.Scalar(view_143, -100);  view_143 = None
        sum_7: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        sum_8: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        div_19: "f32[]" = torch.ops.aten.div.Tensor(sum_8, convert_element_type);  sum_8 = convert_element_type = None
        return (div_19, view_141)
