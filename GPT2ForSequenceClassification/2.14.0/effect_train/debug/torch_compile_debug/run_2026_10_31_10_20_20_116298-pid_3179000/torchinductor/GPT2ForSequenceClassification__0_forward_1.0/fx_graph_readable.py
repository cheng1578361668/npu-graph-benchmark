class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 1024]", primals_2: "f32[50257, 768]", primals_3: "f32[1024, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[2304]", primals_7: "f32[768, 2304]", primals_8: "b8[1, 1, 1024, 1024]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[3072]", primals_14: "f32[768, 3072]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[2304]", primals_20: "f32[768, 2304]", primals_21: "b8[1, 1, 1024, 1024]", primals_22: "f32[768]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[3072]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[3072, 768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[2304]", primals_33: "f32[768, 2304]", primals_34: "b8[1, 1, 1024, 1024]", primals_35: "f32[768]", primals_36: "f32[768, 768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[3072]", primals_40: "f32[768, 3072]", primals_41: "f32[768]", primals_42: "f32[3072, 768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[2304]", primals_46: "f32[768, 2304]", primals_47: "b8[1, 1, 1024, 1024]", primals_48: "f32[768]", primals_49: "f32[768, 768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[3072]", primals_53: "f32[768, 3072]", primals_54: "f32[768]", primals_55: "f32[3072, 768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[2304]", primals_59: "f32[768, 2304]", primals_60: "b8[1, 1, 1024, 1024]", primals_61: "f32[768]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[3072]", primals_66: "f32[768, 3072]", primals_67: "f32[768]", primals_68: "f32[3072, 768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[2304]", primals_72: "f32[768, 2304]", primals_73: "b8[1, 1, 1024, 1024]", primals_74: "f32[768]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[3072]", primals_79: "f32[768, 3072]", primals_80: "f32[768]", primals_81: "f32[3072, 768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[2304]", primals_85: "f32[768, 2304]", primals_86: "b8[1, 1, 1024, 1024]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[3072]", primals_92: "f32[768, 3072]", primals_93: "f32[768]", primals_94: "f32[3072, 768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[2304]", primals_98: "f32[768, 2304]", primals_99: "b8[1, 1, 1024, 1024]", primals_100: "f32[768]", primals_101: "f32[768, 768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[3072]", primals_105: "f32[768, 3072]", primals_106: "f32[768]", primals_107: "f32[3072, 768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[2304]", primals_111: "f32[768, 2304]", primals_112: "b8[1, 1, 1024, 1024]", primals_113: "f32[768]", primals_114: "f32[768, 768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[3072]", primals_118: "f32[768, 3072]", primals_119: "f32[768]", primals_120: "f32[3072, 768]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[2304]", primals_124: "f32[768, 2304]", primals_125: "b8[1, 1, 1024, 1024]", primals_126: "f32[768]", primals_127: "f32[768, 768]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[3072]", primals_131: "f32[768, 3072]", primals_132: "f32[768]", primals_133: "f32[3072, 768]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[2304]", primals_137: "f32[768, 2304]", primals_138: "b8[1, 1, 1024, 1024]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[3072]", primals_144: "f32[768, 3072]", primals_145: "f32[768]", primals_146: "f32[3072, 768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[2304]", primals_150: "f32[768, 2304]", primals_151: "b8[1, 1, 1024, 1024]", primals_152: "f32[768]", primals_153: "f32[768, 768]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[3072]", primals_157: "f32[768, 3072]", primals_158: "f32[768]", primals_159: "f32[3072, 768]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[2, 768]", primals_163: "i64[4]"):
        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:796 in forward, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:797 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:837 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[4, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:838 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:839 in forward, code: hidden_states = inputs_embeds + position_embeds
        add: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:845 in forward, code: hidden_states = self.drop(hidden_states)
        _npu_dropout = torch.ops.npu._npu_dropout.default(add, 0.1);  add = None
        getitem: "f32[4, 1024, 768]" = _npu_dropout[0]
        getitem_1: "u8[393216]" = _npu_dropout[1];  _npu_dropout = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 1024, 1]" = var_mean[0]
        getitem_3: "f32[4, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem_3 = None
        mul: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, primals_4)
        add_2: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[4096, 768]" = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
        addmm: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_6, view_1, primals_7);  primals_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_2: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm, [4, 1024, 2304]);  addmm = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_4: "f32[4, 1024, 768]" = split[0]
        getitem_5: "f32[4, 1024, 768]" = split[1]
        getitem_6: "f32[4, 1024, 768]" = split[2];  split = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_3: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_4, [4, 1024, 12, 64]);  getitem_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_4: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_5, [4, 1024, 12, 64]);  getitem_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_5: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_6, [4, 1024, 12, 64]);  getitem_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_2: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_3: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
        expand: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute, [4, 12, 1024, 64])
        clone: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone, [48, 1024, 64]);  clone = None
        expand_1: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_3, [4, 12, 64, 1024])
        clone_1: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_1, [48, 64, 1024]);  clone_1 = None
        bmm: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm, [4, 12, 1024, 1024]);  bmm = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        div: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_8, full_default);  view_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:201 in _attn, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_8, div, full_default_1);  div = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where, -1, False);  where = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_1 = torch.ops.npu._npu_dropout.default(_softmax, 0.1)
        getitem_7: "f32[4, 12, 1024, 1024]" = _npu_dropout_1[0]
        getitem_8: "u8[6291456]" = _npu_dropout_1[1];  _npu_dropout_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_2: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_7, [4, 12, 1024, 1024])
        view_9: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [48, 1024, 1024]);  expand_2 = None
        expand_3: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_2, [4, 12, 1024, 64])
        clone_2: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_10: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_2, [48, 1024, 64]);  clone_2 = None
        bmm_1: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 1024, 64]);  bmm_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_4: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_3: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_12: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_3, [4, 1024, 768]);  clone_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[4096, 768]" = torch.ops.aten.view.default(view_12, [-1, 768]);  view_12 = None
        addmm_1: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_9, view_13, primals_10);  primals_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_14: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_1, [4, 1024, 768]);  addmm_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_2 = torch.ops.npu._npu_dropout.default(view_14, 0.1);  view_14 = None
        getitem_9: "f32[4, 1024, 768]" = _npu_dropout_2[0]
        getitem_10: "u8[393216]" = _npu_dropout_2[1];  _npu_dropout_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_3: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_9, getitem);  getitem_9 = getitem = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_11: "f32[4, 1024, 1]" = var_mean_1[0]
        getitem_12: "f32[4, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12);  getitem_12 = None
        mul_2: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_11)
        add_5: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_15: "f32[4096, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
        addmm_2: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_13, view_15, primals_14);  primals_13 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_16: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
        pow_1: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_16, 3.0)
        mul_5: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_16, mul_5);  view_16 = mul_5 = None
        mul_6: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_7: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_17: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_15, view_17, primals_16);  primals_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_18: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [4, 1024, 768]);  addmm_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_3 = torch.ops.npu._npu_dropout.default(view_18, 0.1);  view_18 = None
        getitem_13: "f32[4, 1024, 768]" = _npu_dropout_3[0]
        getitem_14: "u8[393216]" = _npu_dropout_3[1];  _npu_dropout_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_8: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_13);  add_3 = getitem_13 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_15: "f32[4, 1024, 1]" = var_mean_2[0]
        getitem_16: "f32[4, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
        rsqrt_2: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16);  getitem_16 = None
        mul_8: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_9: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_17)
        add_10: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_18);  mul_9 = primals_18 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[4096, 768]" = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
        addmm_4: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_19, view_19, primals_20);  primals_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_20: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_4, [4, 1024, 2304]);  addmm_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_20, 768, 2);  view_20 = None
        getitem_17: "f32[4, 1024, 768]" = split_1[0]
        getitem_18: "f32[4, 1024, 768]" = split_1[1]
        getitem_19: "f32[4, 1024, 768]" = split_1[2];  split_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_21: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_17, [4, 1024, 12, 64]);  getitem_17 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_5: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_22: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_18, [4, 1024, 12, 64]);  getitem_18 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_6: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_23: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_19, [4, 1024, 12, 64]);  getitem_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_7: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_8: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
        expand_4: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [4, 12, 1024, 64])
        clone_4: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_24: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_4, [48, 1024, 64]);  clone_4 = None
        expand_5: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_8, [4, 12, 64, 1024])
        clone_5: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_25: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_5, [48, 64, 1024]);  clone_5 = None
        bmm_2: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
        view_26: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [4, 12, 1024, 1024]);  bmm_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_1: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_26, full_default);  view_26 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_1: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_21, div_1, full_default_1);  div_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_1: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_1, -1, False);  where_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_4 = torch.ops.npu._npu_dropout.default(_softmax_1, 0.1)
        getitem_20: "f32[4, 12, 1024, 1024]" = _npu_dropout_4[0]
        getitem_21: "u8[6291456]" = _npu_dropout_4[1];  _npu_dropout_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_6: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_20, [4, 12, 1024, 1024])
        view_27: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [48, 1024, 1024]);  expand_6 = None
        expand_7: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_7, [4, 12, 1024, 64])
        clone_6: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_28: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_6, [48, 1024, 64]);  clone_6 = None
        bmm_3: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_27, view_28);  view_27 = view_28 = None
        view_29: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 1024, 64]);  bmm_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_9: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_7: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_30: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_7, [4, 1024, 768]);  clone_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[4096, 768]" = torch.ops.aten.view.default(view_30, [-1, 768]);  view_30 = None
        addmm_5: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_22, view_31, primals_23);  primals_22 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_32: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [4, 1024, 768]);  addmm_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_5 = torch.ops.npu._npu_dropout.default(view_32, 0.1);  view_32 = None
        getitem_22: "f32[4, 1024, 768]" = _npu_dropout_5[0]
        getitem_23: "u8[393216]" = _npu_dropout_5[1];  _npu_dropout_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_11: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_8);  getitem_22 = add_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 1024, 1]" = var_mean_3[0]
        getitem_25: "f32[4, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25);  getitem_25 = None
        mul_10: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_11: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_24)
        add_13: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_25);  mul_11 = primals_25 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[4096, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
        addmm_6: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_26, view_33, primals_27);  primals_26 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_34: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_2: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_13: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_13);  view_34 = mul_13 = None
        mul_14: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_15: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_28, view_35, primals_29);  primals_28 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_36: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_7, [4, 1024, 768]);  addmm_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_6 = torch.ops.npu._npu_dropout.default(view_36, 0.1);  view_36 = None
        getitem_26: "f32[4, 1024, 768]" = _npu_dropout_6[0]
        getitem_27: "u8[393216]" = _npu_dropout_6[1];  _npu_dropout_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_16: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, getitem_26);  add_11 = getitem_26 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_28: "f32[4, 1024, 1]" = var_mean_4[0]
        getitem_29: "f32[4, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_4: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_4: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  getitem_29 = None
        mul_16: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_17: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_30)
        add_18: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_31);  mul_17 = primals_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[4096, 768]" = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
        addmm_8: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_32, view_37, primals_33);  primals_32 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_38: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_8, [4, 1024, 2304]);  addmm_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_30: "f32[4, 1024, 768]" = split_2[0]
        getitem_31: "f32[4, 1024, 768]" = split_2[1]
        getitem_32: "f32[4, 1024, 768]" = split_2[2];  split_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_39: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_30, [4, 1024, 12, 64]);  getitem_30 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_10: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_40: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_31, [4, 1024, 12, 64]);  getitem_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_11: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_41: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_32, [4, 1024, 12, 64]);  getitem_32 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_12: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_13: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
        expand_8: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_10, [4, 12, 1024, 64])
        clone_8: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_42: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_8, [48, 1024, 64]);  clone_8 = None
        expand_9: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_13, [4, 12, 64, 1024])
        clone_9: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_43: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_9, [48, 64, 1024]);  clone_9 = None
        bmm_4: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None
        view_44: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [4, 12, 1024, 1024]);  bmm_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_2: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_44, full_default);  view_44 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_2: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_34, div_2, full_default_1);  div_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_2: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_2, -1, False);  where_2 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_7 = torch.ops.npu._npu_dropout.default(_softmax_2, 0.1)
        getitem_33: "f32[4, 12, 1024, 1024]" = _npu_dropout_7[0]
        getitem_34: "u8[6291456]" = _npu_dropout_7[1];  _npu_dropout_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_10: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_33, [4, 12, 1024, 1024])
        view_45: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [48, 1024, 1024]);  expand_10 = None
        expand_11: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_12, [4, 12, 1024, 64])
        clone_10: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_46: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_10, [48, 1024, 64]);  clone_10 = None
        bmm_5: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_45, view_46);  view_45 = view_46 = None
        view_47: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 1024, 64]);  bmm_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_14: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        clone_11: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_48: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_11, [4, 1024, 768]);  clone_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[4096, 768]" = torch.ops.aten.view.default(view_48, [-1, 768]);  view_48 = None
        addmm_9: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_35, view_49, primals_36);  primals_35 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_50: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [4, 1024, 768]);  addmm_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_8 = torch.ops.npu._npu_dropout.default(view_50, 0.1);  view_50 = None
        getitem_35: "f32[4, 1024, 768]" = _npu_dropout_8[0]
        getitem_36: "u8[393216]" = _npu_dropout_8[1];  _npu_dropout_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_19: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_35, add_16);  getitem_35 = add_16 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_37: "f32[4, 1024, 1]" = var_mean_5[0]
        getitem_38: "f32[4, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38);  getitem_38 = None
        mul_18: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_19: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_37)
        add_21: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_38);  mul_19 = primals_38 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_51: "f32[4096, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
        addmm_10: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_39, view_51, primals_40);  primals_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_52: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
        pow_3: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_52, 3.0)
        mul_21: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_52, mul_21);  view_52 = mul_21 = None
        mul_22: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_23: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_53: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_41, view_53, primals_42);  primals_41 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_54: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [4, 1024, 768]);  addmm_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_9 = torch.ops.npu._npu_dropout.default(view_54, 0.1);  view_54 = None
        getitem_39: "f32[4, 1024, 768]" = _npu_dropout_9[0]
        getitem_40: "u8[393216]" = _npu_dropout_9[1];  _npu_dropout_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_24: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, getitem_39);  add_19 = getitem_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_41: "f32[4, 1024, 1]" = var_mean_6[0]
        getitem_42: "f32[4, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_6: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_6: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42);  getitem_42 = None
        mul_24: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_25: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_43)
        add_26: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_44);  mul_25 = primals_44 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[4096, 768]" = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
        addmm_12: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_45, view_55, primals_46);  primals_45 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_56: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_12, [4, 1024, 2304]);  addmm_12 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_56, 768, 2);  view_56 = None
        getitem_43: "f32[4, 1024, 768]" = split_3[0]
        getitem_44: "f32[4, 1024, 768]" = split_3[1]
        getitem_45: "f32[4, 1024, 768]" = split_3[2];  split_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_57: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_43, [4, 1024, 12, 64]);  getitem_43 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_15: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_58: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_44, [4, 1024, 12, 64]);  getitem_44 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_16: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_59: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_45, [4, 1024, 12, 64]);  getitem_45 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_17: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_18: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
        expand_12: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_15, [4, 12, 1024, 64])
        clone_12: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_60: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_12, [48, 1024, 64]);  clone_12 = None
        expand_13: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [4, 12, 64, 1024])
        clone_13: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_61: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_13, [48, 64, 1024]);  clone_13 = None
        bmm_6: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
        view_62: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [4, 12, 1024, 1024]);  bmm_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_3: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_62, full_default);  view_62 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_3: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_47, div_3, full_default_1);  div_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_3: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_3, -1, False);  where_3 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_10 = torch.ops.npu._npu_dropout.default(_softmax_3, 0.1)
        getitem_46: "f32[4, 12, 1024, 1024]" = _npu_dropout_10[0]
        getitem_47: "u8[6291456]" = _npu_dropout_10[1];  _npu_dropout_10 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_14: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_46, [4, 12, 1024, 1024])
        view_63: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [48, 1024, 1024]);  expand_14 = None
        expand_15: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [4, 12, 1024, 64])
        clone_14: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_64: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_14, [48, 1024, 64]);  clone_14 = None
        bmm_7: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
        view_65: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 1024, 64]);  bmm_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_19: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_15: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_66: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_15, [4, 1024, 768]);  clone_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[4096, 768]" = torch.ops.aten.view.default(view_66, [-1, 768]);  view_66 = None
        addmm_13: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_48, view_67, primals_49);  primals_48 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_68: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_13, [4, 1024, 768]);  addmm_13 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_11 = torch.ops.npu._npu_dropout.default(view_68, 0.1);  view_68 = None
        getitem_48: "f32[4, 1024, 768]" = _npu_dropout_11[0]
        getitem_49: "u8[393216]" = _npu_dropout_11[1];  _npu_dropout_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_27: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_24);  getitem_48 = add_24 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_50: "f32[4, 1024, 1]" = var_mean_7[0]
        getitem_51: "f32[4, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_7: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51);  getitem_51 = None
        mul_26: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_27: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_50)
        add_29: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_51);  mul_27 = primals_51 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[4096, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
        addmm_14: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_52, view_69, primals_53);  primals_52 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_70: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_4: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_29: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_29);  view_70 = mul_29 = None
        mul_30: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_31: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_54, view_71, primals_55);  primals_54 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_72: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [4, 1024, 768]);  addmm_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_12 = torch.ops.npu._npu_dropout.default(view_72, 0.1);  view_72 = None
        getitem_52: "f32[4, 1024, 768]" = _npu_dropout_12[0]
        getitem_53: "u8[393216]" = _npu_dropout_12[1];  _npu_dropout_12 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_32: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_52);  add_27 = getitem_52 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_54: "f32[4, 1024, 1]" = var_mean_8[0]
        getitem_55: "f32[4, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_8: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_8: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55);  getitem_55 = None
        mul_32: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_33: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_56)
        add_34: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_57);  mul_33 = primals_57 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[4096, 768]" = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
        addmm_16: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_58, view_73, primals_59);  primals_58 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_74: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_16, [4, 1024, 2304]);  addmm_16 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_56: "f32[4, 1024, 768]" = split_4[0]
        getitem_57: "f32[4, 1024, 768]" = split_4[1]
        getitem_58: "f32[4, 1024, 768]" = split_4[2];  split_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_75: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_56, [4, 1024, 12, 64]);  getitem_56 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_20: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_76: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_57, [4, 1024, 12, 64]);  getitem_57 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_21: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_77: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_58, [4, 1024, 12, 64]);  getitem_58 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_22: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_23: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
        expand_16: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_20, [4, 12, 1024, 64])
        clone_16: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_78: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_16, [48, 1024, 64]);  clone_16 = None
        expand_17: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_23, [4, 12, 64, 1024])
        clone_17: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_79: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_17, [48, 64, 1024]);  clone_17 = None
        bmm_8: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
        view_80: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [4, 12, 1024, 1024]);  bmm_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_4: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_80, full_default);  view_80 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_4: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_60, div_4, full_default_1);  div_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_4: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_4, -1, False);  where_4 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_13 = torch.ops.npu._npu_dropout.default(_softmax_4, 0.1)
        getitem_59: "f32[4, 12, 1024, 1024]" = _npu_dropout_13[0]
        getitem_60: "u8[6291456]" = _npu_dropout_13[1];  _npu_dropout_13 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_18: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_59, [4, 12, 1024, 1024])
        view_81: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [48, 1024, 1024]);  expand_18 = None
        expand_19: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_22, [4, 12, 1024, 64])
        clone_18: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_82: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_18, [48, 1024, 64]);  clone_18 = None
        bmm_9: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_81, view_82);  view_81 = view_82 = None
        view_83: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 1024, 64]);  bmm_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_24: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        clone_19: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_84: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_19, [4, 1024, 768]);  clone_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[4096, 768]" = torch.ops.aten.view.default(view_84, [-1, 768]);  view_84 = None
        addmm_17: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_61, view_85, primals_62);  primals_61 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_86: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [4, 1024, 768]);  addmm_17 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_14 = torch.ops.npu._npu_dropout.default(view_86, 0.1);  view_86 = None
        getitem_61: "f32[4, 1024, 768]" = _npu_dropout_14[0]
        getitem_62: "u8[393216]" = _npu_dropout_14[1];  _npu_dropout_14 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_35: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_61, add_32);  getitem_61 = add_32 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_63: "f32[4, 1024, 1]" = var_mean_9[0]
        getitem_64: "f32[4, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_36: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_9: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64);  getitem_64 = None
        mul_34: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_35: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_63)
        add_37: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_64);  mul_35 = primals_64 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_87: "f32[4096, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
        addmm_18: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_65, view_87, primals_66);  primals_65 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_88: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
        pow_5: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_88, 3.0)
        mul_37: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_88, mul_37);  view_88 = mul_37 = None
        mul_38: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_39: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_89: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_67, view_89, primals_68);  primals_67 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_90: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_19, [4, 1024, 768]);  addmm_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_15 = torch.ops.npu._npu_dropout.default(view_90, 0.1);  view_90 = None
        getitem_65: "f32[4, 1024, 768]" = _npu_dropout_15[0]
        getitem_66: "u8[393216]" = _npu_dropout_15[1];  _npu_dropout_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_40: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_35, getitem_65);  add_35 = getitem_65 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_67: "f32[4, 1024, 1]" = var_mean_10[0]
        getitem_68: "f32[4, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_10: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_10: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68);  getitem_68 = None
        mul_40: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_41: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_69)
        add_42: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_70);  mul_41 = primals_70 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[4096, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
        addmm_20: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_71, view_91, primals_72);  primals_71 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_92: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_20, [4, 1024, 2304]);  addmm_20 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_92, 768, 2);  view_92 = None
        getitem_69: "f32[4, 1024, 768]" = split_5[0]
        getitem_70: "f32[4, 1024, 768]" = split_5[1]
        getitem_71: "f32[4, 1024, 768]" = split_5[2];  split_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_93: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_69, [4, 1024, 12, 64]);  getitem_69 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_94: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_70, [4, 1024, 12, 64]);  getitem_70 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_26: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_95: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_71, [4, 1024, 12, 64]);  getitem_71 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_27: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_28: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
        expand_20: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_25, [4, 12, 1024, 64])
        clone_20: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_96: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_20, [48, 1024, 64]);  clone_20 = None
        expand_21: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_28, [4, 12, 64, 1024])
        clone_21: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_97: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_21, [48, 64, 1024]);  clone_21 = None
        bmm_10: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
        view_98: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [4, 12, 1024, 1024]);  bmm_10 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_5: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_98, full_default);  view_98 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_5: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_73, div_5, full_default_1);  div_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_5: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_5, -1, False);  where_5 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_16 = torch.ops.npu._npu_dropout.default(_softmax_5, 0.1)
        getitem_72: "f32[4, 12, 1024, 1024]" = _npu_dropout_16[0]
        getitem_73: "u8[6291456]" = _npu_dropout_16[1];  _npu_dropout_16 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_22: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_72, [4, 12, 1024, 1024])
        view_99: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [48, 1024, 1024]);  expand_22 = None
        expand_23: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_27, [4, 12, 1024, 64])
        clone_22: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_100: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_22, [48, 1024, 64]);  clone_22 = None
        bmm_11: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 1024, 64]);  bmm_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_29: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        clone_23: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_102: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_23, [4, 1024, 768]);  clone_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[4096, 768]" = torch.ops.aten.view.default(view_102, [-1, 768]);  view_102 = None
        addmm_21: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_74, view_103, primals_75);  primals_74 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_104: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [4, 1024, 768]);  addmm_21 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_17 = torch.ops.npu._npu_dropout.default(view_104, 0.1);  view_104 = None
        getitem_74: "f32[4, 1024, 768]" = _npu_dropout_17[0]
        getitem_75: "u8[393216]" = _npu_dropout_17[1];  _npu_dropout_17 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_43: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_74, add_40);  getitem_74 = add_40 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_76: "f32[4, 1024, 1]" = var_mean_11[0]
        getitem_77: "f32[4, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_11: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77);  getitem_77 = None
        mul_42: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_43: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_76)
        add_45: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_77);  mul_43 = primals_77 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[4096, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
        addmm_22: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_78, view_105, primals_79);  primals_78 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_106: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_6: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_45: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_45);  view_106 = mul_45 = None
        mul_46: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_47: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_80, view_107, primals_81);  primals_80 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_108: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [4, 1024, 768]);  addmm_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_18 = torch.ops.npu._npu_dropout.default(view_108, 0.1);  view_108 = None
        getitem_78: "f32[4, 1024, 768]" = _npu_dropout_18[0]
        getitem_79: "u8[393216]" = _npu_dropout_18[1];  _npu_dropout_18 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_48: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_43, getitem_78);  add_43 = getitem_78 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 1024, 1]" = var_mean_12[0]
        getitem_81: "f32[4, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_49: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_12: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_12: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81);  getitem_81 = None
        mul_48: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_49: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_82)
        add_50: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_83);  mul_49 = primals_83 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[4096, 768]" = torch.ops.aten.view.default(add_50, [-1, 768]);  add_50 = None
        addmm_24: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_84, view_109, primals_85);  primals_84 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_110: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_24, [4, 1024, 2304]);  addmm_24 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_82: "f32[4, 1024, 768]" = split_6[0]
        getitem_83: "f32[4, 1024, 768]" = split_6[1]
        getitem_84: "f32[4, 1024, 768]" = split_6[2];  split_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_111: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_82, [4, 1024, 12, 64]);  getitem_82 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_30: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_112: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_83, [4, 1024, 12, 64]);  getitem_83 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_31: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_113: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_84, [4, 1024, 12, 64]);  getitem_84 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_32: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_33: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2])
        expand_24: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_30, [4, 12, 1024, 64])
        clone_24: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_114: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_24, [48, 1024, 64]);  clone_24 = None
        expand_25: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_33, [4, 12, 64, 1024])
        clone_25: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_115: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_25, [48, 64, 1024]);  clone_25 = None
        bmm_12: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [4, 12, 1024, 1024]);  bmm_12 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_6: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_116, full_default);  view_116 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_6: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_86, div_6, full_default_1);  div_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_6: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_6, -1, False);  where_6 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_19 = torch.ops.npu._npu_dropout.default(_softmax_6, 0.1)
        getitem_85: "f32[4, 12, 1024, 1024]" = _npu_dropout_19[0]
        getitem_86: "u8[6291456]" = _npu_dropout_19[1];  _npu_dropout_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_26: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_85, [4, 12, 1024, 1024])
        view_117: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [48, 1024, 1024]);  expand_26 = None
        expand_27: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_32, [4, 12, 1024, 64])
        clone_26: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_118: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_26, [48, 1024, 64]);  clone_26 = None
        bmm_13: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None
        view_119: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 1024, 64]);  bmm_13 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_34: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_27: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_120: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_27, [4, 1024, 768]);  clone_27 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[4096, 768]" = torch.ops.aten.view.default(view_120, [-1, 768]);  view_120 = None
        addmm_25: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_87, view_121, primals_88);  primals_87 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_122: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_25, [4, 1024, 768]);  addmm_25 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_20 = torch.ops.npu._npu_dropout.default(view_122, 0.1);  view_122 = None
        getitem_87: "f32[4, 1024, 768]" = _npu_dropout_20[0]
        getitem_88: "u8[393216]" = _npu_dropout_20[1];  _npu_dropout_20 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_51: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_87, add_48);  getitem_87 = add_48 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_89: "f32[4, 1024, 1]" = var_mean_13[0]
        getitem_90: "f32[4, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_52: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_13: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_90);  getitem_90 = None
        mul_50: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_51: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_89)
        add_53: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_90);  mul_51 = primals_90 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_123: "f32[4096, 768]" = torch.ops.aten.view.default(add_53, [-1, 768]);  add_53 = None
        addmm_26: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_91, view_123, primals_92);  primals_91 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_124: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_26, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.5)
        pow_7: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_124, 3.0)
        mul_53: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_124, mul_53);  view_124 = mul_53 = None
        mul_54: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_55: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_125: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_93, view_125, primals_94);  primals_93 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_126: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [4, 1024, 768]);  addmm_27 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_21 = torch.ops.npu._npu_dropout.default(view_126, 0.1);  view_126 = None
        getitem_91: "f32[4, 1024, 768]" = _npu_dropout_21[0]
        getitem_92: "u8[393216]" = _npu_dropout_21[1];  _npu_dropout_21 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_56: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, getitem_91);  add_51 = getitem_91 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_93: "f32[4, 1024, 1]" = var_mean_14[0]
        getitem_94: "f32[4, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_57: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
        rsqrt_14: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_14: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_94);  getitem_94 = None
        mul_56: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_57: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_95)
        add_58: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_96);  mul_57 = primals_96 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[4096, 768]" = torch.ops.aten.view.default(add_58, [-1, 768]);  add_58 = None
        addmm_28: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_97, view_127, primals_98);  primals_97 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_128: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_28, [4, 1024, 2304]);  addmm_28 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_128, 768, 2);  view_128 = None
        getitem_95: "f32[4, 1024, 768]" = split_7[0]
        getitem_96: "f32[4, 1024, 768]" = split_7[1]
        getitem_97: "f32[4, 1024, 768]" = split_7[2];  split_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_129: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_95, [4, 1024, 12, 64]);  getitem_95 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_35: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_130: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_96, [4, 1024, 12, 64]);  getitem_96 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_131: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_97, [4, 1024, 12, 64]);  getitem_97 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_37: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_38: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2])
        expand_28: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [4, 12, 1024, 64])
        clone_28: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_132: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_28, [48, 1024, 64]);  clone_28 = None
        expand_29: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_38, [4, 12, 64, 1024])
        clone_29: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_133: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_29, [48, 64, 1024]);  clone_29 = None
        bmm_14: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
        view_134: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [4, 12, 1024, 1024]);  bmm_14 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_7: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_134, full_default);  view_134 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_7: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_99, div_7, full_default_1);  div_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_7: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_7, -1, False);  where_7 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_22 = torch.ops.npu._npu_dropout.default(_softmax_7, 0.1)
        getitem_98: "f32[4, 12, 1024, 1024]" = _npu_dropout_22[0]
        getitem_99: "u8[6291456]" = _npu_dropout_22[1];  _npu_dropout_22 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_30: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_98, [4, 12, 1024, 1024])
        view_135: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [48, 1024, 1024]);  expand_30 = None
        expand_31: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_37, [4, 12, 1024, 64])
        clone_30: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_136: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_30, [48, 1024, 64]);  clone_30 = None
        bmm_15: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 1024, 64]);  bmm_15 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_39: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_31: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_138: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_31, [4, 1024, 768]);  clone_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[4096, 768]" = torch.ops.aten.view.default(view_138, [-1, 768]);  view_138 = None
        addmm_29: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_100, view_139, primals_101);  primals_100 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_140: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [4, 1024, 768]);  addmm_29 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_23 = torch.ops.npu._npu_dropout.default(view_140, 0.1);  view_140 = None
        getitem_100: "f32[4, 1024, 768]" = _npu_dropout_23[0]
        getitem_101: "u8[393216]" = _npu_dropout_23[1];  _npu_dropout_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_59: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_56);  getitem_100 = add_56 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_102: "f32[4, 1024, 1]" = var_mean_15[0]
        getitem_103: "f32[4, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_60: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_15: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_103);  getitem_103 = None
        mul_58: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_59: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_102)
        add_61: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_103);  mul_59 = primals_103 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[4096, 768]" = torch.ops.aten.view.default(add_61, [-1, 768]);  add_61 = None
        addmm_30: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_104, view_141, primals_105);  primals_104 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_142: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_30, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_8: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_61: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_61);  view_142 = mul_61 = None
        mul_62: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_63: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_106, view_143, primals_107);  primals_106 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_144: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_31, [4, 1024, 768]);  addmm_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_24 = torch.ops.npu._npu_dropout.default(view_144, 0.1);  view_144 = None
        getitem_104: "f32[4, 1024, 768]" = _npu_dropout_24[0]
        getitem_105: "u8[393216]" = _npu_dropout_24[1];  _npu_dropout_24 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_64: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, getitem_104);  add_59 = getitem_104 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_106: "f32[4, 1024, 1]" = var_mean_16[0]
        getitem_107: "f32[4, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_65: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_16: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_16: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_107);  getitem_107 = None
        mul_64: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_65: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_108)
        add_66: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_109);  mul_65 = primals_109 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_145: "f32[4096, 768]" = torch.ops.aten.view.default(add_66, [-1, 768]);  add_66 = None
        addmm_32: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_110, view_145, primals_111);  primals_110 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_146: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_32, [4, 1024, 2304]);  addmm_32 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_146, 768, 2);  view_146 = None
        getitem_108: "f32[4, 1024, 768]" = split_8[0]
        getitem_109: "f32[4, 1024, 768]" = split_8[1]
        getitem_110: "f32[4, 1024, 768]" = split_8[2];  split_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_147: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_108, [4, 1024, 12, 64]);  getitem_108 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_40: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_148: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_109, [4, 1024, 12, 64]);  getitem_109 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_41: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_149: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_110, [4, 1024, 12, 64]);  getitem_110 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_42: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_43: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2])
        expand_32: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_40, [4, 12, 1024, 64])
        clone_32: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_150: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_32, [48, 1024, 64]);  clone_32 = None
        expand_33: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_43, [4, 12, 64, 1024])
        clone_33: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_151: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_33, [48, 64, 1024]);  clone_33 = None
        bmm_16: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_150, view_151);  view_150 = view_151 = None
        view_152: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [4, 12, 1024, 1024]);  bmm_16 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_8: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_152, full_default);  view_152 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_8: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_112, div_8, full_default_1);  div_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_8: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_8, -1, False);  where_8 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_25 = torch.ops.npu._npu_dropout.default(_softmax_8, 0.1)
        getitem_111: "f32[4, 12, 1024, 1024]" = _npu_dropout_25[0]
        getitem_112: "u8[6291456]" = _npu_dropout_25[1];  _npu_dropout_25 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_34: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_111, [4, 12, 1024, 1024])
        view_153: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [48, 1024, 1024]);  expand_34 = None
        expand_35: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_42, [4, 12, 1024, 64])
        clone_34: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_154: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_34, [48, 1024, 64]);  clone_34 = None
        bmm_17: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_153, view_154);  view_153 = view_154 = None
        view_155: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 1024, 64]);  bmm_17 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_44: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        clone_35: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_156: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_35, [4, 1024, 768]);  clone_35 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_157: "f32[4096, 768]" = torch.ops.aten.view.default(view_156, [-1, 768]);  view_156 = None
        addmm_33: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_113, view_157, primals_114);  primals_113 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_158: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [4, 1024, 768]);  addmm_33 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_26 = torch.ops.npu._npu_dropout.default(view_158, 0.1);  view_158 = None
        getitem_113: "f32[4, 1024, 768]" = _npu_dropout_26[0]
        getitem_114: "u8[393216]" = _npu_dropout_26[1];  _npu_dropout_26 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_67: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_113, add_64);  getitem_113 = add_64 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_115: "f32[4, 1024, 1]" = var_mean_17[0]
        getitem_116: "f32[4, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_68: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_17: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_116);  getitem_116 = None
        mul_66: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_67: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_115)
        add_69: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_116);  mul_67 = primals_116 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_159: "f32[4096, 768]" = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
        addmm_34: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_117, view_159, primals_118);  primals_117 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_160: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
        pow_9: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 3.0)
        mul_69: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_160, mul_69);  view_160 = mul_69 = None
        mul_70: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_71: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_161: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_119, view_161, primals_120);  primals_119 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_162: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [4, 1024, 768]);  addmm_35 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_27 = torch.ops.npu._npu_dropout.default(view_162, 0.1);  view_162 = None
        getitem_117: "f32[4, 1024, 768]" = _npu_dropout_27[0]
        getitem_118: "u8[393216]" = _npu_dropout_27[1];  _npu_dropout_27 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_72: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, getitem_117);  add_67 = getitem_117 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_119: "f32[4, 1024, 1]" = var_mean_18[0]
        getitem_120: "f32[4, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_73: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_18: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_18: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120);  getitem_120 = None
        mul_72: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_73: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_121)
        add_74: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_122);  mul_73 = primals_122 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_163: "f32[4096, 768]" = torch.ops.aten.view.default(add_74, [-1, 768]);  add_74 = None
        addmm_36: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_123, view_163, primals_124);  primals_123 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_164: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_36, [4, 1024, 2304]);  addmm_36 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_164, 768, 2);  view_164 = None
        getitem_121: "f32[4, 1024, 768]" = split_9[0]
        getitem_122: "f32[4, 1024, 768]" = split_9[1]
        getitem_123: "f32[4, 1024, 768]" = split_9[2];  split_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_165: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_121, [4, 1024, 12, 64]);  getitem_121 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_45: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_166: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_122, [4, 1024, 12, 64]);  getitem_122 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_46: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_167: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_123, [4, 1024, 12, 64]);  getitem_123 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_47: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_48: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2])
        expand_36: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_45, [4, 12, 1024, 64])
        clone_36: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_168: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_36, [48, 1024, 64]);  clone_36 = None
        expand_37: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_48, [4, 12, 64, 1024])
        clone_37: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_169: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_37, [48, 64, 1024]);  clone_37 = None
        bmm_18: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [4, 12, 1024, 1024]);  bmm_18 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_9: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_170, full_default);  view_170 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_9: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_125, div_9, full_default_1);  div_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_9: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_9, -1, False);  where_9 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_28 = torch.ops.npu._npu_dropout.default(_softmax_9, 0.1)
        getitem_124: "f32[4, 12, 1024, 1024]" = _npu_dropout_28[0]
        getitem_125: "u8[6291456]" = _npu_dropout_28[1];  _npu_dropout_28 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_38: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_124, [4, 12, 1024, 1024])
        view_171: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [48, 1024, 1024]);  expand_38 = None
        expand_39: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_47, [4, 12, 1024, 64])
        clone_38: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_172: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_38, [48, 1024, 64]);  clone_38 = None
        bmm_19: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
        view_173: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 1024, 64]);  bmm_19 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_49: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        clone_39: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_174: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_39, [4, 1024, 768]);  clone_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_175: "f32[4096, 768]" = torch.ops.aten.view.default(view_174, [-1, 768]);  view_174 = None
        addmm_37: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_126, view_175, primals_127);  primals_126 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_176: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_37, [4, 1024, 768]);  addmm_37 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_29 = torch.ops.npu._npu_dropout.default(view_176, 0.1);  view_176 = None
        getitem_126: "f32[4, 1024, 768]" = _npu_dropout_29[0]
        getitem_127: "u8[393216]" = _npu_dropout_29[1];  _npu_dropout_29 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_75: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_126, add_72);  getitem_126 = add_72 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_128: "f32[4, 1024, 1]" = var_mean_19[0]
        getitem_129: "f32[4, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_76: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_19: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_129);  getitem_129 = None
        mul_74: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_75: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_128)
        add_77: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_129);  mul_75 = primals_129 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_177: "f32[4096, 768]" = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
        addmm_38: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_130, view_177, primals_131);  primals_130 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_178: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_38, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_10: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_77: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_178, mul_77);  view_178 = mul_77 = None
        mul_78: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_79: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_179: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_132, view_179, primals_133);  primals_132 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_180: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [4, 1024, 768]);  addmm_39 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_30 = torch.ops.npu._npu_dropout.default(view_180, 0.1);  view_180 = None
        getitem_130: "f32[4, 1024, 768]" = _npu_dropout_30[0]
        getitem_131: "u8[393216]" = _npu_dropout_30[1];  _npu_dropout_30 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_80: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_75, getitem_130);  add_75 = getitem_130 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_132: "f32[4, 1024, 1]" = var_mean_20[0]
        getitem_133: "f32[4, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_81: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_20: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_20: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_133);  getitem_133 = None
        mul_80: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_81: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_134)
        add_82: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_135);  mul_81 = primals_135 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_181: "f32[4096, 768]" = torch.ops.aten.view.default(add_82, [-1, 768]);  add_82 = None
        addmm_40: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_136, view_181, primals_137);  primals_136 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_182: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_40, [4, 1024, 2304]);  addmm_40 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_182, 768, 2);  view_182 = None
        getitem_134: "f32[4, 1024, 768]" = split_10[0]
        getitem_135: "f32[4, 1024, 768]" = split_10[1]
        getitem_136: "f32[4, 1024, 768]" = split_10[2];  split_10 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_183: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_134, [4, 1024, 12, 64]);  getitem_134 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_50: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_184: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_135, [4, 1024, 12, 64]);  getitem_135 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_51: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_185: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_136, [4, 1024, 12, 64]);  getitem_136 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_52: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_53: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2])
        expand_40: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [4, 12, 1024, 64])
        clone_40: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_186: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_40, [48, 1024, 64]);  clone_40 = None
        expand_41: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_53, [4, 12, 64, 1024])
        clone_41: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_187: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_41, [48, 64, 1024]);  clone_41 = None
        bmm_20: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_186, view_187);  view_186 = view_187 = None
        view_188: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [4, 12, 1024, 1024]);  bmm_20 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_10: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_188, full_default);  view_188 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_10: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_138, div_10, full_default_1);  div_10 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_10: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_10, -1, False);  where_10 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_31 = torch.ops.npu._npu_dropout.default(_softmax_10, 0.1)
        getitem_137: "f32[4, 12, 1024, 1024]" = _npu_dropout_31[0]
        getitem_138: "u8[6291456]" = _npu_dropout_31[1];  _npu_dropout_31 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_42: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_137, [4, 12, 1024, 1024])
        view_189: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [48, 1024, 1024]);  expand_42 = None
        expand_43: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_52, [4, 12, 1024, 64])
        clone_42: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_190: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_42, [48, 1024, 64]);  clone_42 = None
        bmm_21: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_189, view_190);  view_189 = view_190 = None
        view_191: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 1024, 64]);  bmm_21 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_54: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        clone_43: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_192: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_43, [4, 1024, 768]);  clone_43 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_193: "f32[4096, 768]" = torch.ops.aten.view.default(view_192, [-1, 768]);  view_192 = None
        addmm_41: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_139, view_193, primals_140);  primals_139 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_194: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [4, 1024, 768]);  addmm_41 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_32 = torch.ops.npu._npu_dropout.default(view_194, 0.1);  view_194 = None
        getitem_139: "f32[4, 1024, 768]" = _npu_dropout_32[0]
        getitem_140: "u8[393216]" = _npu_dropout_32[1];  _npu_dropout_32 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_83: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_139, add_80);  getitem_139 = add_80 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_141: "f32[4, 1024, 1]" = var_mean_21[0]
        getitem_142: "f32[4, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_84: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_21: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_142);  getitem_142 = None
        mul_82: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_83: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_141)
        add_85: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_142);  mul_83 = primals_142 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_195: "f32[4096, 768]" = torch.ops.aten.view.default(add_85, [-1, 768]);  add_85 = None
        addmm_42: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_143, view_195, primals_144);  primals_143 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_196: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_42, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
        pow_11: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_196, 3.0)
        mul_85: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_196, mul_85);  view_196 = mul_85 = None
        mul_86: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_87: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_197: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_145, view_197, primals_146);  primals_145 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_198: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_43, [4, 1024, 768]);  addmm_43 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_33 = torch.ops.npu._npu_dropout.default(view_198, 0.1);  view_198 = None
        getitem_143: "f32[4, 1024, 768]" = _npu_dropout_33[0]
        getitem_144: "u8[393216]" = _npu_dropout_33[1];  _npu_dropout_33 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_88: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, getitem_143);  add_83 = getitem_143 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_145: "f32[4, 1024, 1]" = var_mean_22[0]
        getitem_146: "f32[4, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_89: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
        rsqrt_22: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_22: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_146);  getitem_146 = None
        mul_88: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_89: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_147)
        add_90: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_148);  mul_89 = primals_148 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_199: "f32[4096, 768]" = torch.ops.aten.view.default(add_90, [-1, 768]);  add_90 = None
        addmm_44: "f32[4096, 2304]" = torch.ops.aten.addmm.default(primals_149, view_199, primals_150);  primals_149 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_200: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_44, [4, 1024, 2304]);  addmm_44 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_200, 768, 2);  view_200 = None
        getitem_147: "f32[4, 1024, 768]" = split_11[0]
        getitem_148: "f32[4, 1024, 768]" = split_11[1]
        getitem_149: "f32[4, 1024, 768]" = split_11[2];  split_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_201: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_147, [4, 1024, 12, 64]);  getitem_147 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_55: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_202: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_148, [4, 1024, 12, 64]);  getitem_148 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_56: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_203: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_149, [4, 1024, 12, 64]);  getitem_149 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_57: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_58: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2])
        expand_44: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_55, [4, 12, 1024, 64])
        clone_44: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_204: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_44, [48, 1024, 64]);  clone_44 = None
        expand_45: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_58, [4, 12, 64, 1024])
        clone_45: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_205: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_45, [48, 64, 1024]);  clone_45 = None
        bmm_22: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [4, 12, 1024, 1024]);  bmm_22 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_11: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_206, full_default);  view_206 = full_default = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_11: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_151, div_11, full_default_1);  div_11 = full_default_1 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        _softmax_11: "f32[4, 12, 1024, 1024]" = torch.ops.aten._softmax.default(where_11, -1, False);  where_11 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        _npu_dropout_34 = torch.ops.npu._npu_dropout.default(_softmax_11, 0.1)
        getitem_150: "f32[4, 12, 1024, 1024]" = _npu_dropout_34[0]
        getitem_151: "u8[6291456]" = _npu_dropout_34[1];  _npu_dropout_34 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_46: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_150, [4, 12, 1024, 1024])
        view_207: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [48, 1024, 1024]);  expand_46 = None
        expand_47: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [4, 12, 1024, 64])
        clone_46: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_208: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_46, [48, 1024, 64]);  clone_46 = None
        bmm_23: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
        view_209: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 1024, 64]);  bmm_23 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_59: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_47: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_210: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_47, [4, 1024, 768]);  clone_47 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_211: "f32[4096, 768]" = torch.ops.aten.view.default(view_210, [-1, 768]);  view_210 = None
        addmm_45: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_152, view_211, primals_153);  primals_152 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_212: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [4, 1024, 768]);  addmm_45 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        _npu_dropout_35 = torch.ops.npu._npu_dropout.default(view_212, 0.1);  view_212 = None
        getitem_152: "f32[4, 1024, 768]" = _npu_dropout_35[0]
        getitem_153: "u8[393216]" = _npu_dropout_35[1];  _npu_dropout_35 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:401 in forward, code: hidden_states = attn_output + residual
        add_91: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_152, add_88);  getitem_152 = add_88 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_154: "f32[4, 1024, 1]" = var_mean_23[0]
        getitem_155: "f32[4, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_92: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_23: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_155);  getitem_155 = None
        mul_90: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_91: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_154)
        add_93: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_155);  mul_91 = primals_155 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_213: "f32[4096, 768]" = torch.ops.aten.view.default(add_93, [-1, 768]);  add_93 = None
        addmm_46: "f32[4096, 3072]" = torch.ops.aten.addmm.default(primals_156, view_213, primals_157);  primals_156 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_214: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [4, 1024, 3072])

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
        pow_12: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 3.0)
        mul_93: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_214, mul_93);  view_214 = mul_93 = None
        mul_94: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_95: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_215: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47: "f32[4096, 768]" = torch.ops.aten.addmm.default(primals_158, view_215, primals_159);  primals_158 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_216: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [4, 1024, 768]);  addmm_47 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        _npu_dropout_36 = torch.ops.npu._npu_dropout.default(view_216, 0.1);  view_216 = None
        getitem_156: "f32[4, 1024, 768]" = _npu_dropout_36[0]
        getitem_157: "u8[393216]" = _npu_dropout_36[1];  _npu_dropout_36 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:429 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_96: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_91, getitem_156);  add_91 = getitem_156 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:914 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_158: "f32[4, 1024, 1]" = var_mean_24[0]
        getitem_159: "f32[4, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        add_97: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_24: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_24: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_159);  add_96 = getitem_159 = None
        mul_96: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_97: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_160)
        add_98: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_161);  mul_97 = primals_161 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1440 in forward, code: logits = self.score(hidden_states)
        permute_60: "f32[768, 2]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        view_218: "f32[4096, 768]" = torch.ops.aten.view.default(add_98, [4096, 768])
        mm: "f32[4096, 2]" = torch.ops.aten.mm.default(view_218, permute_60);  view_218 = permute_60 = None
        view_219: "f32[4, 1024, 2]" = torch.ops.aten.view.default(mm, [4, 1024, 2]);  mm = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1454 in forward, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
        eq: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        convert_element_type_12: "i32[4, 1024]" = torch.ops.prims.convert_element_type.default(eq, torch.int32);  eq = None
        argmax: "i64[4]" = torch.ops.aten.argmax.default(convert_element_type_12, -1);  convert_element_type_12 = None
        sub_25: "i64[4]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1464 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        index: "f32[4, 2]" = torch.ops.aten.index.Tensor(view_219, [iota_1, sub_25]);  view_219 = None

        # File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1484 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        _log_softmax: "f32[4, 2]" = torch.ops.aten._log_softmax.default(index, 1, False)
        ne: "b8[4]" = torch.ops.aten.ne.Scalar(primals_163, -100)
        full_default_24: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_12: "i64[4]" = torch.ops.aten.where.self(ne, primals_163, full_default_24);  full_default_24 = None
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather: "f32[4, 1]" = torch.ops.aten.gather.default(_log_softmax, 1, unsqueeze_1);  unsqueeze_1 = None
        squeeze: "f32[4]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_13: "f32[4]" = torch.ops.aten.where.self(ne, neg, full_default_25);  ne = neg = full_default_25 = None
        ne_2: "b8[4]" = torch.ops.aten.ne.Scalar(primals_163, -100)
        sum_1: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_13: "f32[]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_13);  sum_2 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1484 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        unsqueeze_2: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(primals_163, 1);  primals_163 = None
        ne_3: "b8[4, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, -100)
        full_default_26: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_14: "i64[4, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_2, full_default_26);  unsqueeze_2 = full_default_26 = None

        # No stacktrace found for following nodes
        iota_default: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        view_default: "i64[1, 2]" = torch.ops.aten.view.default(iota_default, [1, 2]);  iota_default = None
        expand_default: "i64[4, 2]" = torch.ops.aten.expand.default(where_14, [4, 2]);  where_14 = None
        eq_tensor: "b8[4, 2]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1440 in forward, code: logits = self.score(hidden_states)
        view_225: "f32[4096, 768]" = torch.ops.aten.view.default(add_98, [-1, 768]);  add_98 = None
        permute_62: "f32[768, 4096]" = torch.ops.aten.permute.default(view_225, [1, 0]);  view_225 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:914 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_14: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_65: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_215, [1, 0]);  view_215 = None
        permute_67: "f32[768, 4096]" = torch.ops.aten.permute.default(view_213, [1, 0]);  view_213 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_15: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_69: "f32[768, 4096]" = torch.ops.aten.permute.default(view_211, [1, 0]);  view_211 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_71: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2])
        expand_49: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_71, [4, 12, 64, 1024]);  permute_71 = None
        clone_49: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_239: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_49, [48, 64, 1024]);  clone_49 = None
        permute_72: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_150, [0, 1, 3, 2]);  getitem_150 = None
        expand_50: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_72, [4, 12, 1024, 1024]);  permute_72 = None
        view_241: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_50, [48, 1024, 1024]);  expand_50 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_73: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_53: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_73, [4, 12, 1024, 64]);  permute_73 = None
        clone_51: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_245: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_51, [48, 1024, 64]);  clone_51 = None
        permute_74: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_55, [0, 1, 3, 2]);  permute_55 = None
        expand_54: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_74, [4, 12, 64, 1024]);  permute_74 = None
        clone_52: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_247: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_52, [48, 64, 1024]);  clone_52 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_80: "f32[768, 4096]" = torch.ops.aten.permute.default(view_199, [1, 0]);  view_199 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_17: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_82: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_197, [1, 0]);  view_197 = None
        permute_84: "f32[768, 4096]" = torch.ops.aten.permute.default(view_195, [1, 0]);  view_195 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_18: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_86: "f32[768, 4096]" = torch.ops.aten.permute.default(view_193, [1, 0]);  view_193 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_88: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2])
        expand_57: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_88, [4, 12, 64, 1024]);  permute_88 = None
        clone_55: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_267: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_55, [48, 64, 1024]);  clone_55 = None
        permute_89: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_137, [0, 1, 3, 2]);  getitem_137 = None
        expand_58: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_89, [4, 12, 1024, 1024]);  permute_89 = None
        view_269: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_58, [48, 1024, 1024]);  expand_58 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_90: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_53, [0, 1, 3, 2]);  permute_53 = None
        expand_61: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_90, [4, 12, 1024, 64]);  permute_90 = None
        clone_57: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_273: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_57, [48, 1024, 64]);  clone_57 = None
        permute_91: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_50, [0, 1, 3, 2]);  permute_50 = None
        expand_62: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_91, [4, 12, 64, 1024]);  permute_91 = None
        clone_58: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_275: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_58, [48, 64, 1024]);  clone_58 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_97: "f32[768, 4096]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_20: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_99: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_179, [1, 0]);  view_179 = None
        permute_101: "f32[768, 4096]" = torch.ops.aten.permute.default(view_177, [1, 0]);  view_177 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_21: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_103: "f32[768, 4096]" = torch.ops.aten.permute.default(view_175, [1, 0]);  view_175 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_105: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2])
        expand_65: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_105, [4, 12, 64, 1024]);  permute_105 = None
        clone_61: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_295: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_61, [48, 64, 1024]);  clone_61 = None
        permute_106: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_124, [0, 1, 3, 2]);  getitem_124 = None
        expand_66: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_106, [4, 12, 1024, 1024]);  permute_106 = None
        view_297: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_66, [48, 1024, 1024]);  expand_66 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_107: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_69: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_107, [4, 12, 1024, 64]);  permute_107 = None
        clone_63: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_301: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_63, [48, 1024, 64]);  clone_63 = None
        permute_108: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_45, [0, 1, 3, 2]);  permute_45 = None
        expand_70: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_108, [4, 12, 64, 1024]);  permute_108 = None
        clone_64: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_303: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_64, [48, 64, 1024]);  clone_64 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_114: "f32[768, 4096]" = torch.ops.aten.permute.default(view_163, [1, 0]);  view_163 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_23: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_116: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_161, [1, 0]);  view_161 = None
        permute_118: "f32[768, 4096]" = torch.ops.aten.permute.default(view_159, [1, 0]);  view_159 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_24: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_120: "f32[768, 4096]" = torch.ops.aten.permute.default(view_157, [1, 0]);  view_157 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_122: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_42, [0, 1, 3, 2])
        expand_73: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_122, [4, 12, 64, 1024]);  permute_122 = None
        clone_67: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_323: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_67, [48, 64, 1024]);  clone_67 = None
        permute_123: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_111, [0, 1, 3, 2]);  getitem_111 = None
        expand_74: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_123, [4, 12, 1024, 1024]);  permute_123 = None
        view_325: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_74, [48, 1024, 1024]);  expand_74 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_124: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_43, [0, 1, 3, 2]);  permute_43 = None
        expand_77: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_124, [4, 12, 1024, 64]);  permute_124 = None
        clone_69: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_329: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_69, [48, 1024, 64]);  clone_69 = None
        permute_125: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_78: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_125, [4, 12, 64, 1024]);  permute_125 = None
        clone_70: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_331: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_70, [48, 64, 1024]);  clone_70 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_131: "f32[768, 4096]" = torch.ops.aten.permute.default(view_145, [1, 0]);  view_145 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_26: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_133: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_143, [1, 0]);  view_143 = None
        permute_135: "f32[768, 4096]" = torch.ops.aten.permute.default(view_141, [1, 0]);  view_141 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_27: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_137: "f32[768, 4096]" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_139: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2])
        expand_81: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_139, [4, 12, 64, 1024]);  permute_139 = None
        clone_73: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_351: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_73, [48, 64, 1024]);  clone_73 = None
        permute_140: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_98, [0, 1, 3, 2]);  getitem_98 = None
        expand_82: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_140, [4, 12, 1024, 1024]);  permute_140 = None
        view_353: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_82, [48, 1024, 1024]);  expand_82 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_141: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        expand_85: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_141, [4, 12, 1024, 64]);  permute_141 = None
        clone_75: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_357: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_75, [48, 1024, 64]);  clone_75 = None
        permute_142: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
        expand_86: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_142, [4, 12, 64, 1024]);  permute_142 = None
        clone_76: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_359: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_76, [48, 64, 1024]);  clone_76 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_148: "f32[768, 4096]" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_29: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_150: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_125, [1, 0]);  view_125 = None
        permute_152: "f32[768, 4096]" = torch.ops.aten.permute.default(view_123, [1, 0]);  view_123 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_30: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_154: "f32[768, 4096]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_156: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_32, [0, 1, 3, 2])
        expand_89: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_156, [4, 12, 64, 1024]);  permute_156 = None
        clone_79: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_379: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_79, [48, 64, 1024]);  clone_79 = None
        permute_157: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_85, [0, 1, 3, 2]);  getitem_85 = None
        expand_90: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_157, [4, 12, 1024, 1024]);  permute_157 = None
        view_381: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_90, [48, 1024, 1024]);  expand_90 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_158: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_33, [0, 1, 3, 2]);  permute_33 = None
        expand_93: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_158, [4, 12, 1024, 64]);  permute_158 = None
        clone_81: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_385: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_81, [48, 1024, 64]);  clone_81 = None
        permute_159: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_30, [0, 1, 3, 2]);  permute_30 = None
        expand_94: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_159, [4, 12, 64, 1024]);  permute_159 = None
        clone_82: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_387: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_82, [48, 64, 1024]);  clone_82 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_165: "f32[768, 4096]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_32: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_167: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_107, [1, 0]);  view_107 = None
        permute_169: "f32[768, 4096]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_33: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_171: "f32[768, 4096]" = torch.ops.aten.permute.default(view_103, [1, 0]);  view_103 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_173: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2])
        expand_97: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_173, [4, 12, 64, 1024]);  permute_173 = None
        clone_85: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_407: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_85, [48, 64, 1024]);  clone_85 = None
        permute_174: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_98: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_174, [4, 12, 1024, 1024]);  permute_174 = None
        view_409: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_98, [48, 1024, 1024]);  expand_98 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_175: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_101: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_175, [4, 12, 1024, 64]);  permute_175 = None
        clone_87: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_413: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_87, [48, 1024, 64]);  clone_87 = None
        permute_176: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_102: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_176, [4, 12, 64, 1024]);  permute_176 = None
        clone_88: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_415: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_88, [48, 64, 1024]);  clone_88 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_182: "f32[768, 4096]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_35: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_184: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_89, [1, 0]);  view_89 = None
        permute_186: "f32[768, 4096]" = torch.ops.aten.permute.default(view_87, [1, 0]);  view_87 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_36: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_188: "f32[768, 4096]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_190: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_22, [0, 1, 3, 2])
        expand_105: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_190, [4, 12, 64, 1024]);  permute_190 = None
        clone_91: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_435: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_91, [48, 64, 1024]);  clone_91 = None
        permute_191: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_59, [0, 1, 3, 2]);  getitem_59 = None
        expand_106: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_191, [4, 12, 1024, 1024]);  permute_191 = None
        view_437: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_106, [48, 1024, 1024]);  expand_106 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_192: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_23, [0, 1, 3, 2]);  permute_23 = None
        expand_109: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_192, [4, 12, 1024, 64]);  permute_192 = None
        clone_93: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_441: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_93, [48, 1024, 64]);  clone_93 = None
        permute_193: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_20, [0, 1, 3, 2]);  permute_20 = None
        expand_110: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_193, [4, 12, 64, 1024]);  permute_193 = None
        clone_94: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_443: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_94, [48, 64, 1024]);  clone_94 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_199: "f32[768, 4096]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_38: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_201: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_203: "f32[768, 4096]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_39: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_205: "f32[768, 4096]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_207: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_17, [0, 1, 3, 2])
        expand_113: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_207, [4, 12, 64, 1024]);  permute_207 = None
        clone_97: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_463: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_97, [48, 64, 1024]);  clone_97 = None
        permute_208: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 3, 2]);  getitem_46 = None
        expand_114: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_208, [4, 12, 1024, 1024]);  permute_208 = None
        view_465: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_114, [48, 1024, 1024]);  expand_114 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_209: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_18, [0, 1, 3, 2]);  permute_18 = None
        expand_117: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_209, [4, 12, 1024, 64]);  permute_209 = None
        clone_99: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_469: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_99, [48, 1024, 64]);  clone_99 = None
        permute_210: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_118: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_210, [4, 12, 64, 1024]);  permute_210 = None
        clone_100: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
        view_471: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_100, [48, 64, 1024]);  clone_100 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_216: "f32[768, 4096]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_41: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_218: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_53, [1, 0]);  view_53 = None
        permute_220: "f32[768, 4096]" = torch.ops.aten.permute.default(view_51, [1, 0]);  view_51 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_42: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_222: "f32[768, 4096]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_224: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_12, [0, 1, 3, 2])
        expand_121: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_224, [4, 12, 64, 1024]);  permute_224 = None
        clone_103: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_491: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_103, [48, 64, 1024]);  clone_103 = None
        permute_225: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_33, [0, 1, 3, 2]);  getitem_33 = None
        expand_122: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_225, [4, 12, 1024, 1024]);  permute_225 = None
        view_493: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_122, [48, 1024, 1024]);  expand_122 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_226: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
        expand_125: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_226, [4, 12, 1024, 64]);  permute_226 = None
        clone_105: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_497: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_105, [48, 1024, 64]);  clone_105 = None
        permute_227: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_10, [0, 1, 3, 2]);  permute_10 = None
        expand_126: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_227, [4, 12, 64, 1024]);  permute_227 = None
        clone_106: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
        view_499: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_106, [48, 64, 1024]);  clone_106 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_233: "f32[768, 4096]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_44: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_235: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_237: "f32[768, 4096]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_45: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_239: "f32[768, 4096]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_241: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_7, [0, 1, 3, 2])
        expand_129: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_241, [4, 12, 64, 1024]);  permute_241 = None
        clone_109: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_519: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_109, [48, 64, 1024]);  clone_109 = None
        permute_242: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_20, [0, 1, 3, 2]);  getitem_20 = None
        expand_130: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_242, [4, 12, 1024, 1024]);  permute_242 = None
        view_521: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_130, [48, 1024, 1024]);  expand_130 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_243: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_8, [0, 1, 3, 2]);  permute_8 = None
        expand_133: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_243, [4, 12, 1024, 64]);  permute_243 = None
        clone_111: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_525: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_111, [48, 1024, 64]);  clone_111 = None
        permute_244: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        expand_134: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_244, [4, 12, 64, 1024]);  permute_244 = None
        clone_112: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_527: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_112, [48, 64, 1024]);  clone_112 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_250: "f32[768, 4096]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_47: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_252: "f32[3072, 4096]" = torch.ops.aten.permute.default(view_17, [1, 0]);  view_17 = None
        permute_254: "f32[768, 4096]" = torch.ops.aten.permute.default(view_15, [1, 0]);  view_15 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_48: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_256: "f32[768, 4096]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_258: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2])
        expand_137: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_258, [4, 12, 64, 1024]);  permute_258 = None
        clone_115: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_547: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_115, [48, 64, 1024]);  clone_115 = None
        permute_259: "f32[4, 12, 1024, 1024]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
        expand_138: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(permute_259, [4, 12, 1024, 1024]);  permute_259 = None
        view_549: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_138, [48, 1024, 1024]);  expand_138 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_260: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_141: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_260, [4, 12, 1024, 64]);  permute_260 = None
        clone_117: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_553: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_117, [48, 1024, 64]);  clone_117 = None
        permute_261: "f32[4, 12, 64, 1024]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
        expand_142: "f32[4, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_261, [4, 12, 64, 1024]);  permute_261 = None
        clone_118: "f32[4, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
        view_555: "f32[48, 64, 1024]" = torch.ops.aten.view.default(clone_118, [48, 64, 1024]);  clone_118 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_267: "f32[768, 4096]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_50: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_12, index, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, permute_31, permute_32, permute_36, permute_37, permute_41, permute_42, permute_46, permute_47, permute_51, permute_52, permute_56, permute_57, primals_1, primals_4, primals_7, primals_8, primals_10, primals_11, primals_14, primals_16, primals_17, primals_20, primals_21, primals_23, primals_24, primals_27, primals_29, primals_30, primals_33, primals_34, primals_36, primals_37, primals_40, primals_42, primals_43, primals_46, primals_47, primals_49, primals_50, primals_53, primals_55, primals_56, primals_59, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_86, primals_88, primals_89, primals_92, primals_94, primals_95, primals_98, primals_99, primals_101, primals_102, primals_105, primals_107, primals_108, primals_111, primals_112, primals_114, primals_115, primals_118, primals_120, primals_121, primals_124, primals_125, primals_127, primals_128, primals_131, primals_133, primals_134, primals_137, primals_138, primals_140, primals_141, primals_144, primals_146, primals_147, primals_150, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_162, unsqueeze, getitem_1, mul, _softmax, getitem_8, getitem_10, mul_2, addmm_2, getitem_14, mul_8, _softmax_1, getitem_21, getitem_23, mul_10, addmm_6, getitem_27, mul_16, _softmax_2, getitem_34, getitem_36, mul_18, addmm_10, getitem_40, mul_24, _softmax_3, getitem_47, getitem_49, mul_26, addmm_14, getitem_53, mul_32, _softmax_4, getitem_60, getitem_62, mul_34, addmm_18, getitem_66, mul_40, _softmax_5, getitem_73, getitem_75, mul_42, addmm_22, getitem_79, mul_48, _softmax_6, getitem_86, getitem_88, mul_50, addmm_26, getitem_92, mul_56, _softmax_7, getitem_99, getitem_101, mul_58, addmm_30, getitem_105, mul_64, _softmax_8, getitem_112, getitem_114, mul_66, addmm_34, getitem_118, mul_72, _softmax_9, getitem_125, getitem_127, mul_74, addmm_38, getitem_131, mul_80, _softmax_10, getitem_138, getitem_140, mul_82, addmm_42, getitem_144, mul_88, _softmax_11, getitem_151, getitem_153, mul_90, addmm_46, getitem_157, mul_96, sub_25, iota_1, _log_softmax, convert_element_type_13, ne_3, eq_tensor, permute_62, div_14, permute_65, permute_67, div_15, permute_69, view_239, view_241, view_245, view_247, permute_80, div_17, permute_82, permute_84, div_18, permute_86, view_267, view_269, view_273, view_275, permute_97, div_20, permute_99, permute_101, div_21, permute_103, view_295, view_297, view_301, view_303, permute_114, div_23, permute_116, permute_118, div_24, permute_120, view_323, view_325, view_329, view_331, permute_131, div_26, permute_133, permute_135, div_27, permute_137, view_351, view_353, view_357, view_359, permute_148, div_29, permute_150, permute_152, div_30, permute_154, view_379, view_381, view_385, view_387, permute_165, div_32, permute_167, permute_169, div_33, permute_171, view_407, view_409, view_413, view_415, permute_182, div_35, permute_184, permute_186, div_36, permute_188, view_435, view_437, view_441, view_443, permute_199, div_38, permute_201, permute_203, div_39, permute_205, view_463, view_465, view_469, view_471, permute_216, div_41, permute_218, permute_220, div_42, permute_222, view_491, view_493, view_497, view_499, permute_233, div_44, permute_235, permute_237, div_45, permute_239, view_519, view_521, view_525, view_527, permute_250, div_47, permute_252, permute_254, div_48, permute_256, view_547, view_549, view_553, view_555, permute_267, div_50)
