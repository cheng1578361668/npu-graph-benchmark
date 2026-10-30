class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128]", primals_2: "f32[30522, 768]", primals_3: "i64[1, 512]", primals_5: "f32[768]", primals_7: "f32[768, 768]", primals_9: "f32[768, 768]", primals_11: "f32[768, 768]", primals_13: "f32[768, 768]", primals_15: "f32[768]", primals_17: "f32[3072, 768]", primals_19: "f32[768, 3072]", primals_21: "f32[768]", primals_23: "f32[768, 768]", primals_25: "f32[768, 768]", primals_27: "f32[768, 768]", primals_29: "f32[768, 768]", primals_31: "f32[768]", primals_33: "f32[3072, 768]", primals_35: "f32[768, 3072]", primals_37: "f32[768]", primals_39: "f32[768, 768]", primals_41: "f32[768, 768]", primals_43: "f32[768, 768]", primals_45: "f32[768, 768]", primals_47: "f32[768]", primals_49: "f32[3072, 768]", primals_51: "f32[768, 3072]", primals_53: "f32[768]", primals_55: "f32[768, 768]", primals_57: "f32[768, 768]", primals_59: "f32[768, 768]", primals_61: "f32[768, 768]", primals_63: "f32[768]", primals_65: "f32[3072, 768]", primals_67: "f32[768, 3072]", primals_69: "f32[768]", primals_71: "f32[768, 768]", primals_73: "f32[768, 768]", primals_75: "f32[768, 768]", primals_77: "f32[768, 768]", primals_79: "f32[768]", primals_81: "f32[3072, 768]", primals_83: "f32[768, 3072]", primals_85: "f32[768]", primals_87: "f32[768, 768]", primals_89: "f32[768, 768]", primals_91: "f32[768, 768]", primals_93: "f32[768, 768]", primals_95: "f32[768]", primals_97: "f32[3072, 768]", primals_99: "f32[768, 3072]", primals_101: "f32[768]", primals_103: "f32[768, 768]", primals_105: "f32[768]", primals_108: "i64[1, 128]", mul: "f32[1, 128, 768]", getitem_3: "u8[12288]", view: "f32[128, 768]", div_1: "f32[1, 12, 128, 128]", getitem_5: "u8[24576]", view_17: "f32[128, 768]", mul_2: "f32[1, 128, 768]", view_19: "f32[128, 768]", addmm_4: "f32[128, 3072]", view_21: "f32[128, 3072]", getitem_9: "u8[12288]", mul_8: "f32[1, 128, 768]", view_23: "f32[128, 768]", div_4: "f32[1, 12, 128, 128]", getitem_13: "u8[24576]", view_40: "f32[128, 768]", mul_10: "f32[1, 128, 768]", view_42: "f32[128, 768]", addmm_10: "f32[128, 3072]", view_44: "f32[128, 3072]", getitem_17: "u8[12288]", mul_16: "f32[1, 128, 768]", view_46: "f32[128, 768]", div_7: "f32[1, 12, 128, 128]", getitem_21: "u8[24576]", view_63: "f32[128, 768]", mul_18: "f32[1, 128, 768]", view_65: "f32[128, 768]", addmm_16: "f32[128, 3072]", view_67: "f32[128, 3072]", getitem_25: "u8[12288]", mul_24: "f32[1, 128, 768]", view_69: "f32[128, 768]", div_10: "f32[1, 12, 128, 128]", getitem_29: "u8[24576]", view_86: "f32[128, 768]", mul_26: "f32[1, 128, 768]", view_88: "f32[128, 768]", addmm_22: "f32[128, 3072]", view_90: "f32[128, 3072]", getitem_33: "u8[12288]", mul_32: "f32[1, 128, 768]", view_92: "f32[128, 768]", div_13: "f32[1, 12, 128, 128]", getitem_37: "u8[24576]", view_109: "f32[128, 768]", mul_34: "f32[1, 128, 768]", view_111: "f32[128, 768]", addmm_28: "f32[128, 3072]", view_113: "f32[128, 3072]", getitem_41: "u8[12288]", mul_40: "f32[1, 128, 768]", view_115: "f32[128, 768]", div_16: "f32[1, 12, 128, 128]", getitem_45: "u8[24576]", view_132: "f32[128, 768]", mul_42: "f32[1, 128, 768]", view_134: "f32[128, 768]", addmm_34: "f32[128, 3072]", view_136: "f32[128, 3072]", getitem_49: "u8[12288]", mul_48: "f32[1, 128, 768]", view_138: "f32[128, 768]", addmm_36: "f32[128, 768]", getitem_53: "f32[1, 128, 1]", rsqrt_13: "f32[1, 128, 1]", view_140: "f32[128, 768]", _log_softmax: "f32[128, 30522]", convert_element_type: "f32[]", div_22: "f32[1, 128, 1]", div_23: "f32[1, 128, 1]", view_162: "f32[12, 64, 128]", view_164: "f32[12, 128, 128]", view_168: "f32[12, 128, 64]", view_170: "f32[12, 64, 128]", div_25: "f32[1, 128, 1]", div_26: "f32[1, 128, 1]", view_196: "f32[12, 64, 128]", view_198: "f32[12, 128, 128]", view_202: "f32[12, 128, 64]", view_204: "f32[12, 64, 128]", div_28: "f32[1, 128, 1]", div_29: "f32[1, 128, 1]", view_230: "f32[12, 64, 128]", view_232: "f32[12, 128, 128]", view_236: "f32[12, 128, 64]", view_238: "f32[12, 64, 128]", div_31: "f32[1, 128, 1]", div_32: "f32[1, 128, 1]", view_264: "f32[12, 64, 128]", view_266: "f32[12, 128, 128]", view_270: "f32[12, 128, 64]", view_272: "f32[12, 64, 128]", div_34: "f32[1, 128, 1]", div_35: "f32[1, 128, 1]", view_298: "f32[12, 64, 128]", view_300: "f32[12, 128, 128]", view_304: "f32[12, 128, 64]", view_306: "f32[12, 64, 128]", div_37: "f32[1, 128, 1]", div_38: "f32[1, 128, 1]", view_332: "f32[12, 64, 128]", view_334: "f32[12, 128, 128]", view_338: "f32[12, 128, 64]", view_340: "f32[12, 64, 128]", div_40: "f32[1, 128, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 30522]"):
        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        div_20: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_143: "i64[128]" = torch.ops.aten.view.default(primals_108, [-1]);  primals_108 = None
        unsqueeze_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_143, 1);  view_143 = None
        ne_3: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default_8: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_8: "i64[128, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default_8);  unsqueeze_1 = full_default_8 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.view.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[128, 30522]" = torch.ops.aten.expand.default(where_8, [128, 30522]);  where_8 = None
        eq_tensor: "b8[128, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:920 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        where_self: "f32[128, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_9: "f32[128, 1]" = torch.ops.aten.where.self(ne_3, div_20, full_default_10);  ne_3 = div_20 = None
        mul_56: "f32[128, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_9);  where_self = where_9 = None
        exp_13: "f32[128, 30522]" = torch.ops.aten.exp.default(_log_softmax);  _log_softmax = None
        sum_9: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_56, [1], True)
        mul_57: "f32[128, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_9);  exp_13 = sum_9 = None
        sub_20: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
        view_144: "f32[1, 128, 30522]" = torch.ops.aten.view.default(sub_20, [1, 128, 30522]);  sub_20 = None
        add_55: "f32[1, 128, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_144);  tangents_2 = view_144 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:916 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_145: "f32[128, 30522]" = torch.ops.aten.view.default(add_55, [128, 30522]);  add_55 = None
        permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_68: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm: "f32[128, 768]" = torch.ops.aten.mm.default(view_145, permute_68);  permute_68 = None
        permute_69: "f32[30522, 128]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_1: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_69, view_140);  permute_69 = view_140 = None
        sum_10: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
        view_146: "f32[30522]" = torch.ops.aten.view.default(sum_10, [30522]);  sum_10 = None
        view_147: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm, [1, 128, 768]);  mm = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:915 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        mul_59: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_147, primals_105);  primals_105 = None
        mul_60: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_59, 768)
        sum_11: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_59, [2], True)

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:913 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_139: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_36, [1, 128, 768]);  addmm_36 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_50: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, view_139)
        mul_51: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, view_139)
        mul_52: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, 0.044715);  mul_51 = None
        add_51: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_139, mul_52);  mul_52 = None
        mul_53: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_51, -1.5957691216057308);  add_51 = None
        exp_12: "f32[1, 128, 768]" = torch.ops.aten.exp.default(mul_53);  mul_53 = None
        add_52: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(exp_12, 1.0);  exp_12 = None
        div_18: "f32[1, 128, 768]" = torch.ops.aten.div.Tensor(view_139, add_52);  add_52 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:915 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_19: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(div_18, getitem_53);  div_18 = getitem_53 = None
        mul_54: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_61: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_59, mul_54);  mul_59 = None
        sum_12: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [2], True);  mul_61 = None
        mul_62: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_12);  sum_12 = None
        sub_22: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_60, sum_11);  mul_60 = sum_11 = None
        sub_23: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_22, mul_62);  sub_22 = mul_62 = None
        div_21: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        mul_63: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_23);  div_21 = sub_23 = None
        mul_64: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_147, mul_54);  mul_54 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_64, [0, 1]);  mul_64 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_147, [0, 1]);  view_147 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_66: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, -0.07135481627260025)
        add_56: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_66, -1.5957691216057308);  mul_66 = None
        mul_67: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, view_139);  add_56 = None
        exp_14: "f32[1, 128, 768]" = torch.ops.aten.exp.default(mul_67);  mul_67 = None
        mul_68: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, 0.2140644488178007);  mul_50 = None
        add_57: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_68, 1.5957691216057308);  mul_68 = None
        mul_69: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_57, view_139);  add_57 = view_139 = None
        add_58: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(exp_14, 1.0)
        reciprocal: "f32[1, 128, 768]" = torch.ops.aten.reciprocal.default(add_58);  add_58 = None
        mul_71: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(exp_14, reciprocal);  exp_14 = None
        mul_72: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_71, mul_69);  mul_71 = mul_69 = None
        mul_73: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_72, reciprocal);  mul_72 = None
        isnan: "b8[1, 128, 768]" = torch.ops.aten.isnan.default(mul_73)
        full_default_11: "f32[1, 128, 768]" = torch.ops.aten.full.default([1, 128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_10: "f32[1, 128, 768]" = torch.ops.aten.where.self(isnan, full_default_11, mul_73);  isnan = full_default_11 = mul_73 = None
        add_59: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(where_10, reciprocal);  where_10 = reciprocal = None
        mul_74: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_63, add_59);  mul_63 = add_59 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:913 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_148: "f32[128, 768]" = torch.ops.aten.view.default(mul_74, [128, 768]);  mul_74 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_2: "f32[128, 768]" = torch.ops.aten.mm.default(view_148, permute_72);  permute_72 = None
        permute_73: "f32[768, 128]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_73, view_138);  permute_73 = view_138 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
        view_149: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
        view_150: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_2, [1, 128, 768]);  mm_2 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_76: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_150, primals_101);  primals_101 = None
        mul_77: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_76, 768)
        sum_16: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [2], True)
        mul_78: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_76, mul_48);  mul_76 = None
        sum_17: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_78, [2], True);  mul_78 = None
        mul_79: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_17);  sum_17 = None
        sub_25: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_77, sum_16);  mul_77 = sum_16 = None
        sub_26: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_25, mul_79);  sub_25 = mul_79 = None
        mul_80: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_26);  div_22 = sub_26 = None
        mul_81: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_150, mul_48);  mul_48 = None
        sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_81, [0, 1]);  mul_81 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_150, [0, 1]);  view_150 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_80, getitem_49, 0.10000000000000009);  getitem_49 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_151: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward, [128, 768]);  npu_dropout_backward = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_4: "f32[128, 3072]" = torch.ops.aten.mm.default(view_151, permute_76);  permute_76 = None
        permute_77: "f32[768, 128]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_136);  permute_77 = view_136 = None
        sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        view_152: "f32[768]" = torch.ops.aten.view.default(sum_20, [768]);  sum_20 = None
        view_153: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_4, [1, 128, 3072]);  mm_4 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_135: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 128, 3072]);  addmm_34 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_44: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, view_135)
        mul_83: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, -0.07135481627260025)
        add_60: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_83, -1.5957691216057308);  mul_83 = None
        mul_84: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_60, view_135);  add_60 = None
        exp_15: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_84);  mul_84 = None
        mul_85: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, 0.2140644488178007);  mul_44 = None
        add_61: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_85, 1.5957691216057308);  mul_85 = None
        mul_86: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_61, view_135);  add_61 = view_135 = None
        add_62: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_15, 1.0)
        reciprocal_1: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_62);  add_62 = None
        mul_88: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_15, reciprocal_1);  exp_15 = None
        mul_89: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_88, mul_86);  mul_88 = mul_86 = None
        mul_90: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_89, reciprocal_1);  mul_89 = None
        isnan_1: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_90)
        full_default_12: "f32[1, 128, 3072]" = torch.ops.aten.full.default([1, 128, 3072], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_11: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_1, full_default_12, mul_90);  isnan_1 = mul_90 = None
        add_63: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_11, reciprocal_1);  where_11 = reciprocal_1 = None
        mul_91: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_153, add_63);  view_153 = add_63 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_154: "f32[128, 3072]" = torch.ops.aten.view.default(mul_91, [128, 3072]);  mul_91 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_6: "f32[128, 768]" = torch.ops.aten.mm.default(view_154, permute_80);  permute_80 = None
        permute_81: "f32[3072, 128]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_134);  permute_81 = view_134 = None
        sum_21: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_155: "f32[3072]" = torch.ops.aten.view.default(sum_21, [3072]);  sum_21 = None
        view_156: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_6, [1, 128, 768]);  mm_6 = None
        add_64: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_80, view_156);  mul_80 = view_156 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_93: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, primals_95);  primals_95 = None
        mul_94: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_93, 768)
        sum_22: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True)
        mul_95: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_93, mul_42);  mul_93 = None
        sum_23: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
        mul_96: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_23);  sum_23 = None
        sub_28: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_94, sum_22);  mul_94 = sum_22 = None
        sub_29: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_28, mul_96);  sub_28 = mul_96 = None
        mul_97: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_29);  div_23 = sub_29 = None
        mul_98: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, mul_42);  mul_42 = None
        sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1]);  mul_98 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_64, [0, 1]);  add_64 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_157: "f32[128, 768]" = torch.ops.aten.view.default(mul_97, [128, 768])
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_8: "f32[128, 768]" = torch.ops.aten.mm.default(view_157, permute_84);  permute_84 = None
        permute_85: "f32[768, 128]" = torch.ops.aten.permute.default(view_157, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_85, view_132);  permute_85 = view_132 = None
        sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
        view_158: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
        view_159: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_8, [1, 128, 768]);  mm_8 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_160: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_159, [1, 128, 12, 64]);  view_159 = None
        permute_88: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_30: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_88, [1, 12, 128, 64]);  permute_88 = None
        view_161: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_30, [12, 128, 64]);  expand_30 = None
        bmm_12: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_161, view_162);  view_162 = None
        view_163: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_12, [1, 12, 128, 128]);  bmm_12 = None
        bmm_13: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_164, view_161);  view_164 = view_161 = None
        view_166: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 128, 64]);  bmm_13 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_1: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_163, getitem_45, 0.10000000000000009);  view_163 = getitem_45 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_99: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_1, div_16);  npu_dropout_backward_1 = None
        sum_27: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [-1], True)
        mul_100: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_16, sum_27);  div_16 = sum_27 = None
        sub_30: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:818 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:242 in _hf_distilbert_multiheadselfattention_forward_new, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)
        eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
        view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq, [1, 1, 1, 128]);  eq = None
        expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_12: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_30);  sub_30 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_34: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_12, [1, 12, 128, 128]);  where_12 = None
        view_167: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_34, [12, 128, 128]);  expand_34 = None
        bmm_14: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_167, view_168);  view_168 = None
        view_169: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_14, [1, 12, 128, 64]);  bmm_14 = None
        bmm_15: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_170, view_167);  view_170 = view_167 = None
        view_172: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_15, [1, 12, 64, 128]);  bmm_15 = None
        permute_93: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_172, [0, 1, 3, 2]);  view_172 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_24: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_169, 8.0);  view_169 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_94: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_6: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_173: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_6, [1, 128, 768]);  clone_6 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_174: "f32[128, 768]" = torch.ops.aten.view.default(view_173, [128, 768]);  view_173 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_10: "f32[128, 768]" = torch.ops.aten.mm.default(view_174, permute_95);  permute_95 = None
        permute_96: "f32[768, 128]" = torch.ops.aten.permute.default(view_174, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_96, view_115);  permute_96 = None
        sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
        view_175: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
        view_176: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_10, [1, 128, 768]);  mm_10 = None
        add_65: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_97, view_176);  mul_97 = view_176 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_99: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_93, [0, 2, 1, 3]);  permute_93 = None
        view_177: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_99, [1, 128, 768]);  permute_99 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_178: "f32[128, 768]" = torch.ops.aten.view.default(view_177, [128, 768]);  view_177 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_12: "f32[128, 768]" = torch.ops.aten.mm.default(view_178, permute_100);  permute_100 = None
        permute_101: "f32[768, 128]" = torch.ops.aten.permute.default(view_178, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_101, view_115);  permute_101 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_178, [0], True);  view_178 = None
        view_179: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
        view_180: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_12, [1, 128, 768]);  mm_12 = None
        add_66: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_65, view_180);  add_65 = view_180 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_104: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_24, [0, 2, 1, 3]);  div_24 = None
        clone_7: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_181: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_7, [1, 128, 768]);  clone_7 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_182: "f32[128, 768]" = torch.ops.aten.view.default(view_181, [128, 768]);  view_181 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_14: "f32[128, 768]" = torch.ops.aten.mm.default(view_182, permute_105);  permute_105 = None
        permute_106: "f32[768, 128]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_106, view_115);  permute_106 = view_115 = None
        sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
        view_183: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
        view_184: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_14, [1, 128, 768]);  mm_14 = None
        add_67: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_66, view_184);  add_66 = view_184 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_102: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, primals_85);  primals_85 = None
        mul_103: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_102, 768)
        sum_31: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
        mul_104: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_102, mul_40);  mul_102 = None
        sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
        mul_105: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_32);  sum_32 = None
        sub_32: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_103, sum_31);  mul_103 = sum_31 = None
        sub_33: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_105);  sub_32 = mul_105 = None
        mul_106: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_33);  div_25 = sub_33 = None
        mul_107: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, mul_40);  mul_40 = None
        sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_107, [0, 1]);  mul_107 = None
        sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_67, [0, 1]);  add_67 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward_2: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_106, getitem_41, 0.10000000000000009);  getitem_41 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_185: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward_2, [128, 768]);  npu_dropout_backward_2 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_16: "f32[128, 3072]" = torch.ops.aten.mm.default(view_185, permute_109);  permute_109 = None
        permute_110: "f32[768, 128]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_110, view_113);  permute_110 = view_113 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        view_186: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
        view_187: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_16, [1, 128, 3072]);  mm_16 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_112: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 128, 3072]);  addmm_28 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_36: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
        mul_109: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, -0.07135481627260025)
        add_68: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_109, -1.5957691216057308);  mul_109 = None
        mul_110: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_68, view_112);  add_68 = None
        exp_16: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_110);  mul_110 = None
        mul_111: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, 0.2140644488178007);  mul_36 = None
        add_69: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_111, 1.5957691216057308);  mul_111 = None
        mul_112: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_69, view_112);  add_69 = view_112 = None
        add_70: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_16, 1.0)
        reciprocal_2: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_70);  add_70 = None
        mul_114: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_16, reciprocal_2);  exp_16 = None
        mul_115: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_114, mul_112);  mul_114 = mul_112 = None
        mul_116: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_115, reciprocal_2);  mul_115 = None
        isnan_2: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_116)
        where_13: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_2, full_default_12, mul_116);  isnan_2 = mul_116 = None
        add_71: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_13, reciprocal_2);  where_13 = reciprocal_2 = None
        mul_117: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_187, add_71);  view_187 = add_71 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_188: "f32[128, 3072]" = torch.ops.aten.view.default(mul_117, [128, 3072]);  mul_117 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_18: "f32[128, 768]" = torch.ops.aten.mm.default(view_188, permute_113);  permute_113 = None
        permute_114: "f32[3072, 128]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_114, view_111);  permute_114 = view_111 = None
        sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        view_189: "f32[3072]" = torch.ops.aten.view.default(sum_36, [3072]);  sum_36 = None
        view_190: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_18, [1, 128, 768]);  mm_18 = None
        add_72: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_106, view_190);  mul_106 = view_190 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_119: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_72, primals_79);  primals_79 = None
        mul_120: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_119, 768)
        sum_37: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True)
        mul_121: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_119, mul_34);  mul_119 = None
        sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
        mul_122: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_38);  sum_38 = None
        sub_35: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_120, sum_37);  mul_120 = sum_37 = None
        sub_36: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_122);  sub_35 = mul_122 = None
        mul_123: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_36);  div_26 = sub_36 = None
        mul_124: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_72, mul_34);  mul_34 = None
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1]);  mul_124 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_72, [0, 1]);  add_72 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_191: "f32[128, 768]" = torch.ops.aten.view.default(mul_123, [128, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_20: "f32[128, 768]" = torch.ops.aten.mm.default(view_191, permute_117);  permute_117 = None
        permute_118: "f32[768, 128]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_118, view_109);  permute_118 = view_109 = None
        sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
        view_192: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
        view_193: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_20, [1, 128, 768]);  mm_20 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_194: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_193, [1, 128, 12, 64]);  view_193 = None
        permute_121: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_38: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_121, [1, 12, 128, 64]);  permute_121 = None
        view_195: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_38, [12, 128, 64]);  expand_38 = None
        bmm_16: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_195, view_196);  view_196 = None
        view_197: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_16, [1, 12, 128, 128]);  bmm_16 = None
        bmm_17: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_198, view_195);  view_198 = view_195 = None
        view_200: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 128, 64]);  bmm_17 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_3: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_197, getitem_37, 0.10000000000000009);  view_197 = getitem_37 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_125: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_3, div_13);  npu_dropout_backward_3 = None
        sum_42: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [-1], True)
        mul_126: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_13, sum_42);  div_13 = sum_42 = None
        sub_37: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_14: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_37);  sub_37 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_42: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_14, [1, 12, 128, 128]);  where_14 = None
        view_201: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_42, [12, 128, 128]);  expand_42 = None
        bmm_18: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_201, view_202);  view_202 = None
        view_203: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_18, [1, 12, 128, 64]);  bmm_18 = None
        bmm_19: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_204, view_201);  view_204 = view_201 = None
        view_206: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_19, [1, 12, 64, 128]);  bmm_19 = None
        permute_126: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_206, [0, 1, 3, 2]);  view_206 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_27: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_203, 8.0);  view_203 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_127: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_8: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_207: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_8, [1, 128, 768]);  clone_8 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_208: "f32[128, 768]" = torch.ops.aten.view.default(view_207, [128, 768]);  view_207 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_22: "f32[128, 768]" = torch.ops.aten.mm.default(view_208, permute_128);  permute_128 = None
        permute_129: "f32[768, 128]" = torch.ops.aten.permute.default(view_208, [1, 0])
        mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_129, view_92);  permute_129 = None
        sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
        view_209: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
        view_210: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_22, [1, 128, 768]);  mm_22 = None
        add_73: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_123, view_210);  mul_123 = view_210 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_132: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_126, [0, 2, 1, 3]);  permute_126 = None
        view_211: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_132, [1, 128, 768]);  permute_132 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_212: "f32[128, 768]" = torch.ops.aten.view.default(view_211, [128, 768]);  view_211 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_24: "f32[128, 768]" = torch.ops.aten.mm.default(view_212, permute_133);  permute_133 = None
        permute_134: "f32[768, 128]" = torch.ops.aten.permute.default(view_212, [1, 0])
        mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_134, view_92);  permute_134 = None
        sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
        view_213: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
        view_214: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_24, [1, 128, 768]);  mm_24 = None
        add_74: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_73, view_214);  add_73 = view_214 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_137: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_27, [0, 2, 1, 3]);  div_27 = None
        clone_9: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_215: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_9, [1, 128, 768]);  clone_9 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_216: "f32[128, 768]" = torch.ops.aten.view.default(view_215, [128, 768]);  view_215 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_26: "f32[128, 768]" = torch.ops.aten.mm.default(view_216, permute_138);  permute_138 = None
        permute_139: "f32[768, 128]" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_92);  permute_139 = view_92 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
        view_217: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
        view_218: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_26, [1, 128, 768]);  mm_26 = None
        add_75: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_74, view_218);  add_74 = view_218 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_128: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_75, primals_69);  primals_69 = None
        mul_129: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_128, 768)
        sum_46: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [2], True)
        mul_130: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_128, mul_32);  mul_128 = None
        sum_47: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True);  mul_130 = None
        mul_131: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_47);  sum_47 = None
        sub_39: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_129, sum_46);  mul_129 = sum_46 = None
        sub_40: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_131);  sub_39 = mul_131 = None
        mul_132: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_40);  div_28 = sub_40 = None
        mul_133: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_75, mul_32);  mul_32 = None
        sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_133, [0, 1]);  mul_133 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_75, [0, 1]);  add_75 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward_4: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_132, getitem_33, 0.10000000000000009);  getitem_33 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_219: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward_4, [128, 768]);  npu_dropout_backward_4 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_28: "f32[128, 3072]" = torch.ops.aten.mm.default(view_219, permute_142);  permute_142 = None
        permute_143: "f32[768, 128]" = torch.ops.aten.permute.default(view_219, [1, 0])
        mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_90);  permute_143 = view_90 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
        view_220: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
        view_221: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_28, [1, 128, 3072]);  mm_28 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_89: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 128, 3072]);  addmm_22 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_28: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, view_89)
        mul_135: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, -0.07135481627260025)
        add_76: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_135, -1.5957691216057308);  mul_135 = None
        mul_136: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_76, view_89);  add_76 = None
        exp_17: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_136);  mul_136 = None
        mul_137: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, 0.2140644488178007);  mul_28 = None
        add_77: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_137, 1.5957691216057308);  mul_137 = None
        mul_138: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_77, view_89);  add_77 = view_89 = None
        add_78: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_17, 1.0)
        reciprocal_3: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_78);  add_78 = None
        mul_140: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_17, reciprocal_3);  exp_17 = None
        mul_141: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_140, mul_138);  mul_140 = mul_138 = None
        mul_142: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_141, reciprocal_3);  mul_141 = None
        isnan_3: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_142)
        where_15: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_3, full_default_12, mul_142);  isnan_3 = mul_142 = None
        add_79: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_15, reciprocal_3);  where_15 = reciprocal_3 = None
        mul_143: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_221, add_79);  view_221 = add_79 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_222: "f32[128, 3072]" = torch.ops.aten.view.default(mul_143, [128, 3072]);  mul_143 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_30: "f32[128, 768]" = torch.ops.aten.mm.default(view_222, permute_146);  permute_146 = None
        permute_147: "f32[3072, 128]" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_88);  permute_147 = view_88 = None
        sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        view_223: "f32[3072]" = torch.ops.aten.view.default(sum_51, [3072]);  sum_51 = None
        view_224: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_30, [1, 128, 768]);  mm_30 = None
        add_80: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_132, view_224);  mul_132 = view_224 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_145: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, primals_63);  primals_63 = None
        mul_146: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_145, 768)
        sum_52: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_145, mul_26);  mul_145 = None
        sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_53);  sum_53 = None
        sub_42: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_146, sum_52);  mul_146 = sum_52 = None
        sub_43: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_148);  sub_42 = mul_148 = None
        mul_149: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_43);  div_29 = sub_43 = None
        mul_150: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, mul_26);  mul_26 = None
        sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1]);  add_80 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_225: "f32[128, 768]" = torch.ops.aten.view.default(mul_149, [128, 768])
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_32: "f32[128, 768]" = torch.ops.aten.mm.default(view_225, permute_150);  permute_150 = None
        permute_151: "f32[768, 128]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_86);  permute_151 = view_86 = None
        sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        view_226: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
        view_227: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_32, [1, 128, 768]);  mm_32 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_228: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_227, [1, 128, 12, 64]);  view_227 = None
        permute_154: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_46: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_154, [1, 12, 128, 64]);  permute_154 = None
        view_229: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_46, [12, 128, 64]);  expand_46 = None
        bmm_20: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230);  view_230 = None
        view_231: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_20, [1, 12, 128, 128]);  bmm_20 = None
        bmm_21: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_232, view_229);  view_232 = view_229 = None
        view_234: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 128, 64]);  bmm_21 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_5: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_231, getitem_29, 0.10000000000000009);  view_231 = getitem_29 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_151: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_5, div_10);  npu_dropout_backward_5 = None
        sum_57: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
        mul_152: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_10, sum_57);  div_10 = sum_57 = None
        sub_44: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_16: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_44);  sub_44 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_50: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_16, [1, 12, 128, 128]);  where_16 = None
        view_235: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_50, [12, 128, 128]);  expand_50 = None
        bmm_22: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_235, view_236);  view_236 = None
        view_237: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_22, [1, 12, 128, 64]);  bmm_22 = None
        bmm_23: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_238, view_235);  view_238 = view_235 = None
        view_240: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_23, [1, 12, 64, 128]);  bmm_23 = None
        permute_159: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_240, [0, 1, 3, 2]);  view_240 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_30: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_237, 8.0);  view_237 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_160: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_10: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_241: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_10, [1, 128, 768]);  clone_10 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_242: "f32[128, 768]" = torch.ops.aten.view.default(view_241, [128, 768]);  view_241 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_34: "f32[128, 768]" = torch.ops.aten.mm.default(view_242, permute_161);  permute_161 = None
        permute_162: "f32[768, 128]" = torch.ops.aten.permute.default(view_242, [1, 0])
        mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_69);  permute_162 = None
        sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
        view_243: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
        view_244: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_34, [1, 128, 768]);  mm_34 = None
        add_81: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_149, view_244);  mul_149 = view_244 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_165: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_245: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_165, [1, 128, 768]);  permute_165 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_246: "f32[128, 768]" = torch.ops.aten.view.default(view_245, [128, 768]);  view_245 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_36: "f32[128, 768]" = torch.ops.aten.mm.default(view_246, permute_166);  permute_166 = None
        permute_167: "f32[768, 128]" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_69);  permute_167 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        view_247: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
        view_248: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_36, [1, 128, 768]);  mm_36 = None
        add_82: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_248);  add_81 = view_248 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_170: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_30, [0, 2, 1, 3]);  div_30 = None
        clone_11: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_249: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_11, [1, 128, 768]);  clone_11 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_250: "f32[128, 768]" = torch.ops.aten.view.default(view_249, [128, 768]);  view_249 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_38: "f32[128, 768]" = torch.ops.aten.mm.default(view_250, permute_171);  permute_171 = None
        permute_172: "f32[768, 128]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_69);  permute_172 = view_69 = None
        sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        view_251: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
        view_252: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_38, [1, 128, 768]);  mm_38 = None
        add_83: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_82, view_252);  add_82 = view_252 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_154: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, primals_53);  primals_53 = None
        mul_155: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_154, 768)
        sum_61: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
        mul_156: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_24);  mul_154 = None
        sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
        mul_157: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_62);  sum_62 = None
        sub_46: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_155, sum_61);  mul_155 = sum_61 = None
        sub_47: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_46, mul_157);  sub_46 = mul_157 = None
        mul_158: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_47);  div_31 = sub_47 = None
        mul_159: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, mul_24);  mul_24 = None
        sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_83, [0, 1]);  add_83 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward_6: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_158, getitem_25, 0.10000000000000009);  getitem_25 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_253: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward_6, [128, 768]);  npu_dropout_backward_6 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_40: "f32[128, 3072]" = torch.ops.aten.mm.default(view_253, permute_175);  permute_175 = None
        permute_176: "f32[768, 128]" = torch.ops.aten.permute.default(view_253, [1, 0])
        mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_67);  permute_176 = view_67 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True);  view_253 = None
        view_254: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
        view_255: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_40, [1, 128, 3072]);  mm_40 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_66: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 128, 3072]);  addmm_16 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, view_66)
        mul_161: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, -0.07135481627260025)
        add_84: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_161, -1.5957691216057308);  mul_161 = None
        mul_162: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_84, view_66);  add_84 = None
        exp_18: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_162);  mul_162 = None
        mul_163: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, 0.2140644488178007);  mul_20 = None
        add_85: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_163, 1.5957691216057308);  mul_163 = None
        mul_164: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_85, view_66);  add_85 = view_66 = None
        add_86: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_18, 1.0)
        reciprocal_4: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_166: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_18, reciprocal_4);  exp_18 = None
        mul_167: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_166, mul_164);  mul_166 = mul_164 = None
        mul_168: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_167, reciprocal_4);  mul_167 = None
        isnan_4: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_168)
        where_17: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_4, full_default_12, mul_168);  isnan_4 = mul_168 = None
        add_87: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_17, reciprocal_4);  where_17 = reciprocal_4 = None
        mul_169: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_255, add_87);  view_255 = add_87 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_256: "f32[128, 3072]" = torch.ops.aten.view.default(mul_169, [128, 3072]);  mul_169 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_256, permute_179);  permute_179 = None
        permute_180: "f32[3072, 128]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_65);  permute_180 = view_65 = None
        sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        view_257: "f32[3072]" = torch.ops.aten.view.default(sum_66, [3072]);  sum_66 = None
        view_258: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_42, [1, 128, 768]);  mm_42 = None
        add_88: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_158, view_258);  mul_158 = view_258 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_171: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, primals_47);  primals_47 = None
        mul_172: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
        sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
        mul_173: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_18);  mul_171 = None
        sum_68: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
        mul_174: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_68);  sum_68 = None
        sub_49: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_67);  mul_172 = sum_67 = None
        sub_50: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_49, mul_174);  sub_49 = mul_174 = None
        mul_175: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_50);  div_32 = sub_50 = None
        mul_176: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_18);  mul_18 = None
        sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_259: "f32[128, 768]" = torch.ops.aten.view.default(mul_175, [128, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_259, permute_183);  permute_183 = None
        permute_184: "f32[768, 128]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_63);  permute_184 = view_63 = None
        sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True);  view_259 = None
        view_260: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
        view_261: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_44, [1, 128, 768]);  mm_44 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_262: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_261, [1, 128, 12, 64]);  view_261 = None
        permute_187: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_54: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_187, [1, 12, 128, 64]);  permute_187 = None
        view_263: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_54, [12, 128, 64]);  expand_54 = None
        bmm_24: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_263, view_264);  view_264 = None
        view_265: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_24, [1, 12, 128, 128]);  bmm_24 = None
        bmm_25: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_266, view_263);  view_266 = view_263 = None
        view_268: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_25, [1, 12, 128, 64]);  bmm_25 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_7: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_265, getitem_21, 0.10000000000000009);  view_265 = getitem_21 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_177: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_7, div_7);  npu_dropout_backward_7 = None
        sum_72: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [-1], True)
        mul_178: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_7, sum_72);  div_7 = sum_72 = None
        sub_51: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_18: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_51);  sub_51 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_58: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_18, [1, 12, 128, 128]);  where_18 = None
        view_269: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_58, [12, 128, 128]);  expand_58 = None
        bmm_26: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_269, view_270);  view_270 = None
        view_271: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_26, [1, 12, 128, 64]);  bmm_26 = None
        bmm_27: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_272, view_269);  view_272 = view_269 = None
        view_274: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_27, [1, 12, 64, 128]);  bmm_27 = None
        permute_192: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_274, [0, 1, 3, 2]);  view_274 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_33: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_271, 8.0);  view_271 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_193: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_12: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_275: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_12, [1, 128, 768]);  clone_12 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_276: "f32[128, 768]" = torch.ops.aten.view.default(view_275, [128, 768]);  view_275 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_276, permute_194);  permute_194 = None
        permute_195: "f32[768, 128]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_46);  permute_195 = None
        sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
        view_278: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_46, [1, 128, 768]);  mm_46 = None
        add_89: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_175, view_278);  mul_175 = view_278 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_198: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_279: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_198, [1, 128, 768]);  permute_198 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_280: "f32[128, 768]" = torch.ops.aten.view.default(view_279, [128, 768]);  view_279 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_280, permute_199);  permute_199 = None
        permute_200: "f32[768, 128]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_46);  permute_200 = None
        sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        view_281: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
        view_282: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_48, [1, 128, 768]);  mm_48 = None
        add_90: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_89, view_282);  add_89 = view_282 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_203: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_33, [0, 2, 1, 3]);  div_33 = None
        clone_13: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_283: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_13, [1, 128, 768]);  clone_13 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_284: "f32[128, 768]" = torch.ops.aten.view.default(view_283, [128, 768]);  view_283 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_50: "f32[128, 768]" = torch.ops.aten.mm.default(view_284, permute_204);  permute_204 = None
        permute_205: "f32[768, 128]" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_46);  permute_205 = view_46 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
        view_285: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
        view_286: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_50, [1, 128, 768]);  mm_50 = None
        add_91: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_90, view_286);  add_90 = view_286 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_180: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, primals_37);  primals_37 = None
        mul_181: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_180, 768)
        sum_76: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
        mul_182: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_180, mul_16);  mul_180 = None
        sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
        mul_183: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_77);  sum_77 = None
        sub_53: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_181, sum_76);  mul_181 = sum_76 = None
        sub_54: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_183);  sub_53 = mul_183 = None
        mul_184: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_54);  div_34 = sub_54 = None
        mul_185: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, mul_16);  mul_16 = None
        sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_91, [0, 1]);  add_91 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward_8: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_184, getitem_17, 0.10000000000000009);  getitem_17 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_287: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward_8, [128, 768]);  npu_dropout_backward_8 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_52: "f32[128, 3072]" = torch.ops.aten.mm.default(view_287, permute_208);  permute_208 = None
        permute_209: "f32[768, 128]" = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_44);  permute_209 = view_44 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        view_288: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
        view_289: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_52, [1, 128, 3072]);  mm_52 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_43: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 128, 3072]);  addmm_10 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
        mul_187: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, -0.07135481627260025)
        add_92: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_187, -1.5957691216057308);  mul_187 = None
        mul_188: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_92, view_43);  add_92 = None
        exp_19: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_188);  mul_188 = None
        mul_189: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, 0.2140644488178007);  mul_12 = None
        add_93: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_189, 1.5957691216057308);  mul_189 = None
        mul_190: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_93, view_43);  add_93 = view_43 = None
        add_94: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_19, 1.0)
        reciprocal_5: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_94);  add_94 = None
        mul_192: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_19, reciprocal_5);  exp_19 = None
        mul_193: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_192, mul_190);  mul_192 = mul_190 = None
        mul_194: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_193, reciprocal_5);  mul_193 = None
        isnan_5: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_194)
        where_19: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_5, full_default_12, mul_194);  isnan_5 = mul_194 = None
        add_95: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_19, reciprocal_5);  where_19 = reciprocal_5 = None
        mul_195: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_289, add_95);  view_289 = add_95 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_290: "f32[128, 3072]" = torch.ops.aten.view.default(mul_195, [128, 3072]);  mul_195 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_290, permute_212);  permute_212 = None
        permute_213: "f32[3072, 128]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_42);  permute_213 = view_42 = None
        sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        view_291: "f32[3072]" = torch.ops.aten.view.default(sum_81, [3072]);  sum_81 = None
        view_292: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_54, [1, 128, 768]);  mm_54 = None
        add_96: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_184, view_292);  mul_184 = view_292 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_197: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_96, primals_31);  primals_31 = None
        mul_198: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
        sum_82: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_10);  mul_197 = None
        sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_83);  sum_83 = None
        sub_56: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_82);  mul_198 = sum_82 = None
        sub_57: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_200);  sub_56 = mul_200 = None
        mul_201: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_57);  div_35 = sub_57 = None
        mul_202: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_96, mul_10);  mul_10 = None
        sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_96, [0, 1]);  add_96 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_293: "f32[128, 768]" = torch.ops.aten.view.default(mul_201, [128, 768])
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_293, permute_216);  permute_216 = None
        permute_217: "f32[768, 128]" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_40);  permute_217 = view_40 = None
        sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        view_294: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
        view_295: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_56, [1, 128, 768]);  mm_56 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_296: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_295, [1, 128, 12, 64]);  view_295 = None
        permute_220: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_62: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_220, [1, 12, 128, 64]);  permute_220 = None
        view_297: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_62, [12, 128, 64]);  expand_62 = None
        bmm_28: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298);  view_298 = None
        view_299: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_28, [1, 12, 128, 128]);  bmm_28 = None
        bmm_29: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_300, view_297);  view_300 = view_297 = None
        view_302: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_29, [1, 12, 128, 64]);  bmm_29 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_9: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_299, getitem_13, 0.10000000000000009);  view_299 = getitem_13 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_203: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_9, div_4);  npu_dropout_backward_9 = None
        sum_87: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [-1], True)
        mul_204: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_4, sum_87);  div_4 = sum_87 = None
        sub_58: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_20: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_58);  sub_58 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_66: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_20, [1, 12, 128, 128]);  where_20 = None
        view_303: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_66, [12, 128, 128]);  expand_66 = None
        bmm_30: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_303, view_304);  view_304 = None
        view_305: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_30, [1, 12, 128, 64]);  bmm_30 = None
        bmm_31: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_306, view_303);  view_306 = view_303 = None
        view_308: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_31, [1, 12, 64, 128]);  bmm_31 = None
        permute_225: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_308, [0, 1, 3, 2]);  view_308 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_36: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_305, 8.0);  view_305 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_226: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_14: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_309: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_14, [1, 128, 768]);  clone_14 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_310: "f32[128, 768]" = torch.ops.aten.view.default(view_309, [128, 768]);  view_309 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_58: "f32[128, 768]" = torch.ops.aten.mm.default(view_310, permute_227);  permute_227 = None
        permute_228: "f32[768, 128]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_23);  permute_228 = None
        sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
        view_312: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_58, [1, 128, 768]);  mm_58 = None
        add_97: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_201, view_312);  mul_201 = view_312 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_231: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_313: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_231, [1, 128, 768]);  permute_231 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_314: "f32[128, 768]" = torch.ops.aten.view.default(view_313, [128, 768]);  view_313 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_60: "f32[128, 768]" = torch.ops.aten.mm.default(view_314, permute_232);  permute_232 = None
        permute_233: "f32[768, 128]" = torch.ops.aten.permute.default(view_314, [1, 0])
        mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_23);  permute_233 = None
        sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
        view_315: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
        view_316: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_60, [1, 128, 768]);  mm_60 = None
        add_98: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_97, view_316);  add_97 = view_316 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_236: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_36, [0, 2, 1, 3]);  div_36 = None
        clone_15: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_317: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_15, [1, 128, 768]);  clone_15 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_318: "f32[128, 768]" = torch.ops.aten.view.default(view_317, [128, 768]);  view_317 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_62: "f32[128, 768]" = torch.ops.aten.mm.default(view_318, permute_237);  permute_237 = None
        permute_238: "f32[768, 128]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_23);  permute_238 = view_23 = None
        sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
        view_320: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_62, [1, 128, 768]);  mm_62 = None
        add_99: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_98, view_320);  add_98 = view_320 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:530 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        mul_206: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_99, primals_21);  primals_21 = None
        mul_207: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_8);  mul_206 = None
        sum_92: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_92);  sum_92 = None
        sub_60: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_91);  mul_207 = sum_91 = None
        sub_61: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_209);  sub_60 = mul_209 = None
        mul_210: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_61);  div_37 = sub_61 = None
        mul_211: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_99, mul_8);  mul_8 = None
        sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_99, [0, 1]);  add_99 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:470 in ff_chunk, code: x = self.dropout(x)
        npu_dropout_backward_10: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_210, getitem_9, 0.10000000000000009);  getitem_9 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:469 in ff_chunk, code: x = self.lin2(x)
        view_321: "f32[128, 768]" = torch.ops.aten.view.default(npu_dropout_backward_10, [128, 768]);  npu_dropout_backward_10 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_64: "f32[128, 3072]" = torch.ops.aten.mm.default(view_321, permute_241);  permute_241 = None
        permute_242: "f32[768, 128]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_21);  permute_242 = view_21 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
        view_323: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_64, [1, 128, 3072]);  mm_64 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_20: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 128, 3072]);  addmm_4 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, view_20)
        mul_213: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, -0.07135481627260025)
        add_100: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_213, -1.5957691216057308);  mul_213 = None
        mul_214: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_100, view_20);  add_100 = None
        exp_20: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_214);  mul_214 = None
        mul_215: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, 0.2140644488178007);  mul_4 = None
        add_101: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_215, 1.5957691216057308);  mul_215 = None
        mul_216: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_101, view_20);  add_101 = view_20 = None
        add_102: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(exp_20, 1.0)
        reciprocal_6: "f32[1, 128, 3072]" = torch.ops.aten.reciprocal.default(add_102);  add_102 = None
        mul_218: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_20, reciprocal_6);  exp_20 = None
        mul_219: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_218, mul_216);  mul_218 = mul_216 = None
        mul_220: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_219, reciprocal_6);  mul_219 = None
        isnan_6: "b8[1, 128, 3072]" = torch.ops.aten.isnan.default(mul_220)
        where_21: "f32[1, 128, 3072]" = torch.ops.aten.where.self(isnan_6, full_default_12, mul_220);  isnan_6 = full_default_12 = mul_220 = None
        add_103: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(where_21, reciprocal_6);  where_21 = reciprocal_6 = None
        mul_221: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_323, add_103);  view_323 = add_103 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:467 in ff_chunk, code: x = self.lin1(input)
        view_324: "f32[128, 3072]" = torch.ops.aten.view.default(mul_221, [128, 3072]);  mul_221 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_66: "f32[128, 768]" = torch.ops.aten.mm.default(view_324, permute_245);  permute_245 = None
        permute_246: "f32[3072, 128]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_19);  permute_246 = view_19 = None
        sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[3072]" = torch.ops.aten.view.default(sum_96, [3072]);  sum_96 = None
        view_326: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_66, [1, 128, 768]);  mm_66 = None
        add_104: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_210, view_326);  mul_210 = view_326 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:526 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_223: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_104, primals_15);  primals_15 = None
        mul_224: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
        sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
        mul_225: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_2);  mul_223 = None
        sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
        mul_226: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_98);  sum_98 = None
        sub_63: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_97);  mul_224 = sum_97 = None
        sub_64: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_226);  sub_63 = mul_226 = None
        mul_227: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_64);  div_38 = sub_64 = None
        mul_228: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_104, mul_2);  mul_2 = None
        sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_104, [0, 1]);  add_104 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:255 in _hf_distilbert_multiheadselfattention_forward_new, code: context = self.out_lin(context)
        view_327: "f32[128, 768]" = torch.ops.aten.view.default(mul_227, [128, 768])
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_68: "f32[128, 768]" = torch.ops.aten.mm.default(view_327, permute_249);  permute_249 = None
        permute_250: "f32[768, 128]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_17);  permute_250 = view_17 = None
        sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        view_328: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
        view_329: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_68, [1, 128, 768]);  mm_68 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:234 in unshape, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        view_330: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_329, [1, 128, 12, 64]);  view_329 = None
        permute_253: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:253 in _hf_distilbert_multiheadselfattention_forward_new, code: context = torch.matmul(weights, v)
        expand_70: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_253, [1, 12, 128, 64]);  permute_253 = None
        view_331: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_70, [12, 128, 64]);  expand_70 = None
        bmm_32: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_331, view_332);  view_332 = None
        view_333: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_32, [1, 12, 128, 128]);  bmm_32 = None
        bmm_33: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_334, view_331);  view_334 = view_331 = None
        view_336: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_33, [1, 12, 128, 64]);  bmm_33 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:248 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = self.dropout(weights)
        npu_dropout_backward_11: "f32[1, 12, 128, 128]" = torch.ops.npu.npu_dropout_backward.default(view_333, getitem_5, 0.10000000000000009);  view_333 = getitem_5 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:247 in _hf_distilbert_multiheadselfattention_forward_new, code: weights = nn.functional.softmax(scores, dim=-1)
        mul_229: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_11, div_1);  npu_dropout_backward_11 = None
        sum_102: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [-1], True)
        mul_230: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(div_1, sum_102);  div_1 = sum_102 = None
        sub_65: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:243 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = scores.masked_fill(
        where_22: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_10, sub_65);  expand_2 = full_default_10 = sub_65 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:241 in _hf_distilbert_multiheadselfattention_forward_new, code: scores = torch.matmul(q, k.transpose(2, 3))
        expand_74: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(where_22, [1, 12, 128, 128]);  where_22 = None
        view_337: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_74, [12, 128, 128]);  expand_74 = None
        bmm_34: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_337, view_338);  view_338 = None
        view_339: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_34, [1, 12, 128, 64]);  bmm_34 = None
        bmm_35: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(view_340, view_337);  view_340 = view_337 = None
        view_342: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_35, [1, 12, 64, 128]);  bmm_35 = None
        permute_258: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_342, [0, 1, 3, 2]);  view_342 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:240 in _hf_distilbert_multiheadselfattention_forward_new, code: q = q / math.sqrt(dim_per_head)
        div_39: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_339, 8.0);  view_339 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_259: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        clone_16: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_343: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_16, [1, 128, 768]);  clone_16 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:238 in _hf_distilbert_multiheadselfattention_forward_new, code: v = shape(self.v_lin(value))
        view_344: "f32[128, 768]" = torch.ops.aten.view.default(view_343, [128, 768]);  view_343 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_70: "f32[128, 768]" = torch.ops.aten.mm.default(view_344, permute_260);  permute_260 = None
        permute_261: "f32[768, 128]" = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view);  permute_261 = None
        sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
        view_345: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
        view_346: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_70, [1, 128, 768]);  mm_70 = None
        add_105: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_227, view_346);  mul_227 = view_346 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_264: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_347: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_264, [1, 128, 768]);  permute_264 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:237 in _hf_distilbert_multiheadselfattention_forward_new, code: k = shape(self.k_lin(key))
        view_348: "f32[128, 768]" = torch.ops.aten.view.default(view_347, [128, 768]);  view_347 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_72: "f32[128, 768]" = torch.ops.aten.mm.default(view_348, permute_265);  permute_265 = None
        permute_266: "f32[768, 128]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view);  permute_266 = None
        sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
        view_349: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
        view_350: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_72, [1, 128, 768]);  mm_72 = None
        add_106: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_105, view_350);  add_105 = view_350 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:231 in shape, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_269: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_39, [0, 2, 1, 3]);  div_39 = None
        clone_17: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_351: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_17, [1, 128, 768]);  clone_17 = None

        # Backward of forward node: File: /opt/ascend_pytorch/pytorch/benchmarks/torchbench/npu_support.py:236 in _hf_distilbert_multiheadselfattention_forward_new, code: q = shape(self.q_lin(query))
        view_352: "f32[128, 768]" = torch.ops.aten.view.default(view_351, [128, 768]);  view_351 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_74: "f32[128, 768]" = torch.ops.aten.mm.default(view_352, permute_270);  permute_270 = None
        permute_271: "f32[768, 128]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view);  permute_271 = view = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
        view_354: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_74, [1, 128, 768]);  mm_74 = None
        add_107: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_106, view_354);  add_106 = view_354 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:158 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        npu_dropout_backward_12: "f32[1, 128, 768]" = torch.ops.npu.npu_dropout_backward.default(add_107, getitem_3, 0.10000000000000009);  add_107 = getitem_3 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:157 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        mul_232: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_12, primals_5);  primals_5 = None
        mul_233: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_232, 768)
        sum_106: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True)
        mul_234: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_232, mul);  mul_232 = None
        sum_107: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True);  mul_234 = None
        mul_235: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, sum_107);  sum_107 = None
        sub_67: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_233, sum_106);  mul_233 = sum_106 = None
        sub_68: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_235);  sub_67 = mul_235 = None
        mul_236: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_68);  div_40 = sub_68 = None
        mul_237: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_12, mul);  mul = None
        sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1]);  mul_237 = None
        sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(npu_dropout_backward_12, [0, 1]);  npu_dropout_backward_12 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:149 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:154 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_dense_backward: "f32[512, 768]" = torch.ops.aten.embedding_dense_backward.default(mul_236, slice_1, 512, -1, False);  slice_1 = None

        # Backward of forward node: File: /opt/npu-envs/tnpu-2.14.0/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:141 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding_dense_backward_1: "f32[30522, 768]" = torch.ops.aten.embedding_dense_backward.default(mul_236, primals_1, 30522, 0, False);  mul_236 = primals_1 = None
        add_108: "f32[30522, 768]" = torch.ops.aten.add.Tensor(mm_1, embedding_dense_backward_1);  mm_1 = embedding_dense_backward_1 = None
        return (None, add_108, None, embedding_dense_backward, sum_108, sum_109, mm_75, view_353, mm_73, view_349, mm_71, view_345, mm_69, view_328, sum_99, sum_100, mm_67, view_325, mm_65, view_322, sum_93, sum_94, mm_63, view_319, mm_61, view_315, mm_59, view_311, mm_57, view_294, sum_84, sum_85, mm_55, view_291, mm_53, view_288, sum_78, sum_79, mm_51, view_285, mm_49, view_281, mm_47, view_277, mm_45, view_260, sum_69, sum_70, mm_43, view_257, mm_41, view_254, sum_63, sum_64, mm_39, view_251, mm_37, view_247, mm_35, view_243, mm_33, view_226, sum_54, sum_55, mm_31, view_223, mm_29, view_220, sum_48, sum_49, mm_27, view_217, mm_25, view_213, mm_23, view_209, mm_21, view_192, sum_39, sum_40, mm_19, view_189, mm_17, view_186, sum_33, sum_34, mm_15, view_183, mm_13, view_179, mm_11, view_175, mm_9, view_158, sum_24, sum_25, mm_7, view_155, mm_5, view_152, sum_18, sum_19, mm_3, view_149, sum_13, sum_14, view_146, None)
