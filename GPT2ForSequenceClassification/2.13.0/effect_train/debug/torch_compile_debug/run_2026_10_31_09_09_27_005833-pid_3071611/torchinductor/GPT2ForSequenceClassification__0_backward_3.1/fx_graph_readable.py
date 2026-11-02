class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 1024]", primals_4: "f32[768]", primals_7: "f32[768, 2304]", primals_8: "b8[1, 1, 1024, 1024]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_14: "f32[768, 3072]", primals_16: "f32[3072, 768]", primals_17: "f32[768]", primals_20: "f32[768, 2304]", primals_21: "b8[1, 1, 1024, 1024]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_27: "f32[768, 3072]", primals_29: "f32[3072, 768]", primals_30: "f32[768]", primals_33: "f32[768, 2304]", primals_34: "b8[1, 1, 1024, 1024]", primals_36: "f32[768, 768]", primals_37: "f32[768]", primals_40: "f32[768, 3072]", primals_42: "f32[3072, 768]", primals_43: "f32[768]", primals_46: "f32[768, 2304]", primals_47: "b8[1, 1, 1024, 1024]", primals_49: "f32[768, 768]", primals_50: "f32[768]", primals_53: "f32[768, 3072]", primals_55: "f32[3072, 768]", primals_56: "f32[768]", primals_59: "f32[768, 2304]", primals_60: "b8[1, 1, 1024, 1024]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_66: "f32[768, 3072]", primals_68: "f32[3072, 768]", primals_69: "f32[768]", primals_72: "f32[768, 2304]", primals_73: "b8[1, 1, 1024, 1024]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_79: "f32[768, 3072]", primals_81: "f32[3072, 768]", primals_82: "f32[768]", primals_85: "f32[768, 2304]", primals_86: "b8[1, 1, 1024, 1024]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_92: "f32[768, 3072]", primals_94: "f32[3072, 768]", primals_95: "f32[768]", primals_98: "f32[768, 2304]", primals_99: "b8[1, 1, 1024, 1024]", primals_101: "f32[768, 768]", primals_102: "f32[768]", primals_105: "f32[768, 3072]", primals_107: "f32[3072, 768]", primals_108: "f32[768]", primals_111: "f32[768, 2304]", primals_112: "b8[1, 1, 1024, 1024]", primals_114: "f32[768, 768]", primals_115: "f32[768]", primals_118: "f32[768, 3072]", primals_120: "f32[3072, 768]", primals_121: "f32[768]", primals_124: "f32[768, 2304]", primals_125: "b8[1, 1, 1024, 1024]", primals_127: "f32[768, 768]", primals_128: "f32[768]", primals_131: "f32[768, 3072]", primals_133: "f32[3072, 768]", primals_134: "f32[768]", primals_137: "f32[768, 2304]", primals_138: "b8[1, 1, 1024, 1024]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_144: "f32[768, 3072]", primals_146: "f32[3072, 768]", primals_147: "f32[768]", primals_150: "f32[768, 2304]", primals_151: "b8[1, 1, 1024, 1024]", primals_153: "f32[768, 768]", primals_154: "f32[768]", primals_157: "f32[768, 3072]", primals_159: "f32[3072, 768]", primals_160: "f32[768]", primals_162: "f32[2, 768]", unsqueeze: "i64[1, 1024]", getitem_1: "u8[393216]", mul: "f32[4, 1024, 768]", _softmax: "f32[4, 12, 1024, 1024]", getitem_8: "u8[6291456]", getitem_10: "u8[393216]", mul_2: "f32[4, 1024, 768]", addmm_2: "f32[4096, 3072]", getitem_14: "u8[393216]", mul_8: "f32[4, 1024, 768]", _softmax_1: "f32[4, 12, 1024, 1024]", getitem_21: "u8[6291456]", getitem_23: "u8[393216]", mul_10: "f32[4, 1024, 768]", addmm_6: "f32[4096, 3072]", getitem_27: "u8[393216]", mul_16: "f32[4, 1024, 768]", _softmax_2: "f32[4, 12, 1024, 1024]", getitem_34: "u8[6291456]", getitem_36: "u8[393216]", mul_18: "f32[4, 1024, 768]", addmm_10: "f32[4096, 3072]", getitem_40: "u8[393216]", mul_24: "f32[4, 1024, 768]", _softmax_3: "f32[4, 12, 1024, 1024]", getitem_47: "u8[6291456]", getitem_49: "u8[393216]", mul_26: "f32[4, 1024, 768]", addmm_14: "f32[4096, 3072]", getitem_53: "u8[393216]", mul_32: "f32[4, 1024, 768]", _softmax_4: "f32[4, 12, 1024, 1024]", getitem_60: "u8[6291456]", getitem_62: "u8[393216]", mul_34: "f32[4, 1024, 768]", addmm_18: "f32[4096, 3072]", getitem_66: "u8[393216]", mul_40: "f32[4, 1024, 768]", _softmax_5: "f32[4, 12, 1024, 1024]", getitem_73: "u8[6291456]", getitem_75: "u8[393216]", mul_42: "f32[4, 1024, 768]", addmm_22: "f32[4096, 3072]", getitem_79: "u8[393216]", mul_48: "f32[4, 1024, 768]", _softmax_6: "f32[4, 12, 1024, 1024]", getitem_86: "u8[6291456]", getitem_88: "u8[393216]", mul_50: "f32[4, 1024, 768]", addmm_26: "f32[4096, 3072]", getitem_92: "u8[393216]", mul_56: "f32[4, 1024, 768]", _softmax_7: "f32[4, 12, 1024, 1024]", getitem_99: "u8[6291456]", getitem_101: "u8[393216]", mul_58: "f32[4, 1024, 768]", addmm_30: "f32[4096, 3072]", getitem_105: "u8[393216]", mul_64: "f32[4, 1024, 768]", _softmax_8: "f32[4, 12, 1024, 1024]", getitem_112: "u8[6291456]", getitem_114: "u8[393216]", mul_66: "f32[4, 1024, 768]", addmm_34: "f32[4096, 3072]", getitem_118: "u8[393216]", mul_72: "f32[4, 1024, 768]", _softmax_9: "f32[4, 12, 1024, 1024]", getitem_125: "u8[6291456]", getitem_127: "u8[393216]", mul_74: "f32[4, 1024, 768]", addmm_38: "f32[4096, 3072]", getitem_131: "u8[393216]", mul_80: "f32[4, 1024, 768]", _softmax_10: "f32[4, 12, 1024, 1024]", getitem_138: "u8[6291456]", getitem_140: "u8[393216]", mul_82: "f32[4, 1024, 768]", addmm_42: "f32[4096, 3072]", getitem_144: "u8[393216]", mul_88: "f32[4, 1024, 768]", _softmax_11: "f32[4, 12, 1024, 1024]", getitem_151: "u8[6291456]", getitem_153: "u8[393216]", mul_90: "f32[4, 1024, 768]", addmm_46: "f32[4096, 3072]", getitem_157: "u8[393216]", mul_96: "f32[4, 1024, 768]", sub_25: "i64[4]", iota_1: "i64[4]", _log_softmax: "f32[4, 2]", convert_element_type_13: "f32[]", ne_3: "b8[4, 1]", eq_tensor: "b8[4, 2]", permute_62: "f32[768, 4096]", div_14: "f32[4, 1024, 1]", permute_65: "f32[3072, 4096]", permute_67: "f32[768, 4096]", div_15: "f32[4, 1024, 1]", permute_69: "f32[768, 4096]", view_239: "f32[48, 64, 1024]", view_241: "f32[48, 1024, 1024]", view_245: "f32[48, 1024, 64]", view_247: "f32[48, 64, 1024]", permute_80: "f32[768, 4096]", div_17: "f32[4, 1024, 1]", permute_82: "f32[3072, 4096]", permute_84: "f32[768, 4096]", div_18: "f32[4, 1024, 1]", permute_86: "f32[768, 4096]", view_267: "f32[48, 64, 1024]", view_269: "f32[48, 1024, 1024]", view_273: "f32[48, 1024, 64]", view_275: "f32[48, 64, 1024]", permute_97: "f32[768, 4096]", div_20: "f32[4, 1024, 1]", permute_99: "f32[3072, 4096]", permute_101: "f32[768, 4096]", div_21: "f32[4, 1024, 1]", permute_103: "f32[768, 4096]", view_295: "f32[48, 64, 1024]", view_297: "f32[48, 1024, 1024]", view_301: "f32[48, 1024, 64]", view_303: "f32[48, 64, 1024]", permute_114: "f32[768, 4096]", div_23: "f32[4, 1024, 1]", permute_116: "f32[3072, 4096]", permute_118: "f32[768, 4096]", div_24: "f32[4, 1024, 1]", permute_120: "f32[768, 4096]", view_323: "f32[48, 64, 1024]", view_325: "f32[48, 1024, 1024]", view_329: "f32[48, 1024, 64]", view_331: "f32[48, 64, 1024]", permute_131: "f32[768, 4096]", div_26: "f32[4, 1024, 1]", permute_133: "f32[3072, 4096]", permute_135: "f32[768, 4096]", div_27: "f32[4, 1024, 1]", permute_137: "f32[768, 4096]", view_351: "f32[48, 64, 1024]", view_353: "f32[48, 1024, 1024]", view_357: "f32[48, 1024, 64]", view_359: "f32[48, 64, 1024]", permute_148: "f32[768, 4096]", div_29: "f32[4, 1024, 1]", permute_150: "f32[3072, 4096]", permute_152: "f32[768, 4096]", div_30: "f32[4, 1024, 1]", permute_154: "f32[768, 4096]", view_379: "f32[48, 64, 1024]", view_381: "f32[48, 1024, 1024]", view_385: "f32[48, 1024, 64]", view_387: "f32[48, 64, 1024]", permute_165: "f32[768, 4096]", div_32: "f32[4, 1024, 1]", permute_167: "f32[3072, 4096]", permute_169: "f32[768, 4096]", div_33: "f32[4, 1024, 1]", permute_171: "f32[768, 4096]", view_407: "f32[48, 64, 1024]", view_409: "f32[48, 1024, 1024]", view_413: "f32[48, 1024, 64]", view_415: "f32[48, 64, 1024]", permute_182: "f32[768, 4096]", div_35: "f32[4, 1024, 1]", permute_184: "f32[3072, 4096]", permute_186: "f32[768, 4096]", div_36: "f32[4, 1024, 1]", permute_188: "f32[768, 4096]", view_435: "f32[48, 64, 1024]", view_437: "f32[48, 1024, 1024]", view_441: "f32[48, 1024, 64]", view_443: "f32[48, 64, 1024]", permute_199: "f32[768, 4096]", div_38: "f32[4, 1024, 1]", permute_201: "f32[3072, 4096]", permute_203: "f32[768, 4096]", div_39: "f32[4, 1024, 1]", permute_205: "f32[768, 4096]", view_463: "f32[48, 64, 1024]", view_465: "f32[48, 1024, 1024]", view_469: "f32[48, 1024, 64]", view_471: "f32[48, 64, 1024]", permute_216: "f32[768, 4096]", div_41: "f32[4, 1024, 1]", permute_218: "f32[3072, 4096]", permute_220: "f32[768, 4096]", div_42: "f32[4, 1024, 1]", permute_222: "f32[768, 4096]", view_491: "f32[48, 64, 1024]", view_493: "f32[48, 1024, 1024]", view_497: "f32[48, 1024, 64]", view_499: "f32[48, 64, 1024]", permute_233: "f32[768, 4096]", div_44: "f32[4, 1024, 1]", permute_235: "f32[3072, 4096]", permute_237: "f32[768, 4096]", div_45: "f32[4, 1024, 1]", permute_239: "f32[768, 4096]", view_519: "f32[48, 64, 1024]", view_521: "f32[48, 1024, 1024]", view_525: "f32[48, 1024, 64]", view_527: "f32[48, 64, 1024]", permute_250: "f32[768, 4096]", div_47: "f32[4, 1024, 1]", permute_252: "f32[3072, 4096]", permute_254: "f32[768, 4096]", div_48: "f32[4, 1024, 1]", permute_256: "f32[768, 4096]", view_547: "f32[48, 64, 1024]", view_549: "f32[48, 1024, 1024]", view_553: "f32[48, 1024, 64]", view_555: "f32[48, 64, 1024]", permute_267: "f32[768, 4096]", div_50: "f32[4, 1024, 1]", tangents_1: "f32[]", tangents_2: "f32[4, 2]", tangents_3: "f32[4, 12, 1024, 64]", tangents_4: "f32[4, 12, 1024, 64]", tangents_5: "f32[4, 12, 1024, 64]", tangents_6: "f32[4, 12, 1024, 64]", tangents_7: "f32[4, 12, 1024, 64]", tangents_8: "f32[4, 12, 1024, 64]", tangents_9: "f32[4, 12, 1024, 64]", tangents_10: "f32[4, 12, 1024, 64]", tangents_11: "f32[4, 12, 1024, 64]", tangents_12: "f32[4, 12, 1024, 64]", tangents_13: "f32[4, 12, 1024, 64]", tangents_14: "f32[4, 12, 1024, 64]", tangents_15: "f32[4, 12, 1024, 64]", tangents_16: "f32[4, 12, 1024, 64]", tangents_17: "f32[4, 12, 1024, 64]", tangents_18: "f32[4, 12, 1024, 64]", tangents_19: "f32[4, 12, 1024, 64]", tangents_20: "f32[4, 12, 1024, 64]", tangents_21: "f32[4, 12, 1024, 64]", tangents_22: "f32[4, 12, 1024, 64]", tangents_23: "f32[4, 12, 1024, 64]", tangents_24: "f32[4, 12, 1024, 64]", tangents_25: "f32[4, 12, 1024, 64]", tangents_26: "f32[4, 12, 1024, 64]"):
        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1484 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_13);  tangents_1 = convert_element_type_13 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1484 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[4, 2]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_15: "f32[4, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_25);  ne_3 = div_13 = None
        mul_98: "f32[4, 2]" = torch.ops.aten.mul.Tensor(where_self, where_15);  where_self = where_15 = None
        exp: "f32[4, 2]" = torch.ops.aten.exp.default(_log_softmax);  _log_softmax = None
        sum_3: "f32[4, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [1], True)
        mul_99: "f32[4, 2]" = torch.ops.aten.mul.Tensor(exp, sum_3);  exp = sum_3 = None
        sub_26: "f32[4, 2]" = torch.ops.aten.sub.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        add_99: "f32[4, 2]" = torch.ops.aten.add.Tensor(tangents_2, sub_26);  tangents_2 = sub_26 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1464 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        full_default_29: "f32[4, 1024, 2]" = torch.ops.aten.full.default([4, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put: "f32[4, 1024, 2]" = torch.ops.aten.index_put.default(full_default_29, [iota_1, sub_25], add_99, True);  full_default_29 = iota_1 = sub_25 = add_99 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1440 in forward, code: logits = self.score(hidden_states)
        view_223: "f32[4096, 2]" = torch.ops.aten.view.default(index_put, [-1, 2]);  index_put = None
        permute_60: "f32[768, 2]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_61: "f32[2, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_1: "f32[4096, 768]" = torch.ops.aten.mm.default(view_223, permute_61);  permute_61 = None
        view_224: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_1, [4, 1024, 768]);  mm_1 = None
        mm_2: "f32[768, 2]" = torch.ops.aten.mm.default(permute_62, view_223);  permute_62 = view_223 = None
        permute_63: "f32[2, 768]" = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:914 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_101: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_224, primals_160);  primals_160 = None
        mul_102: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
        sum_4: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
        mul_103: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_96);  mul_101 = None
        sum_5: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
        mul_104: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_5);  sum_5 = None
        sub_28: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_4);  mul_102 = sum_4 = None
        sub_29: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_28, mul_104);  sub_28 = mul_104 = None
        mul_105: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_29);  div_14 = sub_29 = None
        mul_106: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_224, mul_96);  mul_96 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(mul_105, getitem_157, 0.10000000000000009);  getitem_157 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_228: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward, [4096, 768]);  npu_dropout_backward = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        mm_3: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_228, permute_64);  permute_64 = None
        mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_65, view_228);  permute_65 = None
        sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        view_229: "f32[768]" = torch.ops.aten.view.default(sum_8, [768]);  sum_8 = None
        view_230: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_3, [4, 1024, 3072]);  mm_3 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_214: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [4, 1024, 3072]);  addmm_46 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
        mul_107: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_230, mul_92);  mul_92 = None
        pow_12: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 3.0)
        mul_93: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_214, mul_93);  mul_93 = None
        mul_94: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_95: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_108: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_230, add_95);  view_230 = add_95 = None
        mul_109: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_30: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_109);  mul_109 = None
        mul_110: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_107, sub_30);  mul_107 = sub_30 = None
        mul_111: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_110, 0.7978845608028654);  mul_110 = None
        mul_112: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_111, 0.044715)
        pow_13: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 2.0);  view_214 = None
        mul_113: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
        mul_114: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        add_100: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_111, mul_114);  mul_111 = mul_114 = None
        mul_115: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_108, 0.5);  mul_108 = None
        add_101: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_100, mul_115);  add_100 = mul_115 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_231: "f32[4096, 3072]" = torch.ops.aten.view.default(add_101, [4096, 3072]);  add_101 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        mm_5: "f32[4096, 768]" = torch.ops.aten.mm.default(view_231, permute_66);  permute_66 = None
        mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_67, view_231);  permute_67 = None
        sum_9: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[3072]" = torch.ops.aten.view.default(sum_9, [3072]);  sum_9 = None
        view_233: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_5, [4, 1024, 768]);  mm_5 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_117: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_233, primals_154);  primals_154 = None
        mul_118: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, 768)
        sum_10: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True)
        mul_119: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, mul_90);  mul_117 = None
        sum_11: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
        mul_120: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_11);  sum_11 = None
        sub_32: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_118, sum_10);  mul_118 = sum_10 = None
        sub_33: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_120);  sub_32 = mul_120 = None
        mul_121: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_33);  div_15 = sub_33 = None
        mul_122: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_90);  mul_90 = None
        sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1]);  mul_122 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
        add_102: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_105, mul_121);  mul_105 = mul_121 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_1: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_102, getitem_153, 0.10000000000000009);  getitem_153 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_234: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_1, [4096, 768]);  npu_dropout_backward_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        mm_7: "f32[4096, 768]" = torch.ops.aten.mm.default(view_234, permute_68);  permute_68 = None
        mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_69, view_234);  permute_69 = None
        sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        view_235: "f32[768]" = torch.ops.aten.view.default(sum_14, [768]);  sum_14 = None
        view_236: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_7, [4, 1024, 768]);  mm_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_237: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_236, [4, 1024, 12, 64]);  view_236 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_70: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_48: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_70, [4, 12, 1024, 64]);  permute_70 = None
        clone_48: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_238: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_48, [48, 1024, 64]);  clone_48 = None
        bmm_24: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_238, view_239);  view_239 = None
        view_240: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, [4, 12, 1024, 1024]);  bmm_24 = None
        bmm_25: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_241, view_238);  view_241 = view_238 = None
        view_243: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_25, [4, 12, 1024, 64]);  bmm_25 = None
        add_103: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_26, view_243);  tangents_26 = view_243 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_2: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_240, getitem_151, 0.10000000000000009);  view_240 = getitem_151 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_123: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_2, _softmax_11);  npu_dropout_backward_2 = None
        sum_15: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [-1], True)
        mul_124: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_11, sum_15);  _softmax_11 = sum_15 = None
        sub_34: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_16: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_151, sub_34, full_default_25);  primals_151 = sub_34 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        div_16: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_16, full_default);  where_16 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_52: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_16, [4, 12, 1024, 1024]);  div_16 = None
        view_244: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_52, [48, 1024, 1024]);  expand_52 = None
        bmm_26: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_244, view_245);  view_245 = None
        view_246: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_26, [4, 12, 1024, 64]);  bmm_26 = None
        bmm_27: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_247, view_244);  view_247 = view_244 = None
        view_249: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_27, [4, 12, 64, 1024]);  bmm_27 = None
        permute_75: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_249, [0, 1, 3, 2]);  view_249 = None
        add_104: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, permute_75);  tangents_25 = permute_75 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_76: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_103, [0, 2, 1, 3]);  add_103 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_250: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_76, [4, 1024, 768]);  permute_76 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_77: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_104, [0, 2, 1, 3]);  add_104 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_251: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_77, [4, 1024, 768]);  permute_77 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_78: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_53: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_252: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_53, [4, 1024, 768]);  clone_53 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_252, view_251, view_250], 2);  view_252 = view_251 = view_250 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_253: "f32[4096, 2304]" = torch.ops.aten.view.default(cat, [4096, 2304]);  cat = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_79: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        mm_9: "f32[4096, 768]" = torch.ops.aten.mm.default(view_253, permute_79);  permute_79 = None
        mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_80, view_253);  permute_80 = None
        sum_16: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True);  view_253 = None
        view_254: "f32[2304]" = torch.ops.aten.view.default(sum_16, [2304]);  sum_16 = None
        view_255: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_9, [4, 1024, 768]);  mm_9 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_126: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_255, primals_147);  primals_147 = None
        mul_127: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, 768)
        sum_17: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [2], True)
        mul_128: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, mul_88);  mul_126 = None
        sum_18: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [2], True);  mul_128 = None
        mul_129: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_18);  sum_18 = None
        sub_36: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_127, sum_17);  mul_127 = sum_17 = None
        sub_37: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_36, mul_129);  sub_36 = mul_129 = None
        mul_130: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_37);  div_17 = sub_37 = None
        mul_131: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_255, mul_88);  mul_88 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1]);  mul_131 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 1]);  view_255 = None
        add_105: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_102, mul_130);  add_102 = mul_130 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_3: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_105, getitem_144, 0.10000000000000009);  getitem_144 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_256: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_3, [4096, 768]);  npu_dropout_backward_3 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_81: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        mm_11: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_256, permute_81);  permute_81 = None
        mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_82, view_256);  permute_82 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        view_257: "f32[768]" = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
        view_258: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_11, [4, 1024, 3072]);  mm_11 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_196: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_42, [4, 1024, 3072]);  addmm_42 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
        mul_132: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_258, mul_84);  mul_84 = None
        pow_11: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_196, 3.0)
        mul_85: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_196, mul_85);  mul_85 = None
        mul_86: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_87: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_133: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_258, add_87);  view_258 = add_87 = None
        mul_134: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_38: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_134);  mul_134 = None
        mul_135: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_132, sub_38);  mul_132 = sub_38 = None
        mul_136: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_135, 0.7978845608028654);  mul_135 = None
        mul_137: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_136, 0.044715)
        pow_14: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_196, 2.0);  view_196 = None
        mul_138: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_139: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        add_106: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_139);  mul_136 = mul_139 = None
        mul_140: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_133, 0.5);  mul_133 = None
        add_107: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_106, mul_140);  add_106 = mul_140 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_259: "f32[4096, 3072]" = torch.ops.aten.view.default(add_107, [4096, 3072]);  add_107 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        mm_13: "f32[4096, 768]" = torch.ops.aten.mm.default(view_259, permute_83);  permute_83 = None
        mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_84, view_259);  permute_84 = None
        sum_22: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True);  view_259 = None
        view_260: "f32[3072]" = torch.ops.aten.view.default(sum_22, [3072]);  sum_22 = None
        view_261: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_13, [4, 1024, 768]);  mm_13 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_142: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_261, primals_141);  primals_141 = None
        mul_143: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_142, 768)
        sum_23: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True)
        mul_144: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_142, mul_82);  mul_142 = None
        sum_24: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
        mul_145: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_24);  sum_24 = None
        sub_40: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_143, sum_23);  mul_143 = sum_23 = None
        sub_41: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_145);  sub_40 = mul_145 = None
        mul_146: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_41);  div_18 = sub_41 = None
        mul_147: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_261, mul_82);  mul_82 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_261, [0, 1]);  view_261 = None
        add_108: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_105, mul_146);  add_105 = mul_146 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_4: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_108, getitem_140, 0.10000000000000009);  getitem_140 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_262: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_4, [4096, 768]);  npu_dropout_backward_4 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        mm_15: "f32[4096, 768]" = torch.ops.aten.mm.default(view_262, permute_85);  permute_85 = None
        mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_86, view_262);  permute_86 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
        view_263: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
        view_264: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_15, [4, 1024, 768]);  mm_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_265: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_264, [4, 1024, 12, 64]);  view_264 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_87: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_56: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_87, [4, 12, 1024, 64]);  permute_87 = None
        clone_54: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_266: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_54, [48, 1024, 64]);  clone_54 = None
        bmm_28: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_266, view_267);  view_267 = None
        view_268: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, [4, 12, 1024, 1024]);  bmm_28 = None
        bmm_29: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_269, view_266);  view_269 = view_266 = None
        view_271: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_29, [4, 12, 1024, 64]);  bmm_29 = None
        add_109: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_271);  tangents_24 = view_271 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_5: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_268, getitem_138, 0.10000000000000009);  view_268 = getitem_138 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_148: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_5, _softmax_10);  npu_dropout_backward_5 = None
        sum_28: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [-1], True)
        mul_149: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_10, sum_28);  _softmax_10 = sum_28 = None
        sub_42: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_17: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_138, sub_42, full_default_25);  primals_138 = sub_42 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_19: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_17, full_default);  where_17 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_60: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_19, [4, 12, 1024, 1024]);  div_19 = None
        view_272: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_60, [48, 1024, 1024]);  expand_60 = None
        bmm_30: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_272, view_273);  view_273 = None
        view_274: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_30, [4, 12, 1024, 64]);  bmm_30 = None
        bmm_31: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_275, view_272);  view_275 = view_272 = None
        view_277: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_31, [4, 12, 64, 1024]);  bmm_31 = None
        permute_92: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_277, [0, 1, 3, 2]);  view_277 = None
        add_110: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, permute_92);  tangents_23 = permute_92 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_93: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_109, [0, 2, 1, 3]);  add_109 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_278: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_93, [4, 1024, 768]);  permute_93 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_94: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_279: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_94, [4, 1024, 768]);  permute_94 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_95: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_59: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_280: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_59, [4, 1024, 768]);  clone_59 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_1: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_280, view_279, view_278], 2);  view_280 = view_279 = view_278 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_281: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_1, [4096, 2304]);  cat_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_96: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        mm_17: "f32[4096, 768]" = torch.ops.aten.mm.default(view_281, permute_96);  permute_96 = None
        mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_97, view_281);  permute_97 = None
        sum_29: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
        view_282: "f32[2304]" = torch.ops.aten.view.default(sum_29, [2304]);  sum_29 = None
        view_283: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_17, [4, 1024, 768]);  mm_17 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_151: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_283, primals_134);  primals_134 = None
        mul_152: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_151, 768)
        sum_30: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True)
        mul_153: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_80);  mul_151 = None
        sum_31: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
        mul_154: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_31);  sum_31 = None
        sub_44: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_152, sum_30);  mul_152 = sum_30 = None
        sub_45: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_154);  sub_44 = mul_154 = None
        mul_155: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_45);  div_20 = sub_45 = None
        mul_156: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_283, mul_80);  mul_80 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1]);  mul_156 = None
        sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_283, [0, 1]);  view_283 = None
        add_111: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_108, mul_155);  add_108 = mul_155 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_6: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_111, getitem_131, 0.10000000000000009);  getitem_131 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_284: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_6, [4096, 768]);  npu_dropout_backward_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        mm_19: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_284, permute_98);  permute_98 = None
        mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_99, view_284);  permute_99 = None
        sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
        view_285: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
        view_286: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_19, [4, 1024, 3072]);  mm_19 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_178: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_38, [4, 1024, 3072]);  addmm_38 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        mul_157: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_286, mul_76);  mul_76 = None
        pow_10: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_77: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_178, mul_77);  mul_77 = None
        mul_78: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_79: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_158: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_286, add_79);  view_286 = add_79 = None
        mul_159: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_46: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_159);  mul_159 = None
        mul_160: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_157, sub_46);  mul_157 = sub_46 = None
        mul_161: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_160, 0.7978845608028654);  mul_160 = None
        mul_162: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_161, 0.044715)
        pow_15: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 2.0);  view_178 = None
        mul_163: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_164: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        add_112: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_161, mul_164);  mul_161 = mul_164 = None
        mul_165: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_158, 0.5);  mul_158 = None
        add_113: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_112, mul_165);  add_112 = mul_165 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_287: "f32[4096, 3072]" = torch.ops.aten.view.default(add_113, [4096, 3072]);  add_113 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_100: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        mm_21: "f32[4096, 768]" = torch.ops.aten.mm.default(view_287, permute_100);  permute_100 = None
        mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_101, view_287);  permute_101 = None
        sum_35: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        view_288: "f32[3072]" = torch.ops.aten.view.default(sum_35, [3072]);  sum_35 = None
        view_289: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_21, [4, 1024, 768]);  mm_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_167: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_128);  primals_128 = None
        mul_168: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
        sum_36: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
        mul_169: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_74);  mul_167 = None
        sum_37: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
        mul_170: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_37);  sum_37 = None
        sub_48: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_36);  mul_168 = sum_36 = None
        sub_49: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_170);  sub_48 = mul_170 = None
        mul_171: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_49);  div_21 = sub_49 = None
        mul_172: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_74);  mul_74 = None
        sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
        add_114: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_111, mul_171);  add_111 = mul_171 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_7: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_114, getitem_127, 0.10000000000000009);  getitem_127 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_290: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_7, [4096, 768]);  npu_dropout_backward_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        mm_23: "f32[4096, 768]" = torch.ops.aten.mm.default(view_290, permute_102);  permute_102 = None
        mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_103, view_290);  permute_103 = None
        sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        view_291: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
        view_292: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_23, [4, 1024, 768]);  mm_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_293: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_292, [4, 1024, 12, 64]);  view_292 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_104: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_64: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_104, [4, 12, 1024, 64]);  permute_104 = None
        clone_60: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_294: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_60, [48, 1024, 64]);  clone_60 = None
        bmm_32: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_294, view_295);  view_295 = None
        view_296: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, [4, 12, 1024, 1024]);  bmm_32 = None
        bmm_33: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_297, view_294);  view_297 = view_294 = None
        view_299: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_33, [4, 12, 1024, 64]);  bmm_33 = None
        add_115: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_299);  tangents_22 = view_299 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_8: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_296, getitem_125, 0.10000000000000009);  view_296 = getitem_125 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_173: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_8, _softmax_9);  npu_dropout_backward_8 = None
        sum_41: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [-1], True)
        mul_174: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_9, sum_41);  _softmax_9 = sum_41 = None
        sub_50: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_18: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_125, sub_50, full_default_25);  primals_125 = sub_50 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_22: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_18, full_default);  where_18 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_68: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_22, [4, 12, 1024, 1024]);  div_22 = None
        view_300: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_68, [48, 1024, 1024]);  expand_68 = None
        bmm_34: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_300, view_301);  view_301 = None
        view_302: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_34, [4, 12, 1024, 64]);  bmm_34 = None
        bmm_35: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_303, view_300);  view_303 = view_300 = None
        view_305: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_35, [4, 12, 64, 1024]);  bmm_35 = None
        permute_109: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_305, [0, 1, 3, 2]);  view_305 = None
        add_116: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, permute_109);  tangents_21 = permute_109 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_110: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_115, [0, 2, 1, 3]);  add_115 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_306: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_110, [4, 1024, 768]);  permute_110 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_111: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_116, [0, 2, 1, 3]);  add_116 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_307: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_111, [4, 1024, 768]);  permute_111 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_112: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_65: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_308: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_65, [4, 1024, 768]);  clone_65 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_2: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_308, view_307, view_306], 2);  view_308 = view_307 = view_306 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_309: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_2, [4096, 2304]);  cat_2 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_113: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        mm_25: "f32[4096, 768]" = torch.ops.aten.mm.default(view_309, permute_113);  permute_113 = None
        mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_114, view_309);  permute_114 = None
        sum_42: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
        view_310: "f32[2304]" = torch.ops.aten.view.default(sum_42, [2304]);  sum_42 = None
        view_311: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_25, [4, 1024, 768]);  mm_25 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_176: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, primals_121);  primals_121 = None
        mul_177: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_176, 768)
        sum_43: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
        mul_178: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_176, mul_72);  mul_176 = None
        sum_44: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True);  mul_178 = None
        mul_179: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_44);  sum_44 = None
        sub_52: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_177, sum_43);  mul_177 = sum_43 = None
        sub_53: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_179);  sub_52 = mul_179 = None
        mul_180: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_53);  div_23 = sub_53 = None
        mul_181: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, mul_72);  mul_72 = None
        sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1]);  mul_181 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_311, [0, 1]);  view_311 = None
        add_117: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_114, mul_180);  add_114 = mul_180 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_9: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_117, getitem_118, 0.10000000000000009);  getitem_118 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_312: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_9, [4096, 768]);  npu_dropout_backward_9 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_115: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        mm_27: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_312, permute_115);  permute_115 = None
        mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_116, view_312);  permute_116 = None
        sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
        view_313: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
        view_314: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_27, [4, 1024, 3072]);  mm_27 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_160: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 1024, 3072]);  addmm_34 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
        mul_182: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, mul_68);  mul_68 = None
        pow_9: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 3.0)
        mul_69: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_160, mul_69);  mul_69 = None
        mul_70: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_71: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_183: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, add_71);  view_314 = add_71 = None
        mul_184: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_54: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_184);  mul_184 = None
        mul_185: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_182, sub_54);  mul_182 = sub_54 = None
        mul_186: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_185, 0.7978845608028654);  mul_185 = None
        mul_187: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_186, 0.044715)
        pow_16: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 2.0);  view_160 = None
        mul_188: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_189: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        add_118: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_186, mul_189);  mul_186 = mul_189 = None
        mul_190: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_183, 0.5);  mul_183 = None
        add_119: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_118, mul_190);  add_118 = mul_190 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_315: "f32[4096, 3072]" = torch.ops.aten.view.default(add_119, [4096, 3072]);  add_119 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        mm_29: "f32[4096, 768]" = torch.ops.aten.mm.default(view_315, permute_117);  permute_117 = None
        mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_118, view_315);  permute_118 = None
        sum_48: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_316: "f32[3072]" = torch.ops.aten.view.default(sum_48, [3072]);  sum_48 = None
        view_317: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_29, [4, 1024, 768]);  mm_29 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_192: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, primals_115);  primals_115 = None
        mul_193: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_49: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
        mul_194: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_66);  mul_192 = None
        sum_50: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_50);  sum_50 = None
        sub_56: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_49);  mul_193 = sum_49 = None
        sub_57: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
        mul_196: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_57);  div_24 = sub_57 = None
        mul_197: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, mul_66);  mul_66 = None
        sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_317, [0, 1]);  view_317 = None
        add_120: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_117, mul_196);  add_117 = mul_196 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_10: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_120, getitem_114, 0.10000000000000009);  getitem_114 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_318: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_10, [4096, 768]);  npu_dropout_backward_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        mm_31: "f32[4096, 768]" = torch.ops.aten.mm.default(view_318, permute_119);  permute_119 = None
        mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_120, view_318);  permute_120 = None
        sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
        view_320: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_31, [4, 1024, 768]);  mm_31 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_321: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_320, [4, 1024, 12, 64]);  view_320 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_121: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_72: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_121, [4, 12, 1024, 64]);  permute_121 = None
        clone_66: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_322: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_66, [48, 1024, 64]);  clone_66 = None
        bmm_36: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_322, view_323);  view_323 = None
        view_324: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_36, [4, 12, 1024, 1024]);  bmm_36 = None
        bmm_37: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_325, view_322);  view_325 = view_322 = None
        view_327: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_37, [4, 12, 1024, 64]);  bmm_37 = None
        add_121: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_327);  tangents_20 = view_327 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_11: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_324, getitem_112, 0.10000000000000009);  view_324 = getitem_112 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_198: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_11, _softmax_8);  npu_dropout_backward_11 = None
        sum_54: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [-1], True)
        mul_199: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_8, sum_54);  _softmax_8 = sum_54 = None
        sub_58: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_19: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_112, sub_58, full_default_25);  primals_112 = sub_58 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_25: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_19, full_default);  where_19 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_76: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_25, [4, 12, 1024, 1024]);  div_25 = None
        view_328: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_76, [48, 1024, 1024]);  expand_76 = None
        bmm_38: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_328, view_329);  view_329 = None
        view_330: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_38, [4, 12, 1024, 64]);  bmm_38 = None
        bmm_39: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_331, view_328);  view_331 = view_328 = None
        view_333: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_39, [4, 12, 64, 1024]);  bmm_39 = None
        permute_126: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_333, [0, 1, 3, 2]);  view_333 = None
        add_122: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, permute_126);  tangents_19 = permute_126 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_127: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_334: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_127, [4, 1024, 768]);  permute_127 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_128: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_335: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_128, [4, 1024, 768]);  permute_128 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_129: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_71: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_336: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_71, [4, 1024, 768]);  clone_71 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_3: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_336, view_335, view_334], 2);  view_336 = view_335 = view_334 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_337: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_3, [4096, 2304]);  cat_3 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_130: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        mm_33: "f32[4096, 768]" = torch.ops.aten.mm.default(view_337, permute_130);  permute_130 = None
        mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_131, view_337);  permute_131 = None
        sum_55: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338: "f32[2304]" = torch.ops.aten.view.default(sum_55, [2304]);  sum_55 = None
        view_339: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_33, [4, 1024, 768]);  mm_33 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_201: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_108);  primals_108 = None
        mul_202: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_201, 768)
        sum_56: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
        mul_203: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_64);  mul_201 = None
        sum_57: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
        mul_204: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_57);  sum_57 = None
        sub_60: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_202, sum_56);  mul_202 = sum_56 = None
        sub_61: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_204);  sub_60 = mul_204 = None
        mul_205: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_61);  div_26 = sub_61 = None
        mul_206: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_64);  mul_64 = None
        sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
        sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
        add_123: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_120, mul_205);  add_120 = mul_205 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_12: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_123, getitem_105, 0.10000000000000009);  getitem_105 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_340: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_12, [4096, 768]);  npu_dropout_backward_12 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        mm_35: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_340, permute_132);  permute_132 = None
        mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_133, view_340);  permute_133 = None
        sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
        view_341: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
        view_342: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_35, [4, 1024, 3072]);  mm_35 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_142: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_30, [4, 1024, 3072]);  addmm_30 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        mul_207: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, mul_60);  mul_60 = None
        pow_8: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_61: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_61);  mul_61 = None
        mul_62: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_63: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_208: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, add_63);  view_342 = add_63 = None
        mul_209: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_62: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_209);  mul_209 = None
        mul_210: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_207, sub_62);  mul_207 = sub_62 = None
        mul_211: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_210, 0.7978845608028654);  mul_210 = None
        mul_212: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_211, 0.044715)
        pow_17: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 2.0);  view_142 = None
        mul_213: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_214: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        add_124: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_211, mul_214);  mul_211 = mul_214 = None
        mul_215: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_208, 0.5);  mul_208 = None
        add_125: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_124, mul_215);  add_124 = mul_215 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_343: "f32[4096, 3072]" = torch.ops.aten.view.default(add_125, [4096, 3072]);  add_125 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        mm_37: "f32[4096, 768]" = torch.ops.aten.mm.default(view_343, permute_134);  permute_134 = None
        mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_135, view_343);  permute_135 = None
        sum_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344: "f32[3072]" = torch.ops.aten.view.default(sum_61, [3072]);  sum_61 = None
        view_345: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_37, [4, 1024, 768]);  mm_37 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_217: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_345, primals_102);  primals_102 = None
        mul_218: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_217, 768)
        sum_62: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True)
        mul_219: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_58);  mul_217 = None
        sum_63: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True);  mul_219 = None
        mul_220: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_63);  sum_63 = None
        sub_64: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_218, sum_62);  mul_218 = sum_62 = None
        sub_65: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_220);  sub_64 = mul_220 = None
        mul_221: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_65);  div_27 = sub_65 = None
        mul_222: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_345, mul_58);  mul_58 = None
        sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1]);  mul_222 = None
        sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_345, [0, 1]);  view_345 = None
        add_126: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_123, mul_221);  add_123 = mul_221 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_13: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_126, getitem_101, 0.10000000000000009);  getitem_101 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_346: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_13, [4096, 768]);  npu_dropout_backward_13 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        mm_39: "f32[4096, 768]" = torch.ops.aten.mm.default(view_346, permute_136);  permute_136 = None
        mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_137, view_346);  permute_137 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
        view_347: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
        view_348: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_39, [4, 1024, 768]);  mm_39 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_349: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_348, [4, 1024, 12, 64]);  view_348 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_138: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_80: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [4, 12, 1024, 64]);  permute_138 = None
        clone_72: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_350: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_72, [48, 1024, 64]);  clone_72 = None
        bmm_40: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_350, view_351);  view_351 = None
        view_352: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_40, [4, 12, 1024, 1024]);  bmm_40 = None
        bmm_41: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_353, view_350);  view_353 = view_350 = None
        view_355: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_41, [4, 12, 1024, 64]);  bmm_41 = None
        add_127: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_355);  tangents_18 = view_355 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_14: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_352, getitem_99, 0.10000000000000009);  view_352 = getitem_99 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_223: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_14, _softmax_7);  npu_dropout_backward_14 = None
        sum_67: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [-1], True)
        mul_224: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_7, sum_67);  _softmax_7 = sum_67 = None
        sub_66: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_20: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_99, sub_66, full_default_25);  primals_99 = sub_66 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_28: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_20, full_default);  where_20 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_84: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_28, [4, 12, 1024, 1024]);  div_28 = None
        view_356: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_84, [48, 1024, 1024]);  expand_84 = None
        bmm_42: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_356, view_357);  view_357 = None
        view_358: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_42, [4, 12, 1024, 64]);  bmm_42 = None
        bmm_43: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_359, view_356);  view_359 = view_356 = None
        view_361: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_43, [4, 12, 64, 1024]);  bmm_43 = None
        permute_143: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_361, [0, 1, 3, 2]);  view_361 = None
        add_128: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, permute_143);  tangents_17 = permute_143 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_144: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_362: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_144, [4, 1024, 768]);  permute_144 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_145: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_128, [0, 2, 1, 3]);  add_128 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_363: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_145, [4, 1024, 768]);  permute_145 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_146: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_77: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_364: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_77, [4, 1024, 768]);  clone_77 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_4: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_364, view_363, view_362], 2);  view_364 = view_363 = view_362 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_365: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_4, [4096, 2304]);  cat_4 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_147: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        mm_41: "f32[4096, 768]" = torch.ops.aten.mm.default(view_365, permute_147);  permute_147 = None
        mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_148, view_365);  permute_148 = None
        sum_68: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
        view_366: "f32[2304]" = torch.ops.aten.view.default(sum_68, [2304]);  sum_68 = None
        view_367: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_41, [4, 1024, 768]);  mm_41 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_226: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_367, primals_95);  primals_95 = None
        mul_227: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_226, 768)
        sum_69: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
        mul_228: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_56);  mul_226 = None
        sum_70: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
        mul_229: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_70);  sum_70 = None
        sub_68: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_227, sum_69);  mul_227 = sum_69 = None
        sub_69: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_229);  sub_68 = mul_229 = None
        mul_230: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_69);  div_29 = sub_69 = None
        mul_231: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_367, mul_56);  mul_56 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 1]);  mul_231 = None
        sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_367, [0, 1]);  view_367 = None
        add_129: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_126, mul_230);  add_126 = mul_230 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_15: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_129, getitem_92, 0.10000000000000009);  getitem_92 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_368: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_15, [4096, 768]);  npu_dropout_backward_15 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_149: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        mm_43: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_368, permute_149);  permute_149 = None
        mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_150, view_368);  permute_150 = None
        sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
        view_369: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
        view_370: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_43, [4, 1024, 3072]);  mm_43 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_124: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_26, [4, 1024, 3072]);  addmm_26 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.5)
        mul_232: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_370, mul_52);  mul_52 = None
        pow_7: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_124, 3.0)
        mul_53: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_124, mul_53);  mul_53 = None
        mul_54: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_55: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_233: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_370, add_55);  view_370 = add_55 = None
        mul_234: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_70: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_234);  mul_234 = None
        mul_235: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_232, sub_70);  mul_232 = sub_70 = None
        mul_236: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.7978845608028654);  mul_235 = None
        mul_237: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_236, 0.044715)
        pow_18: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_124, 2.0);  view_124 = None
        mul_238: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_239: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        add_130: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_236, mul_239);  mul_236 = mul_239 = None
        mul_240: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_233, 0.5);  mul_233 = None
        add_131: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_130, mul_240);  add_130 = mul_240 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_371: "f32[4096, 3072]" = torch.ops.aten.view.default(add_131, [4096, 3072]);  add_131 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        mm_45: "f32[4096, 768]" = torch.ops.aten.mm.default(view_371, permute_151);  permute_151 = None
        mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_152, view_371);  permute_152 = None
        sum_74: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
        view_372: "f32[3072]" = torch.ops.aten.view.default(sum_74, [3072]);  sum_74 = None
        view_373: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_45, [4, 1024, 768]);  mm_45 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_242: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_373, primals_89);  primals_89 = None
        mul_243: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
        sum_75: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
        mul_244: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_50);  mul_242 = None
        sum_76: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
        mul_245: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_76);  sum_76 = None
        sub_72: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_75);  mul_243 = sum_75 = None
        sub_73: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_245);  sub_72 = mul_245 = None
        mul_246: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_73);  div_30 = sub_73 = None
        mul_247: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_373, mul_50);  mul_50 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
        sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_373, [0, 1]);  view_373 = None
        add_132: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_129, mul_246);  add_129 = mul_246 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_16: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_132, getitem_88, 0.10000000000000009);  getitem_88 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_374: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_16, [4096, 768]);  npu_dropout_backward_16 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        mm_47: "f32[4096, 768]" = torch.ops.aten.mm.default(view_374, permute_153);  permute_153 = None
        mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_154, view_374);  permute_154 = None
        sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        view_375: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
        view_376: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_47, [4, 1024, 768]);  mm_47 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_377: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_376, [4, 1024, 12, 64]);  view_376 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_155: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_88: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_155, [4, 12, 1024, 64]);  permute_155 = None
        clone_78: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_378: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_78, [48, 1024, 64]);  clone_78 = None
        bmm_44: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_378, view_379);  view_379 = None
        view_380: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_44, [4, 12, 1024, 1024]);  bmm_44 = None
        bmm_45: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_381, view_378);  view_381 = view_378 = None
        view_383: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_45, [4, 12, 1024, 64]);  bmm_45 = None
        add_133: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_383);  tangents_16 = view_383 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_17: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_380, getitem_86, 0.10000000000000009);  view_380 = getitem_86 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_248: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_17, _softmax_6);  npu_dropout_backward_17 = None
        sum_80: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [-1], True)
        mul_249: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_6, sum_80);  _softmax_6 = sum_80 = None
        sub_74: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_21: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_86, sub_74, full_default_25);  primals_86 = sub_74 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_31: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_21, full_default);  where_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_92: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_31, [4, 12, 1024, 1024]);  div_31 = None
        view_384: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_92, [48, 1024, 1024]);  expand_92 = None
        bmm_46: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_384, view_385);  view_385 = None
        view_386: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_46, [4, 12, 1024, 64]);  bmm_46 = None
        bmm_47: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_387, view_384);  view_387 = view_384 = None
        view_389: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_47, [4, 12, 64, 1024]);  bmm_47 = None
        permute_160: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_389, [0, 1, 3, 2]);  view_389 = None
        add_134: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, permute_160);  tangents_15 = permute_160 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_161: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_133, [0, 2, 1, 3]);  add_133 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_390: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_161, [4, 1024, 768]);  permute_161 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_162: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_391: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_162, [4, 1024, 768]);  permute_162 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_163: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_83: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        view_392: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_83, [4, 1024, 768]);  clone_83 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_5: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_392, view_391, view_390], 2);  view_392 = view_391 = view_390 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_393: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_5, [4096, 2304]);  cat_5 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_164: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        mm_49: "f32[4096, 768]" = torch.ops.aten.mm.default(view_393, permute_164);  permute_164 = None
        mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_165, view_393);  permute_165 = None
        sum_81: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_393, [0], True);  view_393 = None
        view_394: "f32[2304]" = torch.ops.aten.view.default(sum_81, [2304]);  sum_81 = None
        view_395: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_49, [4, 1024, 768]);  mm_49 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_251: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_395, primals_82);  primals_82 = None
        mul_252: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
        sum_82: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_48);  mul_251 = None
        sum_83: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_83);  sum_83 = None
        sub_76: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_82);  mul_252 = sum_82 = None
        sub_77: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_254);  sub_76 = mul_254 = None
        mul_255: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_77);  div_32 = sub_77 = None
        mul_256: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_395, mul_48);  mul_48 = None
        sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_395, [0, 1]);  view_395 = None
        add_135: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_132, mul_255);  add_132 = mul_255 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_18: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_135, getitem_79, 0.10000000000000009);  getitem_79 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_396: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_18, [4096, 768]);  npu_dropout_backward_18 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_166: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        mm_51: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_396, permute_166);  permute_166 = None
        mm_52: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_167, view_396);  permute_167 = None
        sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True);  view_396 = None
        view_397: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
        view_398: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_51, [4, 1024, 3072]);  mm_51 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_106: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 1024, 3072]);  addmm_22 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        mul_257: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_398, mul_44);  mul_44 = None
        pow_6: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_45: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_45);  mul_45 = None
        mul_46: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_47: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_258: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_398, add_47);  view_398 = add_47 = None
        mul_259: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_78: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_259);  mul_259 = None
        mul_260: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_257, sub_78);  mul_257 = sub_78 = None
        mul_261: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_260, 0.7978845608028654);  mul_260 = None
        mul_262: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_261, 0.044715)
        pow_19: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 2.0);  view_106 = None
        mul_263: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_264: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        add_136: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_261, mul_264);  mul_261 = mul_264 = None
        mul_265: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_258, 0.5);  mul_258 = None
        add_137: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_136, mul_265);  add_136 = mul_265 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_399: "f32[4096, 3072]" = torch.ops.aten.view.default(add_137, [4096, 3072]);  add_137 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_168: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        mm_53: "f32[4096, 768]" = torch.ops.aten.mm.default(view_399, permute_168);  permute_168 = None
        mm_54: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_169, view_399);  permute_169 = None
        sum_87: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
        view_400: "f32[3072]" = torch.ops.aten.view.default(sum_87, [3072]);  sum_87 = None
        view_401: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_53, [4, 1024, 768]);  mm_53 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_267: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_401, primals_76);  primals_76 = None
        mul_268: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_88: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_42);  mul_267 = None
        sum_89: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_89);  sum_89 = None
        sub_80: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_88);  mul_268 = sum_88 = None
        sub_81: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_270);  sub_80 = mul_270 = None
        mul_271: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_81);  div_33 = sub_81 = None
        mul_272: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_401, mul_42);  mul_42 = None
        sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_401, [0, 1]);  view_401 = None
        add_138: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_135, mul_271);  add_135 = mul_271 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_19: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_138, getitem_75, 0.10000000000000009);  getitem_75 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_402: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_19, [4096, 768]);  npu_dropout_backward_19 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_55: "f32[4096, 768]" = torch.ops.aten.mm.default(view_402, permute_170);  permute_170 = None
        mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_171, view_402);  permute_171 = None
        sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
        view_403: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
        view_404: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_55, [4, 1024, 768]);  mm_55 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_405: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_404, [4, 1024, 12, 64]);  view_404 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_172: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_96: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_172, [4, 12, 1024, 64]);  permute_172 = None
        clone_84: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_406: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_84, [48, 1024, 64]);  clone_84 = None
        bmm_48: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_406, view_407);  view_407 = None
        view_408: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_48, [4, 12, 1024, 1024]);  bmm_48 = None
        bmm_49: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_409, view_406);  view_409 = view_406 = None
        view_411: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_49, [4, 12, 1024, 64]);  bmm_49 = None
        add_139: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_411);  tangents_14 = view_411 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_20: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_408, getitem_73, 0.10000000000000009);  view_408 = getitem_73 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_273: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_20, _softmax_5);  npu_dropout_backward_20 = None
        sum_93: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [-1], True)
        mul_274: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_5, sum_93);  _softmax_5 = sum_93 = None
        sub_82: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_22: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_73, sub_82, full_default_25);  primals_73 = sub_82 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_34: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_22, full_default);  where_22 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_100: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_34, [4, 12, 1024, 1024]);  div_34 = None
        view_412: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_100, [48, 1024, 1024]);  expand_100 = None
        bmm_50: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_412, view_413);  view_413 = None
        view_414: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_50, [4, 12, 1024, 64]);  bmm_50 = None
        bmm_51: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_415, view_412);  view_415 = view_412 = None
        view_417: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_51, [4, 12, 64, 1024]);  bmm_51 = None
        permute_177: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_417, [0, 1, 3, 2]);  view_417 = None
        add_140: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, permute_177);  tangents_13 = permute_177 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_178: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_418: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_178, [4, 1024, 768]);  permute_178 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_179: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_419: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_179, [4, 1024, 768]);  permute_179 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_180: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_89: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_420: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_89, [4, 1024, 768]);  clone_89 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_6: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_420, view_419, view_418], 2);  view_420 = view_419 = view_418 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_421: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_6, [4096, 2304]);  cat_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_181: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        mm_57: "f32[4096, 768]" = torch.ops.aten.mm.default(view_421, permute_181);  permute_181 = None
        mm_58: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_182, view_421);  permute_182 = None
        sum_94: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        view_422: "f32[2304]" = torch.ops.aten.view.default(sum_94, [2304]);  sum_94 = None
        view_423: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_57, [4, 1024, 768]);  mm_57 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_276: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_423, primals_69);  primals_69 = None
        mul_277: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
        sum_95: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
        mul_278: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_40);  mul_276 = None
        sum_96: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
        mul_279: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_96);  sum_96 = None
        sub_84: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_95);  mul_277 = sum_95 = None
        sub_85: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_279);  sub_84 = mul_279 = None
        mul_280: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_85);  div_35 = sub_85 = None
        mul_281: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_423, mul_40);  mul_40 = None
        sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
        sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_423, [0, 1]);  view_423 = None
        add_141: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_138, mul_280);  add_138 = mul_280 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_21: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_141, getitem_66, 0.10000000000000009);  getitem_66 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_424: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_21, [4096, 768]);  npu_dropout_backward_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_183: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        mm_59: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_424, permute_183);  permute_183 = None
        mm_60: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_184, view_424);  permute_184 = None
        sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True);  view_424 = None
        view_425: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
        view_426: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_59, [4, 1024, 3072]);  mm_59 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_88: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, [4, 1024, 3072]);  addmm_18 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
        mul_282: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_426, mul_36);  mul_36 = None
        pow_5: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_88, 3.0)
        mul_37: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_88, mul_37);  mul_37 = None
        mul_38: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_39: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_283: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_426, add_39);  view_426 = add_39 = None
        mul_284: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_86: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_284);  mul_284 = None
        mul_285: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_282, sub_86);  mul_282 = sub_86 = None
        mul_286: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_285, 0.7978845608028654);  mul_285 = None
        mul_287: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_286, 0.044715)
        pow_20: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_88, 2.0);  view_88 = None
        mul_288: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_289: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
        add_142: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_286, mul_289);  mul_286 = mul_289 = None
        mul_290: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_283, 0.5);  mul_283 = None
        add_143: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_142, mul_290);  add_142 = mul_290 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_427: "f32[4096, 3072]" = torch.ops.aten.view.default(add_143, [4096, 3072]);  add_143 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        mm_61: "f32[4096, 768]" = torch.ops.aten.mm.default(view_427, permute_185);  permute_185 = None
        mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_186, view_427);  permute_186 = None
        sum_100: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
        view_428: "f32[3072]" = torch.ops.aten.view.default(sum_100, [3072]);  sum_100 = None
        view_429: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_61, [4, 1024, 768]);  mm_61 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_292: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_429, primals_63);  primals_63 = None
        mul_293: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_292, 768)
        sum_101: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
        mul_294: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_292, mul_34);  mul_292 = None
        sum_102: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
        mul_295: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_102);  sum_102 = None
        sub_88: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_293, sum_101);  mul_293 = sum_101 = None
        sub_89: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_295);  sub_88 = mul_295 = None
        mul_296: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_89);  div_36 = sub_89 = None
        mul_297: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_429, mul_34);  mul_34 = None
        sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_429, [0, 1]);  view_429 = None
        add_144: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_141, mul_296);  add_141 = mul_296 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_22: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_144, getitem_62, 0.10000000000000009);  getitem_62 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_430: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_22, [4096, 768]);  npu_dropout_backward_22 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        mm_63: "f32[4096, 768]" = torch.ops.aten.mm.default(view_430, permute_187);  permute_187 = None
        mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_188, view_430);  permute_188 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        view_431: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
        view_432: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_63, [4, 1024, 768]);  mm_63 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_433: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_432, [4, 1024, 12, 64]);  view_432 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_189: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_104: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_189, [4, 12, 1024, 64]);  permute_189 = None
        clone_90: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_434: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_90, [48, 1024, 64]);  clone_90 = None
        bmm_52: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_434, view_435);  view_435 = None
        view_436: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_52, [4, 12, 1024, 1024]);  bmm_52 = None
        bmm_53: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_437, view_434);  view_437 = view_434 = None
        view_439: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_53, [4, 12, 1024, 64]);  bmm_53 = None
        add_145: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_439);  tangents_12 = view_439 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_23: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_436, getitem_60, 0.10000000000000009);  view_436 = getitem_60 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_298: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_23, _softmax_4);  npu_dropout_backward_23 = None
        sum_106: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [-1], True)
        mul_299: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_4, sum_106);  _softmax_4 = sum_106 = None
        sub_90: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_23: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_60, sub_90, full_default_25);  primals_60 = sub_90 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_37: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_23, full_default);  where_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_108: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_37, [4, 12, 1024, 1024]);  div_37 = None
        view_440: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_108, [48, 1024, 1024]);  expand_108 = None
        bmm_54: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_440, view_441);  view_441 = None
        view_442: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_54, [4, 12, 1024, 64]);  bmm_54 = None
        bmm_55: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_443, view_440);  view_443 = view_440 = None
        view_445: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_55, [4, 12, 64, 1024]);  bmm_55 = None
        permute_194: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_445, [0, 1, 3, 2]);  view_445 = None
        add_146: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, permute_194);  tangents_11 = permute_194 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_195: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_145, [0, 2, 1, 3]);  add_145 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_446: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_195, [4, 1024, 768]);  permute_195 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_196: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_146, [0, 2, 1, 3]);  add_146 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_447: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_196, [4, 1024, 768]);  permute_196 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_197: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_95: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_448: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_95, [4, 1024, 768]);  clone_95 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_7: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_448, view_447, view_446], 2);  view_448 = view_447 = view_446 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_449: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_7, [4096, 2304]);  cat_7 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_198: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        mm_65: "f32[4096, 768]" = torch.ops.aten.mm.default(view_449, permute_198);  permute_198 = None
        mm_66: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_199, view_449);  permute_199 = None
        sum_107: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
        view_450: "f32[2304]" = torch.ops.aten.view.default(sum_107, [2304]);  sum_107 = None
        view_451: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_65, [4, 1024, 768]);  mm_65 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_301: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_451, primals_56);  primals_56 = None
        mul_302: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_301, 768)
        sum_108: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True)
        mul_303: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_301, mul_32);  mul_301 = None
        sum_109: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True);  mul_303 = None
        mul_304: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_109);  sum_109 = None
        sub_92: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_302, sum_108);  mul_302 = sum_108 = None
        sub_93: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_304);  sub_92 = mul_304 = None
        mul_305: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_93);  div_38 = sub_93 = None
        mul_306: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_451, mul_32);  mul_32 = None
        sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_306, [0, 1]);  mul_306 = None
        sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_451, [0, 1]);  view_451 = None
        add_147: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_144, mul_305);  add_144 = mul_305 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_24: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_147, getitem_53, 0.10000000000000009);  getitem_53 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_452: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_24, [4096, 768]);  npu_dropout_backward_24 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_200: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        mm_67: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_452, permute_200);  permute_200 = None
        mm_68: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_201, view_452);  permute_201 = None
        sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True);  view_452 = None
        view_453: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
        view_454: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_67, [4, 1024, 3072]);  mm_67 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_70: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, [4, 1024, 3072]);  addmm_14 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        mul_307: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_454, mul_28);  mul_28 = None
        pow_4: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_29: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_29);  mul_29 = None
        mul_30: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_31: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_308: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_454, add_31);  view_454 = add_31 = None
        mul_309: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_94: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_309);  mul_309 = None
        mul_310: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_307, sub_94);  mul_307 = sub_94 = None
        mul_311: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_310, 0.7978845608028654);  mul_310 = None
        mul_312: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_311, 0.044715)
        pow_21: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 2.0);  view_70 = None
        mul_313: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_314: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        add_148: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_311, mul_314);  mul_311 = mul_314 = None
        mul_315: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_308, 0.5);  mul_308 = None
        add_149: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_148, mul_315);  add_148 = mul_315 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_455: "f32[4096, 3072]" = torch.ops.aten.view.default(add_149, [4096, 3072]);  add_149 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_202: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        mm_69: "f32[4096, 768]" = torch.ops.aten.mm.default(view_455, permute_202);  permute_202 = None
        mm_70: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_203, view_455);  permute_203 = None
        sum_113: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
        view_456: "f32[3072]" = torch.ops.aten.view.default(sum_113, [3072]);  sum_113 = None
        view_457: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_69, [4, 1024, 768]);  mm_69 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_317: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_457, primals_50);  primals_50 = None
        mul_318: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_317, 768)
        sum_114: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_317, mul_26);  mul_317 = None
        sum_115: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_115);  sum_115 = None
        sub_96: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_318, sum_114);  mul_318 = sum_114 = None
        sub_97: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_320);  sub_96 = mul_320 = None
        mul_321: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_97);  div_39 = sub_97 = None
        mul_322: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_457, mul_26);  mul_26 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_457, [0, 1]);  view_457 = None
        add_150: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_147, mul_321);  add_147 = mul_321 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_25: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_150, getitem_49, 0.10000000000000009);  getitem_49 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_458: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_25, [4096, 768]);  npu_dropout_backward_25 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        mm_71: "f32[4096, 768]" = torch.ops.aten.mm.default(view_458, permute_204);  permute_204 = None
        mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_458);  permute_205 = None
        sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        view_459: "f32[768]" = torch.ops.aten.view.default(sum_118, [768]);  sum_118 = None
        view_460: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_71, [4, 1024, 768]);  mm_71 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_461: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_460, [4, 1024, 12, 64]);  view_460 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_206: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_112: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_206, [4, 12, 1024, 64]);  permute_206 = None
        clone_96: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_462: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_96, [48, 1024, 64]);  clone_96 = None
        bmm_56: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_462, view_463);  view_463 = None
        view_464: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_56, [4, 12, 1024, 1024]);  bmm_56 = None
        bmm_57: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_465, view_462);  view_465 = view_462 = None
        view_467: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_57, [4, 12, 1024, 64]);  bmm_57 = None
        add_151: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_467);  tangents_10 = view_467 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_26: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_464, getitem_47, 0.10000000000000009);  view_464 = getitem_47 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_323: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_26, _softmax_3);  npu_dropout_backward_26 = None
        sum_119: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
        mul_324: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_3, sum_119);  _softmax_3 = sum_119 = None
        sub_98: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_24: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_47, sub_98, full_default_25);  primals_47 = sub_98 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_40: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_24, full_default);  where_24 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_116: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_40, [4, 12, 1024, 1024]);  div_40 = None
        view_468: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_116, [48, 1024, 1024]);  expand_116 = None
        bmm_58: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_468, view_469);  view_469 = None
        view_470: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_58, [4, 12, 1024, 64]);  bmm_58 = None
        bmm_59: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_471, view_468);  view_471 = view_468 = None
        view_473: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_59, [4, 12, 64, 1024]);  bmm_59 = None
        permute_211: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_473, [0, 1, 3, 2]);  view_473 = None
        add_152: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, permute_211);  tangents_9 = permute_211 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_212: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_474: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_212, [4, 1024, 768]);  permute_212 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_213: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_475: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_213, [4, 1024, 768]);  permute_213 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_214: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_101: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_476: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_101, [4, 1024, 768]);  clone_101 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_8: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_476, view_475, view_474], 2);  view_476 = view_475 = view_474 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_477: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_8, [4096, 2304]);  cat_8 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_215: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        mm_73: "f32[4096, 768]" = torch.ops.aten.mm.default(view_477, permute_215);  permute_215 = None
        mm_74: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_216, view_477);  permute_216 = None
        sum_120: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
        view_478: "f32[2304]" = torch.ops.aten.view.default(sum_120, [2304]);  sum_120 = None
        view_479: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_73, [4, 1024, 768]);  mm_73 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_326: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_479, primals_43);  primals_43 = None
        mul_327: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_326, 768)
        sum_121: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_326, mul_24);  mul_326 = None
        sum_122: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_122);  sum_122 = None
        sub_100: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_327, sum_121);  mul_327 = sum_121 = None
        sub_101: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_329);  sub_100 = mul_329 = None
        mul_330: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_101);  div_41 = sub_101 = None
        mul_331: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_479, mul_24);  mul_24 = None
        sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_479, [0, 1]);  view_479 = None
        add_153: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_150, mul_330);  add_150 = mul_330 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_27: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_153, getitem_40, 0.10000000000000009);  getitem_40 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_480: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_27, [4096, 768]);  npu_dropout_backward_27 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_217: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        mm_75: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_480, permute_217);  permute_217 = None
        mm_76: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_218, view_480);  permute_218 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_480, [0], True);  view_480 = None
        view_481: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
        view_482: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_75, [4, 1024, 3072]);  mm_75 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_52: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 1024, 3072]);  addmm_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
        mul_332: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_482, mul_20);  mul_20 = None
        pow_3: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_52, 3.0)
        mul_21: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_52, mul_21);  mul_21 = None
        mul_22: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_23: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_333: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_482, add_23);  view_482 = add_23 = None
        mul_334: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_102: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_334);  mul_334 = None
        mul_335: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_332, sub_102);  mul_332 = sub_102 = None
        mul_336: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_335, 0.7978845608028654);  mul_335 = None
        mul_337: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_336, 0.044715)
        pow_22: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_52, 2.0);  view_52 = None
        mul_338: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_339: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        add_154: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_336, mul_339);  mul_336 = mul_339 = None
        mul_340: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_333, 0.5);  mul_333 = None
        add_155: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_154, mul_340);  add_154 = mul_340 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_483: "f32[4096, 3072]" = torch.ops.aten.view.default(add_155, [4096, 3072]);  add_155 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_219: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        mm_77: "f32[4096, 768]" = torch.ops.aten.mm.default(view_483, permute_219);  permute_219 = None
        mm_78: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_220, view_483);  permute_220 = None
        sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
        view_484: "f32[3072]" = torch.ops.aten.view.default(sum_126, [3072]);  sum_126 = None
        view_485: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_77, [4, 1024, 768]);  mm_77 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_342: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_485, primals_37);  primals_37 = None
        mul_343: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
        sum_127: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_18);  mul_342 = None
        sum_128: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_128);  sum_128 = None
        sub_104: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_127);  mul_343 = sum_127 = None
        sub_105: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_345);  sub_104 = mul_345 = None
        mul_346: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_105);  div_42 = sub_105 = None
        mul_347: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_485, mul_18);  mul_18 = None
        sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_485, [0, 1]);  view_485 = None
        add_156: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_153, mul_346);  add_153 = mul_346 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_28: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_156, getitem_36, 0.10000000000000009);  getitem_36 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_486: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_28, [4096, 768]);  npu_dropout_backward_28 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_221: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        mm_79: "f32[4096, 768]" = torch.ops.aten.mm.default(view_486, permute_221);  permute_221 = None
        mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_222, view_486);  permute_222 = None
        sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
        view_487: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
        view_488: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_79, [4, 1024, 768]);  mm_79 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_489: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_488, [4, 1024, 12, 64]);  view_488 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_223: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_120: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_223, [4, 12, 1024, 64]);  permute_223 = None
        clone_102: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_490: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_102, [48, 1024, 64]);  clone_102 = None
        bmm_60: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_490, view_491);  view_491 = None
        view_492: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_60, [4, 12, 1024, 1024]);  bmm_60 = None
        bmm_61: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_493, view_490);  view_493 = view_490 = None
        view_495: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_61, [4, 12, 1024, 64]);  bmm_61 = None
        add_157: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_495);  tangents_8 = view_495 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_29: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_492, getitem_34, 0.10000000000000009);  view_492 = getitem_34 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_348: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_29, _softmax_2);  npu_dropout_backward_29 = None
        sum_132: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [-1], True)
        mul_349: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_2, sum_132);  _softmax_2 = sum_132 = None
        sub_106: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_25: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_34, sub_106, full_default_25);  primals_34 = sub_106 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_43: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_25, full_default);  where_25 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_124: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_43, [4, 12, 1024, 1024]);  div_43 = None
        view_496: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_124, [48, 1024, 1024]);  expand_124 = None
        bmm_62: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_496, view_497);  view_497 = None
        view_498: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_62, [4, 12, 1024, 64]);  bmm_62 = None
        bmm_63: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_499, view_496);  view_499 = view_496 = None
        view_501: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_63, [4, 12, 64, 1024]);  bmm_63 = None
        permute_228: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_501, [0, 1, 3, 2]);  view_501 = None
        add_158: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, permute_228);  tangents_7 = permute_228 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_229: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_157, [0, 2, 1, 3]);  add_157 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_502: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_229, [4, 1024, 768]);  permute_229 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_230: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_158, [0, 2, 1, 3]);  add_158 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_503: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_230, [4, 1024, 768]);  permute_230 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_231: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_107: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_504: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_107, [4, 1024, 768]);  clone_107 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_9: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_504, view_503, view_502], 2);  view_504 = view_503 = view_502 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_505: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_9, [4096, 2304]);  cat_9 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_232: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        mm_81: "f32[4096, 768]" = torch.ops.aten.mm.default(view_505, permute_232);  permute_232 = None
        mm_82: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_233, view_505);  permute_233 = None
        sum_133: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True);  view_505 = None
        view_506: "f32[2304]" = torch.ops.aten.view.default(sum_133, [2304]);  sum_133 = None
        view_507: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_81, [4, 1024, 768]);  mm_81 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_351: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_507, primals_30);  primals_30 = None
        mul_352: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_351, 768)
        sum_134: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True)
        mul_353: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_351, mul_16);  mul_351 = None
        sum_135: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
        mul_354: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_135);  sum_135 = None
        sub_108: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_352, sum_134);  mul_352 = sum_134 = None
        sub_109: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_354);  sub_108 = mul_354 = None
        mul_355: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_109);  div_44 = sub_109 = None
        mul_356: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_507, mul_16);  mul_16 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1]);  mul_356 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_507, [0, 1]);  view_507 = None
        add_159: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_156, mul_355);  add_156 = mul_355 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_30: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_159, getitem_27, 0.10000000000000009);  getitem_27 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_508: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_30, [4096, 768]);  npu_dropout_backward_30 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_234: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        mm_83: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_508, permute_234);  permute_234 = None
        mm_84: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_235, view_508);  permute_235 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True);  view_508 = None
        view_509: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
        view_510: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_83, [4, 1024, 3072]);  mm_83 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_34: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, [4, 1024, 3072]);  addmm_6 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        mul_357: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_510, mul_12);  mul_12 = None
        pow_2: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_13: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_13);  mul_13 = None
        mul_14: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_15: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_358: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_510, add_15);  view_510 = add_15 = None
        mul_359: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_110: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_359);  mul_359 = None
        mul_360: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_357, sub_110);  mul_357 = sub_110 = None
        mul_361: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_360, 0.7978845608028654);  mul_360 = None
        mul_362: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_361, 0.044715)
        pow_23: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 2.0);  view_34 = None
        mul_363: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_364: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
        add_160: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_361, mul_364);  mul_361 = mul_364 = None
        mul_365: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_358, 0.5);  mul_358 = None
        add_161: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_160, mul_365);  add_160 = mul_365 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_511: "f32[4096, 3072]" = torch.ops.aten.view.default(add_161, [4096, 3072]);  add_161 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_236: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        mm_85: "f32[4096, 768]" = torch.ops.aten.mm.default(view_511, permute_236);  permute_236 = None
        mm_86: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_237, view_511);  permute_237 = None
        sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
        view_512: "f32[3072]" = torch.ops.aten.view.default(sum_139, [3072]);  sum_139 = None
        view_513: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_85, [4, 1024, 768]);  mm_85 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_367: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_513, primals_24);  primals_24 = None
        mul_368: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
        sum_140: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_10);  mul_367 = None
        sum_141: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_141);  sum_141 = None
        sub_112: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_140);  mul_368 = sum_140 = None
        sub_113: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_370);  sub_112 = mul_370 = None
        mul_371: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_113);  div_45 = sub_113 = None
        mul_372: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_513, mul_10);  mul_10 = None
        sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_513, [0, 1]);  view_513 = None
        add_162: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_159, mul_371);  add_159 = mul_371 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_31: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_162, getitem_23, 0.10000000000000009);  getitem_23 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_514: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_31, [4096, 768]);  npu_dropout_backward_31 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        mm_87: "f32[4096, 768]" = torch.ops.aten.mm.default(view_514, permute_238);  permute_238 = None
        mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_514);  permute_239 = None
        sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
        view_515: "f32[768]" = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
        view_516: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_87, [4, 1024, 768]);  mm_87 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_517: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_516, [4, 1024, 12, 64]);  view_516 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_240: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_128: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_240, [4, 12, 1024, 64]);  permute_240 = None
        clone_108: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_518: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_108, [48, 1024, 64]);  clone_108 = None
        bmm_64: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_518, view_519);  view_519 = None
        view_520: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_64, [4, 12, 1024, 1024]);  bmm_64 = None
        bmm_65: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_521, view_518);  view_521 = view_518 = None
        view_523: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_65, [4, 12, 1024, 64]);  bmm_65 = None
        add_163: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_523);  tangents_6 = view_523 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_32: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_520, getitem_21, 0.10000000000000009);  view_520 = getitem_21 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_373: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_32, _softmax_1);  npu_dropout_backward_32 = None
        sum_145: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [-1], True)
        mul_374: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax_1, sum_145);  _softmax_1 = sum_145 = None
        sub_114: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_26: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_21, sub_114, full_default_25);  primals_21 = sub_114 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_46: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_26, full_default);  where_26 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_132: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_46, [4, 12, 1024, 1024]);  div_46 = None
        view_524: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_132, [48, 1024, 1024]);  expand_132 = None
        bmm_66: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_524, view_525);  view_525 = None
        view_526: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_66, [4, 12, 1024, 64]);  bmm_66 = None
        bmm_67: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_527, view_524);  view_527 = view_524 = None
        view_529: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_67, [4, 12, 64, 1024]);  bmm_67 = None
        permute_245: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
        add_164: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, permute_245);  tangents_5 = permute_245 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_246: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_163, [0, 2, 1, 3]);  add_163 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_530: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_246, [4, 1024, 768]);  permute_246 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_247: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_164, [0, 2, 1, 3]);  add_164 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_531: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_247, [4, 1024, 768]);  permute_247 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_248: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_113: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_532: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_113, [4, 1024, 768]);  clone_113 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_10: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_532, view_531, view_530], 2);  view_532 = view_531 = view_530 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_533: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_10, [4096, 2304]);  cat_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_249: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        mm_89: "f32[4096, 768]" = torch.ops.aten.mm.default(view_533, permute_249);  permute_249 = None
        mm_90: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_250, view_533);  permute_250 = None
        sum_146: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
        view_534: "f32[2304]" = torch.ops.aten.view.default(sum_146, [2304]);  sum_146 = None
        view_535: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_89, [4, 1024, 768]);  mm_89 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_376: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_535, primals_17);  primals_17 = None
        mul_377: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_376, 768)
        sum_147: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True)
        mul_378: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_376, mul_8);  mul_376 = None
        sum_148: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True);  mul_378 = None
        mul_379: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_148);  sum_148 = None
        sub_116: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_377, sum_147);  mul_377 = sum_147 = None
        sub_117: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_379);  sub_116 = mul_379 = None
        mul_380: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_117);  div_47 = sub_117 = None
        mul_381: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_535, mul_8);  mul_8 = None
        sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 1]);  mul_381 = None
        sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_535, [0, 1]);  view_535 = None
        add_165: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_162, mul_380);  add_162 = mul_380 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:357 in forward, code: hidden_states = self.dropout(hidden_states)
        npu_dropout_backward_33: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_165, getitem_14, 0.10000000000000009);  getitem_14 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_536: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_33, [4096, 768]);  npu_dropout_backward_33 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_251: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        mm_91: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_536, permute_251);  permute_251 = None
        mm_92: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_252, view_536);  permute_252 = None
        sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
        view_537: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
        view_538: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(mm_91, [4, 1024, 3072]);  mm_91 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_16: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, [4, 1024, 3072]);  addmm_2 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
        mul_382: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_538, mul_4);  mul_4 = None
        pow_1: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_16, 3.0)
        mul_5: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_16, mul_5);  mul_5 = None
        mul_6: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_7: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_383: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_538, add_7);  view_538 = add_7 = None
        mul_384: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_118: "f32[4, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_384);  mul_384 = None
        mul_385: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_382, sub_118);  mul_382 = sub_118 = None
        mul_386: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_385, 0.7978845608028654);  mul_385 = None
        mul_387: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_386, 0.044715)
        pow_24: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_16, 2.0);  view_16 = None
        mul_388: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_389: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
        add_166: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_386, mul_389);  mul_386 = mul_389 = None
        mul_390: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_383, 0.5);  mul_383 = None
        add_167: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(add_166, mul_390);  add_166 = mul_390 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_539: "f32[4096, 3072]" = torch.ops.aten.view.default(add_167, [4096, 3072]);  add_167 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        mm_93: "f32[4096, 768]" = torch.ops.aten.mm.default(view_539, permute_253);  permute_253 = None
        mm_94: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_254, view_539);  permute_254 = None
        sum_152: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
        view_540: "f32[3072]" = torch.ops.aten.view.default(sum_152, [3072]);  sum_152 = None
        view_541: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_93, [4, 1024, 768]);  mm_93 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:426 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_392: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_541, primals_11);  primals_11 = None
        mul_393: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_392, 768)
        sum_153: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True)
        mul_394: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_392, mul_2);  mul_392 = None
        sum_154: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [2], True);  mul_394 = None
        mul_395: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_154);  sum_154 = None
        sub_120: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_393, sum_153);  mul_393 = sum_153 = None
        sub_121: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_395);  sub_120 = mul_395 = None
        mul_396: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_121);  div_48 = sub_121 = None
        mul_397: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_541, mul_2);  mul_2 = None
        sum_155: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1]);  mul_397 = None
        sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_541, [0, 1]);  view_541 = None
        add_168: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_165, mul_396);  add_165 = mul_396 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:335 in forward, code: attn_output = self.resid_dropout(attn_output)
        npu_dropout_backward_34: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_168, getitem_10, 0.10000000000000009);  getitem_10 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_542: "f32[4096, 768]" = torch.ops.aten.view.default(npu_dropout_backward_34, [4096, 768]);  npu_dropout_backward_34 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        mm_95: "f32[4096, 768]" = torch.ops.aten.mm.default(view_542, permute_255);  permute_255 = None
        mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_256, view_542);  permute_256 = None
        sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
        view_543: "f32[768]" = torch.ops.aten.view.default(sum_157, [768]);  sum_157 = None
        view_544: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_95, [4, 1024, 768]);  mm_95 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:288 in _merge_heads, code: return tensor.view(new_shape)
        view_545: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(view_544, [4, 1024, 12, 64]);  view_544 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:286 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_257: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:218 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_136: "f32[4, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_257, [4, 12, 1024, 64]);  permute_257 = None
        clone_114: "f32[4, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_546: "f32[48, 1024, 64]" = torch.ops.aten.view.default(clone_114, [48, 1024, 64]);  clone_114 = None
        bmm_68: "f32[48, 1024, 1024]" = torch.ops.aten.bmm.default(view_546, view_547);  view_547 = None
        view_548: "f32[4, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_68, [4, 12, 1024, 1024]);  bmm_68 = None
        bmm_69: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_549, view_546);  view_549 = view_546 = None
        view_551: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_69, [4, 12, 1024, 64]);  bmm_69 = None
        add_169: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_551);  tangents_4 = view_551 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:212 in _attn, code: attn_weights = self.attn_dropout(attn_weights)
        npu_dropout_backward_35: "f32[4, 12, 1024, 1024]" = torch.ops.npu.npu_dropout_backward.default(view_548, getitem_8, 0.10000000000000009);  view_548 = getitem_8 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:208 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_398: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(npu_dropout_backward_35, _softmax);  npu_dropout_backward_35 = None
        sum_158: "f32[4, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [-1], True)
        mul_399: "f32[4, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(_softmax, sum_158);  _softmax = sum_158 = None
        sub_122: "f32[4, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:202 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        where_27: "f32[4, 12, 1024, 1024]" = torch.ops.aten.where.self(primals_8, sub_122, full_default_25);  primals_8 = sub_122 = full_default_25 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:186 in _attn, code: attn_weights = attn_weights / torch.full(
        div_49: "f32[4, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_27, full_default);  where_27 = full_default = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:183 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_140: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_49, [4, 12, 1024, 1024]);  div_49 = None
        view_552: "f32[48, 1024, 1024]" = torch.ops.aten.view.default(expand_140, [48, 1024, 1024]);  expand_140 = None
        bmm_70: "f32[48, 1024, 64]" = torch.ops.aten.bmm.default(view_552, view_553);  view_553 = None
        view_554: "f32[4, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_70, [4, 12, 1024, 64]);  bmm_70 = None
        bmm_71: "f32[48, 64, 1024]" = torch.ops.aten.bmm.default(view_555, view_552);  view_555 = view_552 = None
        view_557: "f32[4, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_71, [4, 12, 64, 1024]);  bmm_71 = None
        permute_262: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_557, [0, 1, 3, 2]);  view_557 = None
        add_170: "f32[4, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, permute_262);  tangents_3 = permute_262 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_263: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_169, [0, 2, 1, 3]);  add_169 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_558: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_263, [4, 1024, 768]);  permute_263 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_264: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        view_559: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_264, [4, 1024, 768]);  permute_264 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_265: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:279 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_119: "f32[4, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
        view_560: "f32[4, 1024, 768]" = torch.ops.aten.view.default(clone_119, [4, 1024, 768]);  clone_119 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_11: "f32[4, 1024, 2304]" = torch.ops.aten.cat.default([view_560, view_559, view_558], 2);  view_560 = view_559 = view_558 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:109 in forward, code: x = x.view(size_out)
        view_561: "f32[4096, 2304]" = torch.ops.aten.view.default(cat_11, [4096, 2304]);  cat_11 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/pytorch_utils.py:108 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_266: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        mm_97: "f32[4096, 768]" = torch.ops.aten.mm.default(view_561, permute_266);  permute_266 = None
        mm_98: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_267, view_561);  permute_267 = None
        sum_159: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
        view_562: "f32[2304]" = torch.ops.aten.view.default(sum_159, [2304]);  sum_159 = None
        view_563: "f32[4, 1024, 768]" = torch.ops.aten.view.default(mm_97, [4, 1024, 768]);  mm_97 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:389 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_401: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_563, primals_4);  primals_4 = None
        mul_402: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_401, 768)
        sum_160: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True)
        mul_403: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_401, mul);  mul_401 = None
        sum_161: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True);  mul_403 = None
        mul_404: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, sum_161);  sum_161 = None
        sub_124: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_402, sum_160);  mul_402 = sum_160 = None
        sub_125: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_124, mul_404);  sub_124 = mul_404 = None
        mul_405: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_125);  div_50 = sub_125 = None
        mul_406: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_563, mul);  mul = None
        sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 1]);  mul_406 = None
        sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1]);  view_563 = None
        add_171: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_168, mul_405);  add_168 = mul_405 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:845 in forward, code: hidden_states = self.drop(hidden_states)
        npu_dropout_backward_36: "f32[4, 1024, 768]" = torch.ops.npu.npu_dropout_backward.default(add_171, getitem_1, 0.10000000000000009);  add_171 = getitem_1 = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:839 in forward, code: hidden_states = inputs_embeds + position_embeds
        sum_164: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(npu_dropout_backward_36, [0], True)

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:838 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_dense_backward: "f32[1024, 768]" = torch.ops.aten.embedding_dense_backward.default(sum_164, unsqueeze, 1024, -1, False);  sum_164 = unsqueeze = None

        # File: /opt/npu-envs/tnpu-2.13.0/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:837 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding_dense_backward_1: "f32[50257, 768]" = torch.ops.aten.embedding_dense_backward.default(npu_dropout_backward_36, primals_1, 50257, -1, False);  npu_dropout_backward_36 = primals_1 = None
        return (None, embedding_dense_backward_1, embedding_dense_backward, sum_162, sum_163, view_562, mm_98, None, view_543, mm_96, sum_155, sum_156, view_540, mm_94, view_537, mm_92, sum_149, sum_150, view_534, mm_90, None, view_515, mm_88, sum_142, sum_143, view_512, mm_86, view_509, mm_84, sum_136, sum_137, view_506, mm_82, None, view_487, mm_80, sum_129, sum_130, view_484, mm_78, view_481, mm_76, sum_123, sum_124, view_478, mm_74, None, view_459, mm_72, sum_116, sum_117, view_456, mm_70, view_453, mm_68, sum_110, sum_111, view_450, mm_66, None, view_431, mm_64, sum_103, sum_104, view_428, mm_62, view_425, mm_60, sum_97, sum_98, view_422, mm_58, None, view_403, mm_56, sum_90, sum_91, view_400, mm_54, view_397, mm_52, sum_84, sum_85, view_394, mm_50, None, view_375, mm_48, sum_77, sum_78, view_372, mm_46, view_369, mm_44, sum_71, sum_72, view_366, mm_42, None, view_347, mm_40, sum_64, sum_65, view_344, mm_38, view_341, mm_36, sum_58, sum_59, view_338, mm_34, None, view_319, mm_32, sum_51, sum_52, view_316, mm_30, view_313, mm_28, sum_45, sum_46, view_310, mm_26, None, view_291, mm_24, sum_38, sum_39, view_288, mm_22, view_285, mm_20, sum_32, sum_33, view_282, mm_18, None, view_263, mm_16, sum_25, sum_26, view_260, mm_14, view_257, mm_12, sum_19, sum_20, view_254, mm_10, None, view_235, mm_8, sum_12, sum_13, view_232, mm_6, view_229, mm_4, sum_6, sum_7, permute_63, None)
