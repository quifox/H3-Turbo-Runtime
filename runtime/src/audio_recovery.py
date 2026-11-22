"""Locate the first invalid audio rows and optionally recompute those FFN rows."""
from contextlib import contextmanager
import os
from unittest.mock import patch

from comfy_kitchen.backends import cuda as native
import comfy.ops as ops
import torch
from torch.nn import functional as F

from w4a8_ops import unpack_for_call
import lora_adapter as routing


def invalid_rows(tensor):
    return (~(torch.isfinite(tensor.amin(dim=1)) & torch.isfinite(tensor.amax(dim=1)))).nonzero().flatten()


def fp32_lora_add(output, x, adapter):
    down = adapter.tensors.get_tensor(adapter.down_key).to(device=x.device, dtype=torch.float32)
    up = adapter.tensors.get_tensor(adapter.up_key).to(device=x.device, dtype=torch.float32)
    output.add_(F.linear(F.linear(x, down), up), alpha=adapter.scale)


def inspect_one_audio_row(module, x, first, second):
    """One-row diagnostic, never an equivalence claim for the chunked GEMM."""
    def summary(value):
        finite = torch.isfinite(value)
        remaining = value[finite]
        return {"bad": int((~finite).count_nonzero()),
                "max_finite_abs": float(remaining.abs().max()) if remaining.numel() else None,
                "dtype": str(value.dtype)}

    def half_delta(value, adapter):
        down = adapter.tensors.get_tensor(adapter.down_key).to(device=x.device, dtype=torch.float16)
        up = adapter.tensors.get_tensor(adapter.up_key).to(device=x.device, dtype=torch.float16)
        return F.linear(F.linear(value.to(torch.float16), down), up)

    w1, s1, group1, dtype1 = unpack_for_call(module.fc1, x)
    w2, s2, group2, dtype2 = unpack_for_call(module.fc2, x)
    raw = native.int8_linear(x, w1, s1, out_dtype=dtype1, convrot=True,
                             convrot_groupsize=group1)
    fp16_delta1 = half_delta(x, first)
    after1 = raw.clone()
    after1.add_(fp16_delta1, alpha=first.scale)
    details = {"fc1_base": summary(raw), "fc1_fp16_delta": summary(fp16_delta1),
               "fc1_after_fp16_lora": summary(after1)}
    if not torch.isfinite(after1).all():
        return details
    hidden2 = ops._eager_input_act(after1, "swiglu")
    details["swiglu"] = summary(hidden2)
    if not torch.isfinite(hidden2).all():
        return details
    output2 = native.int8_linear(after1, w2, s2, out_dtype=dtype2,
                                 convrot=True, convrot_groupsize=group2, input_act="swiglu")
    delta2 = half_delta(hidden2, second)
    final = output2.clone()
    final.add_(delta2, alpha=second.scale)
    details.update(fc2_base=summary(output2), fc2_fp16_delta=summary(delta2),
                   fc2_after_fp16_lora=summary(final))
    return details


def recompute_ffn(module, x, first, second):
    w1, s1, group1, dtype1 = unpack_for_call(module.fc1, x)
    w2, s2, group2, dtype2 = unpack_for_call(module.fc2, x)
    raw = native.int8_linear(x, w1, s1, out_dtype=dtype1, convrot=True, convrot_groupsize=group1)
    fp32_lora_add(raw, x, first)
    value = native.int8_linear(raw, w2, s2, out_dtype=dtype2, convrot=True,
                              convrot_groupsize=group2, input_act="swiglu")
    fp32_lora_add(value, ops._eager_input_act(raw, "swiglu"), second)
    return value


@contextmanager
def watch_audio_rows(model, repair, stats):
    diffusion = model.model.diffusion_model
    original_blocks = diffusion._run_blocks
    original_mlp = routing.chunked_mlp
    span = None
    step = 0
    first_bad = False
    layer_ids = {id(block.mlp): i for i, block in enumerate(diffusion.blocks)}
    handles = []

    def observe(layer, stage, value):
        nonlocal first_bad
        if first_bad or span is None:
            return
        bad = invalid_rows(value[span])
        if bad.numel():
            row = {"step": step, "layer": layer, "stage": stage,
                   "audio_rows": bad.tolist()}
            stats.setdefault("first_invalid_audio_output", []).append(row)
            print("AUDIO_FIRST_INVALID", row, flush=True)
            first_bad = True

    def blocks(h, t_emb, mod_segments, layout, transformer_options):
        nonlocal step, span, first_bad
        step += 1
        first_bad = False
        start, end, _ = next(segment for segment in layout.segments if segment[2] == "audio")
        span = slice(start, end)
        print("AUDIO_SPAN", step, start, end, flush=True)
        try:
            observe(-1, "before_main_blocks", h)
            return original_blocks(h, t_emb, mod_segments, layout, transformer_options)
        finally:
            span = None

    def mlp(module, hidden, rows, first, second):
        output = original_mlp(module, hidden, rows, first, second)
        if span is None or not repair:
            return output
        indices = invalid_rows(output[span]) + span.start
        if not indices.numel():
            return output
        x = hidden.index_select(0, indices)
        if invalid_rows(x).numel():
            return output
        if os.getenv("H3_DIAG_AUDIO") == "1":
            try:
                print("AUDIO_SINGLE_ROW_FP16_DIAGNOSTIC",
                      {"step": step, "layer": layer_ids[id(module)],
                       "rows": indices.tolist(),
                       "stage_stats": inspect_one_audio_row(module, x[:1], first, second)},
                      flush=True)
            except Exception as error:
                print("AUDIO_SINGLE_ROW_DIAGNOSTIC_ERROR", repr(error), flush=True)
        repaired = recompute_ffn(module, x, first, second)
        if invalid_rows(repaired).numel():
            return output
        output.index_copy_(0, indices, repaired)
        row = {"step": step, "layer": layer_ids[id(module)], "rows": indices.tolist()}
        stats.setdefault("audio_ffn_recomputed_rows", []).append(row)
        print("AUDIO_FFN_FP32_RECOMPUTE", row, flush=True)
        return output

    for i, block in enumerate(diffusion.blocks):
        def after_attn(module, inputs, value, i=i):
            observe(i, "attention", value)

        def before_ffn(module, inputs, i=i):
            observe(i, "before_ffn", inputs[0])

        def after_ffn(module, inputs, value, i=i):
            observe(i, "ffn", value)

        def after_block(module, inputs, value, i=i):
            observe(i, "after_gate", value)

        handles.extend((block.attn.register_forward_hook(after_attn),
                        block.mlp.register_forward_pre_hook(before_ffn),
                        block.mlp.register_forward_hook(after_ffn),
                        block.register_forward_hook(after_block)))
    try:
        with patch.object(diffusion, "_run_blocks", blocks), patch.object(routing, "chunked_mlp", mlp):
            yield
    finally:
        for handle in handles:
            handle.remove()
