"""Ada 的双点积 FC1，直接产生 SwiGLU 结果，采用原始 gate/value 权重布局。"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from comfy_kitchen.backends import cuda as native


INPUT = 5376
WIDTH = 14336
ROWS = 1024


@triton.jit
def kitchen_swiglu(gate, value, use_div: tl.constexpr):
    # Kitchen compiles load_input_act<kActSwiGLU> with --use_fast_math.
    # In CUDA, 0fbfb8aa3b is the FP32 encoding of -log2(e).
    if use_div:
        return tl.inline_asm_elementwise(
            """{
            .reg .f32 arg, exponent, denom, logistic;
            mul.ftz.f32 arg, $1, 0fbfb8aa3b;
            ex2.approx.ftz.f32 exponent, arg;
            add.ftz.f32 denom, 0f3f800000, exponent;
            div.approx.ftz.f32 logistic, $1, denom;
            mul.ftz.f32 $0, logistic, $2;
            }""",
            constraints="=f,f,f", args=[gate, value], dtype=tl.float32, is_pure=True, pack=1,
        )
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 arg, exponent, denom, inv, logistic;
        mul.ftz.f32 arg, $1, 0fbfb8aa3b;
        ex2.approx.ftz.f32 exponent, arg;
        add.ftz.f32 denom, 0f3f800000, exponent;
        rcp.approx.ftz.f32 inv, denom;
        mul.ftz.f32 logistic, $1, inv;
        mul.ftz.f32 $0, logistic, $2;
        }""",
        constraints="=f,f,f", args=[gate, value], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _fc1_active(
    a8, w8, row_scale, col_scale, delta0, delta1, out, out_native,
    inputs: tl.constexpr, active: tl.constexpr, half_rows: tl.constexpr,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_m: tl.constexpr,
    native_quant: tl.constexpr, use_div: tl.constexpr, packed_delta: tl.constexpr,
    lora_scale: tl.constexpr,
):
    # 分在同一组的相邻行共享相同的权重面板。
    tile = tl.program_id(0)
    nt: tl.constexpr = active // bn
    row_block = (tile // (group_m * nt)) * group_m + (tile % group_m)
    col_block = (tile // group_m) % nt
    mr = row_block * bm + tl.arange(0, bm)
    nc = col_block * bn + tl.arange(0, bn)
    kc = tl.arange(0, bk)
    a = a8 + mr[:, None] * inputs + kc[None, :]
    b = w8 + kc[:, None] + nc[None, :] * inputs
    gate_acc = tl.full((bm, bn), 0, tl.int32)
    value_acc = tl.full((bm, bn), 0, tl.int32)
    for _ in range(inputs // bk):
        mat_a = tl.load(a)
        gate_acc = tl.dot(mat_a, tl.load(b), gate_acc, out_dtype=tl.int32)
        value_acc = tl.dot(mat_a, tl.load(b + active * inputs), value_acc, out_dtype=tl.int32)
        a += bk
        b += bk

    rs = tl.load(row_scale + mr)
    cg = tl.load(col_scale + nc)
    cv = tl.load(col_scale + nc + active)
    # 对齐原生无偏置访问器的两次乘法和 +0，保持有符号零约定。
    gate = (gate_acc.to(tl.float32) * rs[:, None]) * cg[None, :] + 0.0
    value = (value_acc.to(tl.float32) * rs[:, None]) * cv[None, :] + 0.0
    delta_base = tl.where(mr < half_rows, delta0, delta1)[:, None]
    row_offset = (mr % half_rows)[:, None] * (2 * active)
    if packed_delta:
        selected_delta = delta_base + row_offset + 2 * nc[None, :]
        dg = tl.load(selected_delta).to(tl.float32)
        dv = tl.load(selected_delta + 1).to(tl.float32)
    else:
        selected_delta = delta_base + row_offset + nc[None, :]
        dg = tl.load(selected_delta).to(tl.float32)
        dv = tl.load(selected_delta + active).to(tl.float32)
    gate = tl.fma(dg, lora_scale, gate)
    value = tl.fma(dv, lora_scale, value)
    silu = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[gate, 1.0 + libdevice.exp(-gate)], dtype=tl.float32, is_pure=True, pack=1,
    )
    output_index = mr[:, None] * active + nc[None, :]
    tl.store(out + output_index, silu * value)
    if native_quant:
        fast = kitchen_swiglu(gate, value, use_div)
        if packed_delta:
            # Each column owns exactly one aligned slot. Both half loads precede overwriting that slot.
            tl.store(selected_delta.to(tl.pointer_type(tl.float32)), fast)
        else:
            tl.store(out_native + output_index, fast)


def fc1_active_ada(a8, row_scale, w8, weight_scale, delta0, delta1, *, native_quant=False, use_div=True,
                   lora_half=False, packed_delta=False, lora_scale=0.0625):
    """保留两个原 M1024 LoRA 增量，可另输出原生量化使用的快数学激活。

    packed_delta 模式消费按门/值交错的 delta，并将其原地改写为 FP32 量化输入。
    """
    dev = a8.device
    if (not torch.is_inference_mode_enabled() or not a8.is_cuda
            or torch.cuda.get_device_capability(dev) != (8, 9)):
        raise ValueError("This direct FC1 path requires Ada SM89 inference")
    col_scale = native._int8_weight_scale_arg(weight_scale, dev).reshape(-1)
    row_scale = row_scale.reshape(-1)
    specs = (
        (a8, (2 * ROWS, INPUT), torch.int8), (w8, (2 * WIDTH, INPUT), torch.int8),
        (row_scale, (2 * ROWS,), torch.float32), (col_scale, (2 * WIDTH,), torch.float32),
        (delta0, (ROWS, 2 * WIDTH), torch.float16), (delta1, (ROWS, 2 * WIDTH), torch.float16),
    )
    if not all(t.shape == shape and t.dtype == dtype and t.device == dev and t.is_contiguous()
               for t, shape, dtype in specs):
        raise ValueError("Expected paired H3 FC1 inputs and original LoRA deltas")
    if packed_delta and (
        not native_quant or not lora_half or delta0.data_ptr() % 4 or delta1.data_ptr() % 4
        or abs(delta0.data_ptr() - delta1.data_ptr()) < delta0.numel() * delta0.element_size()
    ):
        raise ValueError("Packed delta reuse requires distinct aligned FP16 buffers and dual output")
    lora_dtype = torch.float16 if lora_half else torch.float32
    out = torch.empty((2 * ROWS, WIDTH), device=dev, dtype=lora_dtype)
    out_native = torch.empty(out.shape, device=dev, dtype=torch.float32) if native_quant and not packed_delta else out
    bm, bn, bk, group = 64, 32, 64, 4
    _fc1_active[(2 * ROWS // bm * (WIDTH // bn),)](
        a8, w8, row_scale, col_scale, delta0, delta1, out, out_native, INPUT, WIDTH, ROWS,
        bm, bn, bk, group, native_quant, use_div, packed_delta, lora_scale,
        num_warps=4, num_stages=3, enable_fp_fusion=False,
    )
    if packed_delta:
        return out, (delta0.view(torch.float32), delta1.view(torch.float32))
    return (out, out_native) if native_quant else out
