"""Opt-in fused FFN dataflow preserving the two original SwiGLU arithmetic paths."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from comfy import ops
from comfy_kitchen.backends import cuda as native

from w4a8_ops import unpack_for_call
from fc1_ada import fc1_active_ada
from fc1_pair import prepare_fc1_pair
from fc2_dual import DualActivationFC2


ROWS = 1024
WIDTH = 5376


def stock_tile(x, destination, w1, s1, w2, s2, g1, g2, first, a1, second, a2):
    raw = native.int8_linear(x, w1, s1, out_dtype=torch.float32, convrot=True, convrot_groupsize=g1)
    first.add(raw, x, a1)
    linear = native.int8_linear(raw, w2, s2, out_dtype=torch.float32,
                                convrot=True, convrot_groupsize=g2, input_act="swiglu")
    gate, value = raw.chunk(2, dim=-1)
    second.add(linear, F.silu(gate).mul_(value), a2)
    destination.copy_(linear)


def paired_ada_mlp(module, hidden, first, second, counters):
    ops.run_every_op()
    w1, s1, group1, d1 = unpack_for_call(module.fc1, hidden)
    w2, s2, group2, d2 = unpack_for_call(module.fc2, hidden)
    a1, a2 = first.weights(hidden.device), second.weights(hidden.device)
    down, up = a1
    if (group1 != 256 or group2 != 256 or d1 != torch.float32 or d2 != torch.float32
            or down.ndim != 2 or up.ndim != 2 or down.shape[1] != 5376
            or up.shape != (28672, down.shape[0])
            or down.dtype != torch.float16 or up.dtype != torch.float16):
        raise RuntimeError("Fused FFN candidate requires the qualified FC1/FC2 geometry")
    out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    probes = first.stats["fp16_delta_relative_l2"]
    start = 0
    if first.role not in probes or second.role not in probes:
        begin = slice(0, ROWS)
        stock_tile(hidden[begin], out[begin], w1, s1, w2, s2, group1, group2, first, a1, second, a2)
        counters["first_stock_probe_rows"] = counters.get("first_stock_probe_rows", 0) + ROWS
        start += ROWS

    rank = down.shape[0]
    packed_up = up.view(2, 14336, rank).transpose(0, 1).contiguous().view(28672, rank)
    consumer = DualActivationFC2(w2, s2, group2, d2, second, a2)
    while start + 2 * ROWS <= hidden.shape[0]:
        stop = start + 2 * ROWS
        pair = hidden[start:stop]
        quant, row_scale = prepare_fc1_pair(pair, group1)
        delta0, delta1 = [F.linear(F.linear(tile.half(), down), packed_up) for tile in pair.split(ROWS)]
        precise, fast = fc1_active_ada(
            quant, row_scale, w1, s1, delta0, delta1,
            native_quant=True, lora_half=True, packed_delta=True, lora_scale=first.scale,
        )
        consumer.stage_dual(precise[:ROWS], fast[0], 0)
        consumer.stage_dual(precise[ROWS:], fast[1], 1)
        del quant, row_scale, precise, fast, delta0, delta1
        consumer.write_pair(out[start:stop])
        counters["fused_pairs"] = counters.get("fused_pairs", 0) + 1
        start = stop

    for pos in range(start, hidden.shape[0], ROWS):
        cut = slice(pos, min(pos + ROWS, hidden.shape[0]))
        stock_tile(hidden[cut], out[cut], w1, s1, w2, s2, group1, group2, first, a1, second, a2)
        counters["stock_tail_rows"] = counters.get("stock_tail_rows", 0) + cut.stop - cut.start
    return out


@contextmanager
def install(counters):
    """Attach to the loaded H3 streamed adapter while preserving other inputs."""
    import lora_adapter as adapter

    original = adapter.chunked_mlp

    def dispatch(module, hidden, rows, first, second):
        if (rows != ROWS or not hidden.is_cuda or hidden.dtype != torch.float32 or hidden.ndim != 2
                or hidden.shape[1] != WIDTH or hidden.shape[0] < 2 * ROWS):
            raise RuntimeError("Selected Ada FFN kernel does not support this input; no implicit backend fallback")
        return paired_ada_mlp(module, hidden, first, second, counters)

    adapter.chunked_mlp = dispatch
    try:
        yield
    finally:
        adapter.chunked_mlp = original
