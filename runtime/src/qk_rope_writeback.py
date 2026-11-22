"""Preserve PyTorch FP32 RMSNorm and fuse only H3's partial RoPE writeback."""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn.functional import rms_norm
import triton
import triton.language as tl


@triton.jit
def rotate_write(X, F, Y, S: tl.constexpr, H: tl.constexpr,
                 XS: tl.constexpr, XH: tl.constexpr, YS: tl.constexpr, YH: tl.constexpr,
                 BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    seq = i // (H * 128)
    head = i // 128 % H
    col = i % 128
    valid = seq < S
    rotate = col < 96
    upper = col < 48
    index = tl.where(upper, col, col - 48)
    partner = tl.where(upper, col + 48, col - 48)
    src = seq * XS + head * XH
    freq = seq * 192 + index * 4
    val = tl.load(X + src + col, valid, other=0)
    other = tl.load(X + src + partner, valid & rotate, other=0)
    diag = tl.load(F + freq + tl.where(upper, 0, 3), valid & rotate, other=0)
    off = tl.load(F + freq + tl.where(upper, 1, 2), valid & rotate, other=0)
    result = tl.fma(other, off, val * diag)
    dst = seq * YS + head * YH
    tl.store(Y + dst + col, tl.where(rotate, result, val), valid)


def supported(q, k, freq, qw, kw, rot_dim):
    return (q.ndim == k.ndim == 4 and q.shape == k.shape
            and q.shape[0] == 1 and q.shape[-1] == 128 and q.shape[2] == 56
            and rot_dim == 96 and freq.shape == (1, q.shape[1], 1, 48, 2, 2)
            and freq.is_contiguous() and q.is_cuda and q.dtype == k.dtype == freq.dtype == torch.float32
            and q.stride(-1) == k.stride(-1) == 1
            and qw.shape == kw.shape == (128,))


def apply(q, k, freq, qw, kw, epsilon, rot_dim):
    for tensor, weight in ((q, qw), (k, kw)):
        normalized = rms_norm(tensor, (128,), weight=weight, eps=epsilon)
        assert normalized.stride(-1) == 1
        _, seq, heads, width = tensor.shape
        rotate_write[(triton.cdiv(seq * heads * width, 1024),)](
            normalized, freq, tensor, seq, heads,
            normalized.stride(1), normalized.stride(2), tensor.stride(1), tensor.stride(2),
            1024, enable_fp_fusion=False,
        )
    return q, k


@contextmanager
def install(active, events):
    import comfy_kitchen as kitchen

    original = kitchen.rms_rope_split_half_

    def patched(q, k, freq, qw, kw=None, epsilon=1e-6, rot_dim=0):
        kw = qw if kw is None else kw
        if not supported(q, k, freq, qw, kw, rot_dim):
            if active:
                raise RuntimeError("Selected fused Q/K kernel does not support this input; no implicit fallback")
            return original(q, k, freq, qw, kw, epsilon, rot_dim)
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record()
        if active:
            out = apply(q, k, freq, qw, kw, epsilon, rot_dim)
        else:
            out = original(q, k, freq, qw, kw, epsilon, rot_dim)
        after.record()
        events.append((before, after))
        return out

    with patch.object(kitchen, "rms_rope_split_half_", patched):
        yield
