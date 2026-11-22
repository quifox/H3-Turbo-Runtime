"""Bound projection and attention workspace by processing independent head groups."""
from contextlib import contextmanager
from unittest.mock import patch

import comfy.model_management
import comfy.quant_ops
import comfy.ops as ops
from comfy.ldm.minimax.model import Attention
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy_kitchen.backends import cuda as native
from comfy_kitchen.backends.eager import quantization as eager
from safetensors import safe_open
import torch
from torch.nn import functional as F

from w4a8_ops import unpack_for_call
from lora_adapter import Adapter
from adapter_spec import targets_from_keys, require_targets, required_stream_attention_targets


def gather_rows(tensor, start, end, inner):
    return torch.cat([tensor[start + offset:end + offset]
                      for offset in (0, inner, 2 * inner)], dim=0)


def attention_inputs(module, projected, heads, rope, norm_weights):
    s = projected.shape[0]
    width = module.head_dim
    q, k, v = [part.view(s, heads, width) for part in projected.split(heads * width, dim=1)]
    if rope is not None:
        qw, kw = norm_weights
        comfy.quant_ops.ck.rms_rope_split_half_(
            q.unsqueeze(0), k.unsqueeze(0), rope, qw, kw,
            epsilon=module.q_norm.eps, rot_dim=rope.shape[-3] * 2)
    else:
        q = module.q_norm(q)
        k = module.k_norm(k)
    return tuple(AttentionTensorContainer(part.transpose(0, 1).unsqueeze(0)) for part in (q, k, v))


def write_projected(xq, xs, transposed, scale, destination):
    acc = eager._int8_matmul_accumulate(xq, transposed)
    torch.mul(acc.float(), xs.float() * scale, out=destination)


def project_eager(quantized, weight, scale, dtype):
    xq, xs = quantized
    result = torch.empty((xq.shape[0], weight.shape[0]), dtype=dtype, device=xq.device)
    transposed = weight.T.contiguous()
    scale = scale.float().reshape(1, -1)
    for begin in range(0, xq.shape[0], 2048):
        subset = slice(begin, begin + 2048)
        write_projected(xq[subset], xs[subset], transposed, scale, result[subset])
    return result


def project_output(layer, hidden, adapter):
    """Compute the original biasless FP32 output projection one token block at a time."""
    ops.run_every_op()
    weight, scale, convrot_size, dtype = unpack_for_call(layer, hidden)
    transposed = weight.T.contiguous()
    del weight
    scale = scale.float().reshape(1, -1)
    rotation = eager._build_hadamard(convrot_size, device=hidden.device, dtype=hidden.dtype)
    down_up = adapter.weights(hidden.device)
    result = torch.empty((hidden.shape[0], layer.out_features), dtype=dtype, device=hidden.device)
    # The base projection has an independent activation scale for every token.
    # Keep the LoRA row boundaries equal to the original adapter hook.
    for begin in range(0, hidden.shape[0], 1024):
        subset = slice(begin, begin + 1024)
        x = hidden[subset]
        xq, xs = eager.quantize_int8_rowwise(eager._rotate_activation(x, rotation, convrot_size))
        write_projected(xq, xs, transposed, scale, result[subset])
        adapter.add(result[subset], x, down_up)
    return result


def stream_attention(module, x, rope, options, adapter, group_size, projection, output_adapter=None):
    if x.ndim != 2 or not x.is_cuda or options.get("optimized_attention_override") is not None:
        raise ValueError("Head-streaming requires unmasked native H3 inference")
    if x.dtype != torch.float32 or module.head_dim != 128 or module.heads % group_size:
        raise ValueError("Unsupported H3 attention geometry")
    ops.run_every_op()
    weight, scale, convrot_size, result_dtype = unpack_for_call(module.qkv_proj, x)
    if result_dtype != x.dtype:
        raise ValueError("Projection arithmetic differs from the FP32 reference")
    quantized = None
    if projection in ("eager-heads", "eager-heads-tiled-output"):
        rotation = eager._build_hadamard(convrot_size, device=x.device, dtype=x.dtype)
        quantized = eager.quantize_int8_rowwise(eager._rotate_activation(x, rotation, convrot_size))
    down, up = adapter.weights(x.device)
    inner = module.heads * module.head_dim
    actual_rows = x.shape[0]
    sums = torch.empty((actual_rows, down.shape[0]), dtype=down.dtype, device=x.device)
    for begin in range(0, actual_rows, 1024):
        subset = slice(begin, begin + 1024)
        sums[subset].copy_(F.linear(x[subset].to(down.dtype), down))
    norm_weights = None
    if rope is not None:
        norm_weights = tuple(comfy.model_management.cast_to(layer.weight, device=x.device)
                             for layer in (module.q_norm, module.k_norm))

    output = torch.empty((actual_rows, inner), dtype=x.dtype, device=x.device)
    for head in range(0, module.heads, group_size):
        start = head * module.head_dim
        end = start + group_size * module.head_dim
        block_weight = gather_rows(weight, start, end, inner)
        block_scale = gather_rows(scale.reshape(-1), start, end, inner)
        if quantized is None:
            projected = native.int8_linear(x, block_weight, block_scale, out_dtype=result_dtype,
                                           convrot=True, convrot_groupsize=convrot_size)
        else:
            projected = project_eager(quantized, block_weight, block_scale, result_dtype)
        del block_weight, block_scale
        expansion = gather_rows(up, start, end, inner)
        for begin in range(0, actual_rows, 1024):
            subset = slice(begin, begin + 1024)
            projected[subset].add_(F.linear(sums[subset], expansion), alpha=adapter.scale)
        inputs = attention_inputs(module, projected, group_size, rope, norm_weights)
        # Comfy Kitchen takes the inputs from the containers. Once it packs
        # them into INT8, the original FP32 projection can be reclaimed.
        del projected, expansion
        attended = optimized_attention(
            *inputs, group_size, preferred_attention=module.comfy_attention, skip_reshape=True,
            mask=None, transformer_options=options,
        ).squeeze(0)
        output[:, start:end].copy_(attended)
        del inputs, attended
    del weight, scale, sums, up, down, quantized
    if output_adapter is None:
        result = module.out_proj(output)
    else:
        result = project_output(module.out_proj, output, output_adapter)
    return result


@contextmanager
def stream_qkv(patcher, path, group_size, stats, projection):
    transformer = patcher.model.diffusion_model
    original = Attention.forward
    with safe_open(path, framework="pt", device="cpu") as tensors:
        if projection == "eager-heads-tiled-output":
            require_targets(targets_from_keys(tensors.keys()),
                            required_stream_attention_targets(len(transformer.blocks)),
                            "Selected head-streaming attention")
        targets = {}
        for index, block in enumerate(transformer.blocks):
            prefix = f"diffusion_model.blocks.{index}.attn"
            qkv = Adapter(tensors, prefix + ".qkv_proj", block.attn.qkv_proj, stats, None)
            out = (Adapter(tensors, prefix + ".out_proj", block.attn.out_proj, stats, None)
                   if projection == "eager-heads-tiled-output" else None)
            targets[id(block.attn)] = (qkv, out)

        def forward(module, x, rope_freqs=None, transformer_options=None):
            options = {} if transformer_options is None else transformer_options
            adapters = targets.get(id(module))
            if adapters is None:
                return original(module, x, rope_freqs=rope_freqs, transformer_options=options)
            adapter, output_adapter = adapters
            stats["streamed_qkv_calls"] = stats.get("streamed_qkv_calls", 0) + 1
            return stream_attention(module, x, rope_freqs, options, adapter, group_size, projection,
                                    output_adapter=output_adapter)

        with patch.object(Attention, "forward", forward):
            yield
