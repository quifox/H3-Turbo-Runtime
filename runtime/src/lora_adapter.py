"""Keep H3's W4A8 linears and apply the LightX2V low-rank residual layer by layer."""
from collections import Counter
from contextlib import contextmanager, ExitStack

from safetensors import safe_open
import torch
from torch.nn import functional as F

import comfy.ops as ops
from comfy.ldm.minimax.model import MLP
from comfy_kitchen.backends import cuda as native
from w4a8_ops import unpack_for_call
from adapter_spec import targets_from_keys, required_ffn_targets, require_targets


class Adapter:
    """Bind one shape-checked H3 LoRA target to a quantized projection module."""
    def __init__(self, tensors, prefix, module, stats, compact):
        self.tensors = tensors
        self.prefix = prefix
        self.down_key = prefix + ".lora_A.weight"
        self.up_key = prefix + ".lora_B.weight"
        rank, width = tensors.get_slice(self.down_key).get_shape()
        output, up_rank = tensors.get_slice(self.up_key).get_shape()
        if (width, output, rank) != (module.in_features, module.out_features, up_rank):
            raise ValueError(f"unexpected adapter shape for {prefix}")
        self.scale = float(tensors.get_tensor(prefix + ".alpha").item()) / rank
        self.stats = stats
        self.role = ".".join(prefix.split(".")[-2:])
        self.compact = compact if self.role == "attn.qkv_proj" else None

    def weights(self, device):
        self.stats["calls"][self.prefix] += 1
        down = self.tensors.get_tensor(self.down_key).to(device=device,dtype=torch.float16)
        source = self.compact if self.compact is not None else self.tensors
        up = source.get_tensor(self.up_key).to(device=device,dtype=torch.float16)
        return down, up

    def add(self, output, x, weights):
        down, up = weights
        low = F.linear(x.to(down.dtype), down)
        if up.ndim == 3:
            groups, channels, rank = up.shape
            projected = torch.bmm(low.reshape(-1,groups,rank).transpose(0,1).contiguous(),up.transpose(1,2))
            if "compact_qkv_bit_equal" not in self.stats:
                original = self.tensors.get_tensor(self.up_key).to(device=x.device,dtype=down.dtype)
                dense = F.linear(low,original).reshape(-1,groups,channels)
                self.stats["compact_qkv_bit_equal"] = bool(torch.equal(projected.transpose(0,1),dense))
            output.reshape(-1,groups,channels).transpose(0,1).add_(projected,alpha=self.scale)
            return
        delta = F.linear(low, up)
        probes = self.stats["fp16_delta_relative_l2"]
        if self.role not in probes:
            x32 = x[:8].float()
            reference = F.linear(F.linear(x32, down.float()), up.float())
            candidate = delta[:8].float()
            error = ((candidate - reference).norm() / reference.norm().clamp_min(1e-12)).item()
            if not (0 <= error < 0.02) or not (float(reference.norm()) > 0):
                raise RuntimeError(f"LoRA numerical check failed for {self.prefix}: {error}")
            probes[self.role] = error
        output.add_(delta, alpha=self.scale)

    def hook(self, rows):
        def apply(module, inputs, output):
            x = inputs[0].reshape(-1, module.in_features)
            y = output.view(-1, module.out_features)
            converted = self.weights(x.device)
            for start in range(0, x.shape[0], rows):
                stop = start + rows
                self.add(y[start:stop], x[start:stop], converted)
        return apply


def chunked_mlp(module, hidden, rows, first, second):
    ops.run_every_op()
    w1, s1, group1, d1 = unpack_for_call(module.fc1, hidden)
    w2, s2, group2, d2 = unpack_for_call(module.fc2, hidden)
    a1 = first.weights(hidden.device) if first is not None else None
    a2 = second.weights(hidden.device) if second is not None else None
    out = torch.empty((hidden.shape[0], module.fc2.out_features), device=hidden.device, dtype=d2)
    for start in range(0, hidden.shape[0], rows):
        stop = start + rows
        x = hidden[start:stop]
        ff1 = native.int8_linear(x, w1, s1, out_dtype=d1, convrot=True, convrot_groupsize=group1)
        if first is not None:
            first.add(ff1, x, a1)
        y = native.int8_linear(ff1, w2, s2, out_dtype=d2, convrot=True,
                               convrot_groupsize=group2, input_act="swiglu")
        if second is not None:
            second.add(y, ops._eager_input_act(ff1, "swiglu"), a2)
        out[start:stop].copy_(y)
        del ff1, y
    return out


@contextmanager
def apply_streamed_adapter(patcher, path, rows, stats, compact_path=None, *, require_all_ffn=False):
    """Install a model-scoped LoRA overlay; restore upstream hooks on exit."""
    transformer = patcher.model.diffusion_model
    with ExitStack() as session:
        tensors = session.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        compact = (session.enter_context(safe_open(str(compact_path), framework="pt", device="cpu"))
                   if compact_path else None)
        names = targets_from_keys(tensors.keys())
        main = {id(block.mlp): f"diffusion_model.blocks.{i}.mlp" for i, block in enumerate(transformer.blocks)}
        main_ffn = required_ffn_targets(len(transformer.blocks))
        stats.update(loaded_targets=len(names), main_ffn_targets=len(main_ffn & names),
                     calls=Counter(), fp16_delta_relative_l2={})
        if require_all_ffn:
            require_targets(names, main_ffn, "Selected Ada Fast FFN")
        adapters = {
            name: Adapter(tensors, name, patcher.model.get_submodule(name), stats, compact)
            for name in sorted(names)
        }
        handles = []
        forced = {}
        original = MLP.forward
        def forward(self, x):
            prefix = main.get(id(self))
            if prefix is None:
                return original(self, x)
            return chunked_mlp(self, x, rows, adapters.get(prefix + ".fc1"), adapters.get(prefix + ".fc2"))
        try:
            for name in sorted(names - main_ffn):
                module = patcher.model.get_submodule(name)
                if name.endswith(".mlp.fc2"):
                    forced[module] = module.comfy_force_forward
                    module.comfy_force_forward = True
                handles.append(module.register_forward_hook(adapters[name].hook(rows)))
            MLP.forward = forward
            yield names
        finally:
            MLP.forward = original
            for handle in handles:
                handle.remove()
            for module, flag in forced.items():
                module.comfy_force_forward = flag
