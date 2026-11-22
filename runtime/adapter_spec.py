"""Pure validation of H3-compatible LoRA target inventories (no CUDA imports).

This layer decides which adapter targets are present; GPU kernels stay specialized.
Missing target requirements are errors for selected backends, never reasons to
silently downgrade the execution path.
"""


def targets_from_keys(keys):
    """List LoRA targets and reject incomplete A/B/alpha tensor triplets."""
    keys = set(keys)
    suffixes = (".lora_A.weight", ".lora_B.weight", ".alpha")
    targets = {key[:-len(suffix)] for key in keys for suffix in suffixes if key.endswith(suffix)}
    if not targets:
        raise ValueError("No H3 LoRA A/B/alpha target triplets were found")
    for target in sorted(targets):
        missing = [suffix for suffix in suffixes if target + suffix not in keys]
        if missing:
            raise ValueError(f"Missing LoRA tensors for {target}: {missing}")
    return targets


def required_ffn_targets(block_count):
    return {f"diffusion_model.blocks.{i}.mlp.fc{j}"
            for i in range(block_count) for j in (1, 2)}


def required_stream_attention_targets(block_count):
    return {f"diffusion_model.blocks.{i}.attn.{name}"
            for i in range(block_count) for name in ("qkv_proj", "out_proj")}


def require_targets(found, required, backend):
    """Fail before execution if a selected backend is missing required LoRA targets."""
    missing = sorted(set(required) - set(found))
    if missing:
        raise ValueError(f"{backend} requires {len(missing)} additional LoRA targets; first: {missing[:5]}")
