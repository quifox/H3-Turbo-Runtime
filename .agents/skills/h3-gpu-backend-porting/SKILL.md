---
name: h3-gpu-backend-porting
description: Guide GPU, CUDA kernel, VRAM and long-video adaptation of the H3 Turbo Runtime without silent fallback or unverifiable changes to the frozen Ada implementation.
---

# H3 GPU/backend adaptation

Use this skill for new NVIDIA architectures, long-duration or high-resolution Fast execution, OOM, unsupported kernels, or hardware-specific tuning.

## Core rules

- Preserve the existing validated LightX2V Turbo W4A8/Ada baseline as an independent reference.
- Treat Fast and Stream as **caller-selected** backends. Never switch automatically after an error, OOM, shape mismatch, or unsupported architecture.
- Preserve the original CUDA/Python error and trace; do not conceal it with synthesized error classes or lower-resolution retries.
- The Fast fused kernel is currently qualified on SM89, not intrinsically restricted to RTX 4060. Port and validate kernels on other architectures before claiming compatibility.
- Do not confuse model invariants (H3 FFN shape, 56 heads, 128-d head, VAE temporal grid) with tuneable device choices (tile, warps, launch grid, register strategy).
- Do not run GPU jobs without user authorization and shared-device occupancy checks.

## Procedure

1. Inspect actual GPU compute capability, free memory, installed PyTorch/CUDA/Triton/comfy-kitchen and selected backend.
2. Reproduce the original error without changing execution strategy; read the corresponding stage log.
3. Locate affected source: runtime/compat.py, runtime/src/ffn_ada.py, fc1_ada.py, fc2_pair.py, qk_rope_writeback.py, attention_stream.py or av_pipeline.py.
4. Identify the precise missing hardware primitive, model geometry, dtype or memory bottleneck.
5. If kernel modification is needed, implement an explicitly chosen architecture-specific specialization. Keep the existing SM89 implementation unchanged until a real A/B warrants replacing it.
6. First run CPU syntax/interface checks. Only after separate permission, compare paired real latent hashes/tolerances, time, peak NVML memory and audio/video quality.
7. Record model/input hashes, device, driver, installed wheels, steps, resolution, backend, kernels selected, errors, time and GPU memory.

Read [backend-contract.md](references/backend-contract.md) for special parameters that should stay in source rather than in the public API.

## Output

Report exact failing component, supported range, source changes, correctness evidence, and whether this GPU/shape is actually verified. A successful import or smoke test does not establish production support.
