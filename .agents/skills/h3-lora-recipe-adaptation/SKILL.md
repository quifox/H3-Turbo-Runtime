---
name: h3-lora-recipe-adaptation
description: Guide integration of community H3 LoRA weights or sampling recipes while preserving the proven LightX2V Turbo default and preventing incompatible LoRA, scheduler, shape or GPU combinations.
---

# H3 LoRA and sampling recipe adaptation

Use when connecting another MiniMax H3 fine-tune, adapting a LoRA rank/alpha, or exploring a non-Turbo Base sampler.

## Rules

- Default validated recipe: H3 FL2VA W4A8, LightX2V FL2VA Turbo 8-step v1.0 adapter, Euler/simple sampler, shifts video=12/audio=3, four or eight steps.
- A LoRA being loadable does not mean it supports four-step distillation. A normal style/subject LoRA can require a Base sampler with many steps.
- Ada Fast requires FC1 and FC2 LoRA on every main H3 MLP block. Missing targets are errors, not reasons for implicit stock FFN.
- Stream's tiled QKV path requires all attention projection targets needed by that implementation. Do not silently disable the selected strategy.
- For other LoRAs, inspect target map, A/B/alpha triplets, rank, scaling, shapes, dtype, variant and author's training-time sampler together.
- Do not change frozen verified input, default LoRA, numerical semantics, quality or backend by accident.
- Do not run unapproved GPU jobs.

## Procedure

1. Confirm FL2VA versus Ref2VA backbone and the fine-tune's original base-model identity.
2. Inspect safetensors keys and scale from runtime/adapter_spec.py and runtime/src/lora_adapter.py. Check every required target's in/out dimensions, rank, alpha and data type.
3. Inspect runtime/src/ffn_ada.py, fc1_ada.py and fc2_pair.py. The Ada kernel is specialized but now accepts compatible LoRA rank/scale data without hard-coding the LightX2V values as semantic requirements.
4. Confirm the author's intended sampler, noise schedule, number of steps and video/audio shifts. Never infer these from LoRA shapes.
5. Set the public --adapter file. If the sampling recipe differs, create a separate recipe JSON using runtime/models/recipes.json as schema and pass --recipe-file; keep the selected model variant consistent.
6. Run CPU-level contract tests and inspect exact error messages. Do not add a fallback to bypass mismatch.
7. Only with GPU authorization, verify video and audio latents, output quality, peak memory, kernel dispatch and generation time. Report unverified configurations as experimental.

Read [adapter-contract.md](references/adapter-contract.md) for the compatibility matrix and known evidence limits.

## Output

Provide a model/adapter provenance sheet, precise target coverage, rank/scale, selected recipe and source of its settings, explicit backend, any raw error, and verification status. This skill guides developers; it is not an automatic adapter converter.
