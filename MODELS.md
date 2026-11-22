# MiniMax H3 Turbo Runtime — Upstream Model Assets and Specifications

[Home](README.md) · [English Home](README.en.md)

This repository is an **optimized inference runtime and compute engine**, not a weight-training or model-hosting distributor. All deep-learning weights required for execution are external open-source assets published by their respective authors. To ensure academic rigor, reproducibility, and intellectual property compliance, this document provides the exact upstream provenance, source repositories, file sizes, and cryptographic SHA-256 digests.

---

## 1. Upstream Engine Architecture & Component Roles

The complete inference pipeline connects the following specialized upstream components:

```
[Multimodal Inputs: Prompt / Keyframes / Media Guides]
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  1. Multimodal Conditioning Frontend                │
│     - Qwen3-VL 4B INT8 (first 24 layers truncated)  │
│     - MiniMax H3 ClipProj v3.1 projection layer     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼ Condition Latents
┌─────────────────────────────────────────────────────┐
│  2. Few-Step Denoising DiT Backbone                 │
│     - MiniMax H3 Base (FL2VA / Ref2VA)              │
│     - Community W4A8 quantized backbone             │
│     - LightX2V Turbo LoRA (4-step / 8-step)         │
│     - Runtime optimization: Triton 148-reg fusion   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼ Video/Audio Latents
┌─────────────────────────────────────────────────────┐
│  3. Native Audiovisual Codec Pipeline               │
│     - Official INT8 ConvRot Video VAE (tiled)       │
│     - Official FP32 Audio VAE                       │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
         [Completed Output: 24fps Native AV MP4]
```

---

## 2. Canonical Upstream Asset Registry (Core 6 Weights)

The default FL2VA task (first/last-frame and text-to-video) requires 6 core weight files totaling **23.67 GB** (22.05 GiB):

| ID | File Name | Upstream Publisher | Canonical Upstream Source (Pinned HuggingFace URL) | File Size | SHA-256 Digest | Recommended Path | GPU Memory Lifecycle |
|---|---|---|---|---:|---|---|---|
| `dit` | `MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors` | joeygambino | [joeygambino/MiniMax-H3-comfy-native-fl2va](https://huggingface.co/joeygambino/MiniMax-H3-comfy-native-fl2va/resolve/d24400b69702a0e242351a945bb23f22f702b36b/MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors) | 12,540,870,488 bytes (~12.54 GB) | `18ce05a9e588618db5ae6dca380f242bf8041d9d47f3ffb5cd80d9dec9f9b1ec` | `weights/` | Resides only during denoise |
| `lora` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | lightx2v | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/c2bf13b837e33bd851b370285fc7db99482d2ab2/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | 1,956,193,000 bytes (~1.96 GB) | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` | `weights/` | Resides only during denoise |
| `video_vae` | `minimax_h3_video_vae_int8_convrot.safetensors` | Kijai | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/c2e7b89b1800123c49a0f620a1932c01d8952256/minimax_h3_video_vae_int8_convrot.safetensors) | 3,171,670,912 bytes (~3.17 GB) | `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410` | `weights/` | Keyframe encode & video decode |
| `audio_vae` | `minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/3f57e82/vae/minimax_h3_audio_vae_fp32.safetensors) | 605,254,808 bytes (~0.61 GB) | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` | `weights/` | Resides only during audio decode |
| `text_encoder` | `qwen3vl_4b_int8_convrot.safetensors` | Stick9190 | [Stick9190/qwen3vl_4b_int8_convrot](https://huggingface.co/Stick9190/qwen3vl_4b_int8_convrot/resolve/d9449494156acb47a84c68777d78bd8d9d5df97d/qwen3vl_4b_int8_convrot.safetensors) | 5,374,169,456 bytes (~5.37 GB) | `ebbb3efdf4a6e1b766e32f6dc4fa92af9530edede0968c070b3e604aeca413aa` | `condition-models/` | Resides only during conditioning |
| `clip_projection` | `mmh3-4b-ClipProj-v3.1.safetensors` | NicoLab28 | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/resolve/main/mmh3-4b-ClipProj-v3.1.safetensors) | 26,256,128 bytes (~26.3 MB) | `0184e5c8d666a131962506d21949c2d8a8c6f33445b7b5e347e9a7e0a5baa819` | `condition-models/` | Resides only during conditioning |

---

## 3. Optional Ref2VA Reference Pair

Tasks requiring identity/audio references across shots (recommended with 8 steps) can swap the backbone/LoRA pair:

| ID | File Name | Upstream Publisher | Canonical Upstream Source (HuggingFace URL) | File Size | SHA-256 Digest |
|---|---|---|---|---:|---|
| `dit` (Ref2VA) | `MiniMax-H3-ref2va-pruned-comfy-w4a8.safetensors` | joeygambino / Kutches | [Kutches/minmax](https://huggingface.co/Kutches/minmax/resolve/0d14132311dd3818bd018ceef294ca8483da02e6/minimaxH3ComfyNativeRef2vaW4a8_v10.safetensors) | 12,540,870,488 bytes (~12.54 GB) | `ddb5e0590ec4702b8024c123c3e8b1d64539ba85446860e86c45783be8c3a09f` |
| `lora` (Ref2VA) | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | lightx2v | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/1a4a150fc6707329409eb52e5d75a0e987ac1928/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors) | 1,956,193,000 bytes (~1.96 GB) | `6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a` |

*(Note: Ref2VA reuses the same Video VAE, Audio VAE, Qwen3-VL text encoder, and ClipProj; no additional files are needed.)*

---

## 4. Local File Layout & Symlink Mounting

### Standard Directory Tree
```text
H3-Turbo-Runtime/
├── weights/
│   ├── MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors
│   ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   └── minimax_h3_audio_vae_fp32.safetensors
└── condition-models/
    ├── qwen3vl_4b_int8_convrot.safetensors
    └── mmh3-4b-ClipProj-v3.1.safetensors
```

### Symlinking Existing ComfyUI Models
If these weights already exist in your local ComfyUI installation, symlink them directly to save storage:
```bash
ln -s /path/to/ComfyUI/models/diffusion_models/MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors weights/
ln -s /path/to/ComfyUI/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors weights/
ln -s /path/to/ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors weights/
ln -s /path/to/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors weights/
ln -s /path/to/ComfyUI/models/text_encoders/qwen3vl_4b_int8_convrot.safetensors condition-models/
ln -s /path/to/ComfyUI/models/clip_projections/mmh3-4b-ClipProj-v3.1.safetensors condition-models/
```

---

## 5. Offline Cryptographic Verification & Retrieval

Use the bundled standard-library Python utility [`scripts/fetch_weights.py`](scripts/fetch_weights.py):

### Offline Hash Verification (Zero Network Traffic)
```bash
uv run python scripts/fetch_weights.py --check --models weights/ --condition-models condition-models/
```
Output upon full verification:
```text
All six weight files verified for variant fl2va
```

### Direct Upstream Download
Download directly from upstream publishers with resume support and atomic write:
```bash
uv run python scripts/fetch_weights.py --fetch --models weights/ --condition-models condition-models/
```
