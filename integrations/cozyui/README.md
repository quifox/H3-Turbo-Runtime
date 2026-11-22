# CozyUI: Dedicated H3 Turbo Runtime Control Tab

[Home](../../README.md) · [English Home](../../README.en.md) · [ComfyUI Integration](../comfyui/README.md)

This integration provides a dedicated control tab within [CozyUI](https://github.com/SD-inst/cozyui) (pinned commit `6491487b20da5422a99c9c15adf0cb2277a7b0a1`).

---

## Installation & Setup

```bash
# 1. Configure ComfyUI custom node backend
python deploy.py configure-comfy --comfy /path/to/ComfyUI

# 2. Patch and configure CozyUI frontend
python integrations/cozyui/configure.py /path/to/cozyui

# 3. Build and launch CozyUI
cd /path/to/cozyui
corepack yarn vite --config vite.h3.config.ts
```

Ensure `H3_RUNTIME_MODELS` is exported in your environment. The frontend proxies to ComfyUI on port 8188 by default (configurable via `COMFYUI_URL`).

---

## Tab Capabilities & Controls

| Control Group | Supported Parameters |
|---|---|
| **Task Mode** | T2VA, I2VA, L2VA, FL2VA, Timeline, Ref2VA |
| **Execution Settings** | Duration (seconds), Steps (4 or 8), Seed, Strategy (Fast / Stream), Variant (auto / fl2va / ref2va) |
| **Resolution** | Explicit pixel dimensions or aspect-ratio presets with 16-pixel spatial alignment |
| **Media Inputs** | First/last keyframe uploads, intermediate timeline image guides, and multimodal references |
| **Ref2VA Conditioning** | Reference images, reference audio, and video tracks with audio passthrough controls |
| **Advanced Tuning** | `head_group`, `reserve_vram`, custom adapter LoRA, and sampling recipe paths |

All generation requests are dispatched to the decoupled H3 Turbo background executor, displaying real-time progress and outputting native audiovisual MP4s.
