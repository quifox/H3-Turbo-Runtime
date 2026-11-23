# H3 Turbo Runtime

[简体中文](README.md) · English

A low-VRAM inference runtime for MiniMax H3 Turbo with **text-to-video, first/last frame and timed multi-frame guidance, configurable durations, and native audio/video output**. The default is five seconds and four denoising steps. **Measured on an RTX 4060 Laptop 8 GiB (hermetic environment managed via `uv` with Python 3.14): a 5-second clip in 159.18 s (sampling 118.92 s) with 5.86 GiB peak VRAM, and native single-pass 15-second generation with approximately 6.83 GiB peak VRAM.** Clips are generated with native video and audio rather than stitching three separate 5-second videos. First-and-last-frame guidance (FL2VA) is the primary benchmark mode.

[Upstream Models & Specs](MODELS.md) · [ComfyUI workflows](integrations/comfyui/README.md) · [CozyUI integration](integrations/cozyui/README.md) · [Showcase Media](demos/README.md) · [Contributing](CONTRIBUTING.en.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Showcase Gallery

Representative audiovisual generations across diverse tasks powered by H3 Turbo Runtime (all media available in [`demos/`](demos/README.md)):

### 1. Nora Turn

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| First/last frame (FL2VA) | 832×480, 5s (120 frames), 4-step Fast | Generates smooth turning motion between start and end frame inputs with synchronized native audio. | [Video](demos/videos/A04-诺拉-4步-我方优化4060.mp4) |

https://github.com/user-attachments/assets/b42f312d-52a6-4ad5-95a4-8a05eb729217

### 2. Native 15s Single-Pass (One More Time)

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| 4-guide timeline (FL2VA) | 832×480, 15s (360 frames), 4-step Stream | Generates a continuous 15-second audiovisual sequence (360 frames) in a single pass guided by four intermediate keyframes. | [Video](demos/videos/C01-再来一次-最终15秒-1080p有声.mp4) |

https://github.com/user-attachments/assets/cc6ee00d-2433-4a59-b455-4c5607a629fa

### 3. Starship Captain

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| Pure Text-to-Video-Audio (T2VA) | 832×480, 10s (240 frames), 4-step Fast | Pure text prompt input generating 10 seconds of cockpit lighting and ambient spatial sound. | [Video](demos/videos/B02-飞船舰长-我方Turbo.mp4) |

https://github.com/user-attachments/assets/1737891d-8ef4-4cfa-8b7e-f7779edf75ef

### 4. Ramen Family

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| Multi-timeline image guides | 832×480, 8s (192 frames), 4-step Fast | Anchors intermediate moments along the timeline to guide multi-character indoor movement and restaurant ambience. | [Video](demos/videos/B04-拉面家庭-我方Turbo.mp4) |

https://github.com/user-attachments/assets/3bafdd3e-0efd-4135-bdfb-5b141d4d36ca

### 5. Nailong (Ref2VA)

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| Cross-modal character & voice reference | 832×480, 5s (120 frames), 8-step Ref2VA | Inputs a single character reference image and audio track to preserve visual identity and vocal timbre. | [Video](demos/videos/C02-奶龙-Ref2VA-有声.mp4) |

https://github.com/user-attachments/assets/b38c0cb4-e39c-4289-855c-61866b0f57e7

### 6. Redelivery (赎命)

| Task Mode | Settings | Description | Media File |
|---|---|---|---|
| Multi-shot narrative sequence (12 shots) | Native 832×480 (5s × 12 shots), RelayVSR 1080p delivered, 44.96s | 12 native 5-second shots generated with H3 Turbo, enhanced via RelayVSR super-resolution and edited into a continuous 44.96s narrative short with original audio. | [Video](demos/videos/C03-赎命-最终44秒-1080p原声.mp4) |

<details>
<summary>Expand 44.96s full narrative clip online player (reduces initial page load)</summary>

https://github.com/user-attachments/assets/fa2eafd7-63ce-4622-8aa4-51c403025516

</details>

## Performance

All example clips use **832×480 at 24 fps, with first-and-last-frame guidance (FL2VA)**. VRAM is the whole-device peak (not just PyTorch allocations). *Sampling* and *completed video* timings have different boundaries; see the [full cross-device benchmark notes](benchmarks/README.md).

### 5 seconds · primary benchmark

| Runtime | GPU | Steps | Denoise / sampling | Completed clip | Peak VRAM |
|---|---|---:|---:|---:|---:|
| Published LightX2V Turbo (INT8 ConvRot) | Tesla T4 16GB | 4 | 1,099.73 s | 1,257.67 s | 14.31 GiB |
| **H3 Turbo Runtime** | Tesla T4 16GB | 4 | **168.92 s** | — | **3.95 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 4 | **118.92 s** | **159.18 s** | **5.86 GiB** |
| Published LightX2V Turbo (INT8 ConvRot) | Tesla T4 16GB | 8 | 2,126.70 s | 2,287.90 s | 14.31 GiB |
| **H3 Turbo Runtime** | Tesla T4 16GB | 8 | **558.19 s** | **670.12 s** | **4.08 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 8 | **223.22 s** | **269.87 s** | **5.33 GiB** |

<small>The 118.92 s RTX 4060 result measures the sampling stage, while its 159.18 s completed-clip time represents full end-to-end processing under a uv-managed hermetic runtime. The T4 four-step optimized path was validated over all four denoising steps and both AV latents, but its archived MP4 has no audio track; the corresponding completed native-AV time is therefore omitted. Upstream and optimized deployments differ in quantization, execution environment, and parts of their timing boundaries, so their observed speed ratios are deployment-level comparisons, not exact isolated-kernel speedups.</small>

### 15 seconds · single-pass long video

| Runtime | GPU | Steps | Four denoising steps | Sampling stage | Peak VRAM |
|---|---|---:|---:|---:|---:|
| **H3 Turbo Runtime** | Tesla T4 16GB | 4 | **986.90 s** | 1,017.26 s | **6.17 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 4 | **586.79 s** | — | **6.83 GiB** |

Both runs generated native 15-second video and audio in **one continuous model run**, with verified MP4 output. The table uses the same two-frame benchmark scene; optional intermediate guides are demonstrated separately in the [15-second verification notes](benchmarks/README.md).

<small>15-second denoising time excludes model startup, conditioning, decoding, and muxing. For a separate four-guide 4060 example, the sum of explicitly timed conditioning, denoising, and AV decoding stages was 698.1 s (11m 38s).</small>

### Features and duration

- **Conditioning:** text-to-video, single-image, first/last-frame, and multiple timed image guides through the original H3 conditioning interface.
- **Duration:** `--duration SECONDS` supports configurable output lengths, including the common **4–15-second** range, without changing checkpoint families. Five seconds is the default. Native frames follow H3's `17k+5` grid and native audio/video is exported at the requested duration. H3 was primarily trained at approximately 5–15 seconds.
- **Execution:** FL2VA Turbo defaults to 4 or 8 steps; Ref2VA recommends 8. The caller explicitly selects Fast or Stream. Fast defaults to SM89-validated kernels and does not silently switch after an error or OOM.
- **Benchmarks:** five and fifteen seconds are representative short/long measurements, not the only allowed input lengths.

**Native inputs:** text-only, single first/last frame, first-and-last frames, timed image/short-video/audio guides, and reference images/videos/audio. They can be combined through the original H3 conditioning pipeline. See the [ComfyUI input and workflow guide](integrations/comfyui/README.md).

## Model lineage

1. **MiniMax H3-Base-FL2VA** is the base audio-video model from MiniMax. The first-and-last-frame path also accepts zero or one image. Native audio-video generation comes from this model. [MiniMax H3 model specifications](https://github.com/MiniMax-AI/MiniMax-H3#model-variants-and-input-specifications)
2. **LightX2V H3 Turbo** uses distilled LoRA weights to compress denoising into four or eight steps, contributed by the LightX2V team. [LightX2V repository](https://github.com/lightx2v/lightx2v)
3. **W4A8 community base:** repackaged 4-bit weights for MiniMax-H3-Base-FL2VA.
4. **Audio/video decoding:** uses the official INT8 video VAE and FP32 audio VAE.
5. **Prompt/vision conditioning:** encodes prompts and reference frames using Qwen3-VL 4B INT8 and aligns features via a projection layer.

The [Upstream Model Specifications](MODELS.md) and [reproduction manifest](runtime/models/manifest.json) lists file names, source repositories, SHA-256 digests, and loading stages for all six weights.

## Quantization and execution

| Stage | Current implementation |
|---|---|
| Backbone weights | Uses a community W4A8 checkpoint. Grouped four-bit weights are unpacked to INT8 when needed by the feed-forward layers and reused across adjacent row tiles. |
| Activations and matrix multiplication | The feed-forward path follows the existing ConvRot group rotation and row-wise INT8 activation quantization. Adapted kernels accumulate in INT32 and apply row and column scales on writeback. The Ada path preserves the reference FP32 output and arithmetic order. |
| Turbo adapter | Uses the LightX2V few-step distilled LoRA (low-rank adaptation) checkpoint. Floating-point residuals are added layer by layer, preserving the quantized backbone layout. |
| Feed-forward network and attention | Processes feed-forward row tiles in pairs and combines work across matrix multiplication, activation, or writeback where the layout permits. The base feed-forward path and the LoRA path retain their respective activation arithmetic. Query and key normalization is preserved before fused rotary embedding and writeback. |
| Resource lifetime | Runs conditioning, denoising, video decoding, and audio decoding in stages, releasing previous weights and workspaces. Video decoding uses an INT8 ConvRot checkpoint with tiled overlap. Audio decoding uses the upstream FP32 checkpoint. |
| Conditioning for new inputs | The runtime uses a community Qwen3-VL 4B INT8 vision-language encoder and a text projection. It keeps the first 24 layers required to extract the target features. |

Staged resource management, weight and adapter residency, tiled fusion, and paired validation can carry over to other models. Each integration needs its own quantization layout, activation and adapter arithmetic, latent interface, and GPU kernels. Measure latency, memory, and output quality with the same input when adapting a model. The published measurements use an RTX 4060 Laptop (SM89).

**Optional Ref2VA pair:** Select joeygambino's native ComfyUI W4A8 Ref2VA backbone together with the original LightX2V Ref2VA eight-step adapter using `--variant ref2va --steps 8`. The original FL2VA pair remains the default, and both reuse the existing Qwen3-VL conditioning encoder and audio/video VAEs. The exact author-published file digest and mirrored download URL are in [the Ref2VA manifest](runtime/models/ref2va.json). Use `uv run python scripts/fetch_weights.py --fetch --variant ref2va` to retrieve and verify the selected weights.

**Interface refactor qualification:** Existing benchmark measurements belong to their archived runs. This refactor has CPU interface tests but no new GPU full-render validation; expanded duration, resolution, third-party LoRA and non-SM89 Fast combinations are not qualified by those benchmark figures.

## Deployment and reproduction

The runtime interface supports flexible invocation via CLI or ComfyUI nodes, offering two primary execution strategies:
- **Fast Mode**: Accelerated via Triton 148-register fused operators, optimized and qualified on NVIDIA Ada architectures (SM 8.9, such as RTX 4060).
- **Stream Mode**: Employs grouped-head streaming attention, tailored for minimal memory overhead and long-duration video generation.

Dependencies: CUDA, PyTorch, Triton, comfy-kitchen, ComfyUI, and FFmpeg. The project recommends `uv` for unified environment management:

The project recommends using `uv` for unified environment management:

```bash
# 1. Initialize environment and synchronize dependencies via uv
uv sync

# 2. Run automated verification test suite
uv run pytest -v

# 3. Verify environment and model assets (does not trigger rendering)
uv run python deploy.py check --comfy /path/to/ComfyUI --models /path/to/weights

# 4. Default baseline: five seconds, four steps, first/last frame conditioning
uv run python deploy.py render --comfy /path/to/ComfyUI --models /path/to/weights \
  --condition /path/to/5s-condition.safetensors --out runs/clip-5s

# 5. Configurable duration: single-pass 15-second generation (prepare 15s condition first)
uv run python deploy.py render --comfy /path/to/ComfyUI --models /path/to/weights \
  --duration 15 --strategy stream --condition /path/to/15s-condition.safetensors --out runs/clip-15s
```

Duration defaults to 5s; width/height default to 832x480, output-fps defaults to 24 (native H3 remains 24fps). Strategy defaults to fast, or explicitly select stream. Additional options: head-group, reserve-vram, adapter, recipe-file. Use project GPU/LoRA Skills to adapt unverified combinations.

**Hardware Adaptation**: Callers explicitly select `--strategy fast` or `stream`. The runtime executes without implicit fallback. To adapt to alternative GPU architectures, refer to the [GPU Porting Guide](.agents/skills/h3-gpu-backend-porting/SKILL.md).

## ComfyUI

The node lives in [`custom_nodes/h3_turbo_runtime/`](custom_nodes/h3_turbo_runtime/). Add it to the active ComfyUI `custom_nodes` folder and set `H3_RUNTIME_MODELS` to your model directory. Optional `H3_RUNTIME_PYTHON` and `H3_RUNTIME_DEPS` select a separate execution environment.

```bash
export H3_RUNTIME_MODELS=/path/to/weights
# Optional: only when runtime uses different python or deps directory
export H3_RUNTIME_PYTHON=/path/to/python
export H3_RUNTIME_DEPS=/path/to/runtime-deps
# Optional: when condition weights reside elsewhere
export H3_RUNTIME_WORKBENCH=/path/to/workbench
uv run python deploy.py configure-comfy --comfy /path/to/ComfyUI
```

The node accepts prompts, first/last frames, duration (five seconds by default), step count and optional timed guides, such as `[{"at_seconds":5,"image":"frame5.png"}]`. Upload images to the ComfyUI input directory. Conditioning uses `ClipProj` and `condition-models`; see the [ComfyUI instructions](integrations/comfyui/README.md).

## Sources and reproducibility

Run `uv run python scripts/fetch_weights.py --fetch` to obtain all six pinned original weight files (23.67 GB total), automatically verifying file sizes and SHA-256. This repository includes the complete model-retrieval recipe without duplicating upstream weights. See the [manifest](benchmarks/reproduction-manifest.json), [cross-device benchmarking](benchmarks/README.md), [implementation notes](benchmarks/implementation.md), [output fidelity](benchmarks/quality.md), and [third-party notices](THIRD_PARTY_NOTICES.md). The published LightX2V Turbo INT8 ConvRot workflow and this project's W4A8 runtime use different quantization/compute stacks, so observed performance improvements are system-level rather than isolated-kernel speedups.

## Complete ComfyUI API and demonstration videos

The [complete default 5-second ComfyUI workflow](integrations/comfyui/complete-5s-workflow.json)
exposes generation settings, optional first/last frames, multiple timed image/video/audio guides, and
Ref2VA reference images/videos/audio. The [API workflow](integrations/comfyui/complete-5s-api.json)
and [CozyUI interface](integrations/cozyui/README.md) share the same default: 5s, 832×480,
24fps, 4 steps, Fast, automatic backbone selection. They do not impose artificial limits on
user durations, canvas dimensions, reference quantities, or sampling steps. Execution may fail
for unsupported resource/model configurations; no fallback strategy is silently selected.

[Public audio-video demo clips](demos/README.md) include upstream-vs-optimized Nora comparisons
and two public-example scenes generated by our optimized runtime. Private creative works are excluded.
