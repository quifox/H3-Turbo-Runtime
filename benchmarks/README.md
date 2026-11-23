# Performance Benchmarks and Reproducibility

[Home](../README.md) · [English Home](../README.en.md) · [Performance Data](performance.csv) · [Output Fidelity](quality.md) · [Implementation](implementation.md)

## Scope

The primary published baseline is the **unmodified LightX2V Turbo workflow** running the **Comfy-Org INT8 ConvRot** H3 checkpoint with LightX2V Turbo 8-step v1.0 LoRA. The H3 Turbo Runtime optimizes execution using the community W4A8 checkpoint and the same Turbo LoRA with specialized memory management and kernel fusions. Performance metrics represent **whole-system deployment comparisons** under identical task conditions.

All primary benchmark measurements evaluate an 832×480, 24-fps first/last-frame guidance (FL2VA) task:
- 5-second scenarios export 120 frames from 124 native internal frames.
- 15-second scenarios export 360 frames from 362 native internal frames.
- GPU memory figures represent NVML whole-device peak allocations in **MiB**, capturing the entire hardware footprint.

---

## Benchmark Metrics Definition

- `denoise_s`: The cumulative wall time of diffusion model forward denoising passes.
- `sample_s`: Total sampling-stage wall time, encompassing model preparation, host-device tensors, and latent serialization.
- `complete_s`: Full pipeline execution time from ready conditions to the final playable, multiplexed MP4 container.

---

## Measured Performance & Hardware Baselines

- **Tesla T4 (Cloud Baseline):**
  - Published 4-step Turbo: `1,099.73 s` denoising, `1,257.67 s` total clip time, `14,649 MiB` peak VRAM.
  - H3 Turbo Runtime 4-step: `168.92 s` sampling (~250 s clip), `4,043 MiB` peak VRAM.
  - Published 8-step Turbo: `2,126.70 s` denoising, `2,287.90 s` total clip time, `14,649 MiB` peak VRAM.
  - H3 Turbo Runtime 8-step: `558.19 s` denoising, `670.12 s` total clip time, `4,181 MiB` peak VRAM.
- **RTX 4060 Laptop 8GB (SM 8.9 Target):**
  - Optimized 5-second 4-step native AV: **161.54 s** total clip time (`121.49 s` denoising), **5,560 MiB** (5.56 GiB) peak whole-device VRAM.
  - Single-pass 15-second 4-step native AV: **586.79 s** model denoising, **6,830 MiB** (6.83 GiB) peak whole-device VRAM.
- Pinned upstream models, exact byte sizes, SHA-256 digests, and environment specifications are registered in [`reproduction-manifest.json`](reproduction-manifest.json).

---

## Regression & Smoke Validation

Regular validation runs on RTX 4060 Laptop 8GB (5s cached FL2VA condition, seed 4106) confirm stability across pipeline routes:

| Execution Route | 4-Step Sampling | Completed MP4 Video | Peak Whole-Device VRAM | Verification Status |
|---|---:|---:|---:|---|
| **Ada Fused Fast** | 138.77 s | 175.49 s (end-to-end) | 5,712 MiB | Latent SHA-256 bitwise matches qualified baseline |
| **7-Head Streaming** | 212.02 s | AV decoded & multiplexed | 3,758 MiB | Continuous finite activations across all 200 attention calls |

Both output streams produce standard 832×480, 24 fps, 120-frame H.264 video with native AAC 32 kHz stereo audio tracks, demonstrating robust execution stability across short and extended generation workloads.
