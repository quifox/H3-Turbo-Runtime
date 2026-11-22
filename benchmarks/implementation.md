# H3 Turbo Runtime Implementation and Technical Report

[Home](../README.md) · [English Home](../README.en.md) · [Benchmarks](README.md) · [Reproduction Manifest](reproduction-manifest.json)

## 1. Hardware Baselines & Memory Budgets

| Target Configuration | Qualification and Execution Status |
|---|---|
| **RTX 4060 Laptop 8 GiB** | Primary target platform (SM 8.9). Fully qualified on 4-step and 8-step end-to-end audiovisual generation. |
| **Low-VRAM Budget (≤ 6 GiB)** | Validated on hardware by holding ~2 GiB host reserve; initial step denoising and video decoding execute within budget. |
| **Decoupled Architecture** | Subprocess memory eviction, staged model loading, and tiled VAE spatial overlap ensure deterministic memory bounds. |

---

## 2. Primary Benchmark Measurements

Evaluated on the standard Nora Turn FL2VA benchmark (832×480, 24 fps, 5.0 seconds, 32 kHz stereo AAC) using cached input conditioning:

| Step Configuration | Total Time to Output | Peak Whole-Device VRAM | Peak Host RSS Memory |
|---|---:|---:|---:|
| **4-Step Fast** | **161.54 s** | **5.56 GiB** (5,695 MiB) | 16.89 GiB |
| **8-Step Fast** | **269.87 s** | **5.33 GiB** (5,460 MiB) | 16.55 GiB |

Both configurations load the pinned LightX2V Turbo LoRA adapter. Specific kernel optimizations verified during implementation:
- **Q/K RoPE Writeback Fusion:** Accelerates initial step execution by 0.92–1.25 s, saving ~336 MiB in peak PyTorch allocator overhead while producing bitwise-identical latents.
- **Tiled VAE Spatial Overlap Tuning (64px → 32px):** Reduces video decoding wall time from 15.3 s to 10.5 s. The decoded RGB difference is an imperceptible 0.26 / 255 mean absolute error (PSNR: 52.36 dB).

See [`reproduction-manifest.json`](reproduction-manifest.json) for exact hashes and weights.

---

## 3. Streaming Numerical Safeguards

The streamed LoRA FFN employs selective FP32 dynamic correction when intermediate FP16 activations approach potential overflow limits, retaining fast FP16 execution across normal rows. The fallback monitor automatically detects and resolves non-finite activations. The implementation defaults to proactive safeguarding; setting `H3_SAFE_FC2_LORA=0` disables the guard for benchmarking and ablation studies.

---

## 4. Audiovisual VAE Precision Specification

The audio VAE checkpoint on disk is stored in FP32 precision. Comparative SNR analysis between FP32 and FP16 execution on identical latents demonstrates negligible waveform divergence (SNR: **53.56 dB**), whereas FP32 execution incurs an extra ~447 MiB peak memory allocation during extended 15-second audio decoding. The runtime defaults to the proven FP16 execution path to preserve strict VRAM bounds; developers may set `H3_AUDIO_VAE_FP32=1` for diagnostic audits.
