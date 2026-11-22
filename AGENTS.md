# AGENTS.md — Contributor and Agent Guidelines

Operational contracts, execution rules, and engineering boundaries for the H3 Turbo Runtime.

---

## 1. Architectural Contracts

1. **Dual Execution Engines**:
   - **Fast Backend**: Fused Triton kernels (SM89 Ada reference baseline, 148-register allocation, paired FC1/FC2, Q/K RoPE writeback). Maximizes throughput on supported hardware.
   - **Stream Backend**: Decoupled, tiled attention and streaming pipeline. Trades peak throughput for bounded memory overhead (~4.0–4.2 GiB VRAM), accommodating constrained hardware environments.
2. **Explicit Backend Selection**:
   - The execution strategy (`fast` or `stream`) is strictly **caller-selected**.
   - **Prohibition on Implicit Fallback**: If a backend encounters unsupported hardware, shape mismatch, or OOM, the runtime must fail immediately with the original trace. Automatic, silent fallback between backends is strictly prohibited.
3. **Reference Design and Hardware Decoupling**:
   - Tested hardware profiles (such as RTX 4060 Laptop 8GB and Tesla T4 16GB) represent **empirical reference designs (参考设计)** for benchmarking and verification, not hard architectural constraints.
   - The compute dispatch interface is hardware-agnostic. The runtime does not artificially restrict resolution, duration, or GPU models.

---

## 2. Extended Capabilities and Hardware Adaptation Skills

- **GPU Architecture and Hardware Adaptation Skill**:
  The skill [`.agents/skills/h3-gpu-backend-porting/`](.agents/skills/h3-gpu-backend-porting/SKILL.md) provides explicit, systematic procedures to guide contributors and autonomous agents in adapting the runtime for different target devices:
  - Adapting and re-tuning Triton kernels for alternative microarchitectures (Turing, Ampere, Ada, Hopper).
  - Configuring dynamic memory parameters (`head_group` dividing the 56 attention heads, `reserve_vram` headroom).
  - Calibrating streaming tile sizes and execution grids for varying VRAM profiles (from 6GB consumer cards to multi-gigabyte datacenter accelerators).
- **LoRA and Sampling Recipe Adaptation Skill**:
  The skill [`.agents/skills/h3-lora-recipe-adaptation/`](.agents/skills/h3-lora-recipe-adaptation/SKILL.md) guides the integration of custom LoRA checkpoints, alternative distillation adapters, or modified scheduler parameters while adhering to interface invariants.

---

## 3. Verification and Testing Boundaries

1. **Bitwise Parity of Fused Kernels**:
   - "Bit-exact" / "逐位一致" strictly denotes: **Isolated fused Triton kernels vs. their unfused reference implementations** (producing bitwise-identical output latents on identical inputs).
   - Whole-system W4A8 quantization exhibits expected sampling trajectory divergence from unquantized 16-bit baselines.
2. **Automated Contract Integrity**:
   - Any PR, edit, or commit must pass `uv run pytest -v` without breaking workflow schema or node contract assertions.
   - `H3TurboRuntime`, `H3TurboSettings`, and `H3TurboMediaCondition` schemas must mirror `integrations/comfyui/complete-5s-workflow.json` exactly.

---

## 4. Repository Cleanliness and Privacy Red Lines

1. **Zero Information Leakage**:
   - No personal usernames, hostnames, private IPs (`192.168.*`, `10.*`), or local absolute filesystem paths may appear in tracked files.
   - All runtime paths and network targets must resolve via environment variables (`COMFYUI_URL`, `H3_RUNTIME_MODELS`, etc.) or relative workspace paths.
2. **Git as Single Source of Truth**:
   - Do not generate, commit, or preserve `.tar.gz` or `.zip` release bundles. The Git repository is the sole distribution mechanism.
   - Caches (`.ruff_cache/`, `.pytest_cache/`, `__pycache__/`, `runs/`) must remain untracked and excluded via `.gitignore`.
3. **Execution Guardrails**:
   - Do not trigger GPU generation jobs, download models, or install system dependencies without explicit user confirmation.

---

## 5. Toolchain and Environment Management (`uv` Standard)

1. **Deterministic Environment via `uv`**:
   - The repository uses `uv` as the authoritative Python environment and package manager.
   - Python version is strictly pinned by [`.python-version`](.python-version) (`3.14.2`), with dependencies declared in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock).
   - Unisolated `pip install` commands are prohibited. All dependency updates must use `uv add`, `uv sync`, or `uv pip`.
2. **Unified Command Invocation**:
   - All CLI verification and execution commands must use `uv run` for predictable runtime isolation:
     - Synchronize dependencies: `uv sync`
     - Test suite: `uv run pytest -v`
     - Check runtime: `uv run python deploy.py check --comfy /path/to/ComfyUI --models /path/to/weights`
     - Render video: `uv run python deploy.py render --comfy /path/to/ComfyUI --models /path/to/weights --condition /path/to/condition.safetensors --duration 5 --steps 4 --out runs/example`
