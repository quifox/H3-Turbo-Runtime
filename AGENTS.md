# AGENTS.md — Contributor and Agent Guidelines

Operational contracts, execution rules, and engineering boundaries for the H3 Turbo Runtime.

---

## 1. Architectural Contracts

1. **Dual Execution Engines**:
   - **Fast Backend**: Fused Triton kernels (SM89 Ada, 148-register allocation, paired FC1/FC2, Q/K RoPE writeback). Maximizes throughput on supported hardware.
   - **Stream Backend**: Decoupled, tiled attention and streaming pipeline. Trades peak throughput for bounded memory overhead.
2. **Explicit Backend Selection**:
   - The execution strategy (`fast` or `stream`) is strictly **caller-selected**.
   - **Prohibition on Implicit Fallback**: If a backend encounters unsupported hardware, shape mismatch, or OOM, the runtime must fail immediately with the original trace. Automatic, silent fallback between backends is strictly prohibited.
3. **Hardware Extensibility**:
   - RTX 4060 Laptop (8GB) and Tesla T4 represent empirical validation baselines.
   - The compute dispatch interface is hardware-decoupled. For adapting to other architectures, refer to [`.agents/skills/h3-gpu-backend-porting/`](.agents/skills/h3-gpu-backend-porting/SKILL.md).

---

## 2. Extended Capabilities and Adaptation Skills

- **GPU Architecture and Hardware Porting**:
  For porting backends, adapting non-SM89 architectures, and tuning memory parameters (`head_group`, `reserve_vram`, streaming tile sizes), refer directly to [`.agents/skills/h3-gpu-backend-porting/`](.agents/skills/h3-gpu-backend-porting/SKILL.md).
- **LoRA and Sampling Recipe Adaptation**:
  For integrating custom LoRA weights, alternative distillation checkpoints, or modified scheduler parameters, refer directly to [`.agents/skills/h3-lora-recipe-adaptation/`](.agents/skills/h3-lora-recipe-adaptation/SKILL.md).

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
