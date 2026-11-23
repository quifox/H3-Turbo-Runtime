# Contributing Guide

[简体中文](CONTRIBUTING.md) · English

Thank you for your interest in the H3 Turbo Runtime project! We welcome community contributions to help advance the open-source low-VRAM audiovisual inference ecosystem.

Before contributing, please review our foundational architectural contracts and engineering boundaries:
👉 **[AGENTS.md](AGENTS.md) — Contributor and Agent Guidelines**

---

## 1. Where to Help

We especially encourage contributions in the following areas:

1. **New Hardware Architectures and Kernel Porting**:
   - Operator tuning and porting for Turing (SM75), Ampere (SM80/SM86), Hopper (SM90), or AMD ROCm/Triton targets.
   - Please follow the [Runtime Optimization Playbook](.agents/skills/h3-runtime-optimization/SKILL.md).
2. **LoRA Adapters and Sampling Recipes**:
   - Integrating new community fine-tunes, fewer-step Turbo LoRA distillation checkpoints, or custom scheduler policies.
   - Adapters and recipes are one lever inside the same [Runtime Optimization Playbook](.agents/skills/h3-runtime-optimization/SKILL.md); verify them through the same loop.
3. **ComfyUI / CozyUI Ergonomics & Workflows**:
   - Enhancing node usability, expanding multi-modal condition inputs, and contributing verified production-ready workflows.
4. **Automated Testing & Engineering Robustness**:
   - Expanding edge-case coverage and strengthening cross-platform (Linux, Windows, macOS) interface and type contracts.
5. **Documentation & Benchmark Replication**:
   - Contributing reproducible end-to-end latency and VRAM figures across diverse hardware setups (with system logs and specifications).
   - Refining deployment tutorials, fixing errata, and expanding translations.
6. **Community Works and Device Reports**:
   - Submitting videos, prompts, or adaptation/optimization notes from your own device; performance work, tooling, and tests are welcome too.
   - Anything another person can roughly reproduce is enough; a video is optional, and topics and formats are not restricted.
   - See [community/README.md](community/README.md) for the layout and templates.

> [!NOTE]
> **Out-of-Scope / Prohibited Changes**:
> - Adding implicit fallbacks or automatic downgrades (e.g., silently falling back from Fast to Stream upon failure), which violates core architectural contracts.
> - Committing large model weights, binary archives, or unverified generated videos directly to the Git tree (use automated scripts for weights and external release assets for showcase videos).
> - Introducing third-party assets with ambiguous licensing or non-commercial closed-source restrictions.

---

## 2. Development Setup

The project uses `uv` as the authoritative environment and package manager.

### Initializing the Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/H3-Turbo-Runtime.git
cd H3-Turbo-Runtime

# 2. Synchronize dependencies and virtual environment (strictly via uv, no unisolated pip install)
uv sync

# 3. Run the automated test suite
uv run pytest -v
```

> [!TIP]
> **CPU-Friendly & Lightweight Contributions**:
> Node contract assertions, ComfyUI schema validation, doc link verification, and syntax checks do **not require a GPU**. Any contributor can run `uv run pytest -v` on a standard CPU machine to validate changes locally.

### Verified Weight Fetching (Optional)

If full end-to-end inference or generation testing is needed on your machine, fetch and verify pinned weights automatically:
```bash
uv run python scripts/fetch_weights.py --fetch
```

---

## 3. Pull Request Workflow

### Step 1: Branching
- Always branch off the latest `main`:
  ```bash
  git checkout -b feat/your-feature-name
  ```

### Step 2: Commit Message Standards (Conventional Commits)
Use clear, descriptive commit messages following standard prefixes:
- `feat:` New features or enhancements
- `fix:` Bug fixes
- `perf:` Performance or operator throughput improvements
- `docs:` Documentation or comment updates
- `test:` Test suite or contract additions
- `refactor:` Code restructuring without external contract changes

### Step 3: Local Quality Gate
Before opening a PR, ensure all local checks pass:
1. **Full Test Suite Passes**:
   ```bash
   uv run pytest -v
   ```
2. **Zero Information Leakage (Strict Boundary)**:
   Ensure no personal usernames, private IPs (`192.168.*`, `10.*`), or local absolute filesystem paths are present in code or comments. All paths must resolve through relative paths or environment variables.
3. **Contract Symmetry**:
   If modifying ComfyUI node schemas, ensure `H3TurboRuntime`, `H3TurboSettings`, and `H3TurboMediaCondition` strictly mirror `integrations/comfyui/complete-5s-workflow.json`.

### Step 4: Submitting the PR
- Open a PR against `main` on GitHub.
- Clearly describe the purpose, affected modules, and verification method.
- For performance or kernel adjustments, attach micro-benchmark measurements and peak VRAM deltas in the PR description.

---

## 4. Reporting Issues

### Bug Reports
Before filing a bug, check existing issues to avoid duplicates. Please include:
1. **Environment Details**: GPU model, driver version, CUDA version, operating system (Linux distro / WSL2 / Windows).
2. **Execution Command**: Full CLI invocation or ComfyUI workflow screenshot.
3. **Full Traceback**: Complete, untruncated Python stack trace.
4. **Execution Strategy**: Specify whether `--strategy fast` or `--strategy stream` was used.

### Feature and Hardware Requests
Suggestions for new hardware targets, alternative schedulers, or UI ergonomics are welcome. Please detail your target hardware profile, VRAM capacity, and expected workflow.
