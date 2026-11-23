"""Validate a user-provided environment and run H3 Turbo at a requested duration."""

from argparse import ArgumentParser
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_runtime(deps):
    """Inspect the user-provided GPU/dependencies without selecting a fallback."""
    if deps is not None:
        if not deps.is_dir():
            raise ValueError("--deps must point to a directory")
        sys.path.insert(0, str(deps))
    sys.path.insert(0, str(ROOT / "runtime"))
    from compat import inspect_environment
    profile, _ = inspect_environment()
    print(f"{profile.name}: SM {profile.capability[0]}.{profile.capability[1]}, "
          f"{profile.memory_gib:.2f} GiB VRAM, {profile.family} runtime")
    return profile


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "render", "configure-comfy"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--strategy", choices=("fast", "stream"), default="fast",
                        help="Explicit execution backend; unsupported Fast kernels fail, never fall back")
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="fl2va",
                        help="Which H3 backbone and matching Turbo LoRA to use")
    parser.add_argument("--head-group", type=int, default=7)
    parser.add_argument("--reserve-vram", type=float, default=None,
                        help="Reserved VRAM in GiB; if omitted, use the verified backend default")
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Output duration in seconds (default: 5); H3 snaps internal frames to 17k+5")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-fps", type=int, default=24,
                        help="Output frame rate; native H3 generation stays at 24 fps")
    parser.add_argument("--adapter", type=Path,
                        help="Optional compatible H3 LoRA safetensors, replacing the bundled Turbo adapter")
    parser.add_argument("--recipe-file", type=Path,
                        help="Optional JSON recipe map for explicit sampler, sigma schedule and shifts")
    parser.add_argument("--comfy", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--condition", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action != "configure-comfy" and args.models is None:
        parser.error(f"{args.action} requires --models")
    if args.action == "render" and args.out is None:
        parser.error("render requires --out with a new directory")
    if args.action == "render" and args.condition is None:
        example = ROOT / "runtime" / "examples" / "nora" / "condition.safetensors"
        if example.is_file():
            args.condition = example
        else:
            parser.error("render requires --condition matching the requested duration; use the ComfyUI node to encode a new input")
    if args.out is not None and args.out.exists():
        parser.error("--out already exists")

    sys.path.insert(0, str(ROOT / "runtime"))
    from geometry import from_seconds
    try:
        geometry = from_seconds(args.duration, width=args.width, height=args.height,
                                output_fps=args.output_fps)
    except ValueError as exc:
        parser.error(str(exc))
    backend = ROOT / "runtime"
    from recipe import load_recipe
    load_recipe(args.variant, args.recipe_file)
    entry = backend / "render.py"
    manifest = json.loads((backend / "models/manifest.json").read_text())
    if args.action == "configure-comfy":
        node = ROOT / "custom_nodes/h3_turbo_runtime"
        target = args.comfy / "custom_nodes" / node.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != node:
            target.symlink_to(node, target_is_directory=True)
        print(f"H3 Turbo Runtime node registered at {target}")
        return
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise ValueError(f"{command} is required")
    verify_weights = runpy.run_path(str(entry))["verify_weights"]
    verify_weights(args.models, manifest, hashes=args.hash_weights, variant=args.variant,
                   adapter=args.adapter)
    profile = check_runtime(args.deps)
    if args.action == "check":
        print("Environment and model assets checked; no backend selected")
        return
    from compat import choose_strategy
    strategy = choose_strategy(profile, requested=args.strategy)
    print(f"H3 Turbo: {args.steps} steps, {geometry.exported_s:.3f}s, {strategy} strategy", flush=True)

    if args.action == "render":
        command = [sys.executable, str(entry), "render", "--models", str(args.models),
                   "--comfy", str(args.comfy), "--out", str(args.out)]
        command += ["--duration", str(args.duration), "--width", str(args.width), "--height", str(args.height),
                    "--output-fps", str(args.output_fps), "--steps", str(args.steps),
                    "--strategy", strategy, "--variant", args.variant, "--head-group", str(args.head_group),
                    "--seed", str(args.seed)]
        if args.adapter is not None:
            command += ["--adapter", str(args.adapter.resolve())]
        if args.recipe_file is not None:
            command += ["--recipe-file", str(args.recipe_file.resolve())]
        if args.reserve_vram is not None:
            command += ["--reserve-vram", str(args.reserve_vram)]
        for name in ("deps", "condition"):
            path = getattr(args, name)
            if path is not None:
                command.extend((f"--{name}", str(path)))
        sys.exit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
