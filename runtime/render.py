"""Run native H3 Turbo audio/video using an explicitly selected GPU backend."""
from argparse import ArgumentParser
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from geometry import as_metadata, from_seconds
from compat import profile_from_capability, choose_strategy
from model_assets import selected_weights

MANIFEST = ROOT / "models" / "manifest.json"
EXAMPLE = ROOT / "examples" / "nora"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_weights(model_root, manifest, *, hashes=False, variant="fl2va", adapter=None):
    """Validate checkpoint sizes and optionally pinned SHA-256 before execution."""
    if adapter is not None and not adapter.is_file():
        raise FileNotFoundError(adapter)
    for asset in selected_weights(manifest, variant):
        if adapter is not None and asset["id"] == "lora":
            continue
        path = model_root / asset["name"]
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"权重不存在或大小不符 {path}")
        if hashes and sha(path) != asset["sha256"]:
            raise ValueError(f"权重 SHA-256 不符 {path}")
        if hashes:
            print(f"通过 {asset['name']}")


def mux(out, geometry):
    """Mux native video/audio and verify duration, dimensions and stream format."""
    video = out / "video.mp4"
    audio = out / "audio.f32le"
    destination = out / "output.mp4"
    temporary = out / "final.partial.mp4"
    samples = geometry.audio_samples
    audio_filter = (f"atempo={geometry.native_frames/(24 * geometry.exported_s):.10f},"
                    f"apad=whole_len={samples},atrim=end_sample={samples}")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(video),
                    "-f", "f32le", "-ar", "32000", "-ac", "2", "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-af", audio_filter, "-movflags", "+faststart", str(temporary)], check=True)
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(temporary)], text=True))
    picture, sound = probe["streams"]
    if (picture["width"], picture["height"], picture["nb_frames"], picture["avg_frame_rate"]) != (geometry.width, geometry.height, str(geometry.export_frames), f"{geometry.fps}/1"):
        raise RuntimeError("MP4 视频流的帧数、帧率或分辨率不符")
    if (sound["channels"], sound["sample_rate"]) != (2, "32000") or \
       abs(float(sound["duration"])-geometry.exported_s) > 0.03 or \
       abs(float(picture["duration"])-geometry.exported_s) > 0.03:
        raise RuntimeError("MP4 音轨参数有误或与画面时长不一致")
    temporary.replace(destination)
    return destination


def qualifies_for_reference_hash(strategy, steps, geometry, variant="fl2va"):
    """The frozen SHA belongs to the original 4-step Ada execution only."""
    return (variant == "fl2va" and strategy == "fast" and steps == 4
            and geometry.native_frames == 124 and geometry.export_frames == 120
            and geometry.width == 832 and geometry.height == 480 and geometry.fps == 24)


def render(args, manifest):
    """Run sample, video and audio in separate processes and record peak GPU use."""
    geometry = from_seconds(args.duration, width=args.width, height=args.height,
                            output_fps=args.output_fps)
    import torch
    profile = profile_from_capability(torch.cuda.get_device_capability())
    strategy = choose_strategy(profile, requested=args.strategy)
    case_file = EXAMPLE / "case.json"
    case = json.loads(case_file.read_text()) if case_file.is_file() else None
    fixture = bool(args.adapter is None and args.recipe_file is None
                   and qualifies_for_reference_hash(strategy, args.steps, geometry, args.variant) and case
                   and sha(args.condition) == case["condition_sha256"]
                   and args.seed == case["seed"])
    args.out.mkdir(parents=True, exist_ok=False)
    if geometry.native_frames < 124 or geometry.native_frames > 362:
        print(f"提示：请求的原生帧数 {geometry.native_frames} 在 H3 声明的约124–362帧训练时长之外；允许尝试，但模型质量与显存尚未验收。", flush=True)

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", PYTORCH_ALLOC_CONF="expandable_segments:True",
               TZ="UTC", PYTHONDONTWRITEBYTECODE="1")
    if args.deps is not None:
        env["PYTHONPATH"] = str(args.deps) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    dynamic = strategy == "stream"
    settings = [str(sys.executable), "-u",
                str(ROOT / "src" / ("stage_stream.py" if dynamic else "stage_fast.py"))]
    flags = ["--comfy", str(args.comfy), "--models", str(args.models), "--condition", str(args.condition),
             "--out", str(args.out), "--seed", str(args.seed)]
    flags += ["--duration", str(args.duration), "--width", str(args.width),
              "--height", str(args.height), "--output-fps", str(args.output_fps),
              "--steps", str(args.steps), "--variant", args.variant]
    if args.adapter is not None:
        flags += ["--adapter", str(args.adapter)]
    if args.recipe_file is not None:
        flags += ["--recipe-file", str(args.recipe_file)]
    if dynamic:
        flags += ["--head-group", str(args.head_group)]
    if args.reserve_vram is not None:
        flags += ["--reserve-vram", str(args.reserve_vram)]
    nvml = shutil.which("nvidia-smi")
    if nvml is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        nvml = "/usr/lib/wsl/lib/nvidia-smi"
    start = perf_counter()
    stages = {}
    with (args.out / "gpu.csv").open("w") as rows:
        monitor = subprocess.Popen([nvml, "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits",
                                    "-lms", "100"], stdout=rows, stderr=subprocess.DEVNULL, env=env) if nvml else None
        try:
            for stage in ("sample", "video", "audio"):
                print(f"运行 {stage}，日志保存到 {args.out / (stage+'.log')}", flush=True)
                with (args.out / f"{stage}.log").open("w") as log:
                    try:
                        subprocess.run([*settings, stage, *flags], cwd=args.comfy, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    except subprocess.CalledProcessError:
                        # Show the real child traceback instead of obscuring it with a new RuntimeError.
                        from collections import deque
                        with (args.out / f"{stage}.log").open(errors="replace") as source:
                            sys.stderr.writelines(deque(source, maxlen=100))
                        raise
                stages[stage] = json.loads((args.out / f"{stage}.json").read_text())
                if stage == "sample" and fixture and stages[stage]["latent_sha256"] != case["expected_latent_sha256"]:
                    raise RuntimeError("The bundled example latents differ from their qualified reference")
            film = mux(args.out, geometry)
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait()
    peaks = [int(row[1]) for row in csv.reader((args.out / "gpu.csv").open(), skipinitialspace=True)
             if len(row) == 2 and row[1].isdigit()]
    report = {"status": "complete", "version": manifest["version"], "wall_s": perf_counter()-start,
              "steps": args.steps, "strategy": strategy, "variant": args.variant,
              "geometry": as_metadata(geometry),
              "peak_device_mib": max(peaks) if peaks else None,
              "peak_stage_rss_mib": max(stage["peak_rss_mib"] for stage in stages.values()),
              "validated_reference_latents": fixture, "seed": args.seed, "stages": stages,
              "final": film.name, "final_sha256": sha(film)}
    (args.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"出片 {film}\n记录 {args.out / 'result.json'}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "render"))
    parser.add_argument("--models", type=Path, default=ROOT/"models")
    parser.add_argument("--comfy", type=Path, default=ROOT/"runtime"/"ComfyUI")
    parser.add_argument("--deps", type=Path, default=ROOT/"runtime"/"deps")
    parser.add_argument("--condition", type=Path, default=EXAMPLE/"condition.safetensors")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--duration", type=float, default=5.0, help="Requested output length in seconds; defaults to the verified 5s example")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-fps", type=int, default=24)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--recipe-file", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--head-group", type=int, choices=(1, 2, 4, 7, 8, 14, 28, 56), default=7)
    parser.add_argument("--strategy", choices=("fast", "stream"), default="fast")
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--reserve-vram", type=float, default=None)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "render":
        from recipe import load_recipe
        load_recipe(args.variant, args.recipe_file)
    verify_weights(args.models.resolve(), manifest, hashes=args.command == "verify",
                   variant=args.variant, adapter=args.adapter)
    if args.command == "verify":
        case_file = EXAMPLE / "case.json"
        if case_file.is_file():
            case = json.loads(case_file.read_text())
            missing = [key for key in case["file_sha256"] if not (EXAMPLE / key).is_file()]
            if missing:
                print(f"Model weights verified; optional example assets not installed: {missing}")
            else:
                for name, expected in case["file_sha256"].items():
                    if sha(EXAMPLE / name) != expected:
                        raise ValueError(f"Example checksum mismatch: {name}")
                print("Model and example SHA-256 verification passed")
        else:
            print("Model weight SHA-256 verification passed")
        return
    if args.out is None:
        parser.error("render requires --out pointing to a new directory")
    if args.reserve_vram is not None and (not math.isfinite(args.reserve_vram) or args.reserve_vram < 0):
        parser.error("--reserve-vram must be nonnegative and finite")
    if not args.condition.is_file():
        parser.error(f"Condition file does not exist: {args.condition}")
    if not (args.comfy / "nodes.py").is_file():
        parser.error("缺少 ComfyUI 源码，请使用 --comfy 指定清单中的固定版本")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            parser.error(f"缺少 {command}")
    for name in ("models", "comfy", "deps", "condition", "out"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    render(args, manifest)


if __name__ == "__main__":
    main()
