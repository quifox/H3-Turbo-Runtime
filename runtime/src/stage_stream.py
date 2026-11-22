"""Run duration-matched H3 Turbo stages with grouped low-VRAM attention."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import from_seconds
from model_assets import model_filenames
from vae_precision import vae_flag_for_stage

CASE = Path(__file__).resolve().parents[1] / "models" / "manifest.json"


def load_case(condition_file, geometry):
    from conditioning import load_condition
    return load_condition(condition_file, geometry)


def sample(args, files):
    from comfy import model_management
    from comfy.ldm.minimax.model import time_shift_sigma
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from safetensors.torch import save_file

    from lora_adapter import apply_streamed_adapter

    conditions, latent = load_case(args.condition, args.geometry)
    cast = model_management.unet_manual_cast

    def select_dtype(weight_dtype, device, supported_dtypes=(torch.float16, torch.bfloat16, torch.float32)):
        if weight_dtype is None and tuple(supported_dtypes) == (torch.bfloat16, torch.float32):
            return torch.float32
        return cast(weight_dtype, device, supported_dtypes)

    with patch.object(model_management, "unet_manual_cast", select_dtype):
        model, = nodes.UNETLoader().load_unet(files["dit"], "default")
    if model.model.get_dtype_inference() != torch.float32 or len(model.model.diffusion_model.blocks) != 50:
        raise RuntimeError("H3 checkpoint has an unexpected inference dtype or layer count")
    model.model.eval().requires_grad_(False)
    from recipe import load_recipe
    recipe = load_recipe(args.variant, args.recipe_file)
    video_shift, audio_shift = recipe["video_shift"], recipe["audio_shift"]
    model = MiniMaxH3SigmaShift.execute(model, video_shift, audio_shift).result[0]
    sigmas = BasicScheduler.execute(model, recipe["scheduler"], args.steps, 1).result[0]
    schedule = {"video": sigmas.tolist(), "audio": time_shift_sigma(sigmas, video_shift, audio_shift).tolist()}
    grid = [i / args.steps for i in range(args.steps, -1, -1)]
    expected = {"video": [video_shift*q/(1+(video_shift-1)*q) for q in grid],
                "audio": [audio_shift*q/(1+(audio_shift-1)*q) for q in grid]}
    for key, values in schedule.items():
        if recipe["scheduler"] == "simple" and any(abs(a-b)>1e-6 for a,b in zip(values, expected[key], strict=True)):
            raise RuntimeError(f"Sampler schedule differs from the selected recipe {schedule}")
    adapter_stats = {}
    adapter = str(args.adapter if args.adapter is not None else args.models / files["lora"])
    from contextlib import ExitStack
    import os
    with ExitStack() as stack:
        names = stack.enter_context(apply_streamed_adapter(model, adapter, 1024, adapter_stats))
        if os.getenv("H3_SAFE_FC2_LORA", "1") == "1":
            from audio_precision import safe_fc2_lora
            stack.enter_context(safe_fc2_lora(adapter_stats.setdefault("fc2_fp32_preempt", [])))
        if args.head_group:
            from attention_stream import stream_qkv
            stack.enter_context(stream_qkv(model, adapter, args.head_group, adapter_stats,
                                           projection="eager-heads-tiled-output"))
        from stage_monitor import monitor_layers
        layer_summary = stack.enter_context(monitor_layers(model))
        audio_diagnosis = {}
        if os.getenv("H3_TRACE_AUDIO", "1") == "1":
            from audio_recovery import watch_audio_rows
            # Rearm only the rare FFN rows that have actually become nonfinite;
            # other tokens retain the original fast FP16 adapter arithmetic.
            stack.enter_context(watch_audio_rows(model, True, audio_diagnosis))
        guider = BasicGuider.execute(model, conditions).result[0]
        euler = KSamplerSelect.execute(recipe["sampler"]).result[0]
        noise = RandomNoise.execute(args.seed).result[0]
        output = SamplerCustomAdvanced.execute(noise, guider, euler, sigmas, latent).result[0]
    touches = adapter_stats["calls"]
    if set(touches) != names or any(count != (args.steps if key.startswith("diffusion_model.blocks.") else 1)
                                   for key, count in touches.items()):
        raise RuntimeError("LoRA 路径与验收配置不一致")
    shapes = args.geometry.shapes
    parts = {name: value.cpu().contiguous() for name, value in zip(shapes, output["samples"].unbind(), strict=True)}
    from stage_monitor import check_latents, tensor_stats
    observed = {name: tensor_stats(tensor) for name, tensor in parts.items()}
    print("RESULT_LATENTS", observed, flush=True)
    if not check_latents(observed, shapes):
        save_file(parts, str(args.out / "diagnostic-latents.safetensors"))
        raise ValueError(f"Nonfinite or mismatched latent: observed={observed}, expected={shapes}")
    save_file(parts, str(args.out / "latents.safetensors"))
    return {"steps": args.steps, "schedule": schedule, "layer_summary": layer_summary, "adapter": adapter_stats, "audio_diagnosis": audio_diagnosis,
            "latent_sha256": {key: hashlib.sha256(memoryview(value.numpy())).hexdigest() for key,value in parts.items()}}


def video(args, files):
    from av_pipeline import decode_video
    return decode_video(args, files, geometry=args.geometry)


def audio(args, files):
    from av_pipeline import decode_audio
    return decode_audio(args, files)


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=("sample", "video", "audio"))
    for option in ("comfy", "models", "condition", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-fps", type=int, default=24)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--recipe-file", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--variant", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--head-group", type=int, choices=(1, 2, 4, 7, 8, 14, 28, 56), default=7)
    parser.add_argument("--reserve-vram", type=float, default=None)
    args = parser.parse_args()
    args.geometry = from_seconds(args.duration, width=args.width, height=args.height,
                                 output_fps=args.output_fps)

    sys.path.insert(1, str(args.comfy))
    sys.argv = [sys.argv[0], "--fp16-unet", vae_flag_for_stage(args.stage), "--reserve-vram",
                str(1.1 if args.reserve_vram is None else args.reserve_vram), "--disable-xformers",
                "--disable-pinned-memory", "--use-ck-attention", "--disable-dynamic-vram"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    global torch, nodes
    import torch
    import nodes

    torch.set_num_threads(2)
    files = model_filenames(json.loads(CASE.read_text()), args.variant)
    for category in ("diffusion_models", "vae"):
        folder_paths.add_model_folder_path(category, str(args.models), is_default=True)
    print("TORCH", torch.__version__, args.stage, flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    operations = {"sample": sample, "video": video, "audio": audio}
    with torch.inference_mode():
        summary = operations[args.stage](args, files)
    torch.cuda.synchronize()
    summary.update(wall_s=perf_counter()-start, device=torch.cuda.get_device_name(),
                   duration_s=args.geometry.exported_s, native_frames=args.geometry.native_frames,
                   export_frames=args.geometry.export_frames,
                   peak_torch_mib=torch.cuda.max_memory_allocated()/2**20,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (args.out / f"{args.stage}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{args.stage} 完成")


if __name__ == "__main__":
    main()
