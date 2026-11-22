"""Encode native H3 AV conditioning: text, frames, timed clips/audio and references."""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from geometry import as_metadata, from_seconds
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from conditioning import encode_condition
from media import image as read_image, video as read_video, audio as read_audio


def load_projection(plugin_root):
    spec = importlib.util.spec_from_file_location(
        "h3_workbench_clipproj", plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS["ClipProjApply"]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reserve-vram", type=float, default=4.5)
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--projection", choices=("v3-mlp", "v3.1"),
                        default=os.getenv("H3_CLIPPROJ_VERSION", "v3.1"),
                        help="ClipProj condition encoder (H3_CLIPPROJ_VERSION for ComfyUI)")
    args = parser.parse_args()
    task = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf-8"))
    geometry = from_seconds(request.get("duration_s", 5.0),
                            width=int(request.get("width", 832)), height=int(request.get("height", 480)),
                            output_fps=int(request.get("output_fps", 24)))
    guides = request.get("guides", [])
    refs = request.get("references", {})
    if not isinstance(guides, list) or not isinstance(refs, dict):
        raise ValueError("guides must be a list and references must be an object")
    comfy = args.comfy
    condition_models = args.workbench / "condition-models"
    archive_models = args.models

    sys.path.insert(1, str(comfy))
    sys.argv = [sys.argv[0], "--fp16-vae", "--reserve-vram", str(args.reserve_vram), "--disable-xformers",
                "--disable-pinned-memory", "--use-ck-attention", "--disable-dynamic-vram"]
    import comfy.options
    comfy.options.enable_args_parsing()

    from safetensors.torch import save_file
    import torch
    import folder_paths
    import nodes
    from comfy import model_management
    from comfy_extras.nodes_minimax_h3 import (
        MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide,
    )
    from prefix_patch import install_prefix24

    torch.set_num_threads(2)
    torch.manual_seed(int(request.get("seed", 4106)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for text and image conditioning")
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()

    projection = load_projection(args.workbench / "ClipProj")
    for category in ("text_encoders", "clip_projections"):
        folder_paths.add_model_folder_path(category, str(condition_models), is_default=True)
    folder_paths.add_model_folder_path("vae", str(archive_models), is_default=True)

    frames = {key: read_image(request[key]) for key in ("first_frame", "last_frame") if request.get(key)}
    timeline = []
    for guide in guides:
        if "frame_idx" in guide:
            index = int(guide["frame_idx"])
        else:
            index = round(float(guide["at_seconds"]) * 24)
        guide_image = None
        if guide.get("image"):
            guide_image = read_image(guide["image"])
        elif guide.get("images"):
            import torch
            guide_image = torch.cat([read_image(path) for path in guide["images"]], dim=0)
        elif guide.get("video"):
            guide_image = read_video(guide["video"])
        guide_audio = read_audio(guide["audio"]) if guide.get("audio") else None
        if guide_image is None and guide_audio is None:
            raise ValueError("A timeline guide needs an image, video or audio")
        timeline.append((index, guide_image, guide_audio))

    reference_images = {f"ref_image_{i}": read_image(item) for i, item in enumerate(refs.get("images", []))}
    reference_videos, reference_video_audio = {}, {}
    for i, item in enumerate(refs.get("videos", [])):
        descriptor = {"path": item} if isinstance(item, str) else item
        reference_videos[f"ref_video_{i}"] = read_video(descriptor["path"])
        if descriptor.get("audio"):
            audio_path = descriptor["path"] if descriptor["audio"] is True else descriptor["audio"]
            reference_video_audio[f"ref_video_audio_{i}"] = read_audio(audio_path)
    reference_audios = {f"ref_audio_{i}": read_audio(item) for i, item in enumerate(refs.get("audios", []))}
    has_references = bool(reference_images or reference_videos or reference_audios)
    needs_video_vae = bool(frames or reference_images or reference_videos or any(img is not None for _, img, _ in timeline))
    needs_audio_vae = bool(reference_audios or reference_video_audio or any(sound is not None for _, _, sound in timeline))

    install_prefix24()
    with torch.inference_mode():
        clip, = nodes.CLIPLoader().load_clip("qwen3vl_4b_int8_convrot.safetensors", "krea2")
        projection_file = {
            "v3-mlp": "mmh3-4b-ClipProj-v3-mlp.safetensors",
            "v3.1": "mmh3-4b-ClipProj-v3.1.safetensors",
        }[args.projection]
        clip, = projection.apply(clip, projection_file)
        vae = None
        audio_vae = None
        if needs_video_vae:
            vae, = nodes.VAELoader().load_vae("minimax_h3_video_vae_int8_convrot.safetensors")
            original_encode = vae.encode
            released = False

            def encode_with_gpu_release(pixels):
                nonlocal released
                if not released:
                    model_management.unload_all_models()
                    gc.collect()
                    model_management.soft_empty_cache()
                    released = True
                return original_encode(pixels)

            vae.encode = encode_with_gpu_release
        if needs_audio_vae:
            if os.getenv("H3_AUDIO_VAE_FP32", "0") == "1":
                with patch.object(model_management.args, "fp16_vae", False), patch.object(
                    model_management.args, "fp32_vae", True
                ):
                    audio_vae, = nodes.VAELoader().load_vae("minimax_h3_audio_vae_fp32.safetensors")
                if audio_vae.vae_dtype != torch.float32:
                    raise RuntimeError("Requested FP32 H3 reference audio VAE was not loaded")
            else:
                audio_vae, = nodes.VAELoader().load_vae("minimax_h3_audio_vae_fp32.safetensors")
        if has_references:
            positive, latent = MiniMaxH3ReferenceToVideo.execute(
                clip, request["prompt"], geometry.width, geometry.height, geometry.native_frames,
                ref_image_size=refs.get("image_size", "match"),
                vae=vae, audio_vae=audio_vae,
                ref_images=reference_images, ref_videos=reference_videos,
                ref_video_audios=reference_video_audio, ref_audios=reference_audios,
            ).result
            if frames.get("first_frame") is not None:
                timeline.insert(0, (0, frames["first_frame"], None))
            if frames.get("last_frame") is not None:
                timeline.append((geometry.native_frames - 1, frames["last_frame"], None))
        else:
            positive, latent = MiniMaxH3ImageToVideo.execute(
                clip, vae, request["prompt"], geometry.width, geometry.height, geometry.native_frames, **frames
            ).result
        for frame_idx, pixels, soundtrack in timeline:
            positive, = MiniMaxH3AddGuide.execute(
                positive, latent, frame_idx, vae=vae, audio_vae=audio_vae,
                image=pixels, audio=soundtrack,
            ).result
        if len(positive) != 1:
            raise RuntimeError("条件编码返回了非单段的提示词")
        text, metadata = positive[0]
        video, audio = latent["samples"].unbind()
        if tuple(video.shape) != geometry.shapes["video"] or tuple(audio.shape) != geometry.shapes["audio"]:
            raise RuntimeError("H3 encoder returned an unexpected output latent shape")
        states, state_metadata = encode_condition(positive, latent)
        save_file(states, str(task / "condition.safetensors"), metadata=state_metadata)

    torch.cuda.synchronize()
    result = {
        "encoder": "Qwen3-VL 4B INT8 Prefix24",
        "projection": args.projection,
        "audio_vae_fp32": os.getenv("H3_AUDIO_VAE_FP32", "0") == "1",
        "anchors": [anchor["resolved_frame_index"] for anchor in metadata.get("minimax_keyframes", ())],
        "references": len(metadata.get("minimax_refs", ())),
        "geometry": as_metadata(geometry),
        "text_tokens": text.shape[1],
        "wall_s": perf_counter() - start,
        "torch_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (task / "condition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
