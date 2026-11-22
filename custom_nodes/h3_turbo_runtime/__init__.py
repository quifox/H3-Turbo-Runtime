"""ComfyUI node for native H3 Turbo audio-video rendering."""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from uuid import uuid4

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar


TEXT = "Text to video"
FRAMES = "First and last frame"
TIMELINE = "Timeline guides"
REFERENCE = "Reference to video"
MODES = ("Auto", TEXT, "First frame to video", "Last frame to video",
         FRAMES, TIMELINE, REFERENCE)
LEGACY_MODES = {"文生视频": TEXT, "首尾帧生视频": FRAMES}



def input_image(name):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not name or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Upload the reference image to the ComfyUI input directory")
    return str(path)


def resolve_guide(guide):
    if not isinstance(guide, dict) or not ("frame_idx" in guide or "at_seconds" in guide):
        raise ValueError("Each guide needs frame_idx or at_seconds")
    resolved = {key: guide[key] for key in ("frame_idx", "at_seconds") if key in guide}
    for field in ("image", "video", "audio"):
        if guide.get(field):
            resolved[field] = input_image(guide[field])
    if guide.get("images"):
        resolved["images"] = [input_image(item) for item in guide["images"]]
    if not any(key in resolved for key in ("image", "images", "video", "audio")):
        raise ValueError("Each guide needs images, a video, or audio")
    return resolved


def resolve_references(references):
    if not isinstance(references, dict):
        raise ValueError("references_json must contain an object")
    resolved = {"images": [input_image(name) for name in references.get("images", [])],
                "audios": [input_image(name) for name in references.get("audios", [])],
                "videos": []}
    for item in references.get("videos", []):
        desc = {"path": item} if isinstance(item, str) else item
        item_path = input_image(desc["path"])
        soundtrack = desc.get("audio")
        if soundtrack and soundtrack is not True:
            soundtrack = input_image(soundtrack)
        resolved["videos"].append({"path": item_path, "audio": soundtrack})
    if references.get("image_size") is not None:
        if references["image_size"] not in ("match", "max"):
            raise ValueError("Reference image_size must be match or max")
        resolved["image_size"] = references["image_size"]
    return resolved


def choose_model_variant(requested, mode, references):
    """Choose checkpoint from the task; callers may override without restrictions."""
    if requested not in ("auto", "fl2va", "ref2va"):
        raise ValueError(f"Unknown H3 model variant: {requested}")
    if requested != "auto":
        return requested
    has_references = any(references.get(key) for key in ("images", "videos", "audios"))
    return "ref2va" if mode == REFERENCE or has_references else "fl2va"


def run_logged(command, cwd, path, env):
    with path.open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while worker.poll() is None:
                try:
                    worker.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    throw_exception_if_processing_interrupted()
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
    if worker.returncode:
        from collections import deque
        with path.open(errors="replace") as output:
            sys.stderr.writelines(deque(output, maxlen=100))
        raise subprocess.CalledProcessError(worker.returncode, command)




class H3TurboSettings:
    """Optional visible settings node; all values default to the 5s Fast preset."""
    CATEGORY = "H3 Turbo / Inputs"
    FUNCTION = "build"
    RETURN_TYPES = ("H3_SETTINGS",)
    RETURN_NAMES = ("settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_s": ("FLOAT", {"default": 5.0}),
            "steps": ("INT", {"default": 4}),
            "strategy": (["fast", "stream"], {"default": "fast"}),
            "seed": ("INT", {"default": 4106}),
            "width": ("INT", {"default": 832}),
            "height": ("INT", {"default": 480}),
            "output_fps": ("INT", {"default": 24}),
            "variant": (["auto", "fl2va", "ref2va"], {"default": "auto"}),
            "canvas_mode": (["explicit", "short_edge_ratio"], {"default": "explicit"}),
            "short_edge": ("FLOAT", {"default": 768.0}),
            "aspect_ratio": ("STRING", {"default": "16:9"}),
            "length_frames": ("INT", {"default": 0}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "head_group": ("INT", {"default": 7}),
            "reserve_vram": ("FLOAT", {"default": 0.0}),
        }}

    def build(self, **kwargs):
        return (dict(kwargs),)


MEDIA_TYPES = (
    "first_frame", "last_frame", "timed_image", "timed_video", "timed_audio",
    "reference_image", "reference_audio", "reference_video",
)


class H3TurboMediaCondition:
    """Composable optional media input, chain multiple nodes for many timestamps."""
    CATEGORY = "H3 Turbo / Inputs"
    FUNCTION = "build"
    RETURN_TYPES = ("H3_MEDIA",)
    RETURN_NAMES = ("media",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False}),
            "kind": (MEDIA_TYPES, {"default": "timed_image"}),
            "filename": ("STRING", {"default": ""}),
            "at_seconds": ("FLOAT", {"default": 2.0}),
            "reference_video_audio": ("BOOLEAN", {"default": False}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
        }, "optional": {"previous": ("H3_MEDIA",)}}

    def build(self, enabled=False, kind="timed_image", filename="",
              at_seconds=2.0, reference_video_audio=False, ref_image_size="match",
              previous=None):
        if previous is not None and not isinstance(previous, dict):
            raise ValueError("Invalid media condition")
        origin = previous or {}
        data = {
            "first_frame": origin.get("first_frame", ""),
            "last_frame": origin.get("last_frame", ""),
            "guides": list(origin.get("guides", [])),
            "references": {
                key: list(origin.get("references", {}).get(key, []))
                for key in ("images", "videos", "audios")
            },
        }
        if origin.get("references", {}).get("image_size"):
            data["references"]["image_size"] = origin["references"]["image_size"]
        if not enabled:
            return (data,)
        if kind not in MEDIA_TYPES:
            raise ValueError("Invalid media type")
        path = input_image(filename)
        if kind in ("first_frame", "last_frame"):
            data[kind] = path
        elif kind.startswith("timed_"):
            data["guides"].append({"at_seconds": at_seconds, kind[6:]: path})
        elif kind == "reference_video":
            data["references"]["videos"].append({"path": path, "audio": str(reference_video_audio).lower() in ("true", "yes", "1")})
        else:
            key = "images" if kind == "reference_image" else "audios"
            data["references"][key].append(path)
            if kind == "reference_image":
                data["references"]["image_size"] = ref_image_size
        return (data,)

def resolve_canvas(width, height, canvas_mode="explicit", short_edge=768.0,
                   aspect_ratio="16:9", first_frame="", reference_image=""):
    """Map official short-edge/ratio API to latent dimensions; explicit pixels unchanged."""
    if canvas_mode == "explicit":
        return width, height
    if canvas_mode != "short_edge_ratio":
        raise ValueError("Unknown canvas selection method")
    if str(aspect_ratio).strip().lower() == "auto":
        picture = first_frame or reference_image
        if picture:
            from PIL import Image
            with Image.open(input_image(picture)) as img:
                ratio = img.width / img.height
        else:
            ratio = float(width) / float(height)
    elif ":" in str(aspect_ratio):
        x, y = str(aspect_ratio).split(":", 1)
        ratio = float(x) / float(y)
    else:
        ratio = float(aspect_ratio)
    # Match the H3 latent spatial grid. We do not cap aspect ratio, pixels or length.
    target_width = float(short_edge) * max(ratio, 1.0)
    target_height = float(short_edge) / min(ratio, 1.0)
    return round(target_width / 16) * 16, round(target_height / 16) * 16


class H3TurboRuntime:
    CATEGORY = "H3 Turbo"
    RETURN_TYPES = ()
    FUNCTION = "render"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (MODES, {"default": "Text to video"}),
            "prompt": ("STRING", {"default": "A single continuous cinematic shot of soft ocean waves rolling onto a quiet beach at sunrise. Natural golden light, gentle surf and wind ambience, stable camera.", "multiline": True}),
            "duration_s": ("FLOAT", {"default": 5.0}),
            "steps": ("INT", {"default": 4}),
            "seed": ("INT", {"default": 4106}),
            "strategy": (["fast", "stream"], {"default": "fast"}),
        }, "optional": {
            "first_frame": ("STRING", {"default": ""}),
            "last_frame": ("STRING", {"default": ""}),
            "guides_json": ("STRING", {"default": "[]", "multiline": True}),
            "references_json": ("STRING", {"default": "{}", "multiline": True}),
            "variant": (["auto", "fl2va", "ref2va"], {"default": "auto"}),
            "width": ("INT", {"default": 832}),
            "height": ("INT", {"default": 480}),
            "output_fps": ("INT", {"default": 24}),
            "adapter_file": ("STRING", {"default": ""}),
            "recipe_file": ("STRING", {"default": ""}),
            "settings": ("H3_SETTINGS",),
            "media": ("H3_MEDIA",),
            "middle_frame": ("STRING", {"default": ""}),
            "middle_at_seconds": ("FLOAT", {"default": 2.0}),
            "reference_image": ("STRING", {"default": ""}),
            "reference_audio": ("STRING", {"default": ""}),
            "reference_video": ("STRING", {"default": ""}),
            "reference_video_audio": ("BOOLEAN", {"default": False}),
            "canvas_mode": (["explicit", "short_edge_ratio"], {"default": "explicit"}),
            "short_edge": ("FLOAT", {"default": 768.0}),
            "aspect_ratio": ("STRING", {"default": "16:9"}),
            "length_frames": ("INT", {"default": 0}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "head_group": ("INT", {"default": 7}),
            "reserve_vram": ("FLOAT", {"default": 0.0}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, mode, prompt, duration_s, steps, seed, strategy,
                        first_frame="", last_frame="", guides_json="[]", references_json="{}", variant="auto",
                        width=832, height=480, output_fps=24, adapter_file="", recipe_file="",
                        settings=None, media=None, middle_frame="", middle_at_seconds=2.0,
                        reference_image="", reference_audio="", reference_video="",
                        reference_video_audio=False, canvas_mode="explicit",
                        short_edge=768.0, aspect_ratio="16:9",
                        length_frames=0, ref_image_size="match", head_group=7,
                        reserve_vram=0.0):
        try:
            if LEGACY_MODES.get(mode, mode) not in MODES:
                raise ValueError("Unknown H3 generation preset")
            if variant not in ("auto", "fl2va", "ref2va"):
                raise ValueError("Model variant must be auto, fl2va or ref2va")
            if strategy not in ("fast", "stream"):
                raise ValueError("Execution strategy must be fast or stream")
            for filename in (first_frame, last_frame):
                if filename:
                    input_image(filename)
            guides = json.loads(guides_json)
            if not isinstance(guides, list):
                raise ValueError("guides_json must contain an array")
            for guide in guides:
                resolve_guide(guide)
            resolve_references(json.loads(references_json))
            for name in (middle_frame, reference_image, reference_audio, reference_video):
                if name:
                    input_image(name)
            if isinstance(settings, dict):
                if settings.get("strategy") not in ("fast", "stream"):
                    raise ValueError("Invalid connected strategy")
            if isinstance(media, dict):
                for field in ("first_frame", "last_frame"):
                    if media.get(field):
                        input_image(media[field])
                for guide in media.get("guides", []):
                    resolve_guide(guide)
                resolve_references(media.get("references", {}))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, **_inputs):
        return float("nan")

    def render(self, mode, prompt, duration_s, steps, seed, strategy,
               first_frame="", last_frame="", guides_json="[]", references_json="{}", variant="auto",
               width=832, height=480, output_fps=24, adapter_file="", recipe_file="",
               settings=None, media=None, middle_frame="", middle_at_seconds=2.0,
               reference_image="", reference_audio="", reference_video="",
               reference_video_audio=False, canvas_mode="explicit",
               short_edge=768.0, aspect_ratio="16:9",
               length_frames=0, ref_image_size="match", head_group=7,
               reserve_vram=0.0):
        if settings is not None:
            if not isinstance(settings, dict):
                raise ValueError("Connected H3 settings must be a dict")
            duration_s = settings["duration_s"]
            steps = settings["steps"]
            strategy = settings["strategy"]
            seed = settings["seed"]
            width = settings["width"]
            height = settings["height"]
            output_fps = settings["output_fps"]
            variant = settings["variant"]
            canvas_mode = settings.get("canvas_mode", canvas_mode)
            short_edge = settings.get("short_edge", short_edge)
            aspect_ratio = settings.get("aspect_ratio", aspect_ratio)
            length_frames = settings.get("length_frames", length_frames)
            ref_image_size = settings.get("ref_image_size", ref_image_size)
            head_group = settings.get("head_group", head_group)
            reserve_vram = settings.get("reserve_vram", reserve_vram)
        if length_frames:
            duration_s = float(length_frames) / 24.0
        width, height = resolve_canvas(width, height, canvas_mode, short_edge,
                                       aspect_ratio, first_frame, reference_image)
        repo = Path(__file__).resolve().parents[2]
        workbench = Path(os.environ.get("H3_RUNTIME_WORKBENCH", str(repo / "workbench"))).resolve()
        comfy = Path(os.environ.get("H3_RUNTIME_COMFY", Path(folder_paths.__file__).resolve().parent)).resolve()
        models = os.environ.get("H3_RUNTIME_MODELS")
        if not models:
            raise ValueError("Configure H3_RUNTIME_MODELS to the directory containing H3 weights")
        python = os.environ.get("H3_RUNTIME_PYTHON", sys.executable)
        deps = os.environ.get("H3_RUNTIME_DEPS")
        output = Path(folder_paths.get_output_directory()) / "h3-turbo"
        output.mkdir(parents=True, exist_ok=True)
        task = output / uuid4().hex
        task.mkdir()
        run_dir = task / "render"
        env = dict(os.environ)
        command = [python, "-u", str(repo / "deploy.py"), "render",
                   "--comfy", str(comfy), "--models", models,
                   "--duration", str(duration_s), "--width", str(width),
                   "--height", str(height), "--output-fps", str(output_fps),
                   "--steps", str(steps), "--strategy", strategy, "--seed", str(seed),
                   "--head-group", str(head_group), "--out", str(run_dir)]
        if reserve_vram != 0.0:
            command.extend(("--reserve-vram", str(reserve_vram)))
        if deps:
            command.extend(("--deps", deps))
        if adapter_file:
            command.extend(("--adapter", adapter_file))
        if recipe_file:
            command.extend(("--recipe-file", recipe_file))
        references = resolve_references(json.loads(references_json))
        references["image_size"] = ref_image_size
        request = {"prompt": prompt.strip(), "duration_s": duration_s, "seed": seed,
                   "width": width, "height": height, "output_fps": output_fps,
                   "guides": [resolve_guide(guide) for guide in json.loads(guides_json)],
                   "references": references}
        if first_frame:
            request["first_frame"] = input_image(first_frame)
        if last_frame:
            request["last_frame"] = input_image(last_frame)
        if middle_frame:
            request["guides"].append({"at_seconds": float(middle_at_seconds),
                                      "image": input_image(middle_frame)})
        if reference_image:
            references["images"].append(input_image(reference_image))
        if reference_audio:
            references["audios"].append(input_image(reference_audio))
        if reference_video:
            references["videos"].append({"path": input_image(reference_video),
                                         "audio": str(reference_video_audio).lower() in ("true", "yes", "1")})
        if media:
            for field in ("first_frame", "last_frame"):
                if media.get(field):
                    if request.get(field):
                        raise ValueError(f"Duplicate {field} condition")
                    request[field] = input_image(media[field])
            request["guides"].extend(resolve_guide(guide) for guide in media.get("guides", []))
            extra = resolve_references(media.get("references", {}))
            for key in ("images", "videos", "audios"):
                references[key].extend(extra[key])
            if extra.get("image_size") is not None:
                references["image_size"] = extra["image_size"]
        selected_variant = choose_model_variant(variant, mode, references)
        command.extend(("--variant", selected_variant))
        request_file = task / "request.json"
        request_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        progress = ProgressBar(3)
        conditioning = [python, "-u", str(Path(__file__).with_name("prepare_condition.py")),
                        "--comfy", str(comfy), "--models", models,
                        "--workbench", str(workbench), "--request", str(request_file)]
        run_logged(conditioning, comfy, task / "condition.log", env)
        progress.update(1)
        command.extend(("--condition", str(task / "condition.safetensors")))
        run_logged(command, repo, task / "render.log", env)
        progress.update(1)
        clip = run_dir / "output.mp4"
        destination = output / f"{task.name}.mp4"
        partial = destination.with_suffix(".partial.mp4")
        shutil.copyfile(clip, partial)
        partial.replace(destination)
        progress.update(1)
        media = {"filename": destination.name, "subfolder": "h3-turbo",
                 "type": "output", "format": "video/mp4"}
        return {"ui": {"images": [media], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {
    "H3TurboRuntime": H3TurboRuntime,
    "H3TurboSettings": H3TurboSettings,
    "H3TurboMediaCondition": H3TurboMediaCondition,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TurboRuntime": "H3 Turbo Runtime · Audio / Video",
    "H3TurboSettings": "H3 · Generation Parameters",
    "H3TurboMediaCondition": "H3 · Optional Image / Audio / Video",
}
