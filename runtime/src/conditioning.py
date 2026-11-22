"""Safetensors interchange for all MiniMax H3 AV conditioning modalities.

The legacy frame_i format is still readable. Schema v2 additionally carries
image/video/audio timeline guides and ordered ref2va blocks as tensors plus
JSON metadata; it preserves the same conditioning structures ComfyUI hands to
MiniMaxH3.extra_conds without relying on pickled Python objects.
"""
import json

SCHEMA_KEY = "h3_conditioning_schema"


def encode_condition(positive, latent):
    """Serialize an upstream H3 conditioning object to tensor-only weights.

    Returns (tensor_mapping, metadata_mapping) for safetensors.save_file.
    """
    import torch
    if len(positive) != 1:
        raise ValueError("Expected a single H3 conditioning segment")
    text, payload = positive[0]
    video, audio = latent["samples"].unbind()
    # The original encoder contract stores projected text as FP32 even when
    # the working CLIP tensors use BF16 or FP16.
    states = {"text": text.float(), "token_tags": payload["minimax_token_tags"].long(),
              "video_init": video, "audio_init": audio}
    guides = []
    for index, guide in enumerate(payload.get("minimax_keyframes", [])):
        descriptor = {"frame": int(guide["resolved_frame_index"])}
        if guide.get("latent") is not None:
            key = f"guide_{index}_video"
            states[key] = guide["latent"]
            descriptor["video_key"] = key
        if guide.get("audio_latent") is not None:
            key = f"guide_{index}_audio"
            states[key] = guide["audio_latent"]
            descriptor["audio_key"] = key
        if "video_key" not in descriptor and "audio_key" not in descriptor:
            raise ValueError("H3 timeline guide needs visual or audio conditioning")
        guides.append(descriptor)
    refs = []
    for index, ref in enumerate(payload.get("minimax_refs", [])):
        descriptor = {key: value for key, value in ref.items() if key not in ("latent", "audio_latent")}
        if ref.get("latent") is not None:
            key = f"ref_{index}_video"
            states[key] = ref["latent"]
            descriptor["video_key"] = key
        if ref.get("audio_latent") is not None:
            key = f"ref_{index}_audio"
            states[key] = ref["audio_latent"]
            descriptor["audio_key"] = key
        refs.append(descriptor)
    cpu = {key: tensor.detach().cpu().contiguous() for key, tensor in states.items()}
    if not torch.isfinite(cpu["text"]).all():
        raise ValueError("H3 text encoding contains non-finite values")
    metadata = {SCHEMA_KEY: "2", "h3_guides": json.dumps(guides, separators=(",", ":")),
                "h3_refs": json.dumps(refs, separators=(",", ":"))}
    return cpu, metadata


def load_condition(path, geometry):
    import torch
    from comfy.nested_tensor import NestedTensor
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        states = {key: file.get_tensor(key) for key in file.keys()}
    text = states["text"]
    if text.ndim != 3 or text.shape[0] != 1 or text.shape[2] != 5120 or text.dtype != torch.float32:
        raise ValueError("H3 text conditioning must have FP32 shape [1, L, 5120]")
    if metadata.get(SCHEMA_KEY) == "2":
        guides = []
        for descriptor in json.loads(metadata["h3_guides"]):
            guide = {"resolved_frame_index": int(descriptor["frame"])}
            for field, data_key in (("latent", "video_key"), ("audio_latent", "audio_key")):
                if data_key in descriptor:
                    guide[field] = states[descriptor[data_key]]
            guides.append(guide)
        refs = []
        for descriptor in json.loads(metadata["h3_refs"]):
            ref = {key: value for key, value in descriptor.items() if key not in ("video_key", "audio_key")}
            for field, data_key in (("latent", "video_key"), ("audio_latent", "audio_key")):
                if data_key in descriptor:
                    ref[field] = states[descriptor[data_key]]
            refs.append(ref)
    else:
        guides = [{"resolved_frame_index": int(key[6:]), "latent": value}
                  for key, value in states.items() if key.startswith("frame_") and key[6:].isdigit()]
        guides.sort(key=lambda guide: guide["resolved_frame_index"])
        refs = []
    for guide in guides:
        frame = guide["resolved_frame_index"]
        if not 0 <= frame < geometry.native_frames:
            raise ValueError("H3 guide lies outside the native output timeline")
        video_latent = guide.get("latent")
        if video_latent is not None:
            if video_latent.ndim != 5 or video_latent.shape[:2] != (1, 24) or video_latent.shape[-2:] != (geometry.height // 16, geometry.width // 16):
                raise ValueError("H3 guide video latent has incompatible spatial shape")
            # A single image covers one frame; a guide clip covers 17k+5 frames.
            length = 1 if video_latent.shape[2] == 1 else (video_latent.shape[2] - 2) // 5 * 17 + 5
            if frame + length > geometry.native_frames:
                raise ValueError("H3 guide clip extends outside the output timeline")
        audio_latent = guide.get("audio_latent")
        if audio_latent is not None and (audio_latent.ndim != 4 or audio_latent.shape[:3] != (1, 32, 2)):
            raise ValueError("H3 guide audio latent must have shape [1,32,2,T]")
    for ref in refs:
        if ref.get("kind") not in ("image", "audio", "video", "video_audio"):
            raise ValueError("Unrecognized H3 reference kind")
        for field in ("latent", "audio_latent"):
            value = ref.get(field)
            if value is not None and value.ndim not in (4, 5):
                raise ValueError("Invalid H3 reference tensor rank")
    parts = []
    for name, shape in geometry.shapes.items():
        latent = states[f"{name}_init"]
        if tuple(latent.shape) != shape or latent.dtype != torch.float32:
            raise ValueError(f"H3 {name} latent does not match requested duration {geometry.requested_s}s: "
                             f"observed={tuple(latent.shape)}, expected={shape}")
        parts.append(latent)
    condition_metadata = {"pooled_output": None, "minimax_token_tags": states["token_tags"],
                          "minimax_keyframes": guides, "minimax_refs": refs}
    return [(text, condition_metadata)], {"samples": NestedTensor(tuple(parts))}
