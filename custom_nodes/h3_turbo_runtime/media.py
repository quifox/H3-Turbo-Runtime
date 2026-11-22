"""Read local user media into the image/audio inputs expected by H3's nodes."""
import json
import shutil
import subprocess


def image(path):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float32).copy() / 255
    return torch.from_numpy(pixels).unsqueeze(0)


def _command(name):
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"{name} is required to decode video or audio conditioning")
    return tool


def video(path):
    """Decode a user reference into H3's 24-fps batch of RGB video frames."""
    import numpy as np
    import torch
    probe = subprocess.run([_command("ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "json", str(path)],
                           check=True, capture_output=True)
    streams = json.loads(probe.stdout)["streams"]
    if not streams:
        raise ValueError("Reference video has no video stream")
    width, height = int(streams[0]["width"]), int(streams[0]["height"])
    result = subprocess.run([_command("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path),
                             "-map", "0:v:0", "-vf", "fps=24", "-pix_fmt", "rgb24",
                             "-f", "rawvideo", "pipe:1"], check=True, capture_output=True)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    unit = width * height * 3
    if not unit or not pixels.size or pixels.size % unit:
        raise ValueError("Reference video does not contain complete RGB frames")
    return torch.from_numpy(pixels.reshape(-1, height, width, 3).copy()).float().div_(255)


def audio(path):
    """Decode a user audio track to ComfyUI's [1,2,N] waveform at 32 kHz."""
    import numpy as np
    import torch
    result = subprocess.run([_command("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path),
                             "-map", "0:a:0", "-ar", "32000", "-ac", "2", "-f", "f32le",
                             "pipe:1"], check=True, capture_output=True)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size == 0 or samples.size % 2:
        raise ValueError("Reference audio is missing or incomplete")
    waveform = torch.from_numpy(samples.reshape(-1, 2).copy()).T.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 32000}
