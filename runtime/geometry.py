"""Pure frame/time geometry for MiniMax H3. No CUDA, ComfyUI or model imports.

The upstream H3 encoder rounds its requested *native* frame count up to the
17k+5 grid. Video and audio latent lengths must use that native count, while
the exported MP4 obeys the user's display duration. Keep both distinct.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

FPS = 24
AUDIO_HZ = 32000
AUDIO_LATENT_FPS = 40
WIDTH = 832
HEIGHT = 480


@dataclass(frozen=True)
class Geometry:
    requested_s: float
    export_frames: int
    native_frames: int
    video_latent_t: int
    audio_latent_t: int
    width: int = WIDTH
    height: int = HEIGHT
    fps: int = FPS

    @property
    def exported_s(self):
        return self.export_frames / self.fps

    @property
    def audio_samples(self):
        return round(self.export_frames * AUDIO_HZ / self.fps)

    @property
    def shapes(self):
        return {"video": (1, 24, self.video_latent_t, self.height // 16, self.width // 16),
                "audio": (1, 32, 2, self.audio_latent_t)}

def from_seconds(duration_s, *, width=WIDTH, height=HEIGHT, output_fps=FPS):
    """Validate output geometry and map a duration to the native 24-fps H3 grid."""
    try:
        value = float(duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("--duration must be a finite positive number of seconds") from exc
    if not isfinite(value) or value <= 0:
        raise ValueError("--duration must be a finite positive number of seconds")
    if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (width, height, output_fps)):
        raise ValueError("Width, height and output FPS must be positive integers")
    if width % 16 or height % 16:
        raise ValueError("H3 latent width and height must be divisible by 16")
    frames = int((Decimal(str(value)) * FPS).to_integral_value(rounding=ROUND_HALF_UP))
    export_frames = int((Decimal(str(value)) * output_fps).to_integral_value(rounding=ROUND_HALF_UP))
    if frames < 1 or export_frames < 1:
        raise ValueError("Duration is shorter than one native/output video frame")
    native = max(5, frames)
    native += (5 - native) % 17
    # Same formulas as ComfyUI's temporal_shape: 124 -> (37,207),
    # 362 -> (107,603). No extra hard-coded duration limit is imposed here.
    video_t = 2 if native <= 5 else ((native - 5) // 17) * 5 + 2
    audio_t = round(native * AUDIO_LATENT_FPS / FPS)
    return Geometry(value, export_frames, native, video_t, audio_t, width, height, output_fps)


def as_metadata(geometry):
    return {"duration_requested_s": geometry.requested_s,
            "duration_exported_s": geometry.exported_s,
            "export_frames": geometry.export_frames,
            "native_frames": geometry.native_frames,
            "video_latent_t": geometry.video_latent_t,
            "audio_latent_t": geometry.audio_latent_t,
            "fps": geometry.fps, "native_fps": FPS, "width": geometry.width, "height": geometry.height}
