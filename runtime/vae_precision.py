"""Select the validated low-memory H3 VAE execution by default.

H3 upstream audio weights support FP32 and can be tested without changing the
default. Same-latent FP16 / FP32 re-decode did not materially improve speech.
"""
import os


def vae_flag_for_stage(stage):
    if stage == "audio" and os.getenv("H3_AUDIO_VAE_FP32", "0") == "1":
        return "--fp32-vae"
    return "--fp16-vae"
