"""Common native audio and video decoding for short and streaming H3 paths."""

def decode_video(args, files, *, geometry):
    import av
    import numpy as np
    import torch
    import nodes
    from safetensors.torch import load_file

    latent = load_file(args.out / "latents.safetensors")["video"]
    vae, = nodes.VAELoader().load_vae(files["video_vae"])
    if vae.first_stage_model.tile_size != 256:
        raise RuntimeError("Video decoder tile size differs from the supported VAE configuration")
    vae.first_stage_model.tile_overlap_min = 32
    images, = nodes.VAEDecode().decode(vae, {"samples": latent})
    if tuple(images.shape) != (geometry.native_frames, geometry.height, geometry.width, 3) or not torch.isfinite(images).all():
        raise ValueError("解码后的画面不完整")
    frame_indexes = np.rint(np.linspace(0, geometry.native_frames - 1,
                                       geometry.export_frames)).astype(np.int64)
    with av.open(str(args.out / "video.mp4"), mode="w") as file:
        stream = file.add_stream("libx264", rate=geometry.fps)
        stream.width = geometry.width
        stream.height = geometry.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "fast", "crf": "20"}
        for index in frame_indexes:
            frame = images[int(index)].mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame.cpu().numpy(), format="rgb24")):
                file.mux(packet)
        for packet in stream.encode():
            file.mux(packet)
    return {"native_frames": geometry.native_frames, "export_frames": geometry.export_frames,
            "minimum_overlap_px": 32}


def decode_audio(args, files):
    from safetensors.torch import load_file
    import torch
    import nodes

    latent = load_file(args.out / "latents.safetensors")["audio"]
    vae, = nodes.VAELoader().load_vae(files["audio_vae"])
    wave = vae.decode(latent).movedim(-1, 1)
    wave = (wave / (5.0 * wave.std(dim=(1, 2), keepdim=True)).clamp_min(1.0)).cpu().float()
    rate = int(vae.audio_sample_rate)
    if wave.shape[:2] != (1, 2) or rate != 32000 or not torch.isfinite(wave).all():
        raise ValueError("Audio decoder output does not match the H3 channel/sample-rate contract")
    wave[0].T.contiguous().numpy().astype("<f4", copy=False).tofile(args.out / "audio.f32le")
    return {"sample_rate_hz": rate, "channels": 2, "native_samples": wave.shape[-1],
            "vae_compute_dtype": str(vae.vae_dtype)}
