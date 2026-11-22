"""Audio-only finite-row safety for the fused Ada FFN without diagnostic hooks.

Keeps the qualified fast kernels unchanged. Only the narrow audio segment of
each FFN output is checked; affected rows are selectively recomputed in FP32.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import lora_adapter as routing
from audio_recovery import recompute_ffn


@contextmanager
def guard_fast_audio_rows(model, stats):
    diffusion = model.model.diffusion_model
    original_blocks = diffusion._run_blocks
    original_ffn = routing.chunked_mlp
    span = None
    step = 0

    def blocks(h, t_emb, mod_segments, layout, transformer_options):
        nonlocal span, step
        step += 1
        start, end, _ = next(segment for segment in layout.segments if segment[2] == "audio")
        span = slice(start, end)
        try:
            return original_blocks(h, t_emb, mod_segments, layout, transformer_options)
        finally:
            span = None

    def mlp(module, hidden, rows, first, second):
        result = original_ffn(module, hidden, rows, first, second)
        if span is None:
            return result
        # Only scan audio, not the full multimodal hidden state. The predicate
        # consumes no GPU parameter copies; for normal rows it does not alter
        # arithmetic, kernel order or output tensors.
        audio = result[span]
        # Check only the audio block, not the entire video/text context.
        # On the verified SM89 workload this predicate was faster and used
        # less peak PyTorch memory than a pair of amin/amax row reductions.
        bad_local = (~torch.isfinite(audio).all(dim=1)).nonzero().flatten()
        if bad_local.numel() == 0:
            return result
        indices = bad_local + span.start
        affected_input = hidden.index_select(0, indices)
        if not torch.isfinite(affected_input).all():
            raise RuntimeError("Nonfinite audio entered the FC1/FC2 block; cannot safely repair it")
        repaired = recompute_ffn(module, affected_input, first, second)
        if not torch.isfinite(repaired).all():
            raise RuntimeError("FP32 FC1/FC2 recovery also produced nonfinite audio")
        result.index_copy_(0, indices, repaired)
        info = {"step": step, "module": first.prefix,
                "audio_rows": bad_local.tolist(), "recomputed_rows": int(indices.numel())}
        stats.setdefault("ffn_recovered_audio", []).append(info)
        print("H3_FAST_AUDIO_ROW_RECOVERED", info, flush=True)
        return result

    # This overrides only the fused FFN dispatch, after it has been installed.
    with patch.object(diffusion, "_run_blocks", blocks), patch.object(routing, "chunked_mlp", mlp):
        yield
