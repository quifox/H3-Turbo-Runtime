"""Observe true H3 denoiser calls and record matched GPU/host measurement windows."""
from contextlib import contextmanager
import json
from time import perf_counter, time

import torch
from comfy import model_management



def tensor_stats(value):
    invalid = ~torch.isfinite(value)
    count = int(torch.count_nonzero(invalid).item())
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "nonfinite_values": count}
    if count:
        result["first_invalid_indices"] = invalid.nonzero()[:8].tolist()
    return result


def check_latents(observed, expected):
    return all(tuple(item["shape"]) == expected[name] and item["dtype"] == "torch.float32"
               and item["nonfinite_values"] == 0 for name, item in observed.items())


@contextmanager
def monitor_layers(model):
    state = {"steps": []}
    elapsed_start = perf_counter()
    handles = []
    diffusion = model.model.diffusion_model

    def before_step(module, inputs):
        torch.cuda.synchronize()
        if not state["steps"]:
            state["before_denoiser_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        step = {"index": len(state["steps"]) + 1, "start_epoch_s": time()}
        state["steps"].append(step)
        print("DENOISE_STEP_START", json.dumps(step), flush=True)

    def after_step(module, inputs, output):
        torch.cuda.synchronize()
        step = state["steps"][-1]
        end = time()
        step.update(end_epoch_s=end, elapsed_s=end - step["start_epoch_s"],
                    peak_torch_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        print("DENOISE_STEP_MEMORY", json.dumps(step), flush=True)
        if isinstance(output, (list, tuple)) and len(output) == 2:
            row = {name: tensor_stats(value) for name, value in zip(("video", "audio"), output, strict=True)}
            step["denoiser_outputs"] = row
            print("DENOISER_LATENTS", step["index"], row, flush=True)

    handles.extend((diffusion.register_forward_pre_hook(before_step), diffusion.register_forward_hook(after_step)))
    for index, layer in enumerate(diffusion.blocks):
        def before(module, inputs, index=index):
            if index == 0:
                state["sequence"] = list(inputs[0].shape)
                device = inputs[0].device
                slots = [round(slot.nbytes / 2**20, 2)
                         for slot in model_management.STREAM_CAST_BUFFERS.values()
                         if slot.device == device]
                state["steps"][-1]["body_entry_cast_slots_mib"] = sorted(slots)
                print("DENOISE_SEQUENCE", state["sequence"], "ALLOC_MIB",
                      round(torch.cuda.memory_allocated(device) / 2**20), "CAST_SLOTS_MIB", slots, flush=True)

        def after(module, inputs, output, index=index):
            if index in (0, 9, 24, 49):
                print("BLOCK_FINISHED", index, round(perf_counter() - elapsed_start, 3),
                      "PEAK_ALLOC_MIB", round(torch.cuda.max_memory_allocated() / 2**20), flush=True)

        handles.extend((layer.register_forward_pre_hook(before), layer.register_forward_hook(after)))
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()
