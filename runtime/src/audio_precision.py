"""Preserve fast FP16 LoRA except for rows that cannot be represented in FP16.

The original 5-second checkpoint already uses FP16 for LoRA A/B. At 15s, an
audio token can exceed 65504 before the FC2 LoRA down-projection: converting
this FP32 activation to FP16 produces infinity and then contaminates the row.
Keep the original path for every safe row, and compute only out-of-range rows
with the original *FP16-stored* LoRA weights converted to FP32 for arithmetic.
This avoids recomputing the main FC1/FC2 W4A8 layers on the normal path.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F

from lora_adapter import Adapter


def add_selective_fp32(self, output, x, weights, original, report):
    if self.role != 'mlp.fc2':
        return original(self, output, x, weights)
    # The output is a FP32 main FC2 projection. The FC2 LoRA activates input
    # SwiGLU in FP32 then immediately casts to FP16 inside Adapter.add.
    absolute_peak = x.abs().amax(dim=-1)
    if not bool(torch.isfinite(absolute_peak).all()):
        raise RuntimeError('Nonfinite inputs reached FC2 LoRA: '+self.prefix)
    # 65504 is the largest finite FP16 number. Use a strict threshold, so all
    # normal rows preserve exactly the original FP16 numerical path.
    unsafe = absolute_peak > 65504.0
    if not bool(unsafe.any()):
        return original(self, output, x, weights)
    bad_idx = unsafe.nonzero().flatten()
    good_idx = (~unsafe).nonzero().flatten()
    if good_idx.numel():
        good_output = output.index_select(0, good_idx)
        original(self, good_output, x.index_select(0, good_idx), weights)
        output.index_copy_(0, good_idx, good_output)
    down, up = weights
    bad_input = x.index_select(0, bad_idx).float()
    # Only the rare offending rows upcast. Keep the checkpoint and regular
    # rows untouched; do not clamp activations or manufacture a new waveform.
    fp32_delta = F.linear(F.linear(bad_input, down.float()), up.float())
    bad_output = output.index_select(0, bad_idx)
    bad_output.add_(fp32_delta, alpha=self.scale)
    output.index_copy_(0, bad_idx, bad_output)
    record = {'prefix': self.prefix, 'rows_in_chunk': bad_idx.tolist(),
              'max_bad_activation': float(absolute_peak.index_select(0, bad_idx).max()),
              'bad_count': int(bad_idx.numel())}
    report.append(record)
    print('H3_FC2_LORA_FP32_PREEMPT', record, flush=True)


@contextmanager
def safe_fc2_lora(report):
    original = Adapter.add

    def replacement(self, output, x, weights):
        return add_selective_fp32(self, output, x, weights, original, report)

    with patch.object(Adapter, 'add', replacement):
        yield
