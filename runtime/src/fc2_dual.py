"""在 Ada 上分别消费 ATen 精确 LoRA 输入和原生快数学量化输入。"""

import torch
import torch.nn.functional as F
from comfy_kitchen.backends import cuda as native

from fc2_pair import GROUP, K, ROWS, PairedFC2


class DualActivationFC2(PairedFC2):
    def stage_dual(self, precise_half, quant_fp32, slot):
        if (slot not in (0, 1) or slot != len(self.low) or self.turing
                or precise_half.dtype != torch.float16 or quant_fp32.dtype != torch.float32
                or precise_half.shape != (ROWS, K) or quant_fp32.shape != (ROWS, K)
                or precise_half.device != self.device or quant_fp32.device != self.device
                or not precise_half.is_contiguous() or not quant_fp32.is_contiguous()
                or not native._convrot_fused_shared_memory_fits(quant_fp32, K, GROUP)):
            raise ValueError("Ada dual FC2 expects a full adjacent tile in both arithmetic forms")
        start = slot * ROWS
        stop = start + ROWS
        wrap = native._wrap_for_dlpack
        stream = torch.cuda.current_stream(self.device).cuda_stream
        native._C.quantize_int8_rowwise_convrot64(
            wrap(quant_fp32), wrap(self.q[start:stop]), wrap(self.scale[start:stop]), GROUP, False,
            native._input_act_code(None), 0, wrap(self.empty), 0.0, stream,
        )
        # The original FP32 LoRA comparison is a one-time validation probe.
        # This route is enabled only after that probe has already been recorded.
        if self.adapter.role not in self.adapter.stats["fp16_delta_relative_l2"]:
            raise RuntimeError("Validate the original FC2 adapter first")
        self.low.append(F.linear(precise_half, self.down))
