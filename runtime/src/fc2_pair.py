"""Consume adjacent H3 FP32 SwiGLU tiles with one native INT8 FC2 launch."""


import torch
import torch.nn.functional as F
from comfy_kitchen.backends import cuda as native


ROWS = 1024
TAIL = 195
K = 14336
N = 5376
GROUP = 256
KITCHEN_SUPPORTED = all(hasattr(native, symbol) for symbol in (
    "_int8_weight_scale_arg", "_empty_cuda_tensor", "_build_hadamard",
    "_cuda_device_should_use_turing_kernels", "_C",
))


class PairedFC2:
    """Keep the baseline quantization and LoRA row geometry and write FC2 in-place."""

    def __init__(self, weight, weight_scale, group, out_dtype, adapter, adapter_weights):
        down, up = adapter_weights
        if (not KITCHEN_SUPPORTED or not torch.is_inference_mode_enabled()
                or not weight.is_cuda or not weight.is_contiguous()
                or weight.shape != (N, K) or weight.dtype != torch.int8
                or group != GROUP or out_dtype != torch.float32
                or down.ndim != 2 or up.ndim != 2
                or down.shape[1] != K or up.shape != (N, down.shape[0])
                or down.dtype != torch.float16 or up.dtype != torch.float16
                or down.device != weight.device or up.device != weight.device):
            raise RuntimeError("Paired FC2 requires the frozen H3 arithmetic contract")

        self.device = weight.device
        self.weight = weight
        self.column_scale = native._int8_weight_scale_arg(weight_scale, self.device).reshape(-1)
        if self.column_scale.shape != (N,):
            raise RuntimeError("Paired FC2 requires per-output-channel scales")
        self.adapter = adapter
        self.down, self.up = adapter_weights
        self.empty = native._empty_cuda_tensor(self.device, torch.float32)
        self.hadamard = native._build_hadamard(GROUP, device=self.device, dtype=torch.float32)
        self.turing = native._cuda_device_should_use_turing_kernels(weight.get_device())
        if not self.turing and native._DISABLE_CUTLASS_INT8:
            raise RuntimeError("Paired FC2 requires a native INT8 GEMM or the Turing kernel path")
        self.q = torch.empty((2 * ROWS, K), dtype=torch.int8, device=self.device)
        self.scale = torch.empty((2 * ROWS, 1), dtype=torch.float32, device=self.device)
        self.low = []
        self.reference = None

    def _prepare(self, active, start, rows, raw=None):
        if (active.device != self.device or active.dtype != torch.float32
                or active.shape != (rows, K) or not active.is_contiguous()):
            raise RuntimeError("Paired FC2 requires a contiguous FP32 SwiGLU tile")
        q = self.q[start:start + rows]
        scale = self.scale[start:start + rows]
        wrap = native._wrap_for_dlpack
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if native._convrot_fused_shared_memory_fits(active, K, GROUP):
            # On SM89 the baseline may fuse stock native SwiGLU and rotation.
            # Passing the LoRA-updated FC1 tile keeps that native arithmetic path.
            source = active if raw is None else raw
            act = None if raw is None else "swiglu"
            if (source.device != self.device or source.dtype != torch.float32
                    or not source.is_contiguous() or source.shape != (rows, K if raw is None else 2 * K)):
                raise RuntimeError("Invalid native rotation source for Paired FC2")
            native._C.quantize_int8_rowwise_convrot64(
                wrap(source), wrap(q), wrap(scale), GROUP, False,
                native._input_act_code(act), 0, wrap(self.empty), 0.0, stream,
            )
        else:
            rotated = native._rotate_activation(active, self.hadamard, GROUP)
            native._C.quantize_int8_rowwise(wrap(rotated), wrap(q), wrap(scale), False, 0, stream)

        probes = self.adapter.stats["fp16_delta_relative_l2"]
        if self.adapter.role not in probes and self.reference is None:
            self.reference = F.linear(F.linear(active[:8].float(), self.down.float()), self.up.float())
        return F.linear(active.to(self.down.dtype), self.down)

    def _add_adapter(self, output, low):
        delta = F.linear(low, self.up)
        if self.reference is not None:
            reference, self.reference = self.reference, None
            error = ((delta[:8].float() - reference).norm() / reference.norm().clamp_min(1e-12)).item()
            if not 0 <= error < 0.02 or float(reference.norm()) <= 0:
                raise RuntimeError(f"LoRA numerical check failed for {self.adapter.prefix}: {error}")
            self.adapter.stats["fp16_delta_relative_l2"][self.adapter.role] = error
        output.add_(delta, alpha=self.adapter.scale)

    def _write(self, out, rows):
        if (out.device != self.device or out.dtype != torch.float32
                or out.shape != (rows, N) or not out.is_contiguous() or out.data_ptr() % 16):
            raise RuntimeError("Paired FC2 requires aligned destination rows")
        q = self.q[:rows]
        scale = self.scale[:rows]
        wrap = native._wrap_for_dlpack
        stream = torch.cuda.current_stream(self.device).cuda_stream
        dtype_code = native.DTYPE_TO_CODE[out.dtype]
        if self.turing and rows == TAIL:
            # The Turing backend uses cuBLAS for the short M195 tail.
            if K % 16 or N % native._cublas_int8_n_alignment(q):
                raise RuntimeError("Paired FC2 tail expects no cuBLAS padding")
            acc = torch.empty((rows, N), dtype=torch.int32, device=self.device)
            native._C.cublas_gemm_int8(
                wrap(q), wrap(self.weight), wrap(acc), wrap(native.get_cublas_workspace()), stream,
            )
            native._C.dequantize_int8_linear(
                wrap(acc), wrap(scale), wrap(self.column_scale), wrap(self.empty), wrap(out), dtype_code, stream,
            )
            return

        gemm = native._C.cutlass_turing_int8_dequant if self.turing else native._C.cutlass_int8_dequant
        row_scale = scale.reshape(-1) if self.turing else scale
        accepted = gemm(
            wrap(q), wrap(self.weight), wrap(row_scale), wrap(self.column_scale), wrap(self.empty),
            wrap(out), dtype_code, stream,
        )
        if not accepted:
            raise RuntimeError(f"Native Paired FC2 rejected M={rows}, N={N}, K={K}")

    def _count(self, rows, num_tiles, key):
        tiles = self.adapter.stats.setdefault("measured_fc2_tiles", {})
        tiles[str(rows)] = tiles.get(str(rows), 0) + num_tiles
        counts = self.adapter.stats.setdefault("paired_fc2_direct", {})
        counts[key] = counts.get(key, 0) + 1

    def stage(self, active, slot, *, raw_for_native=None):
        if slot != len(self.low) or slot not in (0, 1):
            raise RuntimeError("Stage two adjacent FC2 row tiles in order")
        self.low.append(self._prepare(active, ROWS * slot, ROWS, raw_for_native))

    def write_pair(self, out):
        if len(self.low) != 2:
            raise RuntimeError("Both FC2 tiles must be staged before writing")
        self._write(out, 2 * ROWS)
        for slot, low in enumerate(self.low):
            self._add_adapter(out[slot * ROWS:(slot + 1) * ROWS], low)
        self.low.clear()
        self._count(ROWS, 2, "pair_gemm_and_epilogue")

    def write_tail(self, active, out, *, raw_for_native=None):
        if self.low:
            raise RuntimeError("Finish the FC2 pair before writing the tail")
        low = self._prepare(active, 0, TAIL, raw_for_native)
        self._write(out, TAIL)
        self._add_adapter(out, low)
        self._count(TAIL, 1, "tail_gemm_and_epilogue")
