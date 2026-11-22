"""H3 Turbo FC1，两个相邻行块共用一次原生 INT8 GEMM。"""


import torch
from comfy_kitchen.backends import cuda as native


K = 5376
N = 28672
TILE_ROWS = 1024
PAIR_ROWS = 2 * TILE_ROWS
GROUP = 256
KITCHEN_SUPPORTED = all(hasattr(native, symbol) for symbol in (
    "_convrot_fused_shared_memory_fits", "_empty_cuda_tensor", "_wrap_for_dlpack",
    "_int8_weight_scale_arg", "_input_act_code", "_C",
))


def prepare_fc1_pair(x, group):
    """使用与独立行块相同的 ConvRot64 运算准备 FP32 行尺度和 INT8 输入。"""
    if (not torch.is_inference_mode_enabled() or not x.is_cuda
            or x.dtype != torch.float32 or x.shape not in ((TILE_ROWS, K), (PAIR_ROWS, K))
            or group != GROUP or not KITCHEN_SUPPORTED):
        raise ValueError("FC1 pair expects H3's pinned FP32 ConvRot256 CUDA path")
    if not native._convrot_fused_shared_memory_fits(x, K, group):
        raise RuntimeError("FC1 pair shared memory does not fit the stock ConvRot64 kernel")
    x = x.contiguous()
    rows = x.shape[0]
    device = x.device
    a8 = torch.empty((rows, K), dtype=torch.int8, device=device)
    row_scale = torch.empty((rows, 1), dtype=torch.float32, device=device)
    aux = native._empty_cuda_tensor(device, x.dtype)
    wrap = native._wrap_for_dlpack
    native._C.quantize_int8_rowwise_convrot64(
        wrap(x), wrap(a8), wrap(row_scale), group, False, native._input_act_code(None),
        0, wrap(aux), 0.0, torch.cuda.current_stream(device).cuda_stream,
    )
    return a8, row_scale


def gemm_prepared_fc1(a8, row_scale, weight, weight_scale, dtype, out=None):
    """以原生 Tensor Core 计算，保持 FP32 尾运算顺序。支持 256 列对齐的通道切片。"""
    device = a8.device
    rows = a8.shape[0]
    channels = weight.shape[0]
    if (not a8.is_cuda or a8.dtype != torch.int8 or a8.shape != (rows, K)
            or rows not in (TILE_ROWS, PAIR_ROWS) or not a8.is_contiguous()
            or not channels or channels % 256 != 0 or weight.shape != (channels, K)
            or weight.device != device or weight.dtype != torch.int8 or not weight.is_contiguous()
            or row_scale.device != device or row_scale.dtype != torch.float32
            or row_scale.numel() != rows or not row_scale.is_contiguous()
            or dtype != torch.float32):
        raise ValueError("Invalid FC1 INT8 Tensor Core inputs")
    col = native._int8_weight_scale_arg(weight_scale, device).reshape(-1)
    if col.numel() != channels:
        raise ValueError("Expected one FC1 scale per output channel")
    if out is None:
        out = torch.empty((rows, channels), dtype=dtype, device=device)
    elif (out.shape != (rows, channels) or out.device != device
          or out.dtype != dtype or not out.is_contiguous()):
        raise ValueError("Expected a contiguous FP32 FC1 output buffer")

    ext = native._C
    index = a8.get_device()
    if native._cuda_device_should_use_turing_kernels(index):
        gemm = ext.cutlass_turing_int8_dequant
        rows_for_kernel = row_scale.reshape(-1)
    elif (native._cuda_device_supports_cutlass_int8_dequant(a8)
          and not native._DISABLE_CUTLASS_INT8):
        gemm = ext.cutlass_int8_dequant
        rows_for_kernel = row_scale.reshape(rows, 1)
    else:
        raise RuntimeError("FC1 pair requires a native CUTLASS INT8 epilogue")
    bias = native._empty_cuda_tensor(device, dtype)
    wrap = native._wrap_for_dlpack
    used = gemm(
        wrap(a8), wrap(weight), wrap(rows_for_kernel), wrap(col), wrap(bias), wrap(out),
        native.DTYPE_TO_CODE[dtype], torch.cuda.current_stream(device).cuda_stream,
    )
    if not used:
        raise RuntimeError("FC1 pair native INT8 GEMM rejected the selected shape")
    return out


def fc1_pair(x, weight, weight_scale, group, dtype):
    """为两片连续行返回原顺序的投影结果，调用方保持原 LoRA 分块方式。"""
    if x.shape != (PAIR_ROWS, K) or weight.shape != (N, K):
        raise ValueError("Expected aligned H3 paired FC1 rows and the complete weight matrix")
    a8, row_scale = prepare_fc1_pair(x, group)
    return gemm_prepared_fc1(a8, row_scale, weight, weight_scale, dtype)
