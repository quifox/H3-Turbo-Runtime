"""Unpack H3 W4A8 weight tiles for the shared native INT8 runtime."""
import comfy.ops as ops
from comfy_kitchen.backends import cuda as native
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout


def unpack_for_call(layer, x):
    with ops.CastBiasWeightContext(
        layer, x, offloadable=True, compute_dtype=x.dtype, want_requant=True,
    ) as (weight, bias):
        if bias is not None or layer.quant_format != "asym_w4a8_int8":
            raise ValueError(f"unexpected H3 FFN format: {layer.quant_format}, bias={bias is not None}")
        p = weight._params
        qdata, scale, channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
        if p.transposed or p.mma_packed or correction is not None:
            raise ValueError("This H3 W4A8 implementation requires canonical weights without a correction matrix")
        decoded = native._dequant_int4_grouped_to_int8(qdata, scale, codebook, p.group_size)
        # The scales may live in a borrowed ComfyUI cast buffer.
        return decoded, channel.clone(), p.convrot_groupsize, p.orig_dtype
