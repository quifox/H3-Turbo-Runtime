"""CPU-only regression for MiniMax H3 native multimodal conditioning interchange."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "runtime/src"))


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("safetensors")
    and importlib.util.find_spec("comfy"),
    "Requires the H3 PyTorch/ComfyUI environment",
)
class ConditioningCodecTests(unittest.TestCase):
    def test_multimodal_roundtrip_and_legacy_file(self):
        import torch
        from comfy.nested_tensor import NestedTensor
        from safetensors.torch import save_file
        from conditioning import encode_condition, load_condition
        from geometry import from_seconds

        geometry = from_seconds(5)
        tags = torch.zeros(8, dtype=torch.int64)
        text = torch.zeros((1, 8, 5120), dtype=torch.bfloat16)
        video = torch.zeros(geometry.shapes["video"])
        audio = torch.zeros(geometry.shapes["audio"])
        guides = [
            {"resolved_frame_index": 0, "latent": torch.ones((1, 24, 1, 30, 52))},
            {"resolved_frame_index": 42, "audio_latent": torch.ones((1, 32, 2, 20))},
            {"resolved_frame_index": 54, "latent": torch.ones((1, 24, 7, 30, 52))},
        ]
        refs = [
            {
                "kind": "image",
                "latent_h": 12,
                "latent_w": 12,
                "latent": torch.ones((1, 24, 1, 12, 12)),
            },
            {
                "kind": "audio",
                "ref_audio_t": 12,
                "audio_latent": torch.ones((1, 32, 2, 12)),
            },
            {
                "kind": "video_audio",
                "latent_t": 7,
                "latent_h": 12,
                "latent_w": 12,
                "ref_audio_t": 12,
                "latent": torch.ones((1, 24, 7, 12, 12)),
                "audio_latent": torch.ones((1, 32, 2, 12)),
            },
        ]
        positive = [
            (
                text,
                {
                    "minimax_token_tags": tags,
                    "minimax_keyframes": guides,
                    "minimax_refs": refs,
                },
            )
        ]
        latent = {"samples": NestedTensor((video, audio))}
        tensors, metadata = encode_condition(positive, latent)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "combined.safetensors"
            save_file(tensors, str(file), metadata=metadata)
            result, initialized = load_condition(file, geometry)
            payload = result[0][1]
            self.assertEqual(
                [g["resolved_frame_index"] for g in payload["minimax_keyframes"]],
                [0, 42, 54],
            )
            self.assertEqual(payload["minimax_keyframes"][2]["latent"].shape[2], 7)
            self.assertEqual(
                payload["minimax_keyframes"][1]["audio_latent"].shape[-1], 20
            )
            self.assertEqual(
                [item["kind"] for item in payload["minimax_refs"]],
                ["image", "audio", "video_audio"],
            )
            self.assertEqual(payload["minimax_refs"][2]["latent"].shape[2], 7)
            self.assertEqual(result[0][0].dtype, torch.float32)
            self.assertEqual(
                tuple(initialized["samples"].unbind()[0].shape),
                geometry.shapes["video"],
            )

            old = Path(folder) / "legacy.safetensors"
            save_file(
                {
                    "text": text.float(),
                    "token_tags": tags,
                    "frame_0": guides[0]["latent"],
                    "video_init": video,
                    "audio_init": audio,
                },
                str(old),
            )
            old_result, _ = load_condition(old, geometry)
            self.assertEqual(
                old_result[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0
            )
            self.assertEqual(old_result[0][1]["minimax_refs"], [])


if __name__ == "__main__":
    unittest.main()
