"""CPU safety invariants for fast fused FFN recovery; no GPU needed."""

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("comfy"),
    "Requires the H3 ComfyUI/PyTorch runtime",
)
class FastAudioGuardTests(unittest.TestCase):
    def test_unchanged_finite_rows_and_selective_audio_repair(self):
        import torch
        import lora_adapter as routing
        import fast_audio_guard as guard

        class Diffusion:
            def _run_blocks(self, hidden, time_embed, mods, layout, opts):
                return routing.chunked_mlp(
                    None, hidden, 1024, SimpleNamespace(prefix="block.fc1"), None
                )

        diffusion = Diffusion()
        model = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion))
        layout = SimpleNamespace(
            segments=[(0, 2, "text"), (2, 4, "audio"), (4, 5, "video")]
        )
        hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        repairs = []

        def invalid_ffn(*args):
            output = args[1].clone()
            output[3, 1] = float("inf")
            return output

        def restore(module, data, first, second):
            repairs.append(data.clone())
            return data * 2

        stats = {}
        with (
            patch.object(routing, "chunked_mlp", invalid_ffn),
            patch.object(guard, "recompute_ffn", restore),
        ):
            with guard.guard_fast_audio_rows(model, stats):
                value = diffusion._run_blocks(hidden, None, None, layout, {})
        self.assertTrue(torch.equal(value[3], hidden[3] * 2))
        self.assertTrue(torch.equal(value[[0, 1, 2, 4]], hidden[[0, 1, 2, 4]]))
        self.assertEqual(len(repairs), 1)
        self.assertEqual(stats["ffn_recovered_audio"][0]["audio_rows"], [1])

        clean_stats = {}
        with patch.object(routing, "chunked_mlp", lambda m, h, r, a, b: h.clone()):
            with guard.guard_fast_audio_rows(model, clean_stats):
                clean = diffusion._run_blocks(hidden, None, None, layout, {})
        self.assertTrue(torch.equal(clean, hidden))
        self.assertEqual(clean_stats, {})


if __name__ == "__main__":
    unittest.main()
