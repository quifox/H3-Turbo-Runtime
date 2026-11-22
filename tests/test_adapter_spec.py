"""Pure CPU compatibility checks for the external LoRA adapter contract."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from adapter_spec import (
    targets_from_keys,
    required_ffn_targets,
    required_stream_attention_targets,
    require_targets,
)


class AdapterSpecTests(unittest.TestCase):
    def test_complete_targets_not_fixed_to_turbo208(self):
        names = ["diffusion_model.blocks.0.mlp.fc1", "diffusion_model.blocks.0.mlp.fc2"]
        keys = {
            name + suffix
            for name in names
            for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")
        }
        self.assertEqual(targets_from_keys(keys), set(names))
        require_targets(targets_from_keys(keys), required_ffn_targets(1), "fast")

    def test_incomplete_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing LoRA"):
            targets_from_keys({"diffusion_model.blocks.0.mlp.fc1.alpha"})
        with self.assertRaisesRegex(ValueError, "additional LoRA targets"):
            require_targets(set(), required_ffn_targets(1), "fast")
        self.assertEqual(len(required_stream_attention_targets(50)), 100)


if __name__ == "__main__":
    unittest.main()
