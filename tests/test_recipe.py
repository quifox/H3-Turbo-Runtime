"""Validate sample/adapter recipes without initializing GPU frameworks."""

import sys
from pathlib import Path
import tempfile
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from recipe import load_recipe, validate_steps


class RecipeTests(unittest.TestCase):
    def test_default_turbo_preserves_frozen_settings(self):
        for variant in ("fl2va", "ref2va"):
            recipe = load_recipe(variant)
            self.assertEqual(
                (recipe["sampler"], recipe["scheduler"]), ("euler", "simple")
            )
            self.assertEqual(
                (recipe["video_shift"], recipe["audio_shift"]), (12.0, 3.0)
            )
            for steps in (4, 8):
                validate_steps(recipe, steps)
            with self.assertRaises(ValueError):
                validate_steps(recipe, 20)

    def test_external_recipe_requires_consistent_selection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "recipes.json"
            path.write_text(
                json.dumps(
                    {
                        "fl2va": {
                            "name": "local",
                            "supported_steps": [20],
                            "sampler": "res_multistep",
                            "scheduler": "simple",
                            "video_shift": 12,
                            "audio_shift": 3,
                        }
                    }
                )
            )
            recipe = load_recipe("fl2va", path)
            validate_steps(recipe, 20)
            with self.assertRaises(ValueError):
                validate_steps(recipe, 8)


if __name__ == "__main__":
    unittest.main()
