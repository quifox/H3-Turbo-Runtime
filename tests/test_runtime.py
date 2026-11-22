"""CPU-only interface, geometry and packaging checks for the public runtime."""

import ast
import csv
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from geometry import from_seconds
from compat import choose_strategy, profile_from_capability


class DurationGeometryTests(unittest.TestCase):
    def test_verified_lengths_and_audio(self):
        for seconds, frames, native, video, audio in (
            (5, 120, 124, 37, 207),
            (15, 360, 362, 107, 603),
        ):
            with self.subTest(duration=seconds):
                g = from_seconds(seconds)
                self.assertEqual((g.export_frames, g.native_frames), (frames, native))
                self.assertEqual(g.shapes["video"], (1, 24, video, 30, 52))
                self.assertEqual(g.shapes["audio"], (1, 32, 2, audio))
                self.assertEqual(g.audio_samples, seconds * 32000)

    def test_output_geometry_parameters(self):
        g = from_seconds(5, width=1024, height=576, output_fps=30)
        self.assertEqual((g.width, g.height, g.fps), (1024, 576, 30))
        self.assertEqual((g.native_frames, g.export_frames), (124, 150))
        self.assertEqual(g.shapes["video"], (1, 24, 37, 36, 64))
        self.assertEqual(g.audio_samples, 5 * 32000)
        for kw in ({"width": 831}, {"height": 479}, {"output_fps": 0}, {"width": 0}):
            with self.assertRaises(ValueError):
                from_seconds(5, **kw)

    def test_other_lengths_without_hard_bounds(self):
        for sec in (4, 4.25, 6, 10, 12.5, 20):
            with self.subTest(duration=sec):
                g = from_seconds(sec)
                self.assertGreaterEqual(g.native_frames, g.export_frames)
                self.assertEqual((g.native_frames - 5) % 17, 0)
                self.assertEqual(g.export_frames, round(sec * 24))
                self.assertEqual(g.width, 832)
        for bad in (None, 0, -3, "nan", "inf", "oops"):
            with self.subTest(invalid=bad):
                with self.assertRaises(ValueError):
                    from_seconds(bad)


class DeviceRoutingTests(unittest.TestCase):
    def test_explicit_backend_selection_without_fallback(self):
        for sm, family in (
            ((7, 5), "turing"),
            ((8, 6), "ampere"),
            ((8, 9), "ada"),
            ((9, 0), "hopper"),
            ((12, 0), "blackwell_or_newer"),
        ):
            p = profile_from_capability(sm)
            self.assertEqual(p.family, family)
            self.assertEqual(choose_strategy(p, requested="fast"), "fast")
            self.assertEqual(choose_strategy(p, requested="stream"), "stream")
            with self.assertRaises(ValueError):
                choose_strategy(p, requested="auto")
        with self.assertRaises(RuntimeError):
            profile_from_capability((6, 1))


class PackageTests(unittest.TestCase):
    def test_reference_hash_is_path_specific(self):
        from render import qualifies_for_reference_hash

        short = from_seconds(5)
        long = from_seconds(15)
        self.assertTrue(qualifies_for_reference_hash("fast", 4, short))
        self.assertFalse(qualifies_for_reference_hash("fast", 4, short, "ref2va"))
        for strategy, steps, geometry in (
            ("stream", 4, short),
            ("fast", 8, short),
            ("stream", 4, long),
        ):
            self.assertFalse(qualifies_for_reference_hash(strategy, steps, geometry))

    def test_h3_audio_vae_fp32_execution_is_not_overridden_by_video_flag(self):
        from vae_precision import vae_flag_for_stage
        from unittest.mock import patch

        with patch.dict("os.environ", {"H3_AUDIO_VAE_FP32": "0"}):
            for stage in ("sample", "video", "audio"):
                self.assertEqual(vae_flag_for_stage(stage), "--fp16-vae")
        with patch.dict("os.environ", {"H3_AUDIO_VAE_FP32": "1"}):
            self.assertEqual(vae_flag_for_stage("audio"), "--fp32-vae")
            self.assertEqual(vae_flag_for_stage("video"), "--fp16-vae")

    def test_optional_ref2va_uses_same_shared_vaes(self):
        from model_assets import selected_weights, model_filenames

        manifest = json.loads((ROOT / "runtime/models/manifest.json").read_text())
        before = model_filenames(manifest, "fl2va")
        after = model_filenames(manifest, "ref2va")
        self.assertNotEqual(before["dit"], after["dit"])
        self.assertNotEqual(before["lora"], after["lora"])
        self.assertEqual(before["video_vae"], after["video_vae"])
        self.assertEqual(before["audio_vae"], after["audio_vae"])
        ref = selected_weights(manifest, "ref2va")
        self.assertEqual(
            {x["id"] for x in ref}, {"dit", "lora", "video_vae", "audio_vae"}
        )
        self.assertEqual(
            next(x for x in ref if x["id"] == "dit")["sha256"],
            "ddb5e0590ec4702b8024c123c3e8b1d64539ba85446860e86c45783be8c3a09f",
        )

    def test_source_import_syntax(self):
        for source in sorted((ROOT / "runtime").rglob("*.py")):
            with self.subTest(source=source.name):
                ast.parse(source.read_text(), filename=str(source))
        for source in sorted((ROOT / "custom_nodes").rglob("*.py")):
            with self.subTest(source=source.name):
                ast.parse(source.read_text(), filename=str(source))

    def test_release_projection_matches_default_encoder(self):
        manifest = json.loads((ROOT / "runtime/models/manifest.json").read_text())
        asset = next(
            x for x in manifest["conditioning_weights"] if x["id"] == "clip_projection"
        )
        self.assertEqual(asset["name"], "mmh3-4b-ClipProj-v3.1.safetensors")
        self.assertEqual(
            asset["sha256"],
            "0184e5c8d666a131962506d21949c2d8a8c6f33445b7b5e347e9a7e0a5baa819",
        )
        encoder = (
            ROOT / "custom_nodes/h3_turbo_runtime/prepare_condition.py"
        ).read_text()
        self.assertIn('default=os.getenv("H3_CLIPPROJ_VERSION", "v3.1")', encoder)
        self.assertIn(asset["name"], encoder)

    def test_module_names_do_not_refer_to_removed_experiment_imports(self):
        for source in list((ROOT / "runtime").rglob("*.py")):
            text = source.read_text()
            for old in (
                "h3_turbo_stream_adapter_next",
                "h3_step_profile",
                "render_stage_dynamic",
                "H3_4060_ARCHIVE",
                "H3_4060_EIGHT_STEP",
            ):
                self.assertNotIn(old, text, source.name)

    def test_all_user_docs_local_links_exist(self):
        for page in sorted(ROOT.rglob("*.md")):
            for link in re.findall(r"\]\(([^)]+)\)", page.read_text()):
                if (
                    link.startswith(("https://", "http://", "#"))
                    or link == "{{SOURCE_URL}}"
                ):
                    continue
                self.assertTrue(
                    (page.parent / link.split("#")[0]).exists(),
                    (str(page.relative_to(ROOT)), link),
                )

    def test_public_benchmark_scope_and_provenance(self):
        with (ROOT / "benchmarks/performance.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        self.assertEqual({int(x["seconds"]) for x in rows}, {5, 15})
        self.assertEqual(
            {x["gpu"] for x in rows}, {"Tesla T4 16GB", "RTX 4060 Laptop 8GB"}
        )
        self.assertTrue(any("Published LightX2V" in x["runtime"] for x in rows))
        self.assertTrue(all(x["peak_mib"] for x in rows))

    def test_comfyui_class_matches_workflow_json(self):
        modes = {
            "Auto",
            "Text to video",
            "First and last frame",
            "Timeline guides",
            "Reference to video",
        }
        paths = list((ROOT / "integrations/comfyui").glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            if path.name == "complete-5s-workflow.json":
                workflow = json.loads(path.read_text())
                self.assertTrue(workflow["nodes"])
                self.assertTrue(workflow["links"])
                continue
            if path.name == "complete-5s-api.json":
                workflow = json.loads(path.read_text())
                self.assertEqual(workflow["10"]["class_type"], "H3TurboRuntime")
                self.assertEqual(workflow["1"]["class_type"], "H3TurboSettings")
                self.assertEqual(workflow["1"]["inputs"]["duration_s"], 5.0)
                self.assertEqual(workflow["1"]["inputs"]["strategy"], "fast")
                continue
            workflow = json.loads(path.read_text())
            self.assertTrue(workflow)
            for node in workflow.values():
                self.assertEqual(node["class_type"], "H3TurboRuntime")
                self.assertEqual(node["inputs"]["duration_s"], 5.0)
                self.assertIn(node["inputs"]["mode"], modes)
                self.assertIsInstance(json.loads(node["inputs"]["guides_json"]), list)
                self.assertIsInstance(
                    json.loads(node["inputs"]["references_json"]), dict
                )


if __name__ == "__main__":
    unittest.main()
