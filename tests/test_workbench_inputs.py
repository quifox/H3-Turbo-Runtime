"""Import and validate the real ComfyUI node against lightweight interface stubs."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest import TestCase, mock

PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nodes/h3_turbo_runtime/__init__.py"
)


def load_node(input_folder):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(input_folder)
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = lambda: None
    utilities = types.ModuleType("comfy.utils")
    utilities.ProgressBar = lambda _: None
    with mock.patch.dict(
        sys.modules,
        {
            "folder_paths": folder_paths,
            "comfy": comfy,
            "comfy.model_management": management,
            "comfy.utils": utilities,
        },
    ):
        spec = importlib.util.spec_from_file_location("h3_test_custom_node", PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class WorkbenchInputTests(TestCase):
    def test_all_native_modes_and_mixed_media_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            for name in (
                "first.png",
                "last.png",
                "keyframe.png",
                "guide.mp4",
                "guide.wav",
                "reference.png",
                "reference.mp4",
                "reference.wav",
            ):
                (Path(root) / name).touch()
            module = load_node(Path(root))
            guides = json.dumps(
                [
                    {"at_seconds": 2, "image": "keyframe.png"},
                    {"frame_idx": 36, "video": "guide.mp4", "audio": "guide.wav"},
                    {"frame_idx": -1, "image": "last.png"},
                ]
            )
            refs = json.dumps(
                {
                    "images": ["reference.png"],
                    "videos": [{"path": "reference.mp4", "audio": True}],
                    "audios": ["reference.wav"],
                }
            )
            for mode in module.MODES:
                with self.subTest(mode=mode):
                    value = module.H3TurboRuntime.VALIDATE_INPUTS(
                        mode,
                        "",
                        5.0,
                        4,
                        4106,
                        "fast",
                        first_frame="first.png",
                        last_frame="last.png",
                        guides_json=guides,
                        references_json=refs,
                    )
                    self.assertTrue(value is True, value)
            self.assertTrue(
                module.H3TurboRuntime.VALIDATE_INPUTS(
                    "Auto", "", 4.0, 8, 4106, "stream", references_json="{}"
                )
            )
            self.assertTrue(
                module.H3TurboRuntime.VALIDATE_INPUTS(
                    "文生视频", "", 5.0, 4, 4106, "fast"
                )
            )
            self.assertNotEqual(
                module.H3TurboRuntime.VALIDATE_INPUTS(
                    "Auto",
                    "",
                    5.0,
                    4,
                    4106,
                    "fast",
                    guides_json='[{"frame_idx":3,"image":"../../secret.png"}]',
                ),
                True,
            )

    def test_auto_backbone_selection(self):
        with tempfile.TemporaryDirectory() as root:
            node = load_node(Path(root))
            self.assertEqual(node.choose_model_variant("auto", node.TEXT, {}), "fl2va")
            self.assertEqual(
                node.choose_model_variant("auto", node.TIMELINE, {}), "fl2va"
            )
            self.assertEqual(
                node.choose_model_variant("auto", node.REFERENCE, {}), "ref2va"
            )
            self.assertEqual(
                node.choose_model_variant("auto", node.TEXT, {"images": ["a.png"]}),
                "ref2va",
            )
            self.assertEqual(
                node.choose_model_variant(
                    "fl2va", node.REFERENCE, {"images": ["a.png"]}
                ),
                "fl2va",
            )

    def test_references_can_be_empty_and_guide_can_be_audio_only(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "guide.wav").touch()
            module = load_node(Path(root))
            self.assertEqual(module.resolve_references({})["videos"], [])
            resolved = module.resolve_guide({"frame_idx": 12, "audio": "guide.wav"})
            self.assertEqual(resolved["frame_idx"], 12)
            self.assertTrue(resolved["audio"].endswith("guide.wav"))
