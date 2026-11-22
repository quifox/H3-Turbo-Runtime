"""CPU-only ComfyUI contract tests. No model loads, GPU work, or user media."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / "custom_nodes/h3_turbo_runtime/__init__.py"
API = ROOT / "integrations/comfyui/complete-5s-api.json"
GRAPH = ROOT / "integrations/comfyui/complete-5s-workflow.json"


class ProgressBar:
    def __init__(self, steps):
        self.steps = steps
        self.count = 0
    def update(self, n):
        self.count += n


def import_node(input_dir: Path, output_dir: Path):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(input_dir)
    fp.get_output_directory = lambda: str(output_dir)
    fp.__file__ = str(input_dir / "folder_paths.py")
    cm = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    cu = types.ModuleType("comfy.utils")
    cu.ProgressBar = ProgressBar
    sys.modules.update({"folder_paths": fp, "comfy": cm,
                        "comfy.model_management": mm, "comfy.utils": cu})
    spec = importlib.util.spec_from_file_location("h3_interface_under_test", NODE_FILE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ComfyWorkflowContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.inputs = self.base / "input"
        self.outputs = self.base / "output"
        self.inputs.mkdir(); self.outputs.mkdir()
        self.mod = import_node(self.inputs, self.outputs)
        self.fake_model_dir = self.base / "models"
        self.fake_model_dir.mkdir()
        self.old_model_dir = os.environ.get("H3_RUNTIME_MODELS")
        os.environ["H3_RUNTIME_MODELS"] = str(self.fake_model_dir)
        self.addCleanup(self.restore_env)
        self.calls = []

    def restore_env(self):
        if self.old_model_dir is None:
            os.environ.pop("H3_RUNTIME_MODELS", None)
        else:
            os.environ["H3_RUNTIME_MODELS"] = self.old_model_dir

    def fake_run(self, command, cwd, path, env):
        self.calls.append(command)
        path.write_text("dry run\n")
        task = path.parent
        if path.name == "condition.log":
            (task / "condition.safetensors").write_bytes(b"fake condition")
        else:
            render = task / "render"
            render.mkdir(exist_ok=True)
            (render / "output.mp4").write_bytes(b"stub mp4")

    def get_request(self):
        files = list((self.outputs / "h3-turbo").glob("*/request.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def test_cozyui_every_control_has_an_api_field(self):
        cfg = json.loads((ROOT / "integrations/cozyui/config.fragment.json").read_text())
        tab = cfg["tabs"]["H3 Turbo Runtime"]
        workflow = json.loads((ROOT / "integrations/comfyui/text-to-video.json").read_text())
        self.assertEqual(list(workflow), ["1"])
        inputs = workflow["1"]["inputs"]
        self.assertEqual(workflow["1"]["class_type"], "H3TurboRuntime")
        for control, link in tab["controls"].items():
            self.assertEqual(link.get("node_id", link.get("id")), "1" if link["id"] != "handle" else "1")
            self.assertIn(link["field"], inputs, f"Unmapped CozyUI control: {control}")
        self.assertEqual(inputs["duration_s"], 5.0)
        self.assertEqual(inputs["strategy"], "fast")
        self.assertEqual(inputs["variant"], "auto")

    def test_default_workflow_is_5s_unconditioned_fl2va(self):
        graph = json.loads(API.read_text())
        ui = json.loads(GRAPH.read_text())
        self.assertEqual(len(ui["nodes"]), 10)
        self.assertEqual(len(ui["links"]), 9)
        ids = {node["id"] for node in ui["nodes"]}
        for _id, source, _slot, dest, _input, _type in ui["links"]:
            self.assertIn(source, ids)
            self.assertIn(dest, ids)
        settings = self.mod.H3TurboSettings().build(**graph["1"]["inputs"])[0]
        media = None
        for i in range(2, 10):
            value = dict(graph[str(i)]["inputs"])
            value.pop("previous", None)
            media = self.mod.H3TurboMediaCondition().build(previous=media, **value)[0]
        self.assertEqual(media["guides"], [])
        self.assertEqual(media["references"]["images"], [])
        kwargs = dict(graph["10"]["inputs"])
        kwargs.pop("settings")
        kwargs.pop("media")
        kwargs["settings"] = settings
        kwargs["media"] = media
        self.assertEqual(
            self.mod.H3TurboRuntime.VALIDATE_INPUTS(**kwargs), True)
        with patch.object(self.mod, "run_logged", side_effect=self.fake_run):
            result = self.mod.H3TurboRuntime().render(**kwargs)
        request = self.get_request()
        self.assertEqual((request["duration_s"], request["width"], request["height"], request["output_fps"]),
                         (5.0, 832, 480, 24))
        self.assertEqual((request["guides"], request["references"]["images"],
                          request.get("first_frame")), ([], [], None))
        self.assertIn("--variant", self.calls[-1])
        self.assertEqual(self.calls[-1][self.calls[-1].index("--variant")+1], "fl2va")
        self.assertEqual(self.calls[-1][self.calls[-1].index("--steps")+1], "4")
        self.assertEqual(self.calls[-1][self.calls[-1].index("--strategy")+1], "fast")
        self.assertEqual(len(result["ui"]["images"]), 1)
        self.assertEqual(len(list((self.outputs/"h3-turbo").glob("*.mp4"))), 1)

    def test_reference_and_media_chain_routes_ref2va(self):
        for name in ("image.png","audio.wav","video.mp4","timed.png"):
            (self.inputs / name).write_bytes(b"dummy")
        chain = self.mod.H3TurboMediaCondition()
        m = chain.build(enabled=True,kind="reference_image",filename="image.png",
                        ref_image_size="max")[0]
        m = chain.build(enabled=True,kind="reference_audio",filename="audio.wav",previous=m)[0]
        m = chain.build(enabled=True,kind="reference_video",filename="video.mp4",
                        reference_video_audio=True,previous=m)[0]
        m = chain.build(enabled=True,kind="timed_image",filename="timed.png",
                        at_seconds=7.25,previous=m)[0]
        kwargs={"mode":"Reference to video","prompt":"Reference character laugh.",
                "duration_s":15,"steps":8,"seed":6,"strategy":"stream",
                "media":m,"variant":"auto"}
        self.assertEqual(self.mod.H3TurboRuntime.VALIDATE_INPUTS(**kwargs),True)
        with patch.object(self.mod, "run_logged", side_effect=self.fake_run):
            self.mod.H3TurboRuntime().render(**kwargs)
        request=self.get_request()
        self.assertEqual(request["references"]["image_size"],"max")
        self.assertEqual(request["references"]["images"],[str(self.inputs/"image.png")])
        self.assertEqual(request["references"]["audios"],[str(self.inputs/"audio.wav")])
        self.assertEqual(request["references"]["videos"][0]["audio"],True)
        self.assertEqual(request["guides"][0]["at_seconds"],7.25)
        self.assertEqual(self.calls[-1][self.calls[-1].index("--variant")+1],"ref2va")

    def test_free_steps_duration_resolution_and_official_ratios(self):
        kwargs={"mode":"Text to video","prompt":"Any task","duration_s":22.5,
                "steps":17,"seed":18,"strategy":"fast","width":1792,"height":768,
                "variant":"auto","canvas_mode":"explicit"}
        self.assertEqual(self.mod.H3TurboRuntime.VALIDATE_INPUTS(**kwargs),True)
        with patch.object(self.mod,"run_logged",side_effect=self.fake_run):
            self.mod.H3TurboRuntime().render(**kwargs)
        request=self.get_request()
        self.assertEqual((request["duration_s"],request["width"],request["height"]),(22.5,1792,768))
        self.assertEqual(self.calls[-1][self.calls[-1].index("--steps")+1],"17")
        (self.inputs/"first.png").write_bytes(b"dummy")
        self.assertEqual(self.mod.resolve_canvas(832,480,"short_edge_ratio",768,"9:16"),(768,1360))
        self.assertEqual(self.mod.resolve_canvas(832,480,"short_edge_ratio",768,"21:9"),(1792,768))

    def test_explicit_native_frame_count_and_legacy_references(self):
        (self.inputs/"ref.png").write_bytes(b"dummy")
        # A non-preset number of frames is passed to native geometry unchanged
        # as seconds; no UI-made duration cap.
        kwargs={"mode":"First frame to video","prompt":"Move","duration_s":5,
                "steps":4,"seed":5,"strategy":"fast","length_frames":617,
                "reference_image":"ref.png","ref_image_size":"max"}
        with patch.object(self.mod,"run_logged",side_effect=self.fake_run):
            self.mod.H3TurboRuntime().render(**kwargs)
        self.assertAlmostEqual(self.get_request()["duration_s"],617/24)
        self.assertEqual(self.get_request()["references"]["image_size"],"max")
        self.assertEqual(self.calls[-1][self.calls[-1].index("--variant")+1],"ref2va")

if __name__=="__main__":
    unittest.main()
