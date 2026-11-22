"""Pick one H3 backbone/LoRA pair while reusing the original VAEs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("fl2va", "ref2va")


def selected_weights(manifest, variant="fl2va"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown H3 model variant: {variant}")
    if variant == "fl2va":
        return list(manifest["weights"])
    overrides = json.loads((ROOT / "models" / "ref2va.json").read_text())["weights"]
    by_id = {entry["id"]: entry for entry in overrides}
    return [by_id.get(entry["id"], entry) for entry in manifest["weights"]]


def model_filenames(manifest, variant="fl2va"):
    return {entry["id"]: entry["name"] for entry in selected_weights(manifest, variant)}
