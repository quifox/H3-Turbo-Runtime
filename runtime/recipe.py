"""Explicit inference recipes. Nothing here silently changes the selected backend."""
import json
from math import isfinite
from pathlib import Path

DEFAULT_RECIPES = Path(__file__).resolve().parent / "models" / "recipes.json"


def load_recipe(variant, path=None):
    """Read the selected variant recipe; loading alone does not certify new settings."""
    source = Path(path) if path is not None else DEFAULT_RECIPES
    recipes = json.loads(source.read_text())
    recipe = recipes[variant]
    for key in ("sampler", "scheduler"):
        if not isinstance(recipe[key], str) or not recipe[key]:
            raise ValueError(f"Invalid recipe {variant}: {key}")
    steps = recipe["supported_steps"]
    if not isinstance(steps, list) or not steps or any(
        not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in steps
    ):
        raise ValueError(f"Invalid supported_steps in recipe {variant}")
    for key in ("video_shift", "audio_shift"):
        if not isinstance(recipe[key], (float, int)) or not isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f"Invalid {key} in recipe {variant}")
    return recipe


def validate_steps(recipe, steps):
    """Reject inference step counts not explicitly supported by the selected recipe."""
    if steps not in recipe["supported_steps"]:
        raise ValueError(f"Recipe {recipe.get('name')} supports steps {recipe['supported_steps']}, requested {steps}")
