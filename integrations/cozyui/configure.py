"""Install the H3 workflow tab into an eligible CozyUI checkout."""

from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    git = ["git", "-C", str(root)]

    overlay = {
        root / "src/components/tabs/t2v/H3TurboRuntime.tsx": HERE / "H3TurboRuntime.tsx",
        root / "public/api/h3_turbo_runtime.json": HERE.parent / "comfyui/text-to-video.json",
        root / "vite.h3.config.ts": HERE / "vite.h3.config.ts",
    }
    for target, source in overlay.items():
        if target.exists() and target.read_bytes() != source.read_bytes():
            if target.name == "H3TurboRuntime.tsx":
                if "H3TurboRuntimeTab" not in target.read_text():
                    parser.error(f"Existing file is not our H3 tab: {target}")
            elif target.name not in ("h3_turbo_runtime.json", "vite.h3.config.ts"):
                parser.error(f"Existing file differs: {target}")

    config_path = root / "public/conf/config.local.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    tabs = config.setdefault("tabs", {})
    entry = json.loads((HERE / "config.fragment.json").read_text())["tabs"]
    for name, value in entry.items():
        if name in tabs and tabs[name] != value:
            old = tabs[name]
            if not (name == "H3 Turbo Runtime" and isinstance(old, dict)
                    and old.get("api") == "api/h3_turbo_runtime.json"):
                parser.error(f"Existing CozyUI tab is not our H3 tab: {name}")
    config.setdefault("api", "/cui")
    tabs.update(entry)

    command = [*git, "apply"]
    file_upload = root / "src/components/controls/FileUpload.tsx"
    app = root / "src/App.tsx"
    generate = root / "src/components/controls/GenerateButton.tsx"
    upload_store = root / "src/utils/uploadStore.ts"
    required_patches = (
        ("h3-tab.patch", lambda: "H3TurboRuntimeTab" in app.read_text()
         and "serverManagedInputs" in generate.read_text()
         and "backgroundBackup" in upload_store.read_text()),
        ("h3-direct-media.patch", lambda: "directInput" in file_upload.read_text()),
        ("h3-number-fields.patch", lambda: "H3 Turbo Runtime' && typeof val" in generate.read_text()),
    )
    for name, installed in required_patches:
        if installed():
            continue
        patch = str(HERE / name)
        subprocess.run([*command, "--check", patch], check=True)
        subprocess.run([*command, patch], check=True)
    for target, source in overlay.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
    pending = config_path.with_suffix(".tmp")
    pending.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    pending.replace(config_path)
    print("H3 Turbo Runtime tab is configured.")


if __name__ == "__main__":
    main()
