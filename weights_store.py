"""Save trained checkpoints into weights/ and update the local manifest."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"updated_utc": None, "latest": None, "entries": []}


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def save_demo_weights(
    src_weights: str | Path,
    weights_dir: str | Path,
    exp_name: str,
    tag: str = "best",
    also_as_latest: bool = True,
) -> dict:
    """
    Copy a checkpoint into weights/ and append metadata to manifest.json.

    Checkpoints (*.pt) are gitignored. Commit weights/.gitkeep and
    weights/manifest.json so the folder and index stay in the repo.
    """
    src_weights = Path(src_weights).resolve()
    if not src_weights.exists():
        raise FileNotFoundError(f"No weights at {src_weights}")

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / ".gitkeep").touch(exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_name = f"{exp_name}_{tag}_{stamp}.pt"
    dest = weights_dir / dest_name
    shutil.copy2(src_weights, dest)

    if also_as_latest:
        shutil.copy2(src_weights, weights_dir / "latest.pt")

    manifest_path = weights_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    entry = {
        "exp_name": exp_name,
        "tag": tag,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "file": dest_name,
        "source": _relpath(src_weights),
        "bytes": dest.stat().st_size,
    }
    manifest["entries"].append(entry)
    manifest["latest"] = {
        "file": "latest.pt" if also_as_latest else dest_name,
        "points_to": dest_name,
        "exp_name": exp_name,
        "saved_utc": entry["saved_utc"],
    }
    manifest["updated_utc"] = entry["saved_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry
