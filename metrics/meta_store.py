"""Persist per-run report artifacts + a shared compare index."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def run_report_dir(reports_dir: str | Path, exp_name: str) -> Path:
    """reports/<exp_name>/ — one folder per experiment."""
    path = Path(reports_dir) / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_dir(reports_dir: str | Path) -> Path:
    """reports/compare/ — cross-run indexes for meta-plots."""
    path = Path(reports_dir) / "compare"
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").touch(exist_ok=True)
    return path


def _percentiles(samples: list) -> dict:
    if not samples:
        return {"p50_ms": None, "p95_ms": None}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _stage_block(stage: dict | None) -> dict:
    stage = stage or {}
    samples = list(stage.get("samples_ms") or [])
    out = {
        "mean_ms": stage.get("mean_ms"),
        "std_ms": stage.get("std_ms"),
        "p50_ms": stage.get("p50_ms"),
        "p95_ms": stage.get("p95_ms"),
        "n": len(samples),
        "samples_ms": samples,
    }
    if out["p50_ms"] is None:
        out.update(_percentiles(samples))
    return out


def _load_epochs(metrics_dir: Path) -> list:
    path = metrics_dir / "epochs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _upsert_jsonl(path: Path, exp_name: str, row: dict) -> None:
    by_name: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            prev = json.loads(line)
            by_name[prev["exp_name"]] = prev
    by_name[exp_name] = row
    path.write_text("".join(json.dumps(by_name[k]) + "\n" for k in sorted(by_name)))


def build_plot_data(summary: dict, epochs: list, exp_name: str) -> dict:
    lat = summary.get("latency") or {}
    acc = summary.get("accuracy") or {}
    stages = {
        "preprocess": _stage_block(lat.get("preprocess")),
        "infer": _stage_block(lat.get("infer")),
        "post": _stage_block(lat.get("post")),
        "e2e": _stage_block(lat.get("e2e")),
    }
    return {
        "exp_name": exp_name,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "device": summary.get("device") or device_label(),
        "platform": platform.platform(),
        "imgsz": summary.get("imgsz"),
        "weights": summary.get("weights"),
        "quantize": summary.get("quantize") or lat.get("quantize"),
        "backend": summary.get("backend") or lat.get("backend"),
        "plots": {
            "epochs": epochs,
            "latency_e2e_samples_ms": stages["e2e"]["samples_ms"],
            "latency_stages": stages,
            "latency_accuracy_curve": summary.get("latency_accuracy_curve") or [],
        },
        "accuracy": acc,
        "fps": lat.get("fps"),
        "latency_method": lat.get("method"),
        "latency_backend": lat.get("backend") or summary.get("backend"),
        "engine_path": lat.get("engine_path") or summary.get("engine_path"),
        "efficiency": summary.get("efficiency") or {},
        "model": summary.get("model") or {},
    }


def save_run_plot_data(
    summary: dict,
    exp_name: str,
    metrics_dir: str | Path,
    reports_dir: str | Path,
    epochs: list | None = None,
) -> dict:
    """
    Save plot-backing data next to the run report and refresh the compare index.

    Layout:
      reports/<exp>/plot_data.json
      reports/<exp>/epochs.jsonl
      reports/<exp>/latency.json
      reports/compare/latency_accuracy.jsonl
      runs/.../metrics/plot_data.json  (local mirror)
      runs/.../metrics/latency_distribution.json
    """
    metrics_dir = Path(metrics_dir)
    reports_dir = Path(reports_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = run_report_dir(reports_dir, exp_name)
    cmp = compare_dir(reports_dir)

    if epochs is None:
        epochs = _load_epochs(metrics_dir)

    plot_data = build_plot_data(summary, epochs, exp_name)
    stages = plot_data["plots"]["latency_stages"]
    curve = plot_data["plots"]["latency_accuracy_curve"]

    plot_payload = json.dumps(plot_data, indent=2) + "\n"
    local_plot = metrics_dir / "plot_data.json"
    report_plot = out / "plot_data.json"
    local_plot.write_text(plot_payload)
    report_plot.write_text(plot_payload)

    (out / "epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in epochs))

    latency_record = {
        "exp_name": exp_name,
        "saved_utc": plot_data["saved_utc"],
        "device": plot_data["device"],
        "platform": plot_data["platform"],
        "imgsz": plot_data["imgsz"],
        "weights": plot_data["weights"],
        "quantize": plot_data.get("quantize"),
        "method": plot_data.get("latency_method"),
        "backend": plot_data.get("latency_backend"),
        "engine_path": plot_data.get("engine_path"),
        "accuracy": {
            "map50": (plot_data.get("accuracy") or {}).get("map50"),
            "map50_95": (plot_data.get("accuracy") or {}).get("map50_95"),
            "small_object_score": (plot_data.get("accuracy") or {}).get("small_object_score"),
            "small_object_scores": (plot_data.get("accuracy") or {}).get("small_object_scores"),
        },
        "fps": plot_data.get("fps"),
        "stages": stages,
        "latency_accuracy_curve": curve,
        "model": plot_data.get("model") or {},
    }
    latency_payload = json.dumps(latency_record, indent=2) + "\n"
    (metrics_dir / "latency_distribution.json").write_text(latency_payload)
    report_lat = out / "latency.json"
    report_lat.write_text(latency_payload)

    e2e = stages.get("e2e") or {}
    index_row = {
        "exp_name": exp_name,
        "saved_utc": plot_data["saved_utc"],
        "device": plot_data["device"],
        "imgsz": plot_data["imgsz"],
        "method": plot_data.get("latency_method"),
        "backend": plot_data.get("latency_backend"),
        "quantize": plot_data.get("quantize"),
        "map50": (plot_data.get("accuracy") or {}).get("map50"),
        "map50_95": (plot_data.get("accuracy") or {}).get("map50_95"),
        "e2e_mean_ms": e2e.get("mean_ms"),
        "e2e_std_ms": e2e.get("std_ms"),
        "e2e_p50_ms": e2e.get("p50_ms"),
        "e2e_p95_ms": e2e.get("p95_ms"),
        "fps": plot_data.get("fps"),
        "e2e_samples_ms": e2e.get("samples_ms") or [],
        "latency_accuracy_curve": curve,
        "report_dir": exp_name,
        "plot_data_file": f"{exp_name}/plot_data.json",
        "latency_file": f"{exp_name}/latency.json",
        "epochs_file": f"{exp_name}/epochs.jsonl",
    }
    index_path = cmp / "latency_accuracy.jsonl"
    _upsert_jsonl(index_path, exp_name, index_row)

    return {
        "run_dir": str(out),
        "plot_data": str(report_plot),
        "latency": str(report_lat),
        "epochs": str(out / "epochs.jsonl"),
        "index": str(index_path),
        "local_plot_data": str(local_plot),
        "record": plot_data,
    }


def save_latency_distribution(
    summary: dict,
    exp_name: str,
    metrics_dir: str | Path,
    reports_dir: str | Path,
) -> dict:
    return save_run_plot_data(summary, exp_name, metrics_dir, reports_dir)
