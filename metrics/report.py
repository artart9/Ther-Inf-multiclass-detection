"""Write a compact experiment report (markdown + JSON) under reports/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _rel(path: Path, start: Path | None = None) -> str:
    start = start or Path.cwd()
    try:
        return str(Path(path).resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path)


def build_report(
    metrics_dir: str | Path,
    reports_dir: str | Path,
    exp_name: str,
    notes: str = "",
) -> Path:
    """
    Build reports/<exp_name>/report.md and report.json from
    runs/<exp>/metrics/{epochs.jsonl,summary.json}.
    """
    from metrics.meta_store import run_report_dir

    metrics_dir = Path(metrics_dir)
    out_dir = run_report_dir(reports_dir, exp_name)

    summary_path = metrics_dir / "summary.json"
    epochs_path = metrics_dir / "epochs.jsonl"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    epochs = []
    if epochs_path.exists():
        for line in epochs_path.read_text().splitlines():
            if line.strip():
                epochs.append(json.loads(line))

    acc = summary.get("accuracy") or {}
    lat = summary.get("latency") or {}
    model = summary.get("model") or {}
    eff = summary.get("efficiency") or {}
    test = acc.get("test") or {}
    train = acc.get("train") or {}

    weights = summary.get("weights", "—")
    if weights not in (None, "—"):
        weights = _rel(Path(weights))

    lines = [
        f"# Experiment report: `{exp_name}`",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Weights: `{weights}`",
        f"- imgsz: `{summary.get('imgsz', '—')}`",
        "",
        "## Notes",
        notes.strip() or "_Add setup notes (hardware, hypothesis, changes) here._",
        "",
        "## Final accuracy",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP@0.5 | {_fmt(acc.get('map50'))} |",
        f"| mAP@0.5:0.95 | {_fmt(acc.get('map50_95'))} |",
        f"| Test P / R / F1 | {_fmt(test.get('precision'))} / {_fmt(test.get('recall'))} / {_fmt(test.get('f1'))} |",
        f"| Train P / R / F1 | {_fmt(train.get('precision'))} / {_fmt(train.get('recall'))} / {_fmt(train.get('f1'))} |",
        f"| Small-object score | {_fmt(acc.get('small_object_score'))} |",
        "",
        "## Latency (batch=1)",
        "",
        "| Stage | mean ms |",
        "|---|---|",
        f"| preprocess | {_fmt((lat.get('preprocess') or {}).get('mean_ms'), 2)} |",
        f"| infer | {_fmt((lat.get('infer') or {}).get('mean_ms'), 2)} |",
        f"| post | {_fmt((lat.get('post') or {}).get('mean_ms'), 2)} |",
        f"| e2e | {_fmt((lat.get('e2e') or {}).get('mean_ms'), 2)} |",
        f"| FPS | {_fmt(lat.get('fps'), 2)} |",
        "",
        "## Model cost",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Params | {_fmt(model.get('params'), 0)} |",
        f"| GFLOPs | {_fmt(model.get('gflops'), 3)} |",
        f"| Weight MB | {_fmt(model.get('weight_mb'), 2)} |",
        f"| Peak RSS MB | {_fmt(model.get('peak_rss_mb'), 1)} |",
        f"| Peak VRAM MB | {_fmt(model.get('peak_vram_mb'), 1)} |",
        "",
        "## Efficiency",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP / MB | {_fmt(eff.get('map_per_mb'))} |",
        f"| mAP / GFLOP | {_fmt(eff.get('map_per_gflop'))} |",
        f"| mAP × FPS | {_fmt(eff.get('map_times_fps'))} |",
        "",
        "## Latency–accuracy curve",
        "",
        "| imgsz | mAP50 | mAP50-95 | e2e ms | FPS |",
        "|---|---|---|---|---|",
    ]

    for row in summary.get("latency_accuracy_curve") or []:
        lines.append(
            f"| {row.get('imgsz')} | {_fmt(row.get('map50'))} | {_fmt(row.get('map50_95'))} "
            f"| {_fmt(row.get('e2e_mean_ms'), 2)} | {_fmt(row.get('fps'), 2)} |"
        )

    lines += [
        "",
        "## Epoch log",
        "",
        "| epoch | mAP50-95 | P | R | F1 | grad% bb/neck/head |",
        "|---|---|---|---|---|---|",
    ]
    for r in epochs:
        gp = r.get("grad_pct") or {}
        g = "/".join(_fmt(gp.get(k), 1) for k in ("backbone", "neck", "head"))
        lines.append(
            f"| {r.get('epoch')} | {_fmt(r.get('map50_95'))} | {_fmt(r.get('precision'))} "
            f"| {_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} | {g} |"
        )

    lines += [
        "",
        "## Artifacts in this folder",
        "",
        f"- `report.md` / `report.json`",
        f"- `plot_data.json`, `epochs.jsonl`, `latency.json`",
        f"- Compare index: `{_rel(Path(reports_dir) / 'compare' / 'latency_accuracy.jsonl')}`",
        "",
        "## Local artifacts (gitignored)",
        "",
        f"- Metrics: `{_rel(metrics_dir)}`",
        f"- Plots: `{_rel(metrics_dir / 'plots')}`",
        f"- Summary JSON: `{_rel(summary_path)}`",
        "",
    ]

    out = out_dir / "report.md"
    out.write_text("\n".join(lines))

    full = {
        "exp_name": exp_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": summary.get("device"),
        "imgsz": summary.get("imgsz"),
        "weights": summary.get("weights"),
        "summary": {
            "accuracy": acc,
            "latency": lat,
            "model": model,
            "efficiency": eff,
            "latency_accuracy_curve": summary.get("latency_accuracy_curve"),
        },
        "epochs": epochs,
    }
    (out_dir / "report.json").write_text(json.dumps(full, indent=2) + "\n")
    return out
