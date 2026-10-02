"""Plot epoch curves and final latency/accuracy figures (seaborn PNG + plotly HTML)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from metrics.viz_style import (
    PLOTLY_LAYOUT,
    apply_theme,
    finish_ax,
    label_points,
    new_fig,
    palette,
    save_fig,
)


def load_epochs(metrics_dir: Path) -> list:
    path = Path(metrics_dir) / "epochs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_plotly(fig: go.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def plot_epochs(metrics_dir: Path) -> None:
    rows = load_epochs(metrics_dir)
    if not rows:
        print("No epochs.jsonl to plot")
        return

    out = Path(metrics_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    colors = palette()

    # --- Detection metrics ---
    series = {
        "AP@0.5": [r.get("ap", r.get("map50")) for r in rows],
        "mAP50-95": [r.get("map50_95") for r in rows],
        "P": [r.get("precision") for r in rows],
        "R": [r.get("recall") for r in rows],
        "F1": [r.get("f1") for r in rows],
    }
    fig, ax = new_fig()
    for i, (name, ys) in enumerate(series.items()):
        sns.lineplot(x=epochs, y=ys, ax=ax, marker="o", label=name, color=colors[i % len(colors)])
    finish_ax(ax, title="Epoch detection metrics", xlabel="epoch", ylabel="score")
    save_fig(fig, out / "epoch_detection.png")

    fig_p = go.Figure()
    for name, ys in series.items():
        fig_p.add_trace(go.Scatter(x=epochs, y=ys, mode="lines+markers", name=name))
    fig_p.update_layout(title="Epoch detection metrics", xaxis_title="epoch", yaxis_title="score")
    _write_plotly(fig_p, out / "epoch_detection.html")

    # --- Losses ---
    loss_keys = sorted({k for r in rows for k in (r.get("loss") or {})})
    if loss_keys:
        fig, ax = new_fig()
        for i, k in enumerate(loss_keys):
            ys = [((r.get("loss") or {}).get(k)) for r in rows]
            sns.lineplot(x=epochs, y=ys, ax=ax, marker="o", label=k, color=colors[i % len(colors)])
        finish_ax(ax, title="Epoch losses", xlabel="epoch", ylabel="loss")
        save_fig(fig, out / "epoch_losses.png")

        fig_p = go.Figure()
        for k in loss_keys:
            ys = [((r.get("loss") or {}).get(k)) for r in rows]
            fig_p.add_trace(go.Scatter(x=epochs, y=ys, mode="lines+markers", name=k))
        fig_p.update_layout(title="Epoch losses", xaxis_title="epoch", yaxis_title="loss")
        _write_plotly(fig_p, out / "epoch_losses.html")

    # --- Grad % from epoch 2 ---
    grad_rows = [r for r in rows if int(r.get("epoch", 0)) >= 2]
    grad_rows = [
        r for r in grad_rows
        if any((r.get("grad_pct") or {}).get(g) is not None for g in ("backbone", "neck", "head"))
    ]
    if grad_rows:
        xs = [r["epoch"] for r in grad_rows]
        groups = ("backbone", "neck", "head")
        fig, ax = new_fig()
        for i, g in enumerate(groups):
            ys = [(r.get("grad_pct") or {}).get(g) for r in grad_rows]
            sns.lineplot(x=xs, y=ys, ax=ax, marker="o", label=g, color=colors[i % len(colors)])
        finish_ax(
            ax,
            title="Gradient norm change by block (from epoch 2)",
            xlabel="epoch",
            ylabel="% change vs previous epoch",
        )
        save_fig(fig, out / "epoch_grad_pct.png")

        fig_p = go.Figure()
        for g in groups:
            ys = [(r.get("grad_pct") or {}).get(g) for r in grad_rows]
            fig_p.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=g))
        fig_p.update_layout(
            title="Gradient norm change by block (from epoch 2)",
            xaxis_title="epoch",
            yaxis_title="% change vs previous epoch",
        )
        _write_plotly(fig_p, out / "epoch_grad_pct.html")


def plot_summary(metrics_dir: Path) -> None:
    summary_path = Path(metrics_dir) / "summary.json"
    if not summary_path.exists():
        print("No summary.json to plot")
        return

    summary = json.loads(summary_path.read_text())
    out = Path(metrics_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    colors = palette()

    # --- Latency distribution ---
    samples = (((summary.get("latency") or {}).get("e2e") or {}).get("samples_ms")) or []
    if samples:
        arr = np.asarray(samples, dtype=np.float64)
        fig, ax = new_fig()
        sns.histplot(arr, bins=30, stat="density", ax=ax, color=colors[0], alpha=0.55, edgecolor="white", label="histogram")
        try:
            sns.kdeplot(arr, ax=ax, color=colors[1], linewidth=2.2, label="KDE")
        except Exception:
            pass
        finish_ax(ax, title="Latency distribution", xlabel="e2e latency (ms)", ylabel="density")
        save_fig(fig, out / "latency_distribution.png")

        fig_p = go.Figure()
        fig_p.add_trace(go.Histogram(x=arr, histnorm="probability density", name="histogram", opacity=0.65, nbinsx=30))
        fig_p.update_layout(
            title="Latency distribution",
            xaxis_title="e2e latency (ms)",
            yaxis_title="density",
            barmode="overlay",
        )
        _write_plotly(fig_p, out / "latency_distribution.html")

    # --- Latency–accuracy curve (annotate imgsz) ---
    curve = summary.get("latency_accuracy_curve") or []
    if curve:
        # Sort by imgsz for a sensible path
        curve = sorted(curve, key=lambda c: int(c.get("imgsz") or 0))
        xs = [c["e2e_mean_ms"] for c in curve]
        ys = [c["map50"] for c in curve]
        labels = [c["imgsz"] for c in curve]

        fig, ax = new_fig()
        sns.lineplot(x=xs, y=ys, ax=ax, color=colors[0], alpha=0.35, estimator=None, sort=False)
        sns.scatterplot(x=xs, y=ys, ax=ax, s=120, color=colors[0], zorder=3, edgecolor="white", linewidth=1.2)
        label_points(ax, xs, ys, labels)
        finish_ax(
            ax,
            title="Latency–accuracy curve (imgsz sweep)",
            xlabel="e2e latency (ms)",
            ylabel="mAP50",
        )
        save_fig(fig, out / "latency_accuracy_curve.png")

        fig_p = go.Figure(
            go.Scatter(
                x=xs, y=ys, mode="markers+text+lines",
                text=[str(s) for s in labels], textposition="top right",
                marker=dict(size=12),
                hovertemplate="imgsz=%{text}<br>e2e=%{x:.3f} ms<br>mAP50=%{y:.4f}<extra></extra>",
            )
        )
        fig_p.update_layout(
            title="Latency–accuracy curve (imgsz sweep)",
            xaxis_title="e2e latency (ms)",
            yaxis_title="mAP50",
            showlegend=False,
        )
        _write_plotly(fig_p, out / "latency_accuracy_curve.html")


def plot_all(metrics_dir: str | Path) -> None:
    apply_theme()
    metrics_dir = Path(metrics_dir)
    plot_epochs(metrics_dir)
    plot_summary(metrics_dir)
    print(f"Plots written to {metrics_dir / 'plots'}")
