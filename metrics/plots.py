"""Plot epoch curves and final latency/accuracy figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_epochs(metrics_dir: Path) -> list:
    path = Path(metrics_dir) / "epochs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plot_epochs(metrics_dir: Path) -> None:
    rows = load_epochs(metrics_dir)
    if not rows:
        print("No epochs.jsonl to plot")
        return

    out = Path(metrics_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [r.get("map50_95") for r in rows], label="mAP50-95")
    ax.plot(epochs, [r.get("precision") for r in rows], label="P")
    ax.plot(epochs, [r.get("recall") for r in rows], label="R")
    ax.plot(epochs, [r.get("f1") for r in rows], label="F1")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.set_title("Epoch detection metrics")
    fig.tight_layout()
    fig.savefig(out / "epoch_detection.png", dpi=150)
    plt.close(fig)

    # Losses
    loss_keys = sorted({k for r in rows for k in (r.get("loss") or {})})
    if loss_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in loss_keys:
            ax.plot(epochs, [((r.get("loss") or {}).get(k)) for r in rows], label=k)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.set_title("Epoch losses")
        fig.tight_layout()
        fig.savefig(out / "epoch_losses.png", dpi=150)
        plt.close(fig)

    # Grad %
    fig, ax = plt.subplots(figsize=(8, 4))
    for g in ("backbone", "neck", "head"):
        ys = [(r.get("grad_pct") or {}).get(g) for r in rows]
        ax.plot(epochs, ys, label=g)
    ax.set_xlabel("epoch")
    ax.set_ylabel("% change vs prev epoch")
    ax.legend()
    ax.set_title("Gradient norm change by block")
    fig.tight_layout()
    fig.savefig(out / "epoch_grad_pct.png", dpi=150)
    plt.close(fig)


def plot_summary(metrics_dir: Path) -> None:
    summary_path = Path(metrics_dir) / "summary.json"
    if not summary_path.exists():
        print("No summary.json to plot")
        return

    summary = json.loads(summary_path.read_text())
    out = Path(metrics_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # Latency distribution
    samples = (((summary.get("latency") or {}).get("e2e") or {}).get("samples_ms")) or []
    if samples:
        arr = np.asarray(samples, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(arr, bins=30, density=True, alpha=0.75, label="histogram")
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(arr.min(), arr.max(), 200)
            ax.plot(xs, gaussian_kde(arr)(xs), label="KDE")
        except Exception:
            pass
        ax.set_xlabel("e2e latency (ms)")
        ax.set_ylabel("density")
        ax.set_title("Latency distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "latency_distribution.png", dpi=150)
        plt.close(fig)

    # Latency–accuracy curve
    curve = summary.get("latency_accuracy_curve") or []
    if curve:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [c["e2e_mean_ms"] for c in curve]
        ys = [c["map50"] for c in curve]
        ax.plot(xs, ys, marker="o")
        for c in curve:
            ax.annotate(str(c["imgsz"]), (c["e2e_mean_ms"], c["map50"]))
        ax.set_xlabel("e2e latency (ms)")
        ax.set_ylabel("mAP50")
        ax.set_title("Latency–accuracy curve (imgsz sweep)")
        fig.tight_layout()
        fig.savefig(out / "latency_accuracy_curve.png", dpi=150)
        plt.close(fig)


def plot_all(metrics_dir: str | Path) -> None:
    metrics_dir = Path(metrics_dir)
    plot_epochs(metrics_dir)
    plot_summary(metrics_dir)
    print(f"Plots written to {metrics_dir / 'plots'}")
