"""Cross-run meta plots from reports/<exp>/report.json (seaborn PNG + plotly HTML)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.viz_style import (
    FIGSIZE,
    FIGSIZE_WIDE,
    PLOTLY_LAYOUT,
    apply_theme,
    finish_ax,
    label_points,
    new_fig,
    palette,
    save_fig,
)

DEFAULT_IMGSZ = (128, 160, 256, 320)
DEFAULT_NAME_TMPL = "uav2uav_yolo26n_baseline_imgsz{imgsz}"
DEFAULT_QUANT_EXPS = (
    "baseline_imgsz160_fp32",
    "baseline_imgsz160_fp16",
    "baseline_imgsz160_int8",
)
QUANT_ORDER = ("fp32", "fp16", "int8")


def load_quantize_latency_rows(
    reports_dir: Path,
    exp_names: tuple[str, ...] | list[str] = DEFAULT_QUANT_EXPS,
) -> list[dict]:
    """Load per-sample e2e latency from reports/<exp>/latency.json for boxplots."""
    rows: list[dict] = []
    for exp in exp_names:
        path = Path(reports_dir) / exp / "latency.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing latency file: {path}")
        data = json.loads(path.read_text())
        stages = data.get("stages") or {}
        e2e = stages.get("e2e") or {}
        samples = list(e2e.get("samples_ms") or [])
        if not samples:
            raise ValueError(f"No e2e samples_ms in {path}")
        quantize = str(data.get("quantize") or exp.rsplit("_", 1)[-1]).lower()
        for ms in samples:
            rows.append(
                {
                    "exp_name": data.get("exp_name", exp),
                    "quantize": quantize,
                    "backend": data.get("backend"),
                    "device": data.get("device"),
                    "imgsz": data.get("imgsz"),
                    "e2e_ms": float(ms),
                    "e2e_mean_ms": e2e.get("mean_ms"),
                    "fps": data.get("fps"),
                    "map50": (data.get("accuracy") or {}).get("map50"),
                }
            )
    return rows


def plot_quantize_latency_boxplot(
    sample_rows: list[dict],
    out_dir: Path,
    title: str = "Inference latency by TensorRT precision (imgsz=160)",
) -> list[Path]:
    """Bar chart of mean e2e latency across fp32 / fp16 / int8 (y from 0)."""
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = palette()
    written: list[Path] = []

    present = {str(r["quantize"]) for r in sample_rows}
    order = [q for q in QUANT_ORDER if q in present]
    order += sorted(present - set(order))
    by_q: dict[str, list[float]] = {q: [] for q in order}
    for r in sample_rows:
        by_q[str(r["quantize"])].append(float(r["e2e_ms"]))

    means = [sum(by_q[q]) / len(by_q[q]) for q in order]
    labels = [q.upper() for q in order]
    bar_colors = [colors[i % len(colors)] for i in range(len(order))]

    fig, ax = new_fig(FIGSIZE)
    sns.barplot(x=labels, y=means, hue=labels, palette=bar_colors, legend=False, ax=ax, width=0.6)
    ax.set_ylim(0, max(means) * 1.15 if means else 1.0)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="semibold")
    finish_ax(
        ax,
        title=title,
        xlabel="TensorRT precision",
        ylabel="e2e latency (ms)",
    )
    written.append(save_fig(fig, out_dir / "quantize_latency_boxplot.png"))

    fig_p = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=means,
                marker_color=[
                    f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in bar_colors
                ],
                text=[f"{m:.3f}" for m in means],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f} ms<extra></extra>",
            )
        ]
    )
    y_max = max(means) * 1.2 if means else 1.0
    fig_p.update_layout(**PLOTLY_LAYOUT, title=title, showlegend=False)
    fig_p.update_xaxes(title_text="TensorRT precision")
    fig_p.update_yaxes(title_text="e2e latency (ms)", range=[0, y_max])
    html = out_dir / "quantize_latency_boxplot.html"
    fig_p.write_html(html, include_plotlyjs="cdn")
    written.append(html)
    return written


def load_imgsz_rows(
    reports_dir: Path,
    imgsizes: tuple[int, ...] | list[int],
    name_tmpl: str = DEFAULT_NAME_TMPL,
) -> list[dict]:
    rows = []
    for imgsz in imgsizes:
        exp = name_tmpl.format(imgsz=imgsz)
        path = Path(reports_dir) / exp / "report.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing report: {path}")
        report = json.loads(path.read_text())
        summary = report.get("summary") or {}
        lat = summary.get("latency") or {}
        e2e = lat.get("e2e") or {}
        model = summary.get("model") or {}
        acc = summary.get("accuracy") or {}
        rows.append(
            {
                "exp_name": report.get("exp_name", exp),
                "imgsz": int(report.get("imgsz") or imgsz),
                "e2e_mean_ms": e2e.get("mean_ms"),
                "e2e_p50_ms": e2e.get("p50_ms"),
                "fps": lat.get("fps"),
                "gflops": model.get("gflops"),
                "map50": acc.get("map50"),
                "map50_95": acc.get("map50_95"),
                "small_object_score": acc.get("small_object_score"),
                "device": report.get("device"),
            }
        )
    rows.sort(key=lambda r: r["imgsz"])
    return rows


def plot_latency_gflops_vs_imgsz(
    rows: list[dict],
    out_dir: Path,
    title: str = "Latency & compute vs imgsz",
) -> list[Path]:
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = palette()

    xs = [r["imgsz"] for r in rows]
    latency = [r["e2e_mean_ms"] for r in rows]
    gflops = [r["gflops"] for r in rows]
    written: list[Path] = []

    fig, ax1 = new_fig(FIGSIZE)
    sns.lineplot(x=xs, y=latency, marker="o", ax=ax1, color=colors[0], label="e2e latency")
    ax1.set_xlabel("imgsz (px)")
    ax1.set_ylabel("e2e latency (ms)", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    sns.lineplot(x=xs, y=gflops, marker="s", ax=ax2, color=colors[1], label="GFLOPs")
    ax2.set_ylabel("GFLOPs", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    ax1.set_title(title)
    sns.despine(ax=ax1, right=False)
    written.append(save_fig(fig, out_dir / "imgsz_latency_gflops.png"))

    fig_p = make_subplots(specs=[[{"secondary_y": True}]])
    fig_p.add_trace(
        go.Scatter(
            x=xs, y=latency, mode="lines+markers+text",
            name="e2e latency", text=[str(s) for s in xs], textposition="top center",
            marker=dict(size=10),
        ),
        secondary_y=False,
    )
    fig_p.add_trace(
        go.Scatter(
            x=xs, y=gflops, mode="lines+markers",
            name="GFLOPs", marker=dict(size=10, symbol="square"),
        ),
        secondary_y=True,
    )
    fig_p.update_layout(**PLOTLY_LAYOUT, title=title)
    fig_p.update_xaxes(title_text="imgsz (px)", tickvals=xs)
    fig_p.update_yaxes(title_text="e2e latency (ms)", secondary_y=False)
    fig_p.update_yaxes(title_text="GFLOPs", secondary_y=True)
    html = out_dir / "imgsz_latency_gflops.html"
    fig_p.write_html(html, include_plotlyjs="cdn")
    written.append(html)
    return written


def plot_map_vs_latency_gflops(rows: list[dict], out_dir: Path) -> list[Path]:
    """Scatter mAP50-95 vs latency / GFLOPs; label points by imgsz."""
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = palette()

    map_y = [r["map50_95"] for r in rows]
    latency = [r["e2e_mean_ms"] for r in rows]
    gflops = [r["gflops"] for r in rows]
    labels = [r["imgsz"] for r in rows]
    written: list[Path] = []

    def _one_scatter(xs, ys, xlabel, title, color, path: Path) -> Path:
        fig, ax = new_fig(FIGSIZE)
        sns.lineplot(x=xs, y=ys, ax=ax, color=color, alpha=0.35, estimator=None, sort=False)
        sns.scatterplot(x=xs, y=ys, ax=ax, s=130, color=color, zorder=3, edgecolor="white", linewidth=1.2)
        label_points(ax, xs, ys, labels)
        finish_ax(ax, title=title, xlabel=xlabel, ylabel="mAP@0.5:0.95")
        return save_fig(fig, path)

    written.append(
        _one_scatter(
            latency, map_y, "e2e latency (ms)",
            "mAP50-95 vs latency (labeled by imgsz)", colors[0],
            out_dir / "map95_vs_latency.png",
        )
    )
    written.append(
        _one_scatter(
            gflops, map_y, "GFLOPs",
            "mAP50-95 vs compute (labeled by imgsz)", colors[1],
            out_dir / "map95_vs_gflops.png",
        )
    )

    apply_theme()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    sns.lineplot(x=latency, y=map_y, ax=ax_a, color=colors[0], alpha=0.35, estimator=None, sort=False)
    sns.scatterplot(x=latency, y=map_y, ax=ax_a, s=130, color=colors[0], zorder=3, edgecolor="white", linewidth=1.2)
    label_points(ax_a, latency, map_y, labels)
    finish_ax(ax_a, title="vs latency", xlabel="e2e latency (ms)", ylabel="mAP@0.5:0.95")

    sns.lineplot(x=gflops, y=map_y, ax=ax_b, color=colors[1], alpha=0.35, estimator=None, sort=False)
    sns.scatterplot(x=gflops, y=map_y, ax=ax_b, s=130, color=colors[1], zorder=3, edgecolor="white", linewidth=1.2)
    label_points(ax_b, gflops, map_y, labels)
    finish_ax(ax_b, title="vs compute", xlabel="GFLOPs", ylabel=None)
    fig.suptitle("mAP50-95 tradeoffs (imgsz labels)", y=1.02)
    written.append(save_fig(fig, out_dir / "map95_vs_latency_gflops.png"))

    fig_p = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("vs latency", "vs compute"),
        horizontal_spacing=0.08,
    )
    fig_p.add_trace(
        go.Scatter(
            x=latency, y=map_y, mode="markers+text+lines",
            text=[str(s) for s in labels], textposition="top right",
            name="latency", marker=dict(size=12),
            hovertemplate="imgsz=%{text}<br>e2e=%{x:.3f} ms<br>mAP50-95=%{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig_p.add_trace(
        go.Scatter(
            x=gflops, y=map_y, mode="markers+text+lines",
            text=[str(s) for s in labels], textposition="top right",
            name="GFLOPs", marker=dict(size=12),
            hovertemplate="imgsz=%{text}<br>GFLOPs=%{x:.4f}<br>mAP50-95=%{y:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig_p.update_layout(
        **PLOTLY_LAYOUT,
        title="mAP50-95 tradeoffs (imgsz labels)",
        showlegend=False,
        height=480,
        width=980,
    )
    fig_p.update_xaxes(title_text="e2e latency (ms)", row=1, col=1)
    fig_p.update_xaxes(title_text="GFLOPs", row=1, col=2)
    fig_p.update_yaxes(title_text="mAP@0.5:0.95", row=1, col=1)
    html = out_dir / "map95_vs_latency_gflops.html"
    fig_p.write_html(html, include_plotlyjs="cdn")
    written.append(html)
    return written


def plot_small_object_vs_gflops(
    rows: list[dict],
    out_dir: Path,
    title: str = "Small-object score vs compute (labeled by imgsz)",
) -> list[Path]:
    """Scatter/line of small-object F1 vs GFLOPs; points labeled by imgsz."""
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = palette()

    usable = [
        r
        for r in rows
        if r.get("small_object_score") is not None
        and r.get("gflops") is not None
        and float(r["gflops"]) > 0
    ]
    if not usable:
        raise ValueError("No rows with both small_object_score and positive gflops")

    xs = [float(r["gflops"]) for r in usable]
    ys = [float(r["small_object_score"]) for r in usable]
    labels = [r["imgsz"] for r in usable]
    color = colors[2 % len(colors)]
    written: list[Path] = []

    fig, ax = new_fig(FIGSIZE)
    sns.lineplot(x=xs, y=ys, ax=ax, color=color, alpha=0.35, estimator=None, sort=False)
    sns.scatterplot(x=xs, y=ys, ax=ax, s=130, color=color, zorder=3, edgecolor="white", linewidth=1.2)
    label_points(ax, xs, ys, labels)
    finish_ax(ax, title=title, xlabel="GFLOPs", ylabel="Small-object score (F1)")
    ax.set_ylim(0.0, 1.05)
    written.append(save_fig(fig, out_dir / "small_object_vs_gflops.png"))

    fig_p = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text+lines",
                text=[str(s) for s in labels],
                textposition="top right",
                marker=dict(size=12),
                name="small-object F1",
                hovertemplate="imgsz=%{text}<br>GFLOPs=%{x:.4f}<br>small-object F1=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig_p.update_layout(**PLOTLY_LAYOUT, title=title)
    fig_p.update_xaxes(title_text="GFLOPs")
    fig_p.update_yaxes(title_text="Small-object score (F1)", range=[0, 1.05])
    html = out_dir / "small_object_vs_gflops.html"
    fig_p.write_html(html, include_plotlyjs="cdn")
    written.append(html)

    # Remove superseded imgsz-axis plot if present.
    for old in ("small_object_vs_imgsz.png", "small_object_vs_imgsz.html"):
        stale = out_dir / old
        if stale.exists():
            stale.unlink()

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-plot latency, GFLOPs, and small-object vs compute")
    p.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "compare")
    p.add_argument("--imgsz", type=int, nargs="+", default=list(DEFAULT_IMGSZ))
    p.add_argument("--name-tmpl", default=DEFAULT_NAME_TMPL)
    p.add_argument(
        "--quantize-exps",
        nargs="+",
        default=list(DEFAULT_QUANT_EXPS),
        help="Experiment folders for quantization latency boxplot",
    )
    p.add_argument(
        "--only-quantize",
        action="store_true",
        help="Only build the quantization latency boxplot",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    if not args.only_quantize:
        rows = load_imgsz_rows(args.reports_dir, args.imgsz, args.name_tmpl)
        written.extend(plot_latency_gflops_vs_imgsz(rows, out))
        written.extend(plot_map_vs_latency_gflops(rows, out))
        written.extend(plot_small_object_vs_gflops(rows, out))
        data_path = out / "imgsz_latency_gflops.json"
        data_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(f"Data: {data_path}")
        for r in rows:
            small = r.get("small_object_score")
            small_s = f"{small:.4f}" if small is not None else "—"
            gflops = r.get("gflops")
            gflops_s = f"{gflops:.6f}" if gflops is not None else "—"
            e2e = r.get("e2e_mean_ms")
            e2e_s = f"{e2e:.4f}" if e2e is not None else "—"
            print(
                f"  imgsz={r['imgsz']:>4}  mAP50-95={r['map50_95']:.4f}  "
                f"small={small_s}  e2e={e2e_s} ms  GFLOPs={gflops_s}"
            )

    quant_rows = load_quantize_latency_rows(args.reports_dir, args.quantize_exps)
    written.extend(plot_quantize_latency_boxplot(quant_rows, out))
    quant_summary = []
    for q in QUANT_ORDER:
        subset = [r for r in quant_rows if r["quantize"] == q]
        if not subset:
            continue
        vals = [r["e2e_ms"] for r in subset]
        quant_summary.append(
            {
                "quantize": q,
                "n": len(vals),
                "mean_ms": sum(vals) / len(vals),
                "p50_ms": sorted(vals)[len(vals) // 2],
                "fps": subset[0].get("fps"),
                "map50": subset[0].get("map50"),
                "backend": subset[0].get("backend"),
                "device": subset[0].get("device"),
            }
        )
    quant_path = out / "quantize_latency_boxplot.json"
    quant_path.write_text(
        json.dumps({"summary": quant_summary, "n_samples": len(quant_rows)}, indent=2) + "\n"
    )
    print(f"Data: {quant_path}")
    for s in quant_summary:
        print(
            f"  {s['quantize']:>4}  mean={s['mean_ms']:.4f} ms  "
            f"p50={s['p50_ms']:.4f} ms  n={s['n']}  fps={s['fps']:.1f}"
        )

    for p in written:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
