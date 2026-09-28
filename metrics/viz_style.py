"""Shared seaborn / plotly styling for metrics and meta plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SNS_STYLE = {
    "context": "talk",
    "style": "white",
    "palette": "deep",
    "font_scale": 0.85,
}

FIGSIZE = (8.5, 4.8)
FIGSIZE_WIDE = (11.5, 4.8)
DPI = 160

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=40, t=70, b=50),
    font=dict(size=13),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def apply_theme() -> None:
    sns.set_theme(**SNS_STYLE)


def palette(n: int | None = None):
    colors = sns.color_palette("deep")
    return colors if n is None else colors[:n]


def new_fig(figsize=FIGSIZE):
    apply_theme()
    return plt.subplots(figsize=figsize)


def finish_ax(ax, *, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, loc="best")
    sns.despine(ax=ax)


def save_fig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def label_points(ax, xs, ys, labels, *, dx: int = 7, dy: int = 7) -> None:
    for x, y, lab in zip(xs, ys, labels):
        if x is None or y is None:
            continue
        ax.annotate(
            str(lab),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            fontweight="semibold",
            color="0.2",
        )
