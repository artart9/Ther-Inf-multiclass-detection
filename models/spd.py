"""Train YOLO26n-SPD on ThermalUAV2UAV (SPD-Conv stem only)."""

from __future__ import annotations

import argparse
import builtins
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Conv
import ultralytics.nn.tasks as tasks

from metrics.callbacks import MetricsCallback
from metrics.evaluate import run as evaluate_run
from metrics.meta_store import save_run_plot_data
from metrics.plots import plot_all
from metrics.report import build_report
from models.spd_conv import SPDConv
from weights_store import save_demo_weights

DATA_YAML = ROOT / "uav2uav.yaml"
CFG = Path(__file__).resolve().parent / "cfg" / "yolo26n_spd.yaml"
DEFAULT_RUN_NAME = "uav2uav_yolo26n_spd"


def register_spd() -> None:
    """Make SPDConv visible to Ultralytics YAML parsing (idempotent)."""
    tasks.SPDConv = SPDConv
    if getattr(tasks, "_spd_registered", False):
        return

    _orig = tasks.parse_model
    _frozenset = builtins.frozenset

    def parse_model(d, ch, verbose=True):
        # Treat SPDConv like Conv for c1/c2 width scaling inside parse_model.
        def frozenset(iterable=()):
            s = _frozenset(iterable)
            if Conv in s and C3k2 in s:
                return _frozenset(set(s) | {SPDConv})
            return s

        builtins.frozenset = frozenset
        try:
            return _orig(d, ch, verbose)
        finally:
            builtins.frozenset = _frozenset

    tasks.parse_model = parse_model
    tasks._spd_registered = True


def build_cfg(factor: int) -> Path:
    """Write a temp YAML with the SPD stem factor set from the CLI."""
    if factor < 2:
        raise SystemExit(f"--factor must be >= 2, got {factor}")

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    cfg = deepcopy(cfg)
    for row in cfg["backbone"]:
        if row[2] == "SPDConv":
            # args: [c2, k, factor]
            args = list(row[3])
            while len(args) < 3:
                args.append(3 if len(args) == 1 else factor)
            args[2] = factor
            row[3] = args
            break
    else:
        raise SystemExit(f"No SPDConv stem found in {CFG}")

    tmp = Path(tempfile.mkdtemp(prefix="yolo26n_spd_")) / f"yolo26n_spd_f{factor}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26n-SPD on ThermalUAV2UAV")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--factor",
        type=int,
        default=2,
        help="SPD space-to-depth scaling factor (stem downsample; default: 2)",
    )
    p.add_argument(
        "--quantize",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="TensorRT latency export precision (training stays float; default: fp16)",
    )
    p.add_argument("--name", default=DEFAULT_RUN_NAME, help="run / experiment name")
    p.add_argument("--device", default=None, help="cuda, cpu, mps, or device id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    register_spd()

    # Stem /factor then four stride-2 stages → max stride = 16 * factor
    max_stride = 16 * args.factor
    if args.imgsz % max_stride != 0:
        raise SystemExit(
            f"--imgsz={args.imgsz} must be divisible by {max_stride} "
            f"(16 * --factor={args.factor})"
        )

    if not (ROOT / "data" / "ThermalUAV2UAV_Dataset" / "train" / "images").exists():
        raise SystemExit(
            "Dataset missing. Clone it first:\n"
            "  mkdir -p data && git clone "
            "https://github.com/GabryV00/ThermalUAV2UAV_Dataset.git "
            "data/ThermalUAV2UAV_Dataset"
        )

    run_dir = ROOT / "runs" / args.name
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = build_cfg(args.factor)
    model = YOLO(str(cfg_path))
    MetricsCallback(metrics_dir).attach(model)

    train_kw = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )
    if args.device is not None:
        train_kw["device"] = args.device
    model.train(**train_kw)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"

    print("Running final evaluation...")
    summary = evaluate_run(
        weights=best,
        data_yaml=DATA_YAML,
        out_dir=metrics_dir,
        imgsz=args.imgsz,
        quantize=args.quantize,
    )
    plot_all(metrics_dir)

    meta = save_run_plot_data(
        summary=summary,
        exp_name=args.name,
        metrics_dir=metrics_dir,
        reports_dir=ROOT / "reports",
    )

    report_path = build_report(
        metrics_dir=metrics_dir,
        reports_dir=ROOT / "reports",
        exp_name=args.name,
        notes=f"YOLO26n-SPD (SPD-Conv stem, factor={args.factor}) on ThermalUAV2UAV "
        f"(epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"quantize={args.quantize}).",
    )

    demo = save_demo_weights(
        src_weights=best,
        weights_dir=ROOT / "weights",
        exp_name=args.name,
        tag="best",
    )
    print(f"Done. Metrics: {metrics_dir}")
    print(f"Report: {report_path}")
    print(f"Run folder: {meta['run_dir']}")
    print(f"Compare index: {meta['index']}")
    print(f"Demo weights: {demo['file']} (and weights/latest.pt)")


if __name__ == "__main__":
    main()
