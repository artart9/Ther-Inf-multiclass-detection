"""Train YOLO26 on ThermalUAV2UAV and write metrics, report, and demo weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from metrics.callbacks import MetricsCallback
from metrics.evaluate import run as evaluate_run
from metrics.meta_store import save_run_plot_data
from metrics.plots import plot_all
from metrics.report import build_report
from weights_store import save_demo_weights

DATA_YAML = ROOT / "uav2uav.yaml"
DEFAULT_RUN_NAME = "uav2uav_yolo26n_baseline"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26 baseline on ThermalUAV2UAV")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--quantize",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="TensorRT latency export precision (training stays float; default: fp16)",
    )
    p.add_argument("--name", default=DEFAULT_RUN_NAME, help="run / experiment name")
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights or model name")
    p.add_argument("--device", default=None, help="cuda, cpu, mps, or device id")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    model = YOLO(args.model)
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
        notes=f"YOLO26n baseline on ThermalUAV2UAV "
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
