"""Final evaluation after training: accuracy, latency, size, efficiency."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def _split_metrics(results) -> dict:
    if results is None:
        return {}
    box = getattr(results, "box", results)
    p = float(getattr(box, "mp", 0.0) or 0.0)
    r = float(getattr(box, "mr", 0.0) or 0.0)
    return {
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
    }


def _dataset_root(data_yaml: str) -> Path:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = Path(data_yaml).resolve().parent / root
    return root.resolve(), cfg


def _list_images(data_yaml: str, split: str) -> list:
    root, cfg = _dataset_root(data_yaml)
    img_dir = root / cfg[split]
    return sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))


def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def small_object_score(model: YOLO, data_yaml: str, imgsz: int = 640, conf: float = 0.25) -> float:
    """F1 on small boxes only (area < 32^2 on original image). See schema.md."""
    small_thr = 32.0 * 32.0
    root, _ = _dataset_root(data_yaml)
    images = _list_images(data_yaml, "test")
    label_dir = root / "test" / "labels"
    tps = fps = n_gt = 0

    for img_path in images:
        w, h = Image.open(img_path).size
        gts = []
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                pw, ph = bw * w, bh * h
                x1, y1 = cx * w - pw / 2, cy * h - ph / 2
                box = [x1, y1, x1 + pw, y1 + ph]
                if _box_area(box) < small_thr:
                    gts.append(box)
        n_gt += len(gts)

        pred = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            for b in pred.boxes.xyxy.cpu().numpy():
                if _box_area(b) < small_thr:
                    preds.append(b.tolist())

        matched = set()
        for pb in preds:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if j in matched:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                tps += 1
                matched.add(best_j)
            else:
                fps += 1

    fn = max(n_gt - tps, 0)
    prec = tps / (tps + fps) if (tps + fps) else 0.0
    rec = tps / (tps + fn) if (tps + fn) else 0.0
    return _f1(prec, rec)


def measure_latency(model: YOLO, image_paths: list, imgsz: int = 640, warmup: int = 30) -> dict:
    """Time full predict() as e2e; also try staged pre/infer/post when predictor allows."""
    proc = psutil.Process()
    e2e_ms, pre_ms, infer_ms, post_ms = [], [], [], []
    peak_rss = proc.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    warm = image_paths[: min(warmup, len(image_paths))]
    for p in warm:
        model.predict(str(p), imgsz=imgsz, verbose=False)

    for p in image_paths:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_ms.append((time.perf_counter() - t0) * 1000.000)
        peak_rss = max(peak_rss, proc.memory_info().rss)

        # Stage breakdown via predictor internals (best-effort)
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            continue
        try:
            import cv2

            im0 = cv2.imread(str(p))
            t1 = time.perf_counter()
            im = predictor.preprocess([im0])
            t2 = time.perf_counter()
            with torch.no_grad():
                preds = predictor.inference(im)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            predictor.postprocess(preds, im, [im0])
            t4 = time.perf_counter()
            pre_ms.append((t2 - t1) * 1000.000)
            infer_ms.append((t3 - t2) * 1000.000)
            post_ms.append((t4 - t3) * 1000.000)
        except Exception:
            pass

    def pack(xs):
        arr = np.asarray(xs, dtype=np.float64)
        if arr.size == 0:
            return {
                "mean_ms": None,
                "std_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "samples_ms": [],
            }
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "samples_ms": arr.tolist(),
        }

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    e2e = pack(e2e_ms)
    return {
        "preprocess": pack(pre_ms),
        "infer": pack(infer_ms),
        "post": pack(post_ms),
        "e2e": e2e,
        "fps": None if not e2e["mean_ms"] else float(1000.0 / e2e["mean_ms"]),
        "peak_rss_mb": peak_rss / (1024 ** 2),
        "peak_vram_mb": peak_vram_mb,
    }


def model_stats(model: YOLO, weights: Path, imgsz: int) -> dict:
    params = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        pass
    return {
        "params": int(params),
        "gflops": gflops,
        "weight_mb": weights.stat().st_size / (1024 ** 2),
    }


def latency_accuracy_curve(model: YOLO, data_yaml: str, sizes=(320, 480, 640, 800)) -> list:
    images = _list_images(data_yaml, "test")[:40]
    curve = []
    for s in sizes:
        val = model.val(data=data_yaml, split="test", imgsz=s, plots=False, verbose=False)
        m = _split_metrics(val)
        lat = measure_latency(model, images, imgsz=s, warmup=5)
        curve.append(
            {
                "imgsz": s,
                "map50": m.get("map50"),
                "map50_95": m.get("map50_95"),
                "e2e_mean_ms": lat["e2e"]["mean_ms"],
                "fps": lat["fps"],
            }
        )
    return curve


def run(
    weights: str | Path,
    data_yaml: str | Path,
    out_dir: str | Path,
    imgsz: int = 640,
    latency_n: int = 200,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = str(Path(data_yaml).resolve())
    weights = Path(weights).resolve()

    model = YOLO(str(weights))

    test_m = _split_metrics(
        model.val(data=data_yaml, split="test", imgsz=imgsz, plots=False, verbose=False)
    )
    train_m = _split_metrics(
        model.val(data=data_yaml, split="train", imgsz=imgsz, plots=False, verbose=False)
    )
    small = small_object_score(model, data_yaml, imgsz=imgsz)

    stats = model_stats(model, weights, imgsz)
    images = _list_images(data_yaml, "test")[:latency_n]
    latency = measure_latency(model, images, imgsz=imgsz)
    curve = latency_accuracy_curve(model, data_yaml)

    map50 = float(test_m.get("map50") or 0.0)
    weight_mb = stats["weight_mb"]
    gflops = stats["gflops"]
    fps = latency.get("fps")

    try:
        weights_str = str(weights.relative_to(Path.cwd().resolve()))
    except ValueError:
        weights_str = str(weights)

    try:
        device = str(next(model.model.parameters()).device)
    except Exception:
        device = "unknown"

    summary = {
        "weights": weights_str,
        "imgsz": imgsz,
        "device": device,
        "accuracy": {
            "map50": test_m.get("map50"),
            "map50_95": test_m.get("map50_95"),
            "test": test_m,
            "train": {
                "precision": train_m.get("precision"),
                "recall": train_m.get("recall"),
                "f1": train_m.get("f1"),
            },
            "small_object_score": small,
        },
        "latency": latency,
        "model": {
            **stats,
            "peak_rss_mb": latency.get("peak_rss_mb"),
            "peak_vram_mb": latency.get("peak_vram_mb"),
        },
        "efficiency": {
            "map_per_mb": (map50 / weight_mb) if weight_mb else None,
            "map_per_gflop": (map50 / gflops) if gflops else None,
            "map_times_fps": (map50 * fps) if fps else None,
        },
        "latency_accuracy_curve": curve,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
