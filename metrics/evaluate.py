"""Final evaluation after training: accuracy, latency, size, efficiency."""

from __future__ import annotations

import json
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


def _iterative_sigma_clipping(
    data: np.ndarray, sigma: float = 2.0, max_iters: int = 3
) -> np.ndarray:
    """Remove outliers the same way Ultralytics ProfileModels does."""
    data = np.asarray(data, dtype=np.float64).ravel()
    for _ in range(max_iters):
        if data.size < 2:
            break
        mean, std = float(data.mean()), float(data.std())
        if std <= 0:
            break
        clipped = data[(data >= mean - sigma * std) & (data <= mean + sigma * std)]
        if clipped.size == data.size or clipped.size == 0:
            break
        data = clipped
    return data


def _pack_latency(xs) -> dict:
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


def _device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _tensorrt_engine_path(
    weights: Path, imgsz: int, engine_dir: Path, quantize: str
) -> Path:
    engine_dir.mkdir(parents=True, exist_ok=True)
    return engine_dir / f"{weights.stem}_imgsz{imgsz}_{quantize}.engine"


def _export_tensorrt(
    weights: Path,
    imgsz: int,
    engine_path: Path,
    quantize: str,
    data_yaml: str | None = None,
) -> Path | None:
    """Export TensorRT engine at fp32 / fp16 / int8. Returns path or None."""
    if not torch.cuda.is_available():
        return None
    if engine_path.is_file():
        return engine_path
    try:
        model = YOLO(str(weights))
        export_kw: dict = dict(
            format="engine",
            imgsz=imgsz,
            device=0,
            simplify=True,
            verbose=False,
        )
        if quantize == "fp16":
            export_kw["half"] = True
            export_kw["int8"] = False
        elif quantize == "int8":
            export_kw["half"] = False
            export_kw["int8"] = True
            if data_yaml is None:
                raise ValueError("INT8 TensorRT export requires data_yaml for calibration")
            export_kw["data"] = data_yaml
        else:  # fp32
            export_kw["half"] = False
            export_kw["int8"] = False

        exported = model.export(**export_kw)
        src = Path(str(exported))
        if not src.is_file():
            return None
        if src.resolve() != engine_path.resolve():
            engine_path.write_bytes(src.read_bytes())
        return engine_path if engine_path.is_file() else src
    except Exception as exc:
        print(f"TensorRT export ({quantize}) failed ({exc}); falling back to PyTorch speed timing.")
        return None


def _profile_ultralytics_speed(
    model: YOLO,
    imgsz: int,
    warmup: int = 10,
    timed_runs: int = 100,
) -> tuple[list[float], list[float], list[float]]:
    """
    Ultralytics T4 TensorRT10-style timing:
    dummy uint8 input, batch=1, read results[0].speed stages, then sigma-clip later.
    """
    input_data = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(max(warmup, 1)):
        model(input_data, imgsz=imgsz, verbose=False)

    pre_ms, infer_ms, post_ms = [], [], []
    for _ in range(max(timed_runs, 1)):
        results = model(input_data, imgsz=imgsz, verbose=False)
        speed = results[0].speed or {}
        pre_ms.append(float(speed.get("preprocess") or 0.0))
        infer_ms.append(float(speed.get("inference") or 0.0))
        post_ms.append(float(speed.get("postprocess") or 0.0))
    return pre_ms, infer_ms, post_ms


def measure_latency(
    model: YOLO,
    imgsz: int = 640,
    *,
    weights: str | Path | None = None,
    engine_dir: str | Path | None = None,
    data_yaml: str | Path | None = None,
    quantize: str = "fp16",
    warmup: int = 10,
    timed_runs: int = 100,
    prefer_tensorrt: bool = True,
) -> dict:
    """
    Latency à la Ultralytics detect Speed column:
    - Prefer TensorRT engine on CUDA at requested quantize (fp32|fp16|int8)
    - Else PyTorch model on the active device
    - Dummy image, batch=1, warmup + timed runs
    - Primary metric = results[0].speed['inference'] (pre/post excluded), sigma-clipped
    - `e2e` is aliased to inference so existing reports/plots match the table metric
    """
    quantize = str(quantize).lower().strip()
    if quantize not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"quantize must be fp32|fp16|int8, got {quantize!r}")

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    weights_path = Path(weights).resolve() if weights is not None else None
    data_yaml_str = str(Path(data_yaml).resolve()) if data_yaml is not None else None
    backend = "pytorch"
    engine_path = None
    profile_model = model

    if prefer_tensorrt and weights_path is not None and weights_path.is_file():
        cache = Path(engine_dir) if engine_dir is not None else weights_path.parent / "engines"
        candidate = _tensorrt_engine_path(weights_path, imgsz, cache, quantize)
        exported = _export_tensorrt(
            weights_path,
            imgsz,
            candidate,
            quantize=quantize,
            data_yaml=data_yaml_str,
        )
        if exported is not None and exported.is_file():
            try:
                profile_model = YOLO(str(exported))
                backend = f"tensorrt_{quantize}"
                engine_path = str(exported)
            except Exception as exc:
                print(f"TensorRT load failed ({exc}); using PyTorch.")
                profile_model = model
                backend = "pytorch"

    pre_raw, infer_raw, post_raw = _profile_ultralytics_speed(
        profile_model, imgsz=imgsz, warmup=warmup, timed_runs=timed_runs
    )
    peak_rss = max(peak_rss, proc.memory_info().rss)

    pre = _pack_latency(_iterative_sigma_clipping(np.asarray(pre_raw)))
    infer = _pack_latency(_iterative_sigma_clipping(np.asarray(infer_raw)))
    post = _pack_latency(_iterative_sigma_clipping(np.asarray(post_raw)))
    # Headline latency = inference-only (Ultralytics T4 TensorRT10 column).
    e2e = infer

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "method": "ultralytics_speed_inference",
        "quantize": quantize,
        "backend": backend,
        "device": _device_label(),
        "engine_path": engine_path,
        "warmup": warmup,
        "timed_runs": timed_runs,
        "preprocess": pre,
        "infer": infer,
        "post": post,
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


def latency_accuracy_curve(
    model: YOLO,
    data_yaml: str,
    weights: Path,
    engine_dir: Path,
    quantize: str = "fp16",
    sizes=(320, 480, 640, 800),
) -> list:
    curve = []
    for s in sizes:
        val = model.val(data=data_yaml, split="test", imgsz=s, plots=False, verbose=False)
        m = _split_metrics(val)
        lat = measure_latency(
            model,
            imgsz=s,
            weights=weights,
            engine_dir=engine_dir,
            data_yaml=data_yaml,
            quantize=quantize,
            warmup=10,
            timed_runs=50,
        )
        curve.append(
            {
                "imgsz": s,
                "map50": m.get("map50"),
                "map50_95": m.get("map50_95"),
                "e2e_mean_ms": lat["e2e"]["mean_ms"],
                "fps": lat["fps"],
                "backend": lat.get("backend"),
                "quantize": lat.get("quantize"),
            }
        )
    return curve


def run(
    weights: str | Path,
    data_yaml: str | Path,
    out_dir: str | Path,
    imgsz: int = 640,
    latency_n: int = 100,
    quantize: str = "fp16",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = str(Path(data_yaml).resolve())
    weights = Path(weights).resolve()
    engine_dir = out_dir / "engines"
    quantize = str(quantize).lower().strip()

    model = YOLO(str(weights))

    test_m = _split_metrics(
        model.val(data=data_yaml, split="test", imgsz=imgsz, plots=False, verbose=False)
    )
    train_m = _split_metrics(
        model.val(data=data_yaml, split="train", imgsz=imgsz, plots=False, verbose=False)
    )
    small = small_object_score(model, data_yaml, imgsz=imgsz)

    stats = model_stats(model, weights, imgsz)
    latency = measure_latency(
        model,
        imgsz=imgsz,
        weights=weights,
        engine_dir=engine_dir,
        data_yaml=data_yaml,
        quantize=quantize,
        warmup=10,
        timed_runs=latency_n,
    )
    curve = latency_accuracy_curve(
        model,
        data_yaml,
        weights=weights,
        engine_dir=engine_dir,
        quantize=quantize,
    )

    map50 = float(test_m.get("map50") or 0.0)
    weight_mb = stats["weight_mb"]
    gflops = stats["gflops"]
    fps = latency.get("fps")

    try:
        weights_str = str(weights.relative_to(Path.cwd().resolve()))
    except ValueError:
        weights_str = str(weights)

    summary = {
        "weights": weights_str,
        "imgsz": imgsz,
        "quantize": quantize,
        "device": latency.get("device") or _device_label(),
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
