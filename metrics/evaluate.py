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
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    return {
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
        "ap": map50,  # AP@0.5 alias (single-class == mAP@0.5)
        "map50": map50,
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


SMALL_OBJECT_SIDES = (64, 32, 16, 8)  # F1 / AP for boxes with area < side^2


def _match_f1(gts: list, preds: list) -> tuple[float, float, float, int, int, int]:
    """Greedy IoU≥0.5 matching → (f1, precision, recall, tps, fps, n_gt)."""
    tps = fps = 0
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
    n_gt = len(gts)
    fn = max(n_gt - tps, 0)
    prec = tps / (tps + fps) if (tps + fps) else 0.0
    rec = tps / (tps + fn) if (tps + fn) else 0.0
    return _f1(prec, rec), prec, rec, tps, fps, n_gt


def _average_precision(gts: list, preds: list, scores: list, iou_thr: float = 0.5) -> float:
    """VOC-style AP@iou_thr for one image or pooled dets (score-sorted)."""
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0 if preds else 1.0
    if not preds:
        return 0.0

    order = sorted(range(len(preds)), key=lambda i: scores[i], reverse=True)
    matched = set()
    tp = []
    fp = []
    for i in order:
        pb = preds[i]
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in matched:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched.add(best_j)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # All-point interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def small_object_scores(
    model: YOLO,
    data_yaml: str,
    imgsz: int = 640,
    conf: float = 0.25,
    sides: tuple[int, ...] = SMALL_OBJECT_SIDES,
) -> dict:
    """
    Per-threshold small-object F1 and AP@0.5 on the test split.

    For each side in ``sides``, keep GT/pred boxes with area < side^2 (original
    image pixels), match at IoU ≥ 0.5. One predict pass per image.
    Keys are string side sizes, e.g. ``\"32\"`` → metrics for area < 32².
    """
    root, _ = _dataset_root(data_yaml)
    images = _list_images(data_yaml, "test")
    label_dir = root / "test" / "labels"
    thr = {int(s): float(s) * float(s) for s in sides}
    # Per-threshold accumulators (F1 greedy + pooled scored dets for AP)
    buckets = {
        int(s): {"tps": 0, "fps": 0, "n_gt": 0, "preds": [], "scores": [], "gts": []}
        for s in sides
    }

    for img_path in images:
        w, h = Image.open(img_path).size
        all_gts = []
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                pw, ph = bw * w, bh * h
                x1, y1 = cx * w - pw / 2, cy * h - ph / 2
                all_gts.append([x1, y1, x1 + pw, y1 + ph])

        pred = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        all_preds = []
        all_scores = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            for b, s in zip(xyxy, confs):
                all_preds.append(b.tolist())
                all_scores.append(float(s))

        for side, area_thr in thr.items():
            gts = [b for b in all_gts if _box_area(b) < area_thr]
            keep = [i for i, b in enumerate(all_preds) if _box_area(b) < area_thr]
            preds = [all_preds[i] for i in keep]
            scores = [all_scores[i] for i in keep]
            _, _, _, tps, fps, n_gt = _match_f1(gts, preds)
            buckets[side]["tps"] += tps
            buckets[side]["fps"] += fps
            buckets[side]["n_gt"] += n_gt
            # Pool across images for global AP (offset GT indices via separate lists)
            # Compute per-image AP contribution via concatenating with image tags:
            buckets[side]["gts"].append(gts)
            buckets[side]["preds"].append(preds)
            buckets[side]["scores"].append(scores)

    out = {}
    for side in sides:
        b = buckets[int(side)]
        tps, fps, n_gt = b["tps"], b["fps"], b["n_gt"]
        fn = max(n_gt - tps, 0)
        prec = tps / (tps + fps) if (tps + fps) else 0.0
        rec = tps / (tps + fn) if (tps + fn) else 0.0
        # Mean AP over images that have ≥1 small GT (skip empty-GT images)
        aps = []
        for gts, preds, scores in zip(b["gts"], b["preds"], b["scores"]):
            if not gts:
                continue
            aps.append(_average_precision(gts, preds, scores, iou_thr=0.5))
        ap = float(sum(aps) / len(aps)) if aps else 0.0
        out[str(int(side))] = {
            "f1": _f1(prec, rec),
            "ap": ap,
            "precision": prec,
            "recall": rec,
            "tps": tps,
            "fps": fps,
            "n_gt": n_gt,
            "area_lt": int(side) * int(side),
        }
    return out


def small_object_score(model: YOLO, data_yaml: str, imgsz: int = 640, conf: float = 0.25) -> float:
    """Backward-compatible scalar: F1 for area < 32²."""
    table = small_object_scores(model, data_yaml, imgsz=imgsz, conf=conf)
    return float((table.get("32") or {}).get("f1") or 0.0)


def _scalar_small_object(acc: dict | None) -> float | None:
    """Prefer table['32'].f1; fall back to legacy scalar small_object_score."""
    if not acc:
        return None
    table = acc.get("small_object_scores")
    if isinstance(table, dict) and table.get("32") is not None:
        cell = table["32"]
        if isinstance(cell, dict):
            return cell.get("f1")
        return float(cell)
    return acc.get("small_object_score")


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


def _max_stride(model: YOLO) -> int:
    """Largest detection stride (SPD factor>2 raises this above 32)."""
    m = model.model if hasattr(model, "model") else model
    if hasattr(m, "stride"):
        return max(int(m.stride.max()), 1)
    return 32


def _align_imgsz(imgsz: int, stride: int) -> int:
    """Round imgsz up to a multiple of stride (matches Ultralytics predict/val)."""
    stride = max(int(stride), 1)
    imgsz = int(imgsz)
    return imgsz if imgsz % stride == 0 else imgsz + stride - (imgsz % stride)


def _ensure_custom_modules() -> None:
    """Register project custom layers (SPDConv) before YOLO(weights) / export."""
    try:
        from models.spd import register_spd

        register_spd()
    except Exception:
        pass


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
    """Export TensorRT engine at fp32 / fp16 / int8 (batch=1). Returns path or None."""
    if not torch.cuda.is_available():
        return None
    if engine_path.is_file():
        return engine_path
    try:
        _ensure_custom_modules()
        model = YOLO(str(weights))
        export_kw: dict = dict(
            format="engine",
            imgsz=imgsz,
            batch=1,
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
        print(f"TensorRT export ({quantize}) failed ({exc}); falling back to PyTorch.")
        return None


def resolve_deploy_model(
    weights: Path,
    imgsz: int,
    *,
    engine_dir: Path,
    quantize: str,
    data_yaml: str | None = None,
    float_model: YOLO | None = None,
    prefer_tensorrt: bool = True,
) -> dict:
    """
    Load the final deployable model at the requested quantize setting.

    Prefers a TensorRT engine on CUDA (fp32|fp16|int8); otherwise falls back to
    the float PyTorch checkpoint. All accuracy + latency metrics should use the
    returned ``model``.
    """
    quantize = str(quantize).lower().strip()
    if quantize not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"quantize must be fp32|fp16|int8, got {quantize!r}")

    weights = Path(weights).resolve()
    engine_dir = Path(engine_dir)
    _ensure_custom_modules()
    if float_model is None:
        float_model = YOLO(str(weights))

    imgsz = _align_imgsz(int(imgsz), _max_stride(float_model))
    backend = "pytorch"
    engine_path = None
    deploy = float_model
    artifact = weights

    if prefer_tensorrt and weights.is_file():
        candidate = _tensorrt_engine_path(weights, imgsz, engine_dir, quantize)
        exported = _export_tensorrt(
            weights,
            imgsz,
            candidate,
            quantize=quantize,
            data_yaml=data_yaml,
        )
        if exported is not None and exported.is_file():
            try:
                _ensure_custom_modules()
                deploy = YOLO(str(exported))
                backend = f"tensorrt_{quantize}"
                engine_path = str(exported)
                artifact = exported
            except Exception as exc:
                print(f"TensorRT load failed ({exc}); using PyTorch.")
                deploy = float_model
                backend = "pytorch"
                engine_path = None
                artifact = weights

    return {
        "model": deploy,
        "float_model": float_model,
        "backend": backend,
        "quantize": quantize,
        "engine_path": engine_path,
        "artifact_path": Path(artifact),
        "imgsz": imgsz,
    }


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
    deploy: dict | None = None,
) -> dict:
    """
    Latency à la Ultralytics detect Speed column on the final deploy model:
    - Prefer TensorRT engine on CUDA at requested quantize (fp32|fp16|int8)
    - Else PyTorch model on the active device
    - Dummy image, batch=1, warmup + timed runs
    - Primary metric = results[0].speed['inference'] (pre/post excluded), sigma-clipped
    - `e2e` is aliased to inference so existing reports/plots match the table metric

    Pass ``deploy`` from ``resolve_deploy_model`` to avoid re-exporting.
    """
    quantize = str(quantize).lower().strip()
    if quantize not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"quantize must be fp32|fp16|int8, got {quantize!r}")

    imgsz_requested = int(imgsz)
    stride = _max_stride(model)
    imgsz = _align_imgsz(imgsz_requested, stride)
    if imgsz != imgsz_requested:
        print(
            f"Latency imgsz {imgsz_requested} -> {imgsz} "
            f"(aligned to max stride {stride})"
        )

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if deploy is None:
        weights_path = Path(weights).resolve() if weights is not None else None
        data_yaml_str = str(Path(data_yaml).resolve()) if data_yaml is not None else None
        if prefer_tensorrt and weights_path is not None and weights_path.is_file():
            cache = (
                Path(engine_dir) if engine_dir is not None else weights_path.parent / "engines"
            )
            deploy = resolve_deploy_model(
                weights_path,
                imgsz,
                engine_dir=cache,
                quantize=quantize,
                data_yaml=data_yaml_str,
                float_model=model,
                prefer_tensorrt=True,
            )
        else:
            deploy = {
                "model": model,
                "backend": "pytorch",
                "quantize": quantize,
                "engine_path": None,
                "imgsz": imgsz,
            }

    profile_model = deploy["model"]
    backend = deploy.get("backend") or "pytorch"
    engine_path = deploy.get("engine_path")

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
        "imgsz_requested": imgsz_requested,
        "imgsz": imgsz,
        "max_stride": stride,
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


def _get_flops(model: torch.nn.Module, imgsz: int) -> float | None:
    """GFLOPs for imgsz. Uses yaml input channels (not first-weight in_ch).

    Ultralytics ``get_flops`` infers channels from ``next(parameters()).shape[1]``,
    which breaks SPD/Focus stems whose first conv sees ``C * factor**2`` channels.
    """
    try:
        import thop
    except ImportError:
        try:
            import ultralytics.thop as thop  # older / alternate package layout
        except ImportError:
            print("GFLOPs skipped: install ultralytics-thop (pip install ultralytics-thop)")
            return None

    try:
        from ultralytics.nn.modules.block import AAttn, Attention
        from ultralytics.nn.modules.head import RTDETRDecoder
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError as exc:
        print(f"GFLOPs skipped: ultralytics import failed ({exc})")
        return None

    try:
        from ultralytics.utils.torch_utils import _attention_ops
    except ImportError:
        _attention_ops = None

    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        size = [imgsz, imgsz] if not isinstance(imgsz, list) else list(imgsz)
        yaml_cfg = getattr(model, "yaml", None)
        if not isinstance(yaml_cfg, dict):
            yaml_cfg = {}
        ch = int(yaml_cfg.get("channels", yaml_cfg.get("ch", 3)) or 3)
        # Never trust first-weight in_channels (SPD stem is C * factor**2).
        if ch < 1:
            ch = 3

        attn = tuple(m for m in model.modules() if isinstance(m, (Attention, AAttn)))
        rtdetr = any(isinstance(m, RTDETRDecoder) for m in model.modules())
        stride = None if attn else (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32)
        im = torch.empty((1, ch, *size), device=p.device, dtype=p.dtype)
        custom_ops = (
            {Attention: _attention_ops, AAttn: _attention_ops}
            if (attn and _attention_ops is not None)
            else None
        )

        def _profile(ops, use_stride):
            if rtdetr or not use_stride:
                return thop.profile(model, inputs=[im], custom_ops=ops, verbose=False)[0]
            return thop.profile(model, inputs=[im], stride=stride, custom_ops=ops, verbose=False)[0]

        # Prefer attention-aware count; fall back if hooks break on custom stems.
        last_err = None
        for ops, use_stride in (
            (custom_ops, True),
            (custom_ops, False),
            (None, True),
            (None, False),
        ):
            try:
                flops = _profile(ops, use_stride)
                return float(flops) / 1e9 * 2
            except Exception as exc:
                last_err = exc
                continue
        print(f"GFLOPs profiling failed ({last_err})")
        return None
    except Exception as exc:
        print(f"GFLOPs skipped ({exc})")
        return None


def model_stats(
    model: YOLO,
    artifact: Path,
    imgsz: int,
    *,
    float_model: YOLO | None = None,
) -> dict:
    """Params/GFLOPs from the float graph; weight_mb from the deploy artifact."""
    graph = float_model or model
    try:
        params = sum(p.numel() for p in graph.model.parameters())
    except Exception:
        params = None
    gflops = None
    try:
        gflops = _get_flops(graph.model, imgsz=imgsz)
    except Exception:
        gflops = None
    artifact = Path(artifact)
    return {
        "params": int(params) if params is not None else None,
        "gflops": gflops,
        "weight_mb": artifact.stat().st_size / (1024 ** 2) if artifact.is_file() else None,
        "artifact": str(artifact),
    }


def _val_split(model: YOLO, data_yaml: str, split: str, imgsz: int) -> dict:
    """Validate on the deploy model (batch=1 for static TensorRT engines)."""
    return _split_metrics(
        model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=1,
            plots=False,
            verbose=False,
        )
    )


def latency_accuracy_curve(
    float_model: YOLO,
    data_yaml: str,
    weights: Path,
    engine_dir: Path,
    quantize: str = "fp16",
    sizes=(320, 480, 640, 800),
) -> list:
    """Per-imgsz mAP + latency, both on the quantized deploy model for that size."""
    stride = _max_stride(float_model)
    # Align each size; drop duplicates after alignment (e.g. 480 and 512 both -> 512).
    aligned = []
    seen = set()
    for s in sizes:
        a = _align_imgsz(int(s), stride)
        if a not in seen:
            seen.add(a)
            aligned.append(a)

    curve = []
    for s in aligned:
        deploy = resolve_deploy_model(
            weights,
            s,
            engine_dir=engine_dir,
            quantize=quantize,
            data_yaml=data_yaml,
            float_model=float_model,
        )
        m = _val_split(deploy["model"], data_yaml, "test", deploy["imgsz"])
        lat = measure_latency(
            float_model,
            imgsz=deploy["imgsz"],
            weights=weights,
            engine_dir=engine_dir,
            data_yaml=data_yaml,
            quantize=quantize,
            warmup=10,
            timed_runs=50,
            deploy=deploy,
        )
        curve.append(
            {
                "imgsz": lat.get("imgsz", deploy["imgsz"]),
                "imgsz_requested": lat.get("imgsz_requested", s),
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
    """
    Final eval on the post-quantized deploy model (TensorRT when available).

    Accuracy (test/train/small-object), latency, and the latency–accuracy curve
    all use the same backend resolved for ``quantize``. Params/GFLOPs still come
    from the float training graph; ``weight_mb`` is the deploy artifact size.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = str(Path(data_yaml).resolve())
    weights = Path(weights).resolve()
    engine_dir = out_dir / "engines"
    quantize = str(quantize).lower().strip()

    _ensure_custom_modules()
    float_model = YOLO(str(weights))

    imgsz_requested = int(imgsz)
    imgsz = _align_imgsz(imgsz_requested, _max_stride(float_model))
    if imgsz != imgsz_requested:
        print(
            f"Eval imgsz {imgsz_requested} -> {imgsz} "
            f"(aligned to max stride {_max_stride(float_model)})"
        )

    print(f"Resolving deploy model (quantize={quantize})...")
    deploy = resolve_deploy_model(
        weights,
        imgsz,
        engine_dir=engine_dir,
        quantize=quantize,
        data_yaml=data_yaml,
        float_model=float_model,
    )
    model = deploy["model"]
    print(f"Eval backend: {deploy['backend']}")

    test_m = _val_split(model, data_yaml, "test", imgsz)
    train_m = _val_split(model, data_yaml, "train", imgsz)
    small_table = small_object_scores(model, data_yaml, imgsz=imgsz)
    small = float((small_table.get("32") or {}).get("f1") or 0.0)

    stats = model_stats(
        model,
        deploy["artifact_path"],
        imgsz,
        float_model=float_model,
    )
    latency = measure_latency(
        float_model,
        imgsz=imgsz,
        weights=weights,
        engine_dir=engine_dir,
        data_yaml=data_yaml,
        quantize=quantize,
        warmup=10,
        timed_runs=latency_n,
        deploy=deploy,
    )
    curve = latency_accuracy_curve(
        float_model,
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

    small_ap = None
    if isinstance(small_table, dict) and isinstance(small_table.get("32"), dict):
        small_ap = small_table["32"].get("ap")

    summary = {
        "weights": weights_str,
        "imgsz": imgsz,
        "imgsz_requested": imgsz_requested,
        "quantize": quantize,
        "backend": deploy["backend"],
        "engine_path": deploy.get("engine_path"),
        "device": latency.get("device") or _device_label(),
        "accuracy": {
            "ap": test_m.get("ap", map50),
            "map50": test_m.get("map50"),
            "map50_95": test_m.get("map50_95"),
            "test": test_m,
            "train": {
                "precision": train_m.get("precision"),
                "recall": train_m.get("recall"),
                "f1": train_m.get("f1"),
            },
            "small_object_score": small,
            "small_object_ap": small_ap,
            "small_object_scores": small_table,
            "backend": deploy["backend"],
            "quantize": quantize,
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
