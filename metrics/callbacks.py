"""Per-epoch metrics logged during Ultralytics training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def _metric(metrics: dict, *keys: str, default=None):
    for k in keys:
        if k in metrics:
            return metrics[k]
    return default


def _group_indices(model) -> dict:
    """Map DetectionModel layers to backbone / neck / head."""
    layers = model.model if hasattr(model, "model") else model
    n = len(layers)
    head_idx = n - 1
    first_concat = None
    for i, layer in enumerate(layers):
        if type(layer).__name__ == "Concat":
            first_concat = i
            break
    if first_concat is None:
        first_concat = max(n // 2, 1)
    return {
        "backbone": list(range(0, first_concat)),
        "neck": list(range(first_concat, head_idx)),
        "head": [head_idx],
    }


def _unwrap(model):
    """Unwrap DP/DDP so layer indices match DetectionModel.model."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _grad_norm_by_group(model, groups: dict) -> dict:
    model = _unwrap(model)
    layers = model.model if hasattr(model, "model") else model
    out = {}
    for name, idxs in groups.items():
        total = 0.0
        for i in idxs:
            for p in layers[i].parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum().item())
        out[name] = total ** 0.5
    return out


class MetricsCallback:
    """Append one JSON line per training epoch to runs/<exp>/metrics/epochs.jsonl."""

    def __init__(self, metrics_dir: Path):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_path = self.metrics_dir / "epochs.jsonl"
        self.groups = None
        self.prev_grad = {"backbone": None, "neck": None, "head": None}
        self.epoch_grad_sum = {"backbone": 0.0, "neck": 0.0, "head": 0.0}
        self.epoch_grad_count = 0
        self._orig_optimizer_step = None

    def attach(self, model) -> None:
        model.add_callback("on_pretrain_routine_end", self.on_pretrain_routine_end)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def _record_grads(self, trainer) -> None:
        if self.groups is None:
            return
        norms = _grad_norm_by_group(trainer.model, self.groups)
        for k, v in norms.items():
            self.epoch_grad_sum[k] += v
        self.epoch_grad_count += 1

    def on_pretrain_routine_end(self, trainer) -> None:
        self.groups = _group_indices(_unwrap(trainer.model))
        self.epochs_path.write_text("")
        self.prev_grad = {"backbone": None, "neck": None, "head": None}
        self.epoch_grad_sum = {"backbone": 0.0, "neck": 0.0, "head": 0.0}
        self.epoch_grad_count = 0

        # Ultralytics runs on_train_batch_end *after* optimizer_step() → zero_grad(),
        # so grads are already cleared there. Sample just before the real step.
        orig = trainer.optimizer_step
        if not getattr(orig, "_metrics_grad_wrapped", False):

            def _step_with_grad_capture():
                self._record_grads(trainer)
                return orig()

            _step_with_grad_capture._metrics_grad_wrapped = True
            trainer.optimizer_step = _step_with_grad_capture
            self._orig_optimizer_step = orig

    def on_fit_epoch_end(self, trainer) -> None:
        epochs_planned = int(getattr(trainer, "epochs", 0) or 0)
        epoch_1based = int(trainer.epoch) + 1
        # Ignore post-training validation passes (no train batches / past last epoch).
        if epochs_planned and epoch_1based > epochs_planned:
            self.epoch_grad_sum = {"backbone": 0.0, "neck": 0.0, "head": 0.0}
            self.epoch_grad_count = 0
            return
        if self.epoch_grad_count == 0:
            return

        m = trainer.metrics or {}
        p = float(_metric(m, "metrics/precision(B)", "metrics/precision", default=0.0) or 0.0)
        r = float(_metric(m, "metrics/recall(B)", "metrics/recall", default=0.0) or 0.0)
        map50 = _metric(m, "metrics/mAP50(B)", "metrics/mAP50")
        map5095 = _metric(m, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        ap = None if map50 is None else float(map50)  # AP@0.5

        loss = {}
        for key, val in m.items():
            if "loss" in str(key).lower():
                try:
                    loss[str(key).split("/")[-1]] = float(val)
                except (TypeError, ValueError):
                    pass

        tloss = getattr(trainer, "tloss", None)
        loss_names = getattr(trainer, "loss_names", None)
        if tloss is not None:
            if isinstance(tloss, dict):
                for name, val in tloss.items():
                    try:
                        loss[str(name)] = float(val.item() if hasattr(val, "item") else val)
                    except (TypeError, ValueError):
                        continue
            elif loss_names is not None:
                if torch.is_tensor(tloss):
                    vals = tloss.detach().cpu().flatten().tolist()
                else:
                    vals = list(tloss)
                for name, val in zip(list(loss_names), vals):
                    try:
                        loss[str(name)] = float(val)
                    except (TypeError, ValueError):
                        continue

        grad_norm = {}
        grad_pct = {}
        for k in ("backbone", "neck", "head"):
            g = self.epoch_grad_sum[k] / self.epoch_grad_count
            grad_norm[k] = g
            prev = self.prev_grad[k]
            grad_pct[k] = None if prev is None else 100.0 * abs(g - prev) / (prev + 1e-12)
            self.prev_grad[k] = g
        self.epoch_grad_sum = {"backbone": 0.0, "neck": 0.0, "head": 0.0}
        self.epoch_grad_count = 0

        row = {
            "epoch": epoch_1based,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "ap": ap,
            "map50": None if map50 is None else float(map50),
            "map50_95": None if map5095 is None else float(map5095),
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "loss": loss,
            "grad_norm": grad_norm,
            "grad_pct": grad_pct,
        }
        with self.epochs_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self._print_epoch(row)

    @staticmethod
    def _fmt(x, nd: int = 4) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    def _print_epoch(self, row: dict) -> None:
        loss = row.get("loss") or {}
        loss_s = " ".join(f"{k}={self._fmt(v)}" for k, v in loss.items()) or "—"
        gp = row.get("grad_pct") or {}
        grad_s = "/".join(self._fmt(gp.get(k), 4) for k in ("backbone", "neck", "head"))
        print(
            f"\n[metrics] epoch {row['epoch']} (val)  "
            f"AP={self._fmt(row.get('ap'))}  "
            f"mAP50={self._fmt(row.get('map50'))}  "
            f"mAP50-95={self._fmt(row.get('map50_95'))}  "
            f"P={self._fmt(row.get('precision'))}  "
            f"R={self._fmt(row.get('recall'))}  "
            f"F1={self._fmt(row.get('f1'))}  "
            f"grad% bb/neck/head={grad_s}"
            + (f"  |  {loss_s}" if loss_s != "—" else "")
            + f"\n         → {self.epochs_path}\n",
            flush=True,
        )
