"""Normalized Wasserstein Distance (NWD) bbox loss for YOLO26 tiny-object training.

Paper: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection" (https://arxiv.org/abs/2110.13389).

Box regression becomes:
  box = iou_ratio * CIoU + (1 - iou_ratio) * (1 - NWD)
with NWD = exp(-sqrt(W₂²) / C), C default 12.8 (paper AI-TOD average size).

``nwd_loss`` is a separate logged term so MetricsCallback / Ultralytics tloss
surface it each epoch.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


def wasserstein_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
    constant: float = 12.8,
) -> torch.Tensor:
    """NWD similarity in (0, 1] for xyxy boxes; higher = closer.

    Args:
        pred / target: (n, 4) xyxy.
        constant: dataset scale C (paper default 12.8).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred.T
    b2_x1, b2_y1, b2_x2, b2_y2 = target.T
    w1 = (b1_x2 - b1_x1).clamp(min=0) + eps
    h1 = (b1_y2 - b1_y1).clamp(min=0) + eps
    w2 = (b2_x2 - b2_x1).clamp(min=0) + eps
    h2 = (b2_y2 - b2_y1).clamp(min=0) + eps
    cx1, cy1 = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    cx2, cy2 = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5
    center = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)
    wh = ((w1 - w2).pow(2) + (h1 - h2).pow(2)) * 0.25
    return torch.exp(-torch.sqrt(center + wh + eps) / constant)


class NWDBboxLoss(BboxLoss):
    """CIoU + NWD bbox criterion; returns (iou_loss, dfl_or_l1, nwd_loss)."""

    def __init__(self, reg_max: int = 16, constant: float = 12.8):
        super().__init__(reg_max)
        self.constant = float(constant)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fg_idx = fg_mask.nonzero(as_tuple=True)
        if fg_idx[0].numel() == 0:
            zero = pred_bboxes.sum() * 0.0
            return zero, zero, zero

        weight = target_scores[fg_idx].sum(-1, keepdim=True)
        pred_fg = pred_bboxes[fg_idx]
        tgt_fg = target_bboxes[fg_idx]

        iou = bbox_iou(pred_fg, tgt_fg, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        nwd = wasserstein_similarity(pred_fg, tgt_fg, constant=self.constant)
        if nwd.ndim == 1:
            nwd = nwd.unsqueeze(-1)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_idx].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_idx]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_idx], target_ltrb[fg_idx], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl, loss_nwd


class NWDDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with NWD box term; logs ``nwd_loss`` separately."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        constant: float = 12.8,
        iou_ratio: float = 0.5,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.constant = float(constant)
        self.iou_ratio = float(iou_ratio)
        m = model.model[-1]
        self.bbox_loss = NWDBboxLoss(m.reg_max, constant=self.constant).to(self.device)
        # 4 terms so trainer.tloss / MetricsCallback show nwd_loss
        dfl_name = "dfl_loss" if self.use_dfl else "l1_loss"
        self.loss_names = ("box_loss", "cls_loss", dfl_name, "nwd_loss")

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(4, device=self.device)  # box(iou), cls, dfl/l1, nwd
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        loss_iou, loss_dfl, loss_nwd = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        # Split IoU / NWD so loss.sum() does not double-count; both scaled by hyp.box
        r = self.iou_ratio
        loss[0] = loss_iou * r
        loss[3] = loss_nwd * (1.0 - r)
        loss[2] = loss_dfl

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.box

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


def install_nwd_criterion(
    yolo_model,
    constant: float = 12.8,
    iou_ratio: float = 0.5,
) -> None:
    """Patch YOLO DetectionModel so training uses NWDDetectionLoss (E2E-aware)."""

    def make_loss(m, tal_topk: int = 10, tal_topk2: int | None = None):
        return NWDDetectionLoss(
            m,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            constant=constant,
            iou_ratio=iou_ratio,
        )

    def init_criterion(self):
        if getattr(self.model[-1], "one2one_cv2", None) is not None:
            return E2ELoss(self, loss_fn=make_loss)
        return make_loss(self)

    nn_model = yolo_model.model if hasattr(yolo_model, "model") else yolo_model
    nn_model.init_criterion = types.MethodType(init_criterion, nn_model)
    nn_model.criterion = None
