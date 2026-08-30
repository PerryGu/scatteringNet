"""Occupancy classification metrics from logits vs labels.

The model emits raw logits (no sigmoid in ``forward``). Metrics apply
sigmoid only at decision time so they stay consistent with
``BCEWithLogitsLoss`` and with inference (threshold ``0.5``).

Inside (label ``1``) is the product-relevant class: points that belong
in the interior. Precision / recall are therefore reported for that
class only. No extra packages (sklearn, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class OccupancyMetrics:
    """Pointwise occupancy scores for one batch or a full set of queries."""

    accuracy: float
    inside_precision: float
    inside_recall: float
    inside_iou: float
    inside_f1: float


def _safe_div(numerator: float, denominator: float) -> float:
    """Return ``0.0`` when the count in the denominator is zero (no sklearn)."""
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _flatten_pair(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """
    Collapse ``(B, 1)`` or ``(B,)`` logits / labels to a shared 1-D view.

    Last dim of logits is 1 when it comes from ``OccupancyMLP``; labels from
    the Dataset match that. A 1-D label vector is accepted so callers do not
    have to unsqueeze.
    """
    if logits.numel() != labels.numel():
        raise ValueError(
            f"logits and labels must have the same number of elements, "
            f"got logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    return logits.reshape(-1), labels.reshape(-1)


def occupancy_metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    threshold: float = 0.5,
) -> OccupancyMetrics:
    """
    Accuracy plus inside precision / recall from occupancy logits.

    Parameters
    ----------
    logits:
        Unnormalized scores, shape ``(B, 1)`` or ``(B,)``. Positive → inside.
    labels:
        Float ``{0, 1}`` with the same number of elements as ``logits``.
    threshold:
        Decision cut on ``sigmoid(logit)``. Default ``0.5`` matches inference.

    Returns
    -------
    OccupancyMetrics
        Scalar floats on CPU (safe to print or average across batches).
    """
    # Sigmoid here only: training still uses BCE-with-logits on raw logits.
    tp, fp, fn, correct, n = occupancy_counts(logits, labels, threshold=threshold)
    return occupancy_metrics_from_counts(tp=tp, fp=fp, fn=fn, correct=correct, n=n)


def occupancy_counts(
    logits: Tensor,
    labels: Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Return ``(tp, fp, fn, correct, n)`` for a micro-average over points."""
    logits_flat, labels_flat = _flatten_pair(logits, labels)
    pred_inside = logits_flat.sigmoid() >= threshold
    true_inside = labels_flat > 0.5
    pred_f = pred_inside.to(dtype=torch.float32)
    true_f = true_inside.to(dtype=torch.float32)
    tp = float((pred_f * true_f).sum().item())
    fp = float((pred_f * (1.0 - true_f)).sum().item())
    fn = float(((1.0 - pred_f) * true_f).sum().item())
    correct = float((pred_inside == true_inside).to(dtype=torch.float32).sum().item())
    n = float(pred_inside.numel())
    return tp, fp, fn, correct, n


def occupancy_metrics_from_counts(
    *,
    tp: float,
    fp: float,
    fn: float,
    correct: float,
    n: float,
) -> OccupancyMetrics:
    """Build metrics from accumulated confusion counts (point micro-average)."""
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return OccupancyMetrics(
        accuracy=_safe_div(correct, n),
        inside_precision=precision,
        inside_recall=recall,
        inside_iou=_safe_div(tp, tp + fp + fn),
        inside_f1=_safe_div(2.0 * precision * recall, precision + recall),
    )


def accuracy_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Fraction of points whose thresholded sigmoid matches the 0/1 label.

    Parameters
    ----------
    logits, labels, threshold:
        Same meaning as :func:`occupancy_metrics`.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    return occupancy_metrics(logits, labels, threshold=threshold).accuracy


if __name__ == "__main__":
    # Deterministic 8-point batch: 4 true insides, 4 true outsides, all correct.
    demo_logits = torch.tensor(
        [[4.0], [3.0], [2.0], [1.0], [-1.0], [-2.0], [-3.0], [-4.0]]
    )
    demo_labels = torch.tensor(
        [[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]
    )
    scores = occupancy_metrics(demo_logits, demo_labels)
    print(f"accuracy={scores.accuracy:.4f}")
    print(f"inside_precision={scores.inside_precision:.4f}")
    print(f"inside_recall={scores.inside_recall:.4f}")
    print(f"inside_iou={scores.inside_iou:.4f}")
    print(f"inside_f1={scores.inside_f1:.4f}")
