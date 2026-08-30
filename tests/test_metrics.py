"""Tests for occupancy metrics from logits vs labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics import (
    accuracy_from_logits,
    occupancy_counts,
    occupancy_metrics,
    occupancy_metrics_from_counts,
)


class OccupancyMetricsTests(unittest.TestCase):
    def test_perfect_batch_scores_one(self) -> None:
        logits = torch.tensor([[4.0], [3.0], [-2.0], [-1.0]])
        labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        scores = occupancy_metrics(logits, labels)
        self.assertAlmostEqual(scores.accuracy, 1.0)
        self.assertAlmostEqual(scores.inside_precision, 1.0)
        self.assertAlmostEqual(scores.inside_recall, 1.0)
        self.assertAlmostEqual(scores.inside_iou, 1.0)
        self.assertAlmostEqual(scores.inside_f1, 1.0)
        self.assertAlmostEqual(accuracy_from_logits(logits, labels), 1.0)

    def test_known_confusion_counts(self) -> None:
        # Pred: 1, 1, 0, 0   True: 1, 0, 1, 0  → TP=1 FP=1 FN=1 TN=1
        logits = torch.tensor([[2.0], [1.0], [-1.0], [-2.0]])
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        scores = occupancy_metrics(logits, labels)
        self.assertAlmostEqual(scores.accuracy, 0.5)
        self.assertAlmostEqual(scores.inside_precision, 0.5)
        self.assertAlmostEqual(scores.inside_recall, 0.5)
        self.assertAlmostEqual(scores.inside_iou, 1.0 / 3.0)
        self.assertAlmostEqual(scores.inside_f1, 0.5)

    def test_accepts_1d_labels_matching_n(self) -> None:
        logits = torch.tensor([[5.0], [-5.0]])
        labels = torch.tensor([1.0, 0.0])
        scores = occupancy_metrics(logits, labels)
        self.assertAlmostEqual(scores.accuracy, 1.0)

    def test_zero_predicted_insides_precision_is_zero(self) -> None:
        logits = torch.tensor([[-1.0], [-2.0]])
        labels = torch.tensor([[1.0], [0.0]])
        scores = occupancy_metrics(logits, labels)
        self.assertAlmostEqual(scores.inside_precision, 0.0)
        self.assertAlmostEqual(scores.inside_recall, 0.0)
        self.assertAlmostEqual(scores.accuracy, 0.5)

    def test_from_counts_is_point_micro_average(self) -> None:
        # Batch A: 2/2 correct. Batch B: 1/4 correct. Mean of accs = 0.625;
        # point micro-average is 3/6 = 0.5.
        a = occupancy_counts(
            torch.tensor([[4.0], [-4.0]]),
            torch.tensor([[1.0], [0.0]]),
        )
        b = occupancy_counts(
            torch.tensor([[4.0], [4.0], [4.0], [-4.0]]),
            torch.tensor([[1.0], [0.0], [0.0], [1.0]]),
        )
        tp = a[0] + b[0]
        fp = a[1] + b[1]
        fn = a[2] + b[2]
        correct = a[3] + b[3]
        n = a[4] + b[4]
        scores = occupancy_metrics_from_counts(
            tp=tp, fp=fp, fn=fn, correct=correct, n=n
        )
        self.assertAlmostEqual(scores.accuracy, 0.5)
        self.assertNotAlmostEqual(scores.accuracy, 0.625)

    def test_rejects_numel_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            occupancy_metrics(torch.zeros(4, 1), torch.zeros(3, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_metrics_on_cuda(self) -> None:
        logits = torch.tensor([[4.0], [-4.0]], device="cuda")
        labels = torch.tensor([[1.0], [0.0]], device="cuda")
        scores = occupancy_metrics(logits, labels)
        self.assertAlmostEqual(scores.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
