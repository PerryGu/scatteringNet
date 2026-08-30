"""Tests for occupancy metrics from logits vs labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics import accuracy_from_logits, occupancy_metrics


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
