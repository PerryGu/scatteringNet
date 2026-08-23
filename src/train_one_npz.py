"""Overfit OccupancyMLP on a single NPZ (v2 minimal Step 6).

Uses every query point for training (no val split). Checkpoints store the
AABB ``center`` / ``scale`` so inference can reuse the same map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import OccupancyConfig, load_config
from dataset import OccupancyPointDataset, make_dataloader
from occupancy_mlp import OccupancyMLP

CHECKPOINT_KIND = "occupancy_mlp"
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = _REPO_ROOT / "models" / "one_npz.pt"
_SAMPLE_RELATIVE = Path("exports") / "dataset_test" / "sphere__raycast_z_raut_s0.15_inout.npz"


@dataclass
class TrainRunResult:
    """Per-epoch train metrics plus the checkpoint path."""

    out_path: Path
    losses: list[float]
    accuracies: list[float]


def seed_everything(seed: int) -> None:
    """Make the overfit run repeatable (CPU and CUDA RNGs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of points with (sigmoid >= 0.5) matching the 0/1 label."""
    pred = (logits.sigmoid() >= 0.5).to(y.dtype)
    return float((pred == y).float().mean().item())


def save_checkpoint(
    path: Path,
    *,
    model: OccupancyMLP,
    dataset: OccupancyPointDataset,
    hidden: int,
    depth: int,
) -> None:
    """Write weights plus the normalization used to train them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": CHECKPOINT_KIND,
        "state_dict": model.state_dict(),
        "center": np.asarray(dataset.center, dtype=np.float32),
        "scale": float(dataset.scale),
        "hidden": int(hidden),
        "depth": int(depth),
    }
    torch.save(payload, path)


def train_one_npz(
    npz_path: Path,
    cfg: OccupancyConfig,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    out_path: Path = DEFAULT_OUT_PATH,
) -> TrainRunResult:
    """
    Train on all points in ``npz_path`` and save ``models/one_npz.pt``.

    Returns per-epoch mean loss / accuracy and the checkpoint path.
    Prints the same metrics each epoch.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if cfg.device.type != "cuda":
        # Short overfit is allowed on CPU; do not pretend this is a production run.
        print(f"Warning: training on {cfg.device} (CUDA not in use).")

    seed_everything(cfg.seed)
    dataset = OccupancyPointDataset(npz_path)
    loader: DataLoader = make_dataloader(dataset, shuffle=True)

    model = OccupancyMLP(hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    # BCE-with-logits: labels are float {0,1} with shape (B, 1), matching logits.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"npz={npz_path}")
    print(f"N={len(dataset)} device={cfg.device} hidden={cfg.hidden} depth={cfg.depth}")
    print(f"epochs={epochs} batch_size={loader.batch_size} lr={lr}")

    model.train()
    losses: list[float] = []
    accuracies: list[float] = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        for xyz, y in loader:
            xyz = xyz.to(cfg.device, non_blocking=False)
            y = y.to(cfg.device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xyz)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            running_acc += _batch_accuracy(logits.detach(), y)
            n_batches += 1
        mean_loss = running_loss / max(n_batches, 1)
        mean_acc = running_acc / max(n_batches, 1)
        losses.append(mean_loss)
        accuracies.append(mean_acc)
        print(f"epoch {epoch:03d}  loss={mean_loss:.6f}  train_acc={mean_acc:.4f}")

    save_checkpoint(
        out_path,
        model=model,
        dataset=dataset,
        hidden=cfg.hidden,
        depth=cfg.depth,
    )
    print(f"saved={out_path}")
    return TrainRunResult(out_path=out_path, losses=losses, accuracies=accuracies)


if __name__ == "__main__":
    cfg = load_config()
    sample = cfg.data_dir / _SAMPLE_RELATIVE
    train_one_npz(sample, cfg)
