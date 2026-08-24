"""Train OccupancyMLP on one NPZ with a point hold-out (v2 minimal Step 9).

A random fraction of *points* from the same file is held out for val.
AABB ``center`` / ``scale`` are still computed on the full cloud so the
checkpoint map matches Step 8 inference on that NPZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import OccupancyConfig, load_config, sample_npz_path
from dataset import OccupancyPointDataset, make_dataloader, split_train_val_indices
from metrics import occupancy_metrics
from occupancy_mlp import OccupancyMLP

# Checkpoint schema tag (not an experiment knob; stay out of YAML).
CHECKPOINT_KIND = "occupancy_mlp"


@dataclass
class TrainRunResult:
    """Per-epoch train/val metrics plus the checkpoint path."""

    out_path: Path
    losses: list[float]
    accuracies: list[float]
    val_accuracies: list[float]
    n_train: int
    n_val: int


def seed_everything(seed: int) -> None:
    """Make the overfit run repeatable (CPU and CUDA RNGs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _eval_loader(
    model: OccupancyMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Mean occupancy metrics over ``loader`` (no grad)."""
    model.eval()
    running_acc = 0.0
    running_prec = 0.0
    running_rec = 0.0
    n_batches = 0
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            logits = model(xyz)
            scores = occupancy_metrics(logits, y)
            running_acc += scores.accuracy
            running_prec += scores.inside_precision
            running_rec += scores.inside_recall
            n_batches += 1
    denom = max(n_batches, 1)
    return running_acc / denom, running_prec / denom, running_rec / denom


def train_one_npz(
    npz_path: Path,
    cfg: OccupancyConfig,
) -> TrainRunResult:
    """
    Train on a random subset of ``npz_path`` and save ``cfg.checkpoint_path``.

    ``cfg.val_fraction`` of points are held out (same mesh). Returns per-epoch
    train loss / acc, val acc, split sizes, and the checkpoint path.
    """
    epochs = cfg.epochs
    lr = cfg.lr
    out_path = cfg.checkpoint_path
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if cfg.device.type != "cuda":
        # Short overfit is allowed on CPU; do not pretend this is a production run.
        print(f"Warning: training on {cfg.device} (CUDA not in use).")

    seed_everything(cfg.seed)
    dataset = OccupancyPointDataset(npz_path)
    train_idx, val_idx = split_train_val_indices(
        len(dataset), cfg.val_fraction, cfg.seed
    )
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    train_loader: DataLoader = make_dataloader(train_set, shuffle=True)
    val_loader: DataLoader = make_dataloader(val_set, shuffle=False)

    model = OccupancyMLP(hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    # BCE-with-logits: labels are float {0,1} with shape (B, 1), matching logits.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(train_set)
    n_val = len(val_set)
    print(f"npz={npz_path}")
    print(
        f"N={len(dataset)} n_train={n_train} n_val={n_val} "
        f"val_fraction={cfg.val_fraction} device={cfg.device} "
        f"hidden={cfg.hidden} depth={cfg.depth}"
    )
    print(f"epochs={epochs} batch_size={train_loader.batch_size} lr={lr}")

    losses: list[float] = []
    accuracies: list[float] = []
    val_accuracies: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_prec = 0.0
        running_rec = 0.0
        n_batches = 0
        for xyz, y in train_loader:
            xyz = xyz.to(cfg.device, non_blocking=False)
            y = y.to(cfg.device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xyz)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            # Decision metrics live in metrics.py (not ad-hoc sigmoid here).
            batch_scores = occupancy_metrics(logits.detach(), y)
            running_acc += batch_scores.accuracy
            running_prec += batch_scores.inside_precision
            running_rec += batch_scores.inside_recall
            n_batches += 1
        mean_loss = running_loss / max(n_batches, 1)
        mean_acc = running_acc / max(n_batches, 1)
        mean_prec = running_prec / max(n_batches, 1)
        mean_rec = running_rec / max(n_batches, 1)
        val_acc, _val_prec, _val_rec = _eval_loader(model, val_loader, cfg.device)
        losses.append(mean_loss)
        accuracies.append(mean_acc)
        val_accuracies.append(val_acc)
        print(
            f"epoch {epoch:03d}  loss={mean_loss:.6f}  "
            f"train_acc={mean_acc:.4f}  val_acc={val_acc:.4f}  "
            f"inside_prec={mean_prec:.4f}  inside_rec={mean_rec:.4f}"
        )

    save_checkpoint(
        out_path,
        model=model,
        dataset=dataset,
        hidden=cfg.hidden,
        depth=cfg.depth,
    )
    print(f"saved={out_path}")
    return TrainRunResult(
        out_path=out_path,
        losses=losses,
        accuracies=accuracies,
        val_accuracies=val_accuracies,
        n_train=n_train,
        n_val=n_val,
    )


if __name__ == "__main__":
    cfg = load_config()
    train_one_npz(sample_npz_path(cfg), cfg)
