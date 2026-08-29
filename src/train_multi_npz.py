"""Train OccupancyMLP on a catalog of occupancy NPZs.

xyz-only MLP, pooled queries from the catalog, logs in ``runs/<id>/``,
weights in ``models/<id>/best.pt``. Val is a random **point** split of
the pooled cloud (loop health, not shape holdout). YAML ``epochs`` is
the train length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from checkpointing import Checkpointer
from config import OccupancyConfig, as_data_relative, gpu_name, load_config
from dataset import OccupancyMultiNpzDataset, make_dataloader, split_train_val_indices
from metrics import occupancy_metrics
from occupancy_mlp import CHECKPOINT_KIND, OccupancyMLP
from run_tracking import occupancy_config_snapshot, start_run


def seed_everything(seed: int) -> None:
    """Make the train run repeatable (CPU and CUDA RNGs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _make_optimizer(
    name: str,
    model: OccupancyMLP,
    lr: float,
) -> torch.optim.Optimizer:
    """Build the YAML-selected optimizer (adam / adamw / sgd)."""
    params = model.parameters()
    key = str(name).strip().lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"unsupported optimizer {name!r}")


@dataclass
class TrainMultiResult:
    """Metrics and artifact paths for one multi-NPZ train."""

    run_dir: Path
    model_dir: Path
    best_path: Path
    n_files: int
    n_train: int
    n_val: int
    losses: list[float]
    accuracies: list[float]
    val_accuracies: list[float]
    best_epoch: int
    best_metric: float


def _checkpoint_payload(
    model: OccupancyMLP,
    dataset: OccupancyMultiNpzDataset,
    cfg: OccupancyConfig,
) -> dict:
    """Weights plus per-mesh AABB so infer can rebuild the same maps."""
    parts = []
    for part in dataset.parts:
        parts.append(
            {
                "npz": as_data_relative(part.npz_path, cfg.data_dir),
                "center": np.asarray(part.center, dtype=np.float32),
                "scale": float(part.scale),
            }
        )
    return {
        "kind": CHECKPOINT_KIND,
        "state_dict": model.state_dict(),
        "hidden": int(cfg.hidden),
        "depth": int(cfg.depth),
        "npz_paths": [as_data_relative(p, cfg.data_dir) for p in dataset.npz_paths],
        "parts": parts,
    }


def _selection_score(metric_name: str, *, val_acc: float) -> float:
    """Map YAML ``checkpoint_metric`` to the scalar Checkpointer compares."""
    name = metric_name.strip()
    if name == "val_acc":
        return float(val_acc)
    raise ValueError(
        f"unsupported checkpoint_metric {metric_name!r} "
        "(train_multi_npz only logs val_acc)"
    )


def train_multi_npz(
    cfg: OccupancyConfig,
    *,
    npz_paths: Sequence[Path | str] | None = None,
    epochs: int | None = None,
    root: Path | None = None,
    run_name: str | None = None,
) -> TrainMultiResult:
    """
    Train OccupancyMLP on the catalog.

    Parameters
    ----------
    cfg:
        YAML knobs plus detected device. ``epochs`` is the train length.
    npz_paths:
        Explicit files; default is ``resolve_npz_catalog`` from ``cfg``.
    epochs:
        Override; default ``cfg.epochs``.
    root:
        Repo root for ``runs/`` and ``models/`` (tests pass a temp dir).
    run_name:
        Folder suffix; default ``cfg.run_name``.
    """
    n_epochs = int(epochs if epochs is not None else cfg.epochs)
    lr = float(cfg.lr)
    if n_epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {n_epochs}")
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if cfg.device.type != "cuda":
        print(f"Warning: training on {cfg.device} (CUDA not in use).")

    seed_everything(cfg.seed)
    clock_start = datetime.now()
    t0 = time.perf_counter()
    print(f"started={clock_start.isoformat(timespec='seconds')}")
    if npz_paths is None:
        dataset = OccupancyMultiNpzDataset.from_catalog(
            cfg.data_dir,
            npz_glob=cfg.npz_glob,
            npz_paths=cfg.npz_paths or None,
            max_files_per_shape=cfg.max_files_per_shape,
        )
    else:
        dataset = OccupancyMultiNpzDataset(npz_paths)

    train_idx, val_idx = split_train_val_indices(
        len(dataset), cfg.val_fraction, cfg.seed
    )
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    train_loader = make_dataloader(
        train_set, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        val_set, batch_size=cfg.batch_size, shuffle=False
    )

    model = OccupancyMLP(hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(cfg.optimizer, model, lr)

    n_train = len(train_set)
    n_val = len(val_set)
    print(f"files={len(dataset.npz_paths)}")
    print(
        f"N={len(dataset)} n_train={n_train} n_val={n_val} "
        f"val_fraction={cfg.val_fraction} device={cfg.device} "
        f"gpu={gpu_name(cfg.device)} "
        f"hidden={cfg.hidden} depth={cfg.depth}"
    )
    print(
        f"epochs={n_epochs} batch_size={train_loader.batch_size} "
        f"optimizer={cfg.optimizer} lr={lr}"
    )

    name = run_name if run_name is not None else cfg.run_name
    with start_run(name, root=root, t0=t0, clock_start=clock_start) as run:
        snap = occupancy_config_snapshot(cfg)
        snap["total"] = n_epochs
        run.write_config(snap)
        run.write_catalog(dataset.npz_paths, data_dir=cfg.data_dir)
        saver = Checkpointer(
            run,
            total_epochs=n_epochs,
            metric_name=cfg.checkpoint_metric,
            root=root,
        )
        losses: list[float] = []
        accuracies: list[float] = []
        val_accuracies: list[float] = []
        last_result = None
        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_acc = 0.0
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
                running_acc += occupancy_metrics(logits.detach(), y).accuracy
                n_batches += 1
            mean_loss = running_loss / max(n_batches, 1)
            mean_acc = running_acc / max(n_batches, 1)
            val_acc, _prec, _rec = _eval_loader(model, val_loader, cfg.device)
            wall = time.perf_counter() - t0
            losses.append(mean_loss)
            accuracies.append(mean_acc)
            val_accuracies.append(val_acc)
            score = _selection_score(cfg.checkpoint_metric, val_acc=val_acc)
            last_result = saver.save(
                _checkpoint_payload(model, dataset, cfg),
                epoch=epoch,
                metric=score,
            )
            run.log_epoch(
                epoch=epoch,
                loss=mean_loss,
                train_acc=mean_acc,
                val_acc=val_acc,
                wall_seconds=wall,
                best_epoch=last_result.best_epoch,
                best_metric=last_result.best_metric,
            )
            print(
                f"epoch {epoch:03d}  loss={mean_loss:.6f}  "
                f"train_acc={mean_acc:.4f}  val_acc={val_acc:.4f}  "
                f"best_epoch={last_result.best_epoch}  "
                f"best={last_result.best_metric:.4f}"
            )

        assert last_result is not None
        timing = run.write_timing()
        print(f"run_dir={run.dir.resolve()}")
        print(f"model_dir={saver.dir.resolve()}")
        print(f"best={saver.best_path}")
        print(f"wall={timing['wall']} ({timing['wall_seconds']}s)")
        print(f"finished={timing['finished_at']}")
        return TrainMultiResult(
            run_dir=run.dir,
            model_dir=saver.dir,
            best_path=saver.best_path,
            n_files=len(dataset.npz_paths),
            n_train=n_train,
            n_val=n_val,
            losses=losses,
            accuracies=accuracies,
            val_accuracies=val_accuracies,
            best_epoch=last_result.best_epoch,
            best_metric=last_result.best_metric,
        )


if __name__ == "__main__":
    train_multi_npz(load_config())
