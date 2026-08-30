"""Train occupancy on a catalog of NPZs, one shape (file) at a time.

Each file keeps its own NPZ occupancy queries. Envelope points are
sampled on the OBJ when ``shape_encoder`` is ``surface``. Mini-batches
never mix two files. ``test_fraction`` of files is the test set.
Logs in ``runs/<id>/``, weights in ``models/<id>/best.pt``.
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
from torch.utils.data import DataLoader

from checkpointing import Checkpointer
from config import OccupancyConfig, as_data_relative, gpu_name, load_config
from dataset import (
    OccupancyMultiNpzDataset,
    OccupancyPointDataset,
    make_dataloader,
    split_train_test_files,
)
from metrics import occupancy_metrics
from occupancy_encoder import OccupancyEncoder
from occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND
from occupancy_mlp import OccupancyMLP
from occupancy_mlp import CHECKPOINT_KIND as MLP_KIND
from run_tracking import occupancy_config_snapshot, start_run

OccupancyModel = OccupancyMLP | OccupancyEncoder


def seed_everything(seed: int) -> None:
    """Make the train run repeatable (CPU and CUDA RNGs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _uses_surface(cfg: OccupancyConfig) -> bool:
    """True when YAML selected the envelope-conditioned head."""
    return str(cfg.shape_encoder).strip().lower() == "surface"


def _forward_batch(
    model: OccupancyModel,
    batch: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one loader batch to ``device`` and run the active head."""
    xyz = batch[0].to(device, non_blocking=False)
    y = batch[1].to(device, non_blocking=False)
    if len(batch) == 2:
        return model(xyz), y
    envelope = batch[2].to(device, non_blocking=False)
    shape_id = batch[3].to(device, non_blocking=False)
    return model(xyz, envelope, shape_id), y


def _eval_parts(
    model: OccupancyModel,
    parts: Sequence[OccupancyPointDataset],
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float, float, float]:
    """Mean metrics over **files** (each shape scored on its own points)."""
    if not parts:
        raise ValueError("test split is empty")
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    for part in parts:
        loader = make_dataloader(part, batch_size=batch_size, shuffle=False)
        scores = _eval_loader(model, loader, device)
        for i, value in enumerate(scores):
            totals[i] += value
    n = float(len(parts))
    return (
        totals[0] / n,
        totals[1] / n,
        totals[2] / n,
        totals[3] / n,
        totals[4] / n,
    )


def _eval_loader(
    model: OccupancyModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Mean occupancy metrics over ``loader`` (no grad)."""
    model.eval()
    running_acc = 0.0
    running_prec = 0.0
    running_rec = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            logits, y = _forward_batch(model, batch, device)
            scores = occupancy_metrics(logits, y)
            running_acc += scores.accuracy
            running_prec += scores.inside_precision
            running_rec += scores.inside_recall
            running_iou += scores.inside_iou
            running_f1 += scores.inside_f1
            n_batches += 1
    denom = max(n_batches, 1)
    return (
        running_acc / denom,
        running_prec / denom,
        running_rec / denom,
        running_iou / denom,
        running_f1 / denom,
    )


def _make_optimizer(
    name: str,
    model: nn.Module,
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


def _print_mesh_join(
    dataset: OccupancyMultiNpzDataset,
    data_dir: Path,
    *,
    preview: int = 5,
) -> None:
    """Print NPZ → OBJ rows so a batch can be traced to its mesh."""
    parts = dataset.parts
    shown = parts[:preview]
    for part in shown:
        if part.mesh_path is None:
            print(f"  {part.npz_path.name} mesh=None key={part.mesh_key}")
            continue
        obj = as_data_relative(part.mesh_path, data_dir)
        n_v = int(part.vertices.shape[0]) if part.vertices is not None else 0
        n_f = int(part.faces.shape[0]) if part.faces is not None else 0
        print(
            f"  {part.npz_path.name} -> {obj} "
            f"key={part.mesh_key} V={n_v} F={n_f}"
        )
    extra = len(parts) - len(shown)
    if extra > 0:
        print(f"  ... ({extra} more files)")


@dataclass
class TrainMultiResult:
    """Metrics and artifact paths for one multi-NPZ train."""

    run_dir: Path
    model_dir: Path
    best_path: Path
    n_files: int
    n_train: int
    n_test: int
    losses: list[float]
    accuracies: list[float]
    test_accuracies: list[float]
    best_epoch: int
    best_metric: float


def _checkpoint_payload(
    model: OccupancyModel,
    dataset: OccupancyMultiNpzDataset,
    cfg: OccupancyConfig,
) -> dict:
    """Weights plus per-mesh AABB so infer can rebuild the same maps."""
    parts = []
    for part in dataset.parts:
        parts.append(
            {
                "npz": as_data_relative(part.npz_path, cfg.data_dir),
                "mesh": (
                    as_data_relative(part.mesh_path, cfg.data_dir)
                    if part.mesh_path is not None
                    else None
                ),
                "center": np.asarray(part.center, dtype=np.float32),
                "scale": float(part.scale),
            }
        )
    kind = ENCODER_KIND if _uses_surface(cfg) else MLP_KIND
    return {
        "kind": kind,
        "state_dict": model.state_dict(),
        "hidden": int(cfg.hidden),
        "depth": int(cfg.depth),
        "shape_encoder": str(cfg.shape_encoder),
        "n_surface": int(cfg.n_surface),
        "npz_paths": [as_data_relative(p, cfg.data_dir) for p in dataset.npz_paths],
        "parts": parts,
    }


def _selection_score(
    metric_name: str,
    *,
    test_acc: float,
    test_iou: float,
) -> float:
    """Map YAML ``checkpoint_metric`` to the scalar Checkpointer compares."""
    name = metric_name.strip()
    if name in ("test_acc", "val_acc"):
        return float(test_acc)
    if name in ("test_iou", "val_iou"):
        return float(test_iou)
    raise ValueError(
        f"unsupported checkpoint_metric {metric_name!r} "
        "(train_multi_npz logs test_acc and test_iou)"
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
    n_surface = int(cfg.n_surface) if _uses_surface(cfg) else None
    if npz_paths is None:
        dataset = OccupancyMultiNpzDataset.from_catalog(
            cfg.data_dir,
            npz_glob=cfg.npz_glob,
            npz_paths=cfg.npz_paths or None,
            max_files_per_shape=cfg.max_files_per_shape,
            n_surface=n_surface,
            seed=cfg.seed,
        )
    else:
        dataset = OccupancyMultiNpzDataset(
            npz_paths,
            data_dir=cfg.data_dir,
            n_surface=n_surface,
            seed=cfg.seed,
        )

    if len(dataset.parts) < 2:
        raise ValueError(
            f"need at least 2 files to hold out whole shapes, got {len(dataset.parts)}"
        )
    train_idx, test_idx = split_train_test_files(
        len(dataset.parts), cfg.test_fraction, cfg.seed
    )
    train_parts = [dataset.parts[int(i)] for i in train_idx.tolist()]
    test_parts = [dataset.parts[int(i)] for i in test_idx.tolist()]

    if _uses_surface(cfg):
        model: OccupancyModel = OccupancyEncoder(
            hidden=cfg.hidden,
            depth=cfg.depth,
            latent_dim=cfg.hidden,
        ).to(cfg.device)
    else:
        model = OccupancyMLP(hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(cfg.optimizer, model, lr)

    n_train_files = len(train_parts)
    n_test_files = len(test_parts)
    n_train = sum(len(part) for part in train_parts)
    n_test = sum(len(part) for part in test_parts)
    print(f"files={len(dataset.parts)} meshes={dataset.n_meshes}")
    _print_mesh_join(dataset, cfg.data_dir)
    print(
        f"split=shape train_files={n_train_files} test_files={n_test_files} "
        f"N={dataset.n_points} n_train={n_train} n_test={n_test} "
        f"test_fraction={cfg.test_fraction} device={cfg.device} "
        f"gpu={gpu_name(cfg.device)} "
        f"hidden={cfg.hidden} depth={cfg.depth} "
        f"shape_encoder={cfg.shape_encoder} n_surface={cfg.n_surface}"
    )
    print(
        f"epochs={n_epochs} batch_size={cfg.batch_size} "
        f"optimizer={cfg.optimizer} lr={lr}"
    )
    for part in test_parts:
        print(f"  test_file={part.npz_path.name} N={len(part)}")

    name = run_name if run_name is not None else cfg.run_name
    with start_run(name, root=root, t0=t0, clock_start=clock_start) as run:
        snap = occupancy_config_snapshot(cfg)
        snap["total"] = n_epochs
        run.write_config(snap)
        run.write_catalog(dataset.npz_paths, data_dir=cfg.data_dir)
        run.update_config(
            {
                "n_files": len(dataset.parts),
                "n_meshes": dataset.n_meshes,
                "n_points": dataset.n_points,
                "n_train_files": n_train_files,
                "n_test_files": n_test_files,
                "n_train": n_train,
                "n_test": n_test,
                "split": "shape",
            }
        )
        saver = Checkpointer(
            run,
            total_epochs=n_epochs,
            metric_name=cfg.checkpoint_metric,
            root=root,
        )
        losses: list[float] = []
        accuracies: list[float] = []
        test_accuracies: list[float] = []
        last_result = None
        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            n_batches = 0
            # Shuffle file order each epoch; points stay inside their file.
            order = torch.randperm(len(train_parts)).tolist()
            for part_i in order:
                part = train_parts[part_i]
                loader = make_dataloader(
                    part, batch_size=cfg.batch_size, shuffle=True
                )
                for batch in loader:
                    logits, y = _forward_batch(model, batch, cfg.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item())
                    running_acc += occupancy_metrics(logits.detach(), y).accuracy
                    n_batches += 1
            mean_loss = running_loss / max(n_batches, 1)
            mean_acc = running_acc / max(n_batches, 1)
            test_acc, _prec, _rec, test_iou, test_f1 = _eval_parts(
                model, test_parts, cfg.device, cfg.batch_size
            )
            wall = time.perf_counter() - t0
            losses.append(mean_loss)
            accuracies.append(mean_acc)
            test_accuracies.append(test_acc)
            score = _selection_score(
                cfg.checkpoint_metric, test_acc=test_acc, test_iou=test_iou
            )
            last_result = saver.save(
                _checkpoint_payload(model, dataset, cfg),
                epoch=epoch,
                metric=score,
            )
            run.log_epoch(
                epoch=epoch,
                loss=mean_loss,
                train_acc=mean_acc,
                val_acc=test_acc,
                wall_seconds=wall,
                best_epoch=last_result.best_epoch,
                best_metric=last_result.best_metric,
                test_acc=test_acc,
                test_iou=test_iou,
                test_f1=test_f1,
            )
            print(
                f"epoch {epoch:03d}  loss={mean_loss:.6f}  "
                f"train_acc={mean_acc:.4f}  test_acc={test_acc:.4f}  "
                f"test_iou={test_iou:.4f}  test_f1={test_f1:.4f}  "
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
            n_test=n_test,
            losses=losses,
            accuracies=accuracies,
            test_accuracies=test_accuracies,
            best_epoch=last_result.best_epoch,
            best_metric=last_result.best_metric,
        )


if __name__ == "__main__":
    train_multi_npz(load_config())
