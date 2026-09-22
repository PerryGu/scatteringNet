"""Train occupancy on a catalog of NPZs, one shape (file) at a time.

Each file keeps its own NPZ occupancy queries. Envelope points are
sampled on the OBJ when ``shape_encoder`` is ``surface``. Mini-batches
never mix two files. ``val_fraction`` of **meshes** is the val set
(all NPZs of one OBJ stay on one side; used to select ``best.pt``).
Logs in ``runs/<id>/``, weights in ``models/<id>/best.pt``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scatteringnet.checkpointing import Checkpointer
from scatteringnet.config import (
    OccupancyConfig,
    as_data_relative,
    encoder_knn_local_dim,
    encoder_latent_dim,
    gpu_name,
    load_config,
)
from scatteringnet.dataset import (
    OccupancyMultiNpzDataset,
    OccupancyPointDataset,
    make_dataloader,
    mesh_split_key,
    split_train_test_by_mesh,
)
from scatteringnet.infer_multi_npz import load_occupancy_model, resolve_best_pt
from scatteringnet.metrics import occupancy_counts, occupancy_metrics_from_counts
from scatteringnet.occupancy_encoder import OccupancyEncoder
from scatteringnet.occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND
from scatteringnet.occupancy_mlp import OccupancyMLP
from scatteringnet.occupancy_mlp import CHECKPOINT_KIND as MLP_KIND
from scatteringnet.run_tracking import occupancy_config_snapshot, start_run

OccupancyModel = OccupancyMLP | OccupancyEncoder


def seed_everything(seed: int) -> None:
    """Make the train run repeatable (Python, NumPy, CPU and CUDA RNGs)."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _uses_surface(cfg: OccupancyConfig) -> bool:
    """True when YAML selected the envelope-conditioned head."""
    return str(cfg.shape_encoder).strip().lower() == "surface"


def _uses_encoder(cfg: OccupancyConfig) -> bool:
    """True when the occupancy head takes an envelope tensor."""
    return _uses_surface(cfg)


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
    """Point micro-average over every query in ``parts`` (no grad)."""
    if not parts:
        raise ValueError("val split is empty")
    pin = device.type == "cuda"
    tp = fp = fn = correct = n = 0.0
    model.eval()
    with torch.no_grad():
        for part in parts:
            part.ensure_queries()
            try:
                loader = make_dataloader(
                    part, batch_size=batch_size, shuffle=False, pin_memory=pin
                )
                for batch in loader:
                    logits, y = _forward_batch(model, batch, device)
                    c_tp, c_fp, c_fn, c_ok, c_n = occupancy_counts(logits, y)
                    tp += c_tp
                    fp += c_fp
                    fn += c_fn
                    correct += c_ok
                    n += c_n
            finally:
                part.release_queries()
    scores = occupancy_metrics_from_counts(
        tp=tp, fp=fp, fn=fn, correct=correct, n=n
    )
    return (
        scores.accuracy,
        scores.inside_precision,
        scores.inside_recall,
        scores.inside_iou,
        scores.inside_f1,
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
    n_val: int
    losses: list[float]
    accuracies: list[float]
    val_accuracies: list[float]
    best_epoch: int
    best_metric: float


def _checkpoint_payload(
    model: OccupancyModel,
    dataset: OccupancyMultiNpzDataset,
    cfg: OccupancyConfig,
    optimizer: torch.optim.Optimizer | None = None,
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
    kind = ENCODER_KIND if _uses_encoder(cfg) else MLP_KIND
    return {
        "kind": kind,
        "state_dict": model.state_dict(),
        "hidden": int(cfg.hidden),
        "depth": int(cfg.depth),
        "seed": int(cfg.seed),
        "shape_encoder": str(cfg.shape_encoder),
        "n_surface": int(cfg.n_surface),
        "knn_k": int(cfg.knn_k),
        "knn_local_dim": encoder_knn_local_dim(cfg) if int(cfg.knn_k) > 0 else 0,
        "latent_dim": encoder_latent_dim(cfg),
        "envelope_dim": (
            int(model.envelope_dim) if isinstance(model, OccupancyEncoder) else None
        ),
        "npz_paths": [as_data_relative(p, cfg.data_dir) for p in dataset.npz_paths],
        "parts": parts,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
    }


def _selection_score(
    metric_name: str,
    *,
    val_acc: float,
    val_iou: float,
) -> float:
    """Map YAML ``checkpoint_metric`` to the scalar Checkpointer compares."""
    name = metric_name.strip()
    if name in ("val_acc", "test_acc"):
        return float(val_acc)
    if name in ("val_iou", "test_iou"):
        return float(val_iou)
    raise ValueError(
        f"unsupported checkpoint_metric {metric_name!r} "
        "(train_multi_npz logs val_acc and val_iou)"
    )


def train_multi_npz(
    cfg: OccupancyConfig,
    *,
    npz_paths: Sequence[Path | str] | None = None,
    epochs: int | None = None,
    root: Path | None = None,
    run_name: str | None = None,
    resume_checkpoint: Path | str | None = None,
    resume_run_id: str | None = None,
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
    resume_checkpoint, resume_run_id:
        Load ``best.pt`` and run ``epochs`` **more** epochs (numbered
        after the stored epoch). Omit both for a fresh train.
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
    n_surface = int(cfg.n_surface) if _uses_encoder(cfg) else None
    if npz_paths is None:
        dataset = OccupancyMultiNpzDataset.from_catalog(
            cfg.data_dir,
            npz_glob=cfg.npz_glob,
            npz_paths=cfg.npz_paths or None,
            npz_catalog=cfg.npz_catalog or None,
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

    # File-level randperm leaks: lattice + jitter of one OBJ can sit on both sides.
    split_keys = [mesh_split_key(part) for part in dataset.parts]
    n_split_meshes = len(set(split_keys))
    if n_split_meshes < 2:
        raise ValueError(
            f"need at least 2 distinct meshes to hold out a mesh, got {n_split_meshes}"
        )
    train_idx, val_idx = split_train_test_by_mesh(
        split_keys, cfg.val_fraction, cfg.seed
    )
    train_parts = [dataset.parts[int(i)] for i in train_idx.tolist()]
    val_parts = [dataset.parts[int(i)] for i in val_idx.tolist()]
    pin_memory = cfg.device.type == "cuda"

    resume_ckpt: dict | None = None
    resume_path: Path | None = None
    start_epoch = 1
    if resume_checkpoint is not None or resume_run_id is not None:
        resume_path = resolve_best_pt(
            checkpoint=resume_checkpoint,
            run_id=resume_run_id,
            root=root,
        )
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        start_epoch = int(resume_ckpt.get("epoch") or 0) + 1
        if start_epoch < 2:
            raise ValueError(f"resume checkpoint has no epoch: {resume_path}")

    if resume_ckpt is not None:
        model = load_occupancy_model(resume_ckpt, cfg.device)
        model.train()
    elif _uses_encoder(cfg):
        model = OccupancyEncoder(
            hidden=cfg.hidden,
            depth=cfg.depth,
            latent_dim=encoder_latent_dim(cfg),
            shape_encoder=str(cfg.shape_encoder),
            knn_k=int(cfg.knn_k),
            knn_local_dim=encoder_knn_local_dim(cfg) if int(cfg.knn_k) > 0 else None,
        ).to(cfg.device)
    else:
        model = OccupancyMLP(hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(cfg.optimizer, model, lr)
    if resume_ckpt is not None and resume_ckpt.get("optimizer"):
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    n_train_files = len(train_parts)
    n_val_files = len(val_parts)
    n_train_meshes = len({mesh_split_key(part) for part in train_parts})
    n_val_meshes = len({mesh_split_key(part) for part in val_parts})
    n_train = sum(len(part) for part in train_parts)
    n_val = sum(len(part) for part in val_parts)
    print(f"files={len(dataset.parts)} meshes={dataset.n_meshes}")
    if cfg.npz_catalog:
        print(f"npz_catalog_rows={len(cfg.npz_catalog)}")
    _print_mesh_join(dataset, cfg.data_dir)
    print(
        f"split=mesh train_files={n_train_files} val_files={n_val_files} "
        f"train_meshes={n_train_meshes} val_meshes={n_val_meshes} "
        f"N={dataset.n_points} n_train={n_train} n_val={n_val} "
        f"val_fraction={cfg.val_fraction} device={cfg.device} "
        f"gpu={gpu_name(cfg.device)} "
        f"hidden={cfg.hidden} depth={cfg.depth} "
        f"shape_encoder={cfg.shape_encoder} n_surface={cfg.n_surface} "
        f"knn_k={cfg.knn_k} "
        f"latent_dim={encoder_latent_dim(cfg)}"
    )
    last_epoch = start_epoch + n_epochs - 1
    print(
        f"epochs={n_epochs} first_epoch={start_epoch} last_epoch={last_epoch} "
        f"batch_size={cfg.batch_size} optimizer={cfg.optimizer} lr={lr}"
    )
    if resume_path is not None:
        print(f"resume={resume_path} from_epoch={start_epoch - 1}")
    for part in val_parts:
        print(f"  val_file={part.npz_path.name} N={len(part)}")

    name = run_name if run_name is not None else cfg.run_name
    with start_run(name, root=root, t0=t0, clock_start=clock_start) as run:
        snap = occupancy_config_snapshot(cfg)
        snap["total"] = last_epoch
        if resume_path is not None:
            snap["resume_from"] = str(resume_path)
            snap["resume_epoch"] = start_epoch - 1
            snap["extra_epochs"] = n_epochs
        run.write_config(snap)
        run.write_catalog(dataset.npz_paths, data_dir=cfg.data_dir)
        run.update_config(
            {
                "n_files": len(dataset.parts),
                "n_meshes": dataset.n_meshes,
                "n_points": dataset.n_points,
                "n_train_files": n_train_files,
                "n_val_files": n_val_files,
                "n_test_files": n_val_files,
                "n_train_meshes": n_train_meshes,
                "n_val_meshes": n_val_meshes,
                "n_test_meshes": n_val_meshes,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_val,
                "split": "mesh",
            }
        )
        saver = Checkpointer(
            run,
            total_epochs=last_epoch,
            metric_name=cfg.checkpoint_metric,
            root=root,
        )
        if resume_ckpt is not None:
            prior_best = float(
                resume_ckpt["metric"]
                if resume_ckpt.get("metric") is not None
                else resume_ckpt.get("best_metric") or 0.0
            )
            prior_epoch = int(resume_ckpt["epoch"])
            saver.best_metric = prior_best
            saver.best_epoch = prior_epoch
            seeded = _checkpoint_payload(model, dataset, cfg, optimizer)
            seeded["epoch"] = prior_epoch
            seeded["total"] = last_epoch
            seeded["metric"] = prior_best
            seeded["metric_name"] = cfg.checkpoint_metric
            seeded["run_id"] = run.run_id
            torch.save(seeded, saver.best_path)
            saver.run.update_config(
                {
                    "total": last_epoch,
                    "checkpoint": prior_epoch,
                    "best_epoch": prior_epoch,
                    "best_metric": prior_best,
                }
            )
        losses: list[float] = []
        accuracies: list[float] = []
        val_accuracies: list[float] = []
        last_result = None
        for epoch in range(start_epoch, last_epoch + 1):
            t0 = time.perf_counter()
            model.train()
            running_loss = 0.0
            train_correct = 0.0
            train_n = 0.0
            n_batches = 0
            # Shuffle file order each epoch; points stay inside their file.
            order = torch.randperm(len(train_parts)).tolist()
            for part_i in order:
                part = train_parts[part_i]
                part.ensure_queries()
                try:
                    loader = make_dataloader(
                        part,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        pin_memory=pin_memory,
                    )
                    for batch in loader:
                        logits, y = _forward_batch(model, batch, cfg.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        running_loss += float(loss.item())
                        _tp, _fp, _fn, ok, n_pts = occupancy_counts(logits.detach(), y)
                        train_correct += ok
                        train_n += n_pts
                        n_batches += 1
                finally:
                    part.release_queries()
            mean_loss = running_loss / max(n_batches, 1)
            mean_acc = train_correct / max(train_n, 1.0)
            val_acc, _prec, _rec, val_iou, val_f1 = _eval_parts(
                model, val_parts, cfg.device, cfg.batch_size
            )
            wall = time.perf_counter() - t0
            losses.append(mean_loss)
            accuracies.append(mean_acc)
            val_accuracies.append(val_acc)
            score = _selection_score(
                cfg.checkpoint_metric, val_acc=val_acc, val_iou=val_iou
            )
            last_result = saver.save(
                _checkpoint_payload(model, dataset, cfg, optimizer),
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
                val_iou=val_iou,
                val_f1=val_f1,
                test_acc=val_acc,
                test_iou=val_iou,
                test_f1=val_f1,
            )
            print(
                f"epoch {epoch:03d}  loss={mean_loss:.6f}  "
                f"train_acc={mean_acc:.4f}  val_acc={val_acc:.4f}  "
                f"val_iou={val_iou:.4f}  val_f1={val_f1:.4f}  "
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


def main() -> None:
    """CLI: train occupancy on the YAML catalog."""
    import argparse

    parser = argparse.ArgumentParser(description="Train occupancy on the YAML catalog")
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Load models/<id>/best.pt and train cfg.epochs more epochs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to best.pt (overrides --resume-run-id)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs to run (fresh train) or extra epochs (resume)",
    )
    args = parser.parse_args()
    train_multi_npz(
        load_config(),
        epochs=args.epochs,
        resume_checkpoint=args.resume,
        resume_run_id=args.resume_run_id,
    )


if __name__ == "__main__":
    main()
