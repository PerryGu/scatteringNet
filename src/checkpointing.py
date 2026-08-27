"""Checkpoints under ``models/<run_id>/`` — ``last.pt`` and ``best.pt``.

Step 5 keeps weights out of ``runs/`` so logs stay commitable. The same
``run_id`` joins the two trees. ``best.pt`` is the apex: written only when
the selection metric **strictly improves**. Last epoch is often not best.

``runs/<id>/config.yaml`` is updated on every save:

- ``total`` — planned epoch count for this run
- ``checkpoint`` — epoch index of the last ``last.pt`` write (this saved moment)

This module does not import OccupancyMLP and does not train.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from run_tracking import RunTracker

# Repo root: src/checkpointing.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PHASE1_CKPT = _REPO_ROOT / "models" / "one_npz.pt"


@dataclass(frozen=True)
class CheckpointSaveResult:
    """Outcome of one ``Checkpointer.save`` call."""

    last_path: Path
    best_path: Path
    epoch: int
    is_best: bool
    best_epoch: int
    best_metric: float


class Checkpointer:
    """Write ``last.pt`` always and ``best.pt`` on a strict metric improve."""

    def __init__(
        self,
        run: RunTracker,
        *,
        total_epochs: int,
        metric_name: str = "val_acc",
        root: Path | None = None,
    ) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        self.run = run
        self.run_id = run.run_id
        self.total_epochs = int(total_epochs)
        self.metric_name = str(metric_name).strip() or "val_acc"
        self._root = (root or _REPO_ROOT).resolve()
        self.dir = self._root / "models" / self.run_id
        self.last_path = self.dir / "last.pt"
        self.best_path = self.dir / "best.pt"
        # Never clobber the Phase 1 one-NPZ file.
        if self.last_path.resolve() == _PHASE1_CKPT.resolve():
            raise ValueError(f"refusing to overwrite Phase 1 checkpoint {_PHASE1_CKPT}")
        if self.best_path.resolve() == _PHASE1_CKPT.resolve():
            raise ValueError(f"refusing to overwrite Phase 1 checkpoint {_PHASE1_CKPT}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        # Pointer from logs → weights; no .pt under runs/.
        pointer = self.dir.resolve()
        try:
            rel = pointer.relative_to(self._root).as_posix()
        except ValueError:
            rel = pointer.as_posix()
        (self.run.dir / "checkpoint_dir.txt").write_text(rel + "\n", encoding="utf-8")
        self.run.update_config(
            {
                "total": self.total_epochs,
                "checkpoint": None,
                "checkpoint_metric": self.metric_name,
            }
        )

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        epoch: int,
        metric: float,
    ) -> CheckpointSaveResult:
        """
        Write ``last.pt``. Write ``best.pt`` only if ``metric`` strictly improves.

        Parameters
        ----------
        payload:
            Caller-owned checkpoint dict (dummy weights in this step; model
            state in Step 6). ``epoch`` / ``total`` are stamped here.
        epoch:
            1-based epoch index of this saved moment.
        metric:
            Selection scalar (default ``val_acc``).
        """
        if epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {epoch}")
        score = float(metric)
        blob: dict[str, Any] = dict(payload)
        blob["epoch"] = int(epoch)
        blob["total"] = self.total_epochs
        blob["metric_name"] = self.metric_name
        blob["metric"] = score
        blob["run_id"] = self.run_id

        torch.save(blob, self.last_path)
        is_best = self.best_metric is None or score > self.best_metric
        if is_best:
            self.best_metric = score
            self.best_epoch = int(epoch)
            torch.save(blob, self.best_path)

        assert self.best_epoch is not None and self.best_metric is not None
        self.run.update_config(
            {
                "total": self.total_epochs,
                "checkpoint": int(epoch),
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
            }
        )
        self.run.log_scalars(
            int(epoch),
            {
                "checkpoint/last_epoch": float(epoch),
                "checkpoint/best_epoch": float(self.best_epoch),
                "checkpoint/best_metric": float(self.best_metric),
            },
        )
        return CheckpointSaveResult(
            last_path=self.last_path,
            best_path=self.best_path,
            epoch=int(epoch),
            is_best=is_best,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
        )


if __name__ == "__main__":
    # Dummy 3-epoch smoke: no OccupancyMLP. Epoch 2 is the apex.
    from config import load_config
    from run_tracking import occupancy_config_snapshot, start_run
    import yaml

    cfg = load_config()
    with start_run("ckpt_dummy") as run:
        run.write_config(occupancy_config_snapshot(cfg))
        saver = Checkpointer(
            run,
            total_epochs=3,
            metric_name=cfg.checkpoint_metric,
        )
        # val_acc peaks at epoch 2 so best.pt must keep those weights.
        vals = (0.50, 0.90, 0.60)
        for epoch, val_acc in enumerate(vals, start=1):
            tag = torch.tensor([float(epoch)])
            result = saver.save({"w": tag}, epoch=epoch, metric=val_acc)
            run.log_epoch(
                epoch=epoch,
                loss=1.0 / epoch,
                train_acc=0.5,
                val_acc=val_acc,
                wall_seconds=0.01 * epoch,
                best_epoch=result.best_epoch,
                best_metric=result.best_metric,
            )
        last = torch.load(saver.last_path, map_location="cpu", weights_only=False)
        best = torch.load(saver.best_path, map_location="cpu", weights_only=False)
        snap = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
        print(f"run_dir={run.dir.resolve()}")
        print(f"model_dir={saver.dir.resolve()}")
        print(f"last_epoch={last['epoch']} best_epoch={best['epoch']}")
        print(f"snapshot_total={snap['total']} snapshot_checkpoint={snap['checkpoint']}")
        print(f"runs_has_pt={any(run.dir.rglob('*.pt'))}")
        print(f"phase1_untouched={_PHASE1_CKPT}")
