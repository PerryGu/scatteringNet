"""Per-run log folders under ``runs/<timestamp>_<name>/``.

Step 4 is logs only: config snapshot, ``metrics.jsonl``, TensorBoard scalars.
Weights stay out of ``runs/`` so JSON + YAML can be committed without
gitignore-ing the whole tree (``.pt`` files belong in ``models/`` in Step 5).

Layout::

    runs/<YYYY-MM-DD_HH-MM-SS>_<name>/
      config.yaml          # knobs + device + seed + NPZ list when known
      metrics.jsonl        # one JSON object per epoch
      events.out.tfevents* # TensorBoard (gitignored)

This module does not import OccupancyMLP and does not write checkpoints.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from torch.utils.tensorboard import SummaryWriter

from config import OccupancyConfig

# Repo root: src/run_tracking.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_DEFAULT_RUN_NAME = "run"

# TensorBoard tags stay split so curves are comparable across later train steps.
_TB_TRAIN_LOSS = "train/loss"
_TB_TRAIN_ACC = "train/acc"
_TB_VAL_ACC = "val/acc"
_TB_WALL = "time/wall_seconds"


def _sanitize_name(name: str) -> str:
    """Folder suffix: keep alphanumerics, dots, hyphens, underscores."""
    text = _SAFE_NAME.sub("_", str(name).strip())
    text = text.strip("._-")
    return text[:80] if text else _DEFAULT_RUN_NAME


def _jsonable(value: Any) -> Any:
    """YAML/JSON snapshot must not dump Path / device objects as Python tags."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def occupancy_config_snapshot(
    cfg: OccupancyConfig,
    *,
    npz_paths: Sequence[Path | str] | None = None,
) -> dict[str, Any]:
    """
    Flatten OccupancyConfig into a JSON-safe dict for the run snapshot.

    ``device`` is recorded here (runtime) even though it is not a YAML knob.
    ``npz_paths`` is optional so Step 3 catalogs can be attached without
    coupling this helper to the catalog loader.
    """
    payload: dict[str, Any] = {
        "data_dir": cfg.data_dir,
        "device": str(cfg.device),
        "hidden": int(cfg.hidden),
        "depth": int(cfg.depth),
        "seed": int(cfg.seed),
        "epochs": int(cfg.epochs),
        # ``total`` is the planned epoch count; ``checkpoint`` is filled by Step 5
        # with the epoch index of the last ``last.pt`` write (null until then).
        "total": int(cfg.epochs),
        "checkpoint": None,
        "lr": float(cfg.lr),
        "checkpoint_path": cfg.checkpoint_path,
        "sample_npz": cfg.sample_npz,
        "val_fraction": float(cfg.val_fraction),
        "npz_glob": str(cfg.npz_glob),
        "npz_paths": list(cfg.npz_paths),
        "max_files_per_shape": cfg.max_files_per_shape,
        "run_name": str(cfg.run_name),
        "checkpoint_metric": str(cfg.checkpoint_metric),
    }
    if npz_paths is not None:
        payload["catalog_npz_paths"] = [str(p) for p in npz_paths]
    return _jsonable(payload)


class RunTracker:
    """Create one dated ``runs/<id>/`` folder and append epoch metrics to it."""

    def __init__(
        self,
        name: str = _DEFAULT_RUN_NAME,
        *,
        root: Path | None = None,
        created_at: datetime | None = None,
    ) -> None:
        when = created_at or datetime.now()
        stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
        safe = _sanitize_name(name)
        self.run_id = f"{stamp}_{safe}"
        self._root = (root or _REPO_ROOT).resolve()
        self.dir = self._root / "runs" / self.run_id
        suffix = 1
        # Never overwrite an existing run id (same-second collisions).
        while self.dir.exists():
            self.run_id = f"{stamp}_{safe}_{suffix}"
            self.dir = self._root / "runs" / self.run_id
            suffix += 1
        self.dir.mkdir(parents=True, exist_ok=False)
        self.config_path = self.dir / "config.yaml"
        self.metrics_path = self.dir / "metrics.jsonl"
        # SummaryWriter writes events.out.tfevents* into this directory.
        self._tb: SummaryWriter | None = SummaryWriter(log_dir=str(self.dir))

    @property
    def relative_dir(self) -> str:
        """Repo-relative posix path, or the absolute path if outside the repo."""
        try:
            return self.dir.resolve().relative_to(self._root).as_posix()
        except ValueError:
            return self.dir.resolve().as_posix()

    def write_config(self, config: Mapping[str, Any]) -> Path:
        """
        Snapshot knobs into ``config.yaml``.

        Always stamps ``run_id`` / ``run_dir`` so logs stay joinable to
        ``models/<id>/`` in Step 5 without putting ``.pt`` here.
        """
        payload = _jsonable(dict(config))
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("run_dir", self.relative_dir)
        text = yaml.safe_dump(
            payload,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
        )
        self.config_path.write_text(text, encoding="utf-8")
        if self._tb is None:
            raise RuntimeError("RunTracker is closed")
        brief = "\n".join(f"{k}: {payload[k]}" for k in sorted(payload))
        self._tb.add_text("config", brief, 0)
        self._tb.flush()
        return self.config_path

    def log_epoch(
        self,
        *,
        epoch: int,
        loss: float,
        train_acc: float,
        val_acc: float,
        wall_seconds: float,
        **extra: Any,
    ) -> None:
        """
        Append one JSON line and matching TensorBoard scalars.

        Parameters
        ----------
        epoch:
            Zero-based or one-based index; stored as given.
        loss:
            Train loss for the epoch.
        train_acc:
            Train occupancy accuracy.
        val_acc:
            Validation occupancy accuracy.
        wall_seconds:
            Wall time for this epoch (or cumulative — caller decides).
        extra:
            Optional extra numeric fields (e.g. later IoU). Written to JSONL;
            numeric values also go to TensorBoard under their key name.
        """
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "loss": float(loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "wall_seconds": float(wall_seconds),
        }
        for key, value in extra.items():
            row[key] = _jsonable(value)

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        if self._tb is None:
            raise RuntimeError("RunTracker is closed")
        step = int(epoch)
        self._tb.add_scalar(_TB_TRAIN_LOSS, float(loss), step)
        self._tb.add_scalar(_TB_TRAIN_ACC, float(train_acc), step)
        self._tb.add_scalar(_TB_VAL_ACC, float(val_acc), step)
        self._tb.add_scalar(_TB_WALL, float(wall_seconds), step)
        for key, value in extra.items():
            try:
                self._tb.add_scalar(str(key), float(value), step)
            except (TypeError, ValueError):
                continue
        self._tb.flush()

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        """Write extra TensorBoard scalars (e.g. ``checkpoint/best_metric``)."""
        if self._tb is None:
            raise RuntimeError("RunTracker is closed")
        for key, value in values.items():
            self._tb.add_scalar(str(key), float(value), int(step))
        self._tb.flush()

    def update_config(self, updates: Mapping[str, Any]) -> Path:
        """
        Merge keys into the existing snapshot and rewrite ``config.yaml``.

        Used by Step 5 to stamp ``total`` (planned epochs) and ``checkpoint``
        (epoch index of the last ``last.pt`` write) without dropping knobs.
        """
        current: dict[str, Any] = {}
        if self.config_path.is_file():
            loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = dict(loaded)
        current.update(_jsonable(dict(updates)))
        return self.write_config(current)

    def close(self) -> None:
        """Flush and close TensorBoard so event files are complete on disk."""
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def start_run(
    name: str = _DEFAULT_RUN_NAME,
    *,
    root: Path | None = None,
    created_at: datetime | None = None,
) -> RunTracker:
    """Create a ``RunTracker`` under ``root/runs`` (default: repo root)."""
    return RunTracker(name=name, root=root, created_at=created_at)


def _has_pt_under(path: Path) -> bool:
    return any(path.rglob("*.pt"))


def _has_tfevents(path: Path) -> bool:
    return any(path.glob("events.out.tfevents*"))


if __name__ == "__main__":
    # Dummy 3-epoch smoke: metrics only, no OccupancyMLP, no .pt.
    from config import load_config

    cfg = load_config()
    with start_run("dummy") as run:
        run.write_config(occupancy_config_snapshot(cfg))
        for epoch in range(3):
            run.log_epoch(
                epoch=epoch,
                loss=1.0 / (epoch + 1),
                train_acc=0.5 + 0.1 * epoch,
                val_acc=0.4 + 0.1 * epoch,
                wall_seconds=0.01 * (epoch + 1),
            )
        print(f"run_dir={run.dir.resolve()}")
        print(f"run_id={run.run_id}")
        print(f"metrics={run.metrics_path}")
        print(f"config={run.config_path}")
    print(f"has_pt={_has_pt_under(run.dir)}")
    print(f"has_tfevents={_has_tfevents(run.dir)}")
