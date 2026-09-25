"""Per-run log folders under ``runs/<timestamp>_<name>/``.

Logs only: config snapshot, ``metrics.jsonl``, TensorBoard scalars.
Weights stay out of ``runs/`` so JSON + YAML can be committed without
gitignore-ing the whole tree (``.pt`` files belong in ``models/``).

Layout::

    runs/<YYYY-MM-DD_HH-MM-SS>_<name>/
      config.yaml          # knobs (no resolved file list)
      catalog.txt          # one data-relative NPZ path per line
      metrics.jsonl        # one JSON object per epoch
      events.out.tfevents* # TensorBoard (gitignored)

This module does not import OccupancyMLP and does not write checkpoints.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from torch.utils.tensorboard import SummaryWriter

from scatteringnet.config import OccupancyConfig, as_data_relative, as_repo_relative, gpu_name, repo_root

# Repo root: src/run_tracking.py → parents[1].
_REPO_ROOT = repo_root()
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_CATALOG_FILENAME = "catalog.txt"
_DEFAULT_RUN_NAME = "run"

# TensorBoard tags stay split so curves are comparable across later train steps.
_TB_TRAIN_LOSS = "train/loss"
_TB_TRAIN_ACC = "train/acc"
_TB_VAL_ACC = "val/acc"
_TB_WALL = "time/wall_seconds"
_TB_RUN_WALL = "time/run_wall_seconds"


def format_duration(seconds: float) -> str:
    """Human-readable wall time for the terminal and the run snapshot."""
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total - hours * 3600 - minutes * 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:05.2f}s"
    if minutes > 0:
        return f"{minutes}m {secs:05.2f}s"
    return f"{secs:.2f}s"


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
) -> dict[str, Any]:
    """
    Flatten OccupancyConfig into a JSON-safe dict for the run snapshot.

    Paths under the repo or ``data_dir`` are stored relative (POSIX).
    ``data_dir`` itself is the dataset root: relative to the git repo when
    it lives there, otherwise absolute POSIX (data disk vs git disk).
    ``device`` and ``gpu`` are recorded here (runtime) even though they
    are not YAML knobs. ``gpu`` is the CUDA card name, or null on CPU.

    The resolved train file list is **not** stored here (see
    :meth:`RunTracker.write_catalog`). ``npz_paths`` is the YAML knob:
    empty means the glob was used.
    """
    payload: dict[str, Any] = {
        "data_dir": as_repo_relative(cfg.data_dir),
        "device": str(cfg.device),
        "gpu": gpu_name(cfg.device),
        "hidden": int(cfg.hidden),
        "depth": int(cfg.depth),
        "seed": int(cfg.seed),
        "epochs": int(cfg.epochs),
        # ``total`` is the planned epoch count; ``checkpoint`` is filled
        # with the epoch index stored in ``best.pt`` (null until then).
        "total": int(cfg.epochs),
        "checkpoint": None,
        "lr": float(cfg.lr),
        "val_fraction": float(cfg.val_fraction),
        "latent_dim": cfg.latent_dim,
        "npz_glob": Path(str(cfg.npz_glob)).as_posix(),
        "max_files_per_shape": cfg.max_files_per_shape,
        "run_name": str(cfg.run_name),
        "checkpoint_metric": str(cfg.checkpoint_metric),
        "batch_size": int(cfg.batch_size),
        "optimizer": str(cfg.optimizer),
        "n_surface": int(cfg.n_surface),
        "knn_k": int(cfg.knn_k),
        "knn_local_dim": cfg.knn_local_dim,
        "knn_pool": str(cfg.knn_pool),
        "shape_encoder": str(cfg.shape_encoder),
        "pos_weight": cfg.pos_weight,
        "pos_weight_auto": bool(cfg.pos_weight_auto),
    }
    # Explicit list knob: only snapshot it when it actually replaced the glob.
    explicit = [as_data_relative(p, cfg.data_dir) for p in cfg.npz_paths]
    if explicit:
        payload["npz_paths"] = explicit
    if cfg.npz_catalog:
        payload["npz_catalog"] = [
            {"glob": glob_s, "max_shapes": max_s} for glob_s, max_s in cfg.npz_catalog
        ]
    return _jsonable(payload)


class RunTracker:
    """Create one dated ``runs/<id>/`` folder and append epoch metrics to it."""

    def __init__(
        self,
        name: str = _DEFAULT_RUN_NAME,
        *,
        root: Path | None = None,
        created_at: datetime | None = None,
        t0: float | None = None,
        clock_start: datetime | None = None,
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
        self.catalog_path = self.dir / _CATALOG_FILENAME
        # SummaryWriter writes events.out.tfevents* into this directory.
        self._tb: SummaryWriter | None = SummaryWriter(log_dir=str(self.dir))
        # Wall clock for the whole workout. Train can pass ``t0`` /
        # ``clock_start`` so catalog load is included before this folder exists.
        self.started_at = clock_start or datetime.now()
        self._t0 = t0 if t0 is not None else time.perf_counter()
        self._timing_written = False

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
        ``models/<id>/`` without putting ``.pt`` here.
        """
        payload = _jsonable(dict(config))
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("run_dir", self.relative_dir)
        payload.setdefault(
            "started_at",
            self.started_at.isoformat(timespec="seconds"),
        )
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

    def write_catalog(
        self,
        paths: Sequence[Path | str],
        *,
        data_dir: Path | str,
    ) -> Path:
        """
        Write resolved train NPZs to ``catalog.txt`` (one relative path per line).

        Keeps ``config.yaml`` small: the snapshot only records ``catalog_file``
        and ``catalog_n``. Paths are data-dir relative.
        """
        rel = [as_data_relative(p, data_dir) for p in paths]
        text = "\n".join(rel)
        if rel:
            text += "\n"
        self.catalog_path.write_text(text, encoding="utf-8")
        self.update_config(
            {
                "catalog_file": _CATALOG_FILENAME,
                "catalog_n": len(rel),
            }
        )
        return self.catalog_path

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

    def write_timing(self) -> dict[str, Any]:
        """
        Stamp whole-run wall time into ``config.yaml`` and TensorBoard.

        Call at the end of a train (or let ``__exit__`` do it). ``wall_seconds``
        is the precise elapsed time; ``wall`` is the same value for humans.
        """
        elapsed = time.perf_counter() - self._t0
        finished = datetime.now()
        payload = {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "wall_seconds": round(elapsed, 3),
            "wall": format_duration(elapsed),
        }
        self.update_config(payload)
        if self._tb is not None:
            self._tb.add_scalar(_TB_RUN_WALL, float(elapsed), 0)
            self._tb.flush()
        self._timing_written = True
        return payload

    def update_config(self, updates: Mapping[str, Any]) -> Path:
        """
        Merge keys into the existing snapshot and rewrite ``config.yaml``.

        Used to stamp ``total`` (planned epochs) and ``checkpoint``
        (epoch index stored in ``best.pt``) without dropping knobs.
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
        if not self._timing_written:
            self.write_timing()
        self.close()


def start_run(
    name: str = _DEFAULT_RUN_NAME,
    *,
    root: Path | None = None,
    created_at: datetime | None = None,
    t0: float | None = None,
    clock_start: datetime | None = None,
) -> RunTracker:
    """Create a ``RunTracker`` under ``root/runs`` (default: repo root)."""
    return RunTracker(
        name=name,
        root=root,
        created_at=created_at,
        t0=t0,
        clock_start=clock_start,
    )


def _has_pt_under(path: Path) -> bool:
    return any(path.rglob("*.pt"))


def _has_tfevents(path: Path) -> bool:
    return any(path.glob("events.out.tfevents*"))


if __name__ == "__main__":
    # Dummy 3-epoch smoke: metrics only, no OccupancyMLP, no .pt.
    from scatteringnet.config import load_config

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
        timing = run.write_timing()
        print(f"run_dir={run.dir.resolve()}")
        print(f"run_id={run.run_id}")
        print(f"metrics={run.metrics_path}")
        print(f"config={run.config_path}")
        print(f"wall={timing['wall']} ({timing['wall_seconds']}s)")
    print(f"has_pt={_has_pt_under(run.dir)}")
    print(f"has_tfevents={_has_tfevents(run.dir)}")
