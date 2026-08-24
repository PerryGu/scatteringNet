"""Infer occupancy labels from a trained checkpoint (v2 minimal Step 8).

Uses the AABB ``center`` / ``scale`` stored in the checkpoint so the
query XYZ map matches training. Does not recompute bounds from the NPZ
(``OccupancyPointDataset`` would do that and could drift).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from config import OccupancyConfig, load_config, sample_npz_path
from data_npz import load_points_labels
from dataset import DEFAULT_BATCH_SIZE
from metrics import occupancy_metrics
from normalize import apply_normalization
from occupancy_mlp import OccupancyMLP
from train_one_npz import CHECKPOINT_KIND

# Decision cut on sigmoid(logit). Same as metrics.py / train prints.
_DECISION_THRESHOLD = 0.5
_REQUIRED_CKPT_KEYS = ("kind", "state_dict", "center", "scale", "hidden", "depth")


@dataclass
class InferRunResult:
    """Counts and accuracy for one NPZ evaluated with a checkpoint."""

    ckpt_path: Path
    npz_path: Path
    n: int
    n_pred_inside: int
    n_pred_outside: int
    accuracy: float
    pred_path: Path | None


def _pred_npz_path(ckpt_path: Path) -> Path:
    """Write predictions next to the checkpoint, not over it."""
    return ckpt_path.with_name(f"{ckpt_path.stem}_pred.npz")


def load_occupancy_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """
    Load ``models/one_npz.pt`` (or any occupancy_mlp payload).

    ``weights_only=False`` is required: the file stores numpy ``center`` plus
    a ``state_dict``, not a raw tensor.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must be a mapping, got {type(payload).__name__}")
    missing = [k for k in _REQUIRED_CKPT_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"Checkpoint missing keys: {', '.join(missing)} in {ckpt_path}"
        )
    if payload["kind"] != CHECKPOINT_KIND:
        raise ValueError(
            f"Checkpoint kind must be {CHECKPOINT_KIND!r}, "
            f"got {payload['kind']!r} in {ckpt_path}"
        )
    return dict(payload)


def _logits_for_points(
    model: OccupancyMLP,
    xyz_np: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    """Forward in chunks so a large NPZ does not need one giant (N, 3) batch."""
    n = int(xyz_np.shape[0])
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(xyz_np[start : start + batch_size]).to(device)
            chunks.append(model(batch).cpu())
    return torch.cat(chunks, dim=0)  # (N, 1)


def infer_one_npz(
    npz_path: Path,
    cfg: OccupancyConfig,
    *,
    ckpt_path: Path | None = None,
    write_pred: bool = True,
) -> InferRunResult:
    """
    Classify one occupancy NPZ with a saved ``OccupancyMLP``.

    Architecture (hidden/depth) comes from the checkpoint, not from YAML,
    so a file trained with other knobs still loads. Device comes from ``cfg``.
    """
    ckpt_file = Path(ckpt_path) if ckpt_path is not None else cfg.checkpoint_path
    payload = load_occupancy_checkpoint(ckpt_file, cfg.device)

    hidden = int(payload["hidden"])
    depth = int(payload["depth"])
    center = np.asarray(payload["center"], dtype=np.float32)
    scale = float(payload["scale"])

    points, labels = load_points_labels(Path(npz_path))
    # Checkpoint AABB, not a fresh compute_center_scale(points).
    xyz_norm = apply_normalization(points, center, scale)

    model = OccupancyMLP(hidden=hidden, depth=depth).to(cfg.device)
    model.load_state_dict(payload["state_dict"])

    logits = _logits_for_points(model, xyz_norm, cfg.device)
    y = torch.from_numpy(labels).unsqueeze(1)  # (N, 1) to match logits
    scores = occupancy_metrics(logits, y, threshold=_DECISION_THRESHOLD)

    pred_inside = logits.reshape(-1).sigmoid() >= _DECISION_THRESHOLD
    n = int(pred_inside.numel())
    n_inside = int(pred_inside.sum().item())
    n_outside = n - n_inside
    pred_u8 = pred_inside.to(dtype=torch.uint8).numpy()

    pred_path: Path | None = None
    if write_pred:
        pred_path = _pred_npz_path(ckpt_file)
        np.savez(pred_path, pred_labels=pred_u8)

    print(f"npz={npz_path}")
    print(f"ckpt={ckpt_file}")
    print(f"N={n} device={cfg.device} hidden={hidden} depth={depth}")
    print(f"pred_inside={n_inside} pred_outside={n_outside}")
    print(f"accuracy={scores.accuracy:.4f}")
    if pred_path is not None:
        print(f"saved_pred={pred_path}")

    return InferRunResult(
        ckpt_path=ckpt_file,
        npz_path=Path(npz_path),
        n=n,
        n_pred_inside=n_inside,
        n_pred_outside=n_outside,
        accuracy=scores.accuracy,
        pred_path=pred_path,
    )


if __name__ == "__main__":
    import sys

    cfg = load_config()
    # Optional path argument; default is the YAML sample NPZ (the train file).
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else sample_npz_path(cfg)
    infer_one_npz(npz, cfg)
