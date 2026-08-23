"""Load occupancy query points and labels from a scatter NPZ (Step 3).

Only ``points`` and ``labels`` are used. Other keys (``mesh_path``, ``tags``,
``method``, …) are ignored on purpose so this MVP stays xyz + occupancy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Labels are converted here (not deferred to the Dataset) so every caller gets
# the same dtypes: float32 XYZ and float32 {0, 1} occupancy.
PointsArray = NDArray[np.float32]
LabelsArray = NDArray[np.float32]


def load_points_labels(path: Path) -> tuple[PointsArray, LabelsArray]:
    """
    Read query coordinates and inside/outside labels from one NPZ file.

    Parameters
    ----------
    path:
        Path to a ``.npz`` with arrays ``points`` ``(N, 3)`` and
        ``labels`` ``(N,)`` (typically uint8 0/1).

    Returns
    -------
    points:
        ``float32`` array of shape ``(N, 3)``.
    labels:
        ``float32`` array of shape ``(N,)`` with values in ``{0.0, 1.0}``
        (0 = outside, 1 = inside).
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # allow_pickle=False: we only need numeric arrays, not object payloads.
    with np.load(npz_path, allow_pickle=False) as raw:
        files = set(raw.files)
        if "points" not in files or "labels" not in files:
            raise KeyError(
                f"NPZ must contain 'points' and 'labels', got {sorted(files)} "
                f"in {npz_path}"
            )
        points = np.asarray(raw["points"])
        labels = np.asarray(raw["labels"])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {tuple(points.shape)} in {npz_path}"
        )
    n = int(points.shape[0])
    if labels.shape != (n,):
        raise ValueError(
            f"labels must have shape (N,) with N={n}, got {tuple(labels.shape)} "
            f"in {npz_path}"
        )

    points_f32 = np.asarray(points, dtype=np.float32)
    labels_f32 = np.asarray(labels, dtype=np.float32)
    unique = np.unique(labels_f32)
    if not np.all((unique == 0.0) | (unique == 1.0)):
        raise ValueError(
            f"labels must be in {{0, 1}}, got unique={unique.tolist()} in {npz_path}"
        )
    return points_f32, labels_f32


def _summarize(points: PointsArray, labels: LabelsArray) -> str:
    n = int(points.shape[0])
    inside = float(labels.mean()) if n else float("nan")
    xyz_min = points.min(axis=0) if n else np.full(3, np.nan, dtype=np.float32)
    xyz_max = points.max(axis=0) if n else np.full(3, np.nan, dtype=np.float32)
    return (
        f"N={n}\n"
        f"inside_fraction={inside:.6f}\n"
        f"xyz_min={xyz_min.tolist()}\n"
        f"xyz_max={xyz_max.tolist()}"
    )


# Default smoke-check file from the v2 plan (dataset_test sphere).
_SAMPLE_RELATIVE = Path("exports") / "dataset_test" / "sphere__raycast_z_raut_s0.15_inout.npz"


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    sample = cfg.data_dir / _SAMPLE_RELATIVE
    pts, labs = load_points_labels(sample)
    print(f"file={sample}")
    print(_summarize(pts, labs))
