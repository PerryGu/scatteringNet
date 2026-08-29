"""AABB normalization for occupancy XYZ.

Maps a point cloud into a roughly ``[-1, 1]^3`` cube so the MLP sees
comparable coordinates across differently sized meshes.

  center = midpoint of the axis-aligned bounding box
  scale  = maximum half-extent (longest AABB side / 2)

Normalized point: ``(xyz - center) / scale``.
This module does not touch the occupancy model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PointsArray = NDArray[np.float32]


def compute_center_scale(points: np.ndarray) -> tuple[PointsArray, float]:
    """
    AABB center and max half-extent for an ``(N, 3)`` point array.

    Parameters
    ----------
    points:
        Query XYZ, shape ``(N, 3)``, at least one row.

    Returns
    -------
    center:
        ``float32`` vector of shape ``(3,)``.
    scale:
        Positive float (max half-extent). Raises if the cloud has no extent.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(pts.shape)}")
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least one row")

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    half_extents = 0.5 * (xyz_max - xyz_min)
    scale = float(np.max(half_extents))
    if scale <= 0.0:
        raise ValueError(
            "scale must be > 0; all points appear to share the same location"
        )
    return center.astype(np.float32, copy=False), scale


def apply_normalization(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> PointsArray:
    """
    Return ``(points - center) / scale`` as ``float32 (N, 3)``.

    Parameters
    ----------
    points:
        Query XYZ, shape ``(N, 3)``.
    center:
        AABB midpoint, shape ``(3,)``.
    scale:
        Positive max half-extent.

    Returns
    -------
    ndarray
        Normalized points, ``float32 (N, 3)``.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(pts.shape)}")
    c = np.asarray(center, dtype=np.float32).reshape(3)
    return (pts - c) / np.float32(scale)


if __name__ == "__main__":
    from config import load_config
    from data_npz import load_points_labels

    sample = (
        load_config().data_dir
        / "exports"
        / "dataset_test"
        / "sphere__raycast_z_raut_s0.15_inout.npz"
    )
    points, _labels = load_points_labels(sample)
    center, scale = compute_center_scale(points)
    normed = apply_normalization(points, center, scale)
    recovered = normed[:3] * np.float32(scale) + center
    print(f"file={sample}")
    print(f"center={center.tolist()} scale={scale:.6f}")
    print(f"normed_min={normed.min(axis=0).tolist()}")
    print(f"normed_max={normed.max(axis=0).tolist()}")
    print(f"inverse_ok={np.allclose(recovered, points[:3], rtol=1e-5, atol=1e-5)}")
