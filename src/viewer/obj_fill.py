"""Job B: unlabeled AABB lattice inside an uploaded OBJ (no occupancy GT).

Lattice math matches ``scatter_generation.raycast_scatter`` occupancy grid
(``_padded_bounds`` / ``_uniform_grid_points``) without importing that module
(Open3D / package ``__init__``). Does not import torch and does not raycast-label.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import numpy as np


from scatteringnet.geometry.mesh_io import load_obj_triangles  # noqa: E402

MAX_FILL_POINTS = 200_000
SPACING_MIN = 0.04
SPACING_MAX = 0.50
# Slider 0 = coarse (0.40), 100 = dense (0.05). Training occupancy often uses 0.15.
SPACING_COARSE = 0.40
SPACING_FINE = 0.05


def spacing_from_slider(value: float) -> float:
    """Map UI density 0–100 to lattice spacing (higher = denser = smaller step)."""
    t = min(1.0, max(0.0, float(value) / 100.0))
    return float(SPACING_COARSE + (SPACING_FINE - SPACING_COARSE) * t)


def clamp_spacing(spacing: float) -> float:
    s = float(spacing)
    if s < SPACING_MIN or s > SPACING_MAX:
        raise ValueError(
            f"spacing must be between {SPACING_MIN} and {SPACING_MAX}, got {s}"
        )
    return s


def _padded_bounds(bounds: np.ndarray, pad: float) -> np.ndarray:
    """Expand AABB by ``pad`` on every side (same as occupancy lattice)."""
    bounds = np.asarray(bounds, dtype=np.float64)
    out = bounds.copy()
    out[0] -= pad
    out[1] += pad
    return out


def _uniform_grid_points(
    bounds: np.ndarray,
    spacing: float,
    *,
    max_points: int = MAX_FILL_POINTS,
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """Regular XYZ lattice; coarsen spacing by 1.25 until under ``max_points``."""
    if spacing <= 0:
        raise ValueError("point_spacing must be > 0")
    bmin = bounds[0].astype(np.float64)
    bmax = bounds[1].astype(np.float64)
    extents = np.maximum(bmax - bmin, 1e-12)
    used = float(spacing)

    def counts(step: float) -> tuple[int, int, int]:
        return tuple(max(2, int(np.floor(extents[i] / step)) + 1) for i in range(3))

    nx, ny, nz = counts(used)
    while nx * ny * nz > max_points:
        used *= 1.25
        nx, ny, nz = counts(used)

    xs = np.linspace(bmin[0], bmax[0], nx, dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], ny, dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], nz, dtype=np.float64)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, used, (nx, ny, nz)


def triangles_from_obj_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse Wavefront text via a temp file (same loader as training)."""
    raw = str(text or "")
    if not raw.strip():
        raise ValueError("OBJ is empty")
    fd, path = tempfile.mkstemp(suffix=".obj")
    try:
        os.write(fd, raw.encode("utf-8"))
        os.close(fd)
        fd = -1
        return load_obj_triangles(path, cache=False)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass


def fill_aabb_lattice(
    vertices: np.ndarray,
    spacing: float,
    *,
    max_points: int = MAX_FILL_POINTS,
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """
    Regular grid in a padded mesh AABB. Pad = spacing (outside shell), no jitter.
    """
    step = clamp_spacing(spacing)
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 1:
        raise ValueError(f"vertices must be (V, 3), got {tuple(verts.shape)}")
    bounds = np.stack([verts.min(axis=0), verts.max(axis=0)])
    sample_bounds = _padded_bounds(bounds, step)
    points, used, grid = _uniform_grid_points(
        sample_bounds, step, max_points=int(max_points)
    )
    return np.ascontiguousarray(points, dtype=np.float32), float(used), grid


def fill_from_obj_text(
    obj_text: str,
    spacing: float,
) -> dict:
    """Return lattice points (float32) and grid metadata."""
    vertices, _faces = triangles_from_obj_text(obj_text)
    points, used, grid = fill_aabb_lattice(vertices, spacing)
    return {
        "n": int(points.shape[0]),
        "used_spacing": used,
        "grid": [int(grid[0]), int(grid[1]), int(grid[2])],
        "points_b64": base64.b64encode(np.ascontiguousarray(points)).decode("ascii"),
    }
