"""
Volume occupancy sampling: place query points, then label inside/outside.

Two placement methods:

* ``occupancy`` — regular 3D lattice in a padded AABB (default NPZ path).
* ``raycast`` — samples along rays from one AABB face (torus-correct hit pairs).

``random_range`` / ``jitter``: after placement, each point is offset by
independent ``U[-r, +r]`` on X, Y, and Z. Labels are computed on the *moved*
points. ``r = 0`` leaves the lattice unchanged.

Occupancy labels are a multi-direction odd-hit vote (not Open3D winding
number). One open face must not paint a ghost inside-slab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import trimesh

from scatteringnet.scatter_generation.mesh_loader import load_mesh, to_data_relative, trimesh_to_open3d

AxisName = Literal["x", "y", "z"]
ScatterMethod = Literal["raycast", "occupancy"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Inside = odd triangle hits along most of these directions. Fibonacci
# directions avoid lining up with CAD axes (vertex / edge hits flip parity).
_OCC_N_RAYS = 24
# 75%: a hole can spoil a few rays; a ghost slab only wins a few.
_OCC_VOTE_MIN = 18


@dataclass
class ScatterResult:
    """
    Labeled query points for one mesh.

    Shapes
    ------
    points:
        ``(N, 3)`` float64 XYZ in mesh coordinates.
    labels:
        ``(N,)`` uint8; ``1`` = inside, ``0`` = outside.

    Attributes
    ----------
    axis:
        Ray axis ``x|y|z``, or ``n`` for the occupancy grid (no ray axis).
    ray_grid:
        Ray counts ``(n0, n1)``, or occupancy lattice ``(nx, ny)`` (``nz`` is
        not stored here).
    point_spacing:
        World-space step used to place points (may be coarsened by ``max_points``).
    num_inside, num_outside:
        Class counts; ``num_inside + num_outside == N``.
    method:
        ``occupancy`` or ``raycast``.
    jitter:
        Per-axis half-width of the uniform offset (same value as ``random_range``).
    occupancy_verified:
        True when labels came from per-point occupancy (grid path, or raycast
        after jitter).
    """

    points: np.ndarray  # (N, 3) float64
    labels: np.ndarray  # (N,) uint8 — 1 inside, 0 outside
    axis: str
    ray_grid: tuple[int, int]
    point_spacing: float
    num_inside: int
    num_outside: int
    method: str = "occupancy"
    jitter: float = 0.0
    # True when labels came from per-point occupancy (grid path, or raycast after jitter).
    occupancy_verified: bool = False


def _make_result(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    axis: str,
    ray_grid: tuple[int, int],
    point_spacing: float,
    method: str,
    jitter: float,
    occupancy_verified: bool,
) -> ScatterResult:
    """
    Build a :class:`ScatterResult` with consistent dtypes and class counts.

    Parameters
    ----------
    points:
        Query XYZ, shape ``(N, 3)``.
    labels:
        Inside/outside flags, length ``N``.
    axis, ray_grid, point_spacing, method, jitter, occupancy_verified:
        Copied onto the result (see :class:`ScatterResult`).

    Returns
    -------
    ScatterResult
        ``num_inside`` / ``num_outside`` are derived from ``labels``.
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(labels) != len(points):
        raise ValueError("points/labels length mismatch")
    return ScatterResult(
        points=points,
        labels=labels,
        axis=str(axis),
        ray_grid=(int(ray_grid[0]), int(ray_grid[1])),
        point_spacing=float(point_spacing),
        num_inside=int(np.sum(labels == 1)),
        num_outside=int(np.sum(labels == 0)),
        method=str(method),
        jitter=float(jitter),
        occupancy_verified=bool(occupancy_verified),
    )


def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v / ||v||``. Raises if the vector is degenerate."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length direction")
    return v / n


def _merge_close_hits(ts: np.ndarray, eps: float) -> np.ndarray:
    """
    Sort hit distances and merge values closer than ``eps``.

    Shared triangle edges otherwise produce duplicate hits that break
    enter/exit pairing.

    Parameters
    ----------
    ts:
        Unsorted ray-parameter hits.
    eps:
        Merge threshold in the same units as ``ts``.

    Returns
    -------
    ndarray
        Sorted, deduplicated 1-D array (possibly empty).
    """
    if ts.size == 0:
        return ts
    ts = np.sort(ts.astype(np.float64))
    kept = [float(ts[0])]
    for t in ts[1:]:
        if abs(float(t) - kept[-1]) > eps:
            kept.append(float(t))
    return np.asarray(kept, dtype=np.float64)


def _sample_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    t0: float,
    t1: float,
    spacing: float,
    *,
    inset: float,
) -> np.ndarray:
    """
    Sample points on ``[t0, t1]`` along the ray, slightly inset from endpoints.

    Parameters
    ----------
    origin, direction:
        Ray origin ``(3,)`` and unit direction ``(3,)``.
    t0, t1:
        Interval along the ray (``t1`` should be ``>= t0``).
    spacing:
        Approximate distance between samples.
    inset:
        Trim this much from each end so samples do not sit on the surface.

    Returns
    -------
    ndarray
        Points ``(M, 3)``, or empty if the interval is invalid.
    """
    if t1 <= t0:
        return np.zeros((0, 3), dtype=np.float64)
    a = t0 + inset
    b = t1 - inset
    if b <= a:
        mid = 0.5 * (t0 + t1)
        return (origin + mid * direction).reshape(1, 3)
    length = b - a
    n = max(1, int(np.floor(length / spacing)) + 1)
    ts = np.linspace(a, b, n, dtype=np.float64)
    return origin[None, :] + ts[:, None] * direction[None, :]


def _ray_grid_on_bbox(
    bounds: np.ndarray,
    axis: int,
    ray_grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build ray origins on the min face of ``axis``, direction along +axis.

    Parameters
    ----------
    bounds:
        AABB ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``.
    axis:
        Coordinate index ``0|1|2``.
    ray_grid:
        Counts ``(n0, n1)`` on the two axes orthogonal to ``axis``.

    Returns
    -------
    origins:
        ``(n0 * n1, 3)`` starting points.
    direction:
        Unit vector along +axis, shape ``(3,)``.
    length:
        AABB extent along ``axis``.
    """
    bmin = bounds[0].astype(np.float64)
    bmax = bounds[1].astype(np.float64)
    n0, n1 = ray_grid
    if n0 < 1 or n1 < 1:
        raise ValueError("ray_grid dimensions must be >= 1")

    other = [i for i in range(3) if i != axis]
    u_ax, v_ax = other

    def centers(lo: float, hi: float, n: int) -> np.ndarray:
        edges = np.linspace(lo, hi, n + 1, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    us = centers(float(bmin[u_ax]), float(bmax[u_ax]), n0)
    vs = centers(float(bmin[v_ax]), float(bmax[v_ax]), n1)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    origins = np.zeros((n0 * n1, 3), dtype=np.float64)
    origins[:, u_ax] = uu.ravel()
    origins[:, v_ax] = vv.ravel()
    origins[:, axis] = bmin[axis]

    direction = np.zeros(3, dtype=np.float64)
    direction[axis] = 1.0
    length = float(bmax[axis] - bmin[axis])
    return origins, direction, length


def auto_ray_grid_from_bbox(
    bounds: np.ndarray,
    axis: AxisName | int,
    spacing: float,
    *,
    min_rays: int = 2,
    max_rays: int = 128,
) -> tuple[int, int]:
    """
    Choose a ray grid so face spacing ≈ ``spacing`` in world units.

    Parameters
    ----------
    bounds:
        Mesh AABB, shape ``(2, 3)``.
    axis:
        Ray axis name ``x|y|z`` or index ``0|1|2``.
    spacing:
        Target world-space step (must be ``> 0``).
    min_rays, max_rays:
        Clamp for each face dimension.

    Returns
    -------
    tuple of int
        ``(n0, n1)`` ray counts on the two face axes.
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    if min_rays < 1:
        raise ValueError("min_rays must be >= 1")
    if max_rays < min_rays:
        raise ValueError("max_rays must be >= min_rays")

    if isinstance(axis, str):
        if axis not in _AXIS_INDEX:
            raise ValueError(f"axis must be x|y|z, got {axis!r}")
        axis_i = _AXIS_INDEX[axis]
    else:
        axis_i = int(axis)
        if axis_i not in (0, 1, 2):
            raise ValueError("axis index must be 0, 1, or 2")

    bounds = np.asarray(bounds, dtype=np.float64)
    extents = np.maximum(bounds[1] - bounds[0], 1e-12)
    other = [i for i in range(3) if i != axis_i]
    n0 = int(np.clip(int(np.round(extents[other[0]] / spacing)), min_rays, max_rays))
    n1 = int(np.clip(int(np.round(extents[other[1]] / spacing)), min_rays, max_rays))
    return (n0, n1)


def _fit_raycast_to_budget(
    bounds: np.ndarray,
    axis_i: int,
    ray_grid: tuple[int, int],
    spacing: float,
    *,
    max_points: int,
    min_rays: int = 2,
) -> tuple[tuple[int, int], float]:
    """
    Shrink the ray grid and/or coarsen spacing so a worst-case sample
    count stays under ``max_points``.

    Parameters
    ----------
    bounds:
        Mesh AABB, shape ``(2, 3)``.
    axis_i:
        Ray axis index ``0|1|2``.
    ray_grid:
        Starting ``(n0, n1)``.
    spacing:
        Starting point spacing along each ray.
    max_points:
        Hard cap (must be ``>= 8``).
    min_rays:
        Floor for each grid dimension.

    Returns
    -------
    ray_grid:
        Possibly reduced ``(n0, n1)``.
    spacing:
        Possibly increased step.
    """
    if max_points < 8:
        raise ValueError("max_points must be >= 8")
    bounds = np.asarray(bounds, dtype=np.float64)
    bbox_len = float(max(bounds[1, axis_i] - bounds[0, axis_i], 1e-12))
    n0, n1 = int(ray_grid[0]), int(ray_grid[1])
    used = float(spacing)

    def estimate(a: int, b: int, sp: float) -> int:
        per_ray = max(1, int(np.floor(bbox_len / sp)) + 1)
        return int(a) * int(b) * per_ray

    for _ in range(80):
        if estimate(n0, n1, used) <= max_points:
            return (max(min_rays, n0), max(min_rays, n1)), used
        if n0 * n1 > min_rays * min_rays:
            scale = float(np.sqrt(max_points / max(estimate(n0, n1, used), 1)))
            scale = min(scale, 0.9)
            n0 = max(min_rays, int(np.floor(n0 * scale)))
            n1 = max(min_rays, int(np.floor(n1 * scale)))
        else:
            used *= 1.25
    return (max(min_rays, n0), max(min_rays, n1)), used


def scatter_volume_raycast(
    mesh: trimesh.Trimesh,
    *,
    axis: AxisName = "z",
    ray_grid: tuple[int, int] = (32, 32),
    point_spacing: float = 0.05,
    include_outside: bool = True,
    auto_rays: bool = False,
    min_rays: int = 2,
    max_rays: int = 128,
    max_points: int = 1_500_000,
    hit_merge_eps: Optional[float] = None,
    endpoint_inset: Optional[float] = None,
) -> ScatterResult:
    """
    Sample along bbox-face rays. Inside = paired hit intervals (0-1), (2-3), …

    ``auto_rays`` derives the 2D ray count from bbox face size / spacing.
    ``max_points`` caps a worst-case budget by shrinking the grid or spacing.

    Parameters
    ----------
    mesh:
        Triangle mesh in world coordinates.
    axis:
        Ray direction ``x|y|z``.
    ray_grid:
        Face sampling ``(n0, n1)`` unless ``auto_rays`` is True.
    point_spacing:
        Step along each ray (must be ``> 0``).
    include_outside:
        If True, also sample complementary intervals inside the AABB.
    auto_rays:
        Derive ``ray_grid`` from AABB face size / ``point_spacing``.
    min_rays, max_rays:
        Clamps for auto grid.
    max_points:
        Worst-case sample budget.
    hit_merge_eps:
        Duplicate-hit merge distance; default is derived from spacing / AABB.
    endpoint_inset:
        Trim from segment ends; default is derived from spacing / AABB.

    Returns
    -------
    ScatterResult
        ``method='raycast'``, ``jitter=0``, ``occupancy_verified=False``.
        Apply :func:`scatter_volume` with ``random_range > 0`` to jitter
        and relabel.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be x|y|z, got {axis!r}")
    if point_spacing <= 0:
        raise ValueError("point_spacing must be > 0")

    axis_i = _AXIS_INDEX[axis]
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if auto_rays:
        ray_grid = auto_ray_grid_from_bbox(
            bounds, axis_i, point_spacing, min_rays=min_rays, max_rays=max_rays
        )
    ray_grid, point_spacing = _fit_raycast_to_budget(
        bounds, axis_i, ray_grid, point_spacing, max_points=max_points, min_rays=min_rays
    )
    origins, direction, bbox_len = _ray_grid_on_bbox(bounds, axis_i, ray_grid)
    direction = _unit(direction)

    if hit_merge_eps is None:
        hit_merge_eps = max(point_spacing * 1e-3, bbox_len * 1e-6, 1e-8)
    if endpoint_inset is None:
        endpoint_inset = min(point_spacing * 0.25, bbox_len * 1e-3)

    o3d_mesh = trimesh_to_open3d(mesh)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    rays_np = np.concatenate(
        [origins, np.repeat(direction[None, :], len(origins), axis=0)],
        axis=1,
    ).astype(np.float32)
    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
    hits = scene.list_intersections(rays)

    t_all = hits["t_hit"].numpy().astype(np.float64)
    splits = hits["ray_splits"].numpy().astype(np.int64)

    inside_parts: list[np.ndarray] = []
    outside_parts: list[np.ndarray] = []

    for r in range(len(origins)):
        a = int(splits[r])
        b = int(splits[r + 1])
        ts = _merge_close_hits(t_all[a:b], hit_merge_eps)
        ts = ts[(ts >= 0.0) & (ts <= bbox_len)]
        origin = origins[r]

        for i in range(0, len(ts) - 1, 2):
            seg = _sample_segment(
                origin, direction, float(ts[i]), float(ts[i + 1]),
                point_spacing, inset=endpoint_inset,
            )
            if len(seg):
                inside_parts.append(seg)

        if not include_outside:
            continue

        out_intervals: list[tuple[float, float]] = []
        if len(ts) == 0:
            out_intervals.append((0.0, bbox_len))
        else:
            out_intervals.append((0.0, float(ts[0])))
            for i in range(1, len(ts) - 1, 2):
                out_intervals.append((float(ts[i]), float(ts[i + 1])))
            out_intervals.append((float(ts[-1]), bbox_len))

        for t0, t1 in out_intervals:
            seg = _sample_segment(
                origin, direction, t0, t1, point_spacing, inset=endpoint_inset
            )
            if len(seg):
                outside_parts.append(seg)

    inside = (
        np.concatenate(inside_parts, axis=0)
        if inside_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    outside = (
        np.concatenate(outside_parts, axis=0)
        if outside_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    points = np.concatenate([inside, outside], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(inside), dtype=np.uint8),
            np.zeros(len(outside), dtype=np.uint8),
        ],
        axis=0,
    )
    return _make_result(
        points,
        labels,
        axis=axis,
        ray_grid=(int(ray_grid[0]), int(ray_grid[1])),
        point_spacing=float(point_spacing),
        method="raycast",
        jitter=0.0,
        occupancy_verified=False,
    )


def _padded_bounds(bounds: np.ndarray, pad: float) -> np.ndarray:
    """
    Expand an AABB by ``pad`` on every side.

    Parameters
    ----------
    bounds:
        ``(2, 3)`` min/max corners.
    pad:
        World-space margin (may be 0).

    Returns
    -------
    ndarray
        Copy of ``bounds`` with min decreased and max increased by ``pad``.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    out = bounds.copy()
    out[0] -= pad
    out[1] += pad
    return out


def _uniform_grid_points(
    bounds: np.ndarray,
    spacing: float,
    *,
    max_points: int = 500_000,
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """
    Axis-aligned 3D lattice inside ``bounds``. Coarsen spacing if over budget.

    Parameters
    ----------
    bounds:
        ``(2, 3)`` sample AABB.
    spacing:
        Target step on each axis (must be ``> 0``).
    max_points:
        If ``nx * ny * nz`` would exceed this, spacing is multiplied by 1.25
        until it fits.

    Returns
    -------
    points:
        ``(N, 3)`` float64 lattice.
    used_spacing:
        Spacing actually used (``>= spacing``).
    grid:
        ``(nx, ny, nz)`` counts.
    """
    if spacing <= 0:
        raise ValueError("point_spacing must be > 0")
    bmin = bounds[0].astype(np.float64)
    bmax = bounds[1].astype(np.float64)
    extents = np.maximum(bmax - bmin, 1e-12)
    used = float(spacing)

    def counts(step: float) -> tuple[int, int, int]:
        return tuple(max(2, int(np.floor(extents[i] / step)) + 1) for i in range(3))  # type: ignore[return-value]

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


def apply_point_jitter(
    points: np.ndarray,
    jitter: float,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Independent uniform offset ``[-jitter, +jitter]`` on each axis.

    Labels must be recomputed after this call.

    Parameters
    ----------
    points:
        ``(N, 3)`` positions.
    jitter:
        Half-width of the cube offset (must be ``>= 0``). ``0`` is a no-op.
    seed:
        Used only when ``rng`` is omitted.
    rng:
        Optional NumPy Generator; otherwise ``default_rng(seed)``.

    Returns
    -------
    ndarray
        Jittered copy of ``points`` (same shape).
    """
    if jitter < 0:
        raise ValueError("jitter must be >= 0")
    points = np.asarray(points, dtype=np.float64)
    if jitter == 0.0 or len(points) == 0:
        return points.copy()
    if rng is None:
        rng = np.random.default_rng(seed)
    noise = rng.uniform(-jitter, jitter, size=points.shape)
    return points + noise


def _fibonacci_unit_dirs(n: int) -> np.ndarray:
    """
    Evenly spaced unit directions on the sphere (Fibonacci / golden spiral).

    Parameters
    ----------
    n:
        Number of directions (must be ``>= 2``).

    Returns
    -------
    ndarray
        ``(n, 3)`` float64 unit vectors.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    i = np.arange(n, dtype=np.float64)
    # Even y-band spacing; golden-angle azimuth. Not axis-aligned on purpose.
    y = 1.0 - (i / (n - 1.0)) * 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    dirs = np.stack((np.cos(theta) * r, y, np.sin(theta) * r), axis=1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _occupancy_labels(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    chunk_size: int = 250_000,
    n_rays: int = _OCC_N_RAYS,
    vote_min: int = _OCC_VOTE_MIN,
) -> np.ndarray:
    """
    Per-point inside/outside via multi-ray odd-hit vote (1=inside, 0=outside).

    Open3D ``compute_occupancy`` is generalized winding number. On an open
    or non-manifold shell that fills a ghost slab through the hole (the
    nr4/nr5 Truth leaks). Here each query fires ``n_rays`` directions and
    counts triangle hits. Odd = that ray thinks inside. The point is inside
    only if at least ``vote_min`` rays agree.

    Parameters
    ----------
    mesh:
        Triangle mesh.
    points:
        Query XYZ ``(N, 3)``.
    chunk_size:
        Max query points per Open3D call before splitting (ray count is
        ``chunk * n_rays``).
    n_rays:
        Directions on the sphere (``>= 2``).
    vote_min:
        Odd-hit rays required to label inside (``1..n_rays``).

    Returns
    -------
    ndarray
        uint8 labels, shape ``(N,)``.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    if n_rays < 2:
        raise ValueError("n_rays must be >= 2")
    if vote_min < 1 or vote_min > n_rays:
        raise ValueError("vote_min must be in 1..n_rays")

    o3d_mesh = trimesh_to_open3d(mesh)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    dirs = _fibonacci_unit_dirs(n_rays)
    # Cap batched rays (~0.5M) so a dense lattice does not allocate a huge tensor.
    point_chunk = max(1, min(int(chunk_size), max(1, 500_000 // int(n_rays))))

    def _label_chunk(pts: np.ndarray) -> np.ndarray:
        m = len(pts)
        # (M, D, 3) origins / dirs → (M*D, 6) Open3D rays.
        origins = np.repeat(pts[:, None, :], n_rays, axis=1)
        directions = np.broadcast_to(dirs, (m, n_rays, 3))
        rays = np.concatenate((origins, directions), axis=2).reshape(m * n_rays, 6)
        ray_t = o3d.core.Tensor(rays.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        counts = scene.count_intersections(ray_t).numpy().reshape(m, n_rays)
        odd = (counts % 2) == 1
        return (odd.sum(axis=1) >= vote_min).astype(np.uint8)

    if n <= point_chunk:
        return _label_chunk(points)

    labels = np.empty(n, dtype=np.uint8)
    for start in range(0, n, point_chunk):
        end = min(start + point_chunk, n)
        labels[start:end] = _label_chunk(points[start:end])
    return labels


def scatter_volume_occupancy(
    mesh: trimesh.Trimesh,
    *,
    point_spacing: float = 0.05,
    include_outside: bool = True,
    bbox_pad: Optional[float] = None,
    max_points: int = 500_000,
    jitter: float = 0.0,
    seed: Optional[int] = None,
) -> ScatterResult:
    """
    Fill a padded AABB with a 3D grid, optionally jitter, then occupancy-label.

    Pad is ``point_spacing + jitter`` when keeping outside samples so a solid
    that fills its bbox still has an outside shell, and jittered points can
    leave the unpadded box.

    Parameters
    ----------
    mesh:
        Triangle mesh.
    point_spacing:
        Lattice step (must be ``> 0``).
    include_outside:
        If False, drop points labeled outside after occupancy.
    bbox_pad:
        Extra AABB margin. Default ``point_spacing + jitter`` when
        ``include_outside`` else ``jitter``.
    max_points:
        Lattice budget (must be ``>= 8``).
    jitter:
        Per-axis ``U[-j, +j]`` applied **before** labeling (``>= 0``).
    seed:
        RNG seed for jitter.

    Returns
    -------
    ScatterResult
        ``method='occupancy'``, ``occupancy_verified=True``.
    """
    if point_spacing <= 0:
        raise ValueError("point_spacing must be > 0")
    if max_points < 8:
        raise ValueError("max_points must be >= 8")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bbox_pad is None:
        base = float(point_spacing) if include_outside else 0.0
        bbox_pad = base + float(jitter)
    if bbox_pad < 0:
        raise ValueError("bbox_pad must be >= 0")

    sample_bounds = _padded_bounds(bounds, float(bbox_pad))
    points, used_spacing, grid = _uniform_grid_points(
        sample_bounds, point_spacing, max_points=max_points
    )
    # Order: lattice → offset → labels on the moved points.
    points = apply_point_jitter(points, jitter, seed=seed)
    labels = _occupancy_labels(mesh, points)

    if not include_outside:
        keep = labels == 1
        points = points[keep]
        labels = labels[keep]

    return _make_result(
        points,
        labels,
        axis="n",
        ray_grid=(int(grid[0]), int(grid[1])),
        point_spacing=float(used_spacing),
        method="occupancy",
        jitter=float(jitter),
        occupancy_verified=True,
    )


def _jitter_and_relabel(
    mesh: trimesh.Trimesh,
    result: ScatterResult,
    *,
    jitter: float,
    seed: Optional[int],
    include_outside: bool,
) -> ScatterResult:
    """
    Keep the sample count/pattern, move points, then occupancy-label.

    Parameters
    ----------
    mesh:
        Triangle mesh used for occupancy.
    result:
        Existing samples (typically raycast lattice).
    jitter:
        Per-axis offset half-width.
    seed:
        RNG seed for jitter.
    include_outside:
        If False, drop outside points after relabel.

    Returns
    -------
    ScatterResult
        Same ``method`` / ``axis`` / spacing; ``occupancy_verified=True``.
    """
    points = apply_point_jitter(result.points, jitter, seed=seed)
    labels = _occupancy_labels(mesh, points)
    if not include_outside:
        keep = labels == 1
        points = points[keep]
        labels = labels[keep]
    return _make_result(
        points,
        labels,
        axis=result.axis,
        ray_grid=result.ray_grid,
        point_spacing=result.point_spacing,
        method=result.method,
        jitter=float(jitter),
        occupancy_verified=True,
    )


def scatter_volume(
    mesh: trimesh.Trimesh,
    *,
    method: ScatterMethod | str = "occupancy",
    axis: AxisName = "z",
    ray_grid: tuple[int, int] = (32, 32),
    point_spacing: float = 0.05,
    include_outside: bool = True,
    auto_rays: bool = False,
    min_rays: int = 2,
    max_rays: int = 128,
    bbox_pad: Optional[float] = None,
    max_points: int = 1_500_000,
    jitter: float = 0.0,
    random_range: Optional[float] = None,
    seed: Optional[int] = None,
    hit_merge_eps: Optional[float] = None,
    endpoint_inset: Optional[float] = None,
) -> ScatterResult:
    """
    Place labeled occupancy queries.

    ``random_range`` is the protocol name for ``jitter``. If both are passed,
    ``random_range`` wins. After placement: offset (no-op if ``r = 0``), then
    classify the moved points (occupancy path always; raycast path when ``r > 0``).

    Parameters
    ----------
    mesh:
        Triangle mesh.
    method:
        ``occupancy`` (3D lattice) or ``raycast`` (bbox-face rays).
    axis, ray_grid, auto_rays, min_rays, max_rays, hit_merge_eps, endpoint_inset:
        Raycast knobs; ignored by the occupancy lattice.
    point_spacing:
        Density step.
    include_outside:
        Keep outside samples (default True).
    bbox_pad:
        Occupancy-grid AABB pad; default depends on spacing/jitter.
    max_points:
        Cap sample count by coarsening density.
    jitter:
        Per-axis uniform half-width. Overridden by ``random_range`` if set.
    random_range:
        Protocol alias for ``jitter``.
    seed:
        RNG seed for the offset.

    Returns
    -------
    ScatterResult
        Points and labels; labels are always computed on final positions
        when ``jitter > 0``.
    """
    method = str(method).lower().strip()
    if random_range is not None:
        jitter = float(random_range)
    if jitter < 0:
        raise ValueError("jitter / random_range must be >= 0")

    if method == "raycast":
        result = scatter_volume_raycast(
            mesh,
            axis=axis,
            ray_grid=ray_grid,
            point_spacing=point_spacing,
            include_outside=include_outside,
            auto_rays=auto_rays,
            min_rays=min_rays,
            max_rays=max_rays,
            max_points=max_points,
            hit_merge_eps=hit_merge_eps,
            endpoint_inset=endpoint_inset,
        )
        if jitter > 0:
            result = _jitter_and_relabel(
                mesh,
                result,
                jitter=jitter,
                seed=seed,
                include_outside=include_outside,
            )
        return result

    if method == "occupancy":
        return scatter_volume_occupancy(
            mesh,
            point_spacing=point_spacing,
            include_outside=include_outside,
            bbox_pad=bbox_pad,
            max_points=max_points,
            jitter=jitter,
            seed=seed,
        )

    raise ValueError(f"Unknown scatter method {method!r}; use 'raycast' or 'occupancy'")


def export_scatter_npz(
    result: ScatterResult,
    out_path: str | Path,
    *,
    mesh_path: str | None = None,
) -> Path:
    """
    Write ``points``, ``labels``, ``mesh_path``, and sampling metadata.

    Parameters
    ----------
    result:
        Scatter cloud to serialize.
    out_path:
        Destination ``.npz`` (suffix is added if missing).
    mesh_path:
        Source mesh path stored as repo-relative when possible.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": result.points.astype(np.float32),
        "labels": result.labels.astype(np.uint8),
        "axis": np.asarray(result.axis),
        "ray_grid": np.asarray(result.ray_grid, dtype=np.int32),
        "point_spacing": np.asarray(result.point_spacing, dtype=np.float32),
        "method": np.asarray(result.method),
        "jitter": np.asarray(result.jitter, dtype=np.float32),
        "random_range": np.asarray(result.jitter, dtype=np.float32),
        "occupancy_verified": np.asarray(bool(result.occupancy_verified)),
    }
    if mesh_path is not None:
        payload["mesh_path"] = np.asarray(to_data_relative(mesh_path))
    np.savez(out_path, **payload)
    return out_path.resolve()


def load_scatter_npz(path: str | Path) -> tuple[ScatterResult, str | None]:
    """
    Load a scatter NPZ.

    Parameters
    ----------
    path:
        ``.npz`` written by :func:`export_scatter_npz`.

    Returns
    -------
    result:
        :class:`ScatterResult` reconstructed from arrays.
    mesh_path:
        Stored mesh path string, or ``None`` if the key is absent.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "points" not in data or "labels" not in data:
        raise ValueError(f"NPZ missing points/labels: {path}")

    axis = str(np.asarray(data["axis"]).item()) if "axis" in data else "z"
    ray_grid = (0, 0)
    if "ray_grid" in data:
        rg = np.asarray(data["ray_grid"]).reshape(-1)
        if len(rg) >= 2:
            ray_grid = (int(rg[0]), int(rg[1]))
    spacing = (
        float(np.asarray(data["point_spacing"]).item()) if "point_spacing" in data else 0.0
    )
    method = str(np.asarray(data["method"]).item()) if "method" in data else "occupancy"
    jitter = float(np.asarray(data["jitter"]).item()) if "jitter" in data else 0.0
    if jitter == 0.0 and "random_range" in data:
        jitter = float(np.asarray(data["random_range"]).item())
    occupancy_verified = (
        bool(np.asarray(data["occupancy_verified"]).item())
        if "occupancy_verified" in data
        else False
    )
    mesh_path = (
        str(np.asarray(data["mesh_path"]).item()) if "mesh_path" in data else None
    )
    result = _make_result(
        data["points"],
        data["labels"],
        axis=axis,
        ray_grid=ray_grid,
        point_spacing=spacing,
        method=method,
        jitter=jitter,
        occupancy_verified=occupancy_verified,
    )
    return result, mesh_path


def export_occupancy_npz(
    mesh_path: str | Path,
    out_path: str | Path,
    *,
    method: ScatterMethod | str = "occupancy",
    point_spacing: float = 0.1,
    random_range: float = 0.0,
    seed: Optional[int] = None,
    include_outside: bool = True,
    axis: AxisName = "z",
    auto_rays: bool = True,
    max_points: int = 50_000,
) -> Path:
    """
    Load one OBJ, scatter queries, write a Phase-1 NPZ.

    Parameters
    ----------
    mesh_path:
        Input mesh file.
    out_path:
        Destination NPZ.
    method:
        ``occupancy`` (default) or ``raycast``.
    point_spacing:
        Density step.
    random_range:
        Per-axis offset after the lattice (``0`` = no move).
    seed:
        RNG seed for ``random_range``.
    include_outside:
        Keep outside points (default True).
    axis, auto_rays:
        Raycast knobs.
    max_points:
        Sample budget.

    Returns
    -------
    Path
        Resolved NPZ path. Arrays include ``points``, ``labels``, ``mesh_path``.
    """
    mesh, _info = load_mesh(mesh_path)
    result = scatter_volume(
        mesh,
        method=method,
        axis=axis,
        point_spacing=float(point_spacing),
        include_outside=include_outside,
        auto_rays=auto_rays,
        max_points=int(max_points),
        random_range=float(random_range),
        seed=seed,
    )
    return export_scatter_npz(result, out_path, mesh_path=str(mesh_path))
