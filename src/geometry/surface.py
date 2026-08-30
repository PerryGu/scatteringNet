"""Area-weighted surface envelope samples from a triangle mesh.

World-space clouds are cached per ``(mesh_key, n_surface, seed)``.
AABB normalization is applied by the caller so the envelope shares
the same ``center`` / ``scale`` as that file's query XYZ.
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.sample import sample_surface

PointsArray = NDArray[np.float32]

# Same OBJ + count + seed → same world samples (several NPZs can share a mesh).
_ENVELOPE_CACHE: dict[tuple[str, int, int], PointsArray] = {}


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_surface: int,
    *,
    seed: int = 1,
    cache_key: str | None = None,
) -> PointsArray:
    """
    Sample ``n_surface`` points on the triangle mesh (area-weighted).

    Parameters
    ----------
    vertices:
        ``(V, 3)`` triangle corners.
    faces:
        ``(T, 3)`` integer vertex indices.
    n_surface:
        How many envelope points to draw. YAML ``n_surface`` (must be ``>= 1``).
    seed:
        Passed to trimesh so the cloud is repeatable.
    cache_key:
        Optional cache identity (resolved OBJ path). ``None`` skips the cache.

    Returns
    -------
    ndarray
        ``float32`` array of shape ``(n_surface, 3)`` in **world** coordinates.
    """
    count = int(n_surface)
    if count < 1:
        raise ValueError(f"n_surface must be >= 1, got {count}")
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {tuple(verts.shape)}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must have shape (T, 3), got {tuple(tris.shape)}")

    key: tuple[str, int, int] | None = None
    if cache_key is not None:
        key = (str(cache_key), count, int(seed))
        cached = _ENVELOPE_CACHE.get(key)
        if cached is not None:
            return cached

    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    # Area-weighted face sampling; seed keeps catalog loads deterministic.
    points, _face_idx = sample_surface(mesh, count, seed=int(seed))
    out = np.asarray(points, dtype=np.float32)
    if out.shape != (count, 3):
        raise ValueError(
            f"expected envelope shape {(count, 3)}, got {tuple(out.shape)}"
        )
    if key is not None:
        _ENVELOPE_CACHE[key] = out
    return out
