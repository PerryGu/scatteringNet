"""Surface envelope samples from a triangle mesh.

Area-weighted face darts (larger triangles get more samples). Each
sample is ``(x, y, z, nx, ny, nz)``: position plus the unit normal of
the triangle it sits on. World clouds are cached per
``(mesh_key, n_surface, seed)``. AABB is applied to XYZ only.
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.sample import sample_surface

from normalize import apply_normalization

PointsArray = NDArray[np.float32]
ENVELOPE_XYZ_DIM = 3
ENVELOPE_FEAT_DIM = 6

# Same OBJ + count + seed → same world samples (always 6-D).
_ENVELOPE_CACHE: dict[tuple[str, int, int], PointsArray] = {}


def clear_envelope_cache() -> None:
    """Drop cached world-space envelopes (tests / long-lived notebooks)."""
    _ENVELOPE_CACHE.clear()


def _triangle_normals(verts: np.ndarray, faces: np.ndarray) -> NDArray[np.float64]:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    ok = length[:, 0] > 1e-12
    normals = np.zeros_like(cross)
    normals[ok] = cross[ok] / length[ok]
    return normals


def _pack_xyz_normal(xyz: np.ndarray, normals: np.ndarray) -> PointsArray:
    """Concatenate XYZ with unit face normals → ``(N, 6)``."""
    pos = np.asarray(xyz, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32)
    if pos.shape != nrm.shape or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"xyz/normals must be (N, 3), got {tuple(pos.shape)} / {tuple(nrm.shape)}"
        )
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    ok = length[:, 0] > 1e-12
    unit = np.zeros_like(nrm)
    unit[ok] = nrm[ok] / length[ok]
    return np.concatenate([pos, unit], axis=1)


def apply_envelope_aabb(
    env: np.ndarray, center: np.ndarray, scale: float
) -> PointsArray:
    """AABB-normalize XYZ; leave normals as unit directions."""
    arr = np.asarray(env, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (ENVELOPE_XYZ_DIM, ENVELOPE_FEAT_DIM):
        raise ValueError(
            f"envelope must be (N, 3) or (N, 6), got {tuple(arr.shape)}"
        )
    out = arr.copy()
    out[:, :3] = apply_normalization(out[:, :3], center, scale)
    return out


def undo_envelope_aabb(
    env: np.ndarray, center: np.ndarray, scale: float
) -> PointsArray:
    """Undo AABB on XYZ only."""
    arr = np.asarray(env, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (ENVELOPE_XYZ_DIM, ENVELOPE_FEAT_DIM):
        raise ValueError(
            f"envelope must be (N, 3) or (N, 6), got {tuple(arr.shape)}"
        )
    out = arr.copy()
    c = np.asarray(center, dtype=np.float32).reshape(3)
    out[:, :3] = out[:, :3] * np.float32(scale) + c
    return out


def project_envelope_dim(env: np.ndarray, dim: int) -> PointsArray:
    """Keep XYZ+normal or drop to XYZ for an older checkpoint."""
    arr = np.asarray(env, dtype=np.float32)
    want = int(dim)
    if want == ENVELOPE_FEAT_DIM:
        if arr.ndim != 2 or arr.shape[1] != ENVELOPE_FEAT_DIM:
            raise ValueError(f"expected (N, 6) envelope, got {tuple(arr.shape)}")
        return arr
    if want == ENVELOPE_XYZ_DIM:
        if arr.ndim != 2 or arr.shape[1] not in (ENVELOPE_XYZ_DIM, ENVELOPE_FEAT_DIM):
            raise ValueError(f"expected (N, 3|6) envelope, got {tuple(arr.shape)}")
        return arr[:, :3]
    raise ValueError(f"envelope dim must be 3 or 6, got {want}")


def _sample_area(
    verts: np.ndarray, tris: np.ndarray, count: int, seed: int
) -> PointsArray:
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    points, face_idx = sample_surface(mesh, count, seed=int(seed))
    nrm = _triangle_normals(verts, tris)[np.asarray(face_idx, dtype=np.int64)]
    out = _pack_xyz_normal(points, nrm)
    if out.shape != (count, ENVELOPE_FEAT_DIM):
        raise ValueError(
            f"expected envelope shape {(count, ENVELOPE_FEAT_DIM)}, got {tuple(out.shape)}"
        )
    return out


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_surface: int,
    *,
    seed: int = 1,
    cache_key: str | None = None,
) -> PointsArray:
    """
    Sample ``n_surface`` envelope points as ``(N, 6)`` XYZ + unit normal.

    Face-area weighted: larger triangles receive more darts. No crease
    or fold path — unused ``envelope_mix`` on old checkpoints is ignored.
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

    out = _sample_area(verts, tris, count, int(seed))
    if out.shape != (count, ENVELOPE_FEAT_DIM):
        raise ValueError(
            f"expected envelope shape {(count, ENVELOPE_FEAT_DIM)}, got {tuple(out.shape)}"
        )
    if key is not None:
        _ENVELOPE_CACHE[key] = out
    return out
