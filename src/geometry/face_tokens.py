"""Triangle face tokens: three corners plus a unit normal.

Used by the viewer Faces overlay, not by occupancy train/infer.
``FACE_FEAT_DIM = 12``: ``[v0(3), v1(3), v2(3), n(3)]`` in the mesh frame.
Too many faces → keep the largest by area. Too few → repeat rows so the
length matches the requested count. World-space arrays are cached per mesh key.
AABB is applied to the three corners only; normals stay unit directions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TokensArray = NDArray[np.float32]

FACE_FEAT_DIM = 12

# Same resolved OBJ + count → same world tokens (several NPZs share a mesh).
_TOKEN_CACHE: dict[tuple[str, int], TokensArray] = {}


def clear_face_token_cache() -> None:
    """Drop cached world-space tokens (tests / long-lived notebooks)."""
    _TOKEN_CACHE.clear()


def apply_face_aabb(
    tokens: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> TokensArray:
    """
    Phase 1 AABB on the three corners. Normals are not translated or scaled.

    Parameters
    ----------
    tokens:
        ``(F, 12)`` face features.
    center, scale:
        Same AABB used on query XYZ.

    Returns
    -------
    ndarray
        ``float32 (F, 12)``.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    out = np.asarray(tokens, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != FACE_FEAT_DIM:
        raise ValueError(f"tokens must have shape (F, {FACE_FEAT_DIM}), got {tuple(out.shape)}")
    c = np.asarray(center, dtype=np.float32).reshape(3)
    s = np.float32(scale)
    # Corners occupy columns 0:3, 3:6, 6:9. Last three are the unit normal.
    for k in (0, 3, 6):
        out[:, k : k + 3] = (out[:, k : k + 3] - c) / s
    return out


def undo_face_aabb(
    tokens: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> TokensArray:
    """Inverse of :func:`apply_face_aabb` (corners only)."""
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    out = np.asarray(tokens, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != FACE_FEAT_DIM:
        raise ValueError(f"tokens must have shape (F, {FACE_FEAT_DIM}), got {tuple(out.shape)}")
    c = np.asarray(center, dtype=np.float32).reshape(3)
    s = np.float32(scale)
    for k in (0, 3, 6):
        out[:, k : k + 3] = out[:, k : k + 3] * s + c
    return out


def _resize_rows(feats: np.ndarray, n_faces: int) -> TokensArray:
    """Keep largest faces by area, or tile rows, so the count is ``n_faces``."""
    n = int(feats.shape[0])
    count = int(n_faces)
    if n == count:
        return feats.astype(np.float32, copy=False)
    if n > count:
        v0 = feats[:, 0:3]
        v1 = feats[:, 3:6]
        v2 = feats[:, 6:9]
        # Triangle area from the same cross product that built the normal.
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        order = np.argsort(-areas, kind="stable")
        return feats[order[:count]].astype(np.float32, copy=False)
    reps = int(np.ceil(count / n))
    tiled = np.tile(feats, (reps, 1))
    return tiled[:count].astype(np.float32, copy=False)


def face_tokens_from_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_faces: int,
    *,
    cache_key: str | None = None,
) -> TokensArray:
    """
    Build ``(n_faces, 12)`` tokens in **world** coordinates.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` triangle corners.
    faces:
        ``(T, 3)`` integer vertex indices.
    n_faces:
        YAML length (must be ``>= 1``).
    cache_key:
        Optional cache identity (resolved OBJ path). ``None`` skips the cache.

    Returns
    -------
    ndarray
        ``float32`` array of shape ``(n_faces, 12)``.
    """
    count = int(n_faces)
    if count < 1:
        raise ValueError(f"n_faces must be >= 1, got {count}")
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {tuple(verts.shape)}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must have shape (T, 3), got {tuple(tris.shape)}")
    if int(tris.shape[0]) < 1:
        raise ValueError("need at least one triangle")

    key: tuple[str, int] | None = None
    if cache_key is not None:
        key = (str(cache_key), count)
        cached = _TOKEN_CACHE.get(key)
        if cached is not None:
            return cached

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / np.clip(nlen, 1e-12, None)
    feats = np.concatenate(
        [
            v0.astype(np.float32),
            v1.astype(np.float32),
            v2.astype(np.float32),
            normals.astype(np.float32),
        ],
        axis=1,
    )
    out = _resize_rows(feats, count)
    if out.shape != (count, FACE_FEAT_DIM):
        raise ValueError(
            f"expected token shape {(count, FACE_FEAT_DIM)}, got {tuple(out.shape)}"
        )
    if key is not None:
        _TOKEN_CACHE[key] = out
    return out
