"""Surface envelope samples from a triangle mesh.

``mix`` 0 is area-weighted (Step 8). ``mix`` 100 is crease-hugging (same
lottery as the viewer Mix slider). Values in between split ``n_surface``.
World clouds are cached per ``(mesh_key, n_surface, seed, mix)``.
AABB normalization is applied by the caller.
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.sample import sample_surface

PointsArray = NDArray[np.float32]

# Shared-edge dihedral at or above this is a crease (cube walls are 90°).
_SHARP_RAD = float(np.deg2rad(20.0))
_FALLOFF_FRAC = 0.02
_ON_EDGE_FRAC = 0.65

# Same OBJ + count + seed + mix → same world samples.
_ENVELOPE_CACHE: dict[tuple[str, int, int, int], PointsArray] = {}


def clear_envelope_cache() -> None:
    """Drop cached world-space envelopes (tests / long-lived notebooks)."""
    _ENVELOPE_CACHE.clear()


def clamp_envelope_mix(mix: int) -> int:
    """0 = all face-area samples, 100 = all crease samples."""
    n = int(mix)
    if n < 0 or n > 100:
        raise ValueError(f"envelope_mix must be between 0 and 100, got {n}")
    return n


def split_envelope_counts(n_surface: int, mix: int) -> tuple[int, int]:
    """Return ``(n_area, n_crease)`` summing to ``n_surface``."""
    count = int(n_surface)
    m = clamp_envelope_mix(mix)
    n_crease = int(round(count * m / 100.0))
    n_crease = min(count, max(0, n_crease))
    return count - n_crease, n_crease


def weld_vertices(
    vertices: np.ndarray, faces: np.ndarray, *, ndigits: int = 6
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Merge corners that share rounded XYZ (Maya often emits unique indices)."""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    keys = np.round(verts, decimals=int(ndigits))
    uniq, remap = np.unique(keys, axis=0, return_inverse=True)
    welded = remap[tris]
    keep = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 2] != welded[:, 0])
    )
    return uniq.astype(np.float64, copy=False), welded[keep]


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


def _edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(fi)
    return edge_faces


def _sharp_edges(
    verts: np.ndarray, faces: np.ndarray
) -> list[tuple[int, int, list[int]]]:
    normals = _triangle_normals(verts, faces)
    sharp: list[tuple[int, int, list[int]]] = []
    for (i0, i1), incident in _edge_map(faces).items():
        if len(incident) != 2:
            continue
        a, b = incident[0], incident[1]
        dot = float(np.clip(np.dot(normals[a], normals[b]), -1.0, 1.0))
        if float(np.arccos(dot)) >= _SHARP_RAD:
            sharp.append((i0, i1, incident))
    return sharp


def n_sharp_edges(vertices: np.ndarray, faces: np.ndarray) -> int:
    """How many welded dihedral creases the overlay/status can report."""
    verts, tris = weld_vertices(vertices, faces)
    return int(len(_sharp_edges(verts, tris)))


def _offset_into_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    dist: float,
) -> np.ndarray:
    if dist <= 0.0:
        return p
    edge = b - a
    elen = float(np.linalg.norm(edge))
    if elen < 1e-12:
        return p
    nrm = np.cross(edge, c - a)
    inward = np.cross(nrm, edge)
    ilen = float(np.linalg.norm(inward))
    if ilen < 1e-12:
        return p
    inward = inward / ilen
    if float(np.dot(inward, c - p)) < 0.0:
        inward = -inward
    height = float(np.linalg.norm(nrm)) / elen
    step = min(float(dist), 0.35 * height)
    return p + inward * step


def _sample_area(
    verts: np.ndarray, tris: np.ndarray, count: int, seed: int
) -> PointsArray:
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    points, _face_idx = sample_surface(mesh, count, seed=int(seed))
    out = np.asarray(points, dtype=np.float32)
    if out.shape != (count, 3):
        raise ValueError(f"expected envelope shape {(count, 3)}, got {tuple(out.shape)}")
    return out


def _sample_crease(
    verts: np.ndarray, tris: np.ndarray, count: int, seed: int
) -> PointsArray:
    """Weld, then sample on/near sharp edges. Area fallback if no creases."""
    welded_v, welded_f = weld_vertices(verts, tris)
    sharp = _sharp_edges(welded_v, welded_f)
    if not sharp:
        return _sample_area(verts, tris, count, seed)
    lengths = np.array(
        [float(np.linalg.norm(welded_v[i1] - welded_v[i0])) for i0, i1, _inc in sharp],
        dtype=np.float64,
    )
    total = float(lengths.sum())
    if total <= 1e-12:
        lengths = np.ones((len(sharp),), dtype=np.float64)
        total = float(len(sharp))
    probs = lengths / total
    extent = welded_v.max(axis=0) - welded_v.min(axis=0)
    sigma = max(float(np.linalg.norm(extent)) * _FALLOFF_FRAC, 1e-4)
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(len(sharp), size=count, p=probs)
    ts = rng.random(count)
    on_edge = rng.random(count) < _ON_EDGE_FRAC
    dists = np.where(
        on_edge,
        0.0,
        -sigma * np.log(np.clip(rng.random(count), 1e-12, 1.0)),
    )
    side = rng.integers(0, 2, size=count)
    out = np.empty((count, 3), dtype=np.float32)
    for k in range(count):
        i0, i1, incident = sharp[int(picks[k])]
        a = welded_v[i0]
        b = welded_v[i1]
        p = a + float(ts[k]) * (b - a)
        fi = incident[int(side[k]) % len(incident)]
        tri = welded_f[fi]
        c_idx = int(tri[0] + tri[1] + tri[2] - i0 - i1)
        q = _offset_into_triangle(p, a, b, welded_v[c_idx], float(dists[k]))
        out[k] = q.astype(np.float32)
    return out


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_surface: int,
    *,
    seed: int = 1,
    mix: int = 0,
    cache_key: str | None = None,
) -> PointsArray:
    """
    Sample ``n_surface`` envelope points.

    ``mix`` 0 is the original area-weighted cloud (old checkpoints).
    ``mix`` 100 is crease-hugging. In between, Count is split like the viewer Mix slider.
    """
    count = int(n_surface)
    if count < 1:
        raise ValueError(f"n_surface must be >= 1, got {count}")
    mix_v = clamp_envelope_mix(mix)
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {tuple(verts.shape)}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must have shape (T, 3), got {tuple(tris.shape)}")

    key: tuple[str, int, int, int] | None = None
    if cache_key is not None:
        key = (str(cache_key), count, int(seed), mix_v)
        cached = _ENVELOPE_CACHE.get(key)
        if cached is not None:
            return cached

    if mix_v == 0:
        out = _sample_area(verts, tris, count, int(seed))
    else:
        n_area, n_crease = split_envelope_counts(count, mix_v)
        chunks: list[np.ndarray] = []
        if n_area > 0:
            chunks.append(_sample_area(verts, tris, n_area, int(seed)))
        if n_crease > 0:
            chunks.append(_sample_crease(verts, tris, n_crease, int(seed) + 1))
        out = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if out.shape != (count, 3):
        raise ValueError(
            f"expected envelope shape {(count, 3)}, got {tuple(out.shape)}"
        )
    if key is not None:
        _ENVELOPE_CACHE[key] = out
    return out
