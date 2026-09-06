"""Surface envelope samples from a triangle mesh.

``mix`` 0 is area-weighted (Step 8). ``mix`` 100 is crease-hugging (same
lottery as the viewer Mix slider). Values in between split ``n_surface``.
If sharp edges are only a small fraction of the mesh, unused crease
slots spill back to face-area samples so a few folds cannot take most
of the cloud. Each sample is ``(x, y, z, nx, ny, nz)``: position plus
the unit normal of the triangle it sits on. World clouds are cached per
``(mesh_key, n_surface, seed, mix)``. AABB is applied to XYZ only.
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

# Shared-edge dihedral at or above this is a crease (cube walls are 90°).
_SHARP_RAD = float(np.deg2rad(20.0))
_FALLOFF_FRAC = 0.02
_ON_EDGE_FRAC = 0.65
# A fold that is 10% of interior edges may take at most 20% of the envelope.
_CREASE_OVERREP = 2.0

# Same OBJ + count + seed + mix → same world samples (always 6-D).
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
    """Return Mix-only ``(n_area, n_crease)`` summing to ``n_surface``."""
    count = int(n_surface)
    m = clamp_envelope_mix(mix)
    n_crease = int(round(count * m / 100.0))
    n_crease = min(count, max(0, n_crease))
    return count - n_crease, n_crease


def plan_envelope_counts(
    n_surface: int, mix: int, crease_frac: float
) -> tuple[int, int]:
    """
    Mix split, then cap crease dots by how much of the mesh is sharp.

    ``crease_frac`` is sharp-edge length / interior-edge length (0–1).
    Leftover crease slots spill to area so ``n_surface`` stays exact.
    """
    count = int(n_surface)
    n_area, n_crease = split_envelope_counts(count, mix)
    if n_crease < 1:
        return n_area, 0
    frac = float(crease_frac)
    if frac < 0.0:
        frac = 0.0
    mix_frac = float(clamp_envelope_mix(mix)) / 100.0
    # Do not let Mix request more crease share than 2× the sharp-edge fraction.
    cap_frac = min(mix_frac, _CREASE_OVERREP * frac)
    n_crease_cap = int(round(count * cap_frac))
    n_crease_cap = min(n_crease, max(0, n_crease_cap))
    return n_area + (n_crease - n_crease_cap), n_crease_cap


def crease_length_fraction(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Sharp interior-edge length / all interior-edge length after weld."""
    verts, tris = weld_vertices(vertices, faces)
    edge_map = _edge_map(tris)
    interior = 0.0
    sharp = 0.0
    normals = _triangle_normals(verts, tris)
    for (i0, i1), incident in edge_map.items():
        # Border edges have one face; they are not folds.
        if len(incident) != 2:
            continue
        elen = float(np.linalg.norm(verts[i1] - verts[i0]))
        interior += elen
        a, b = incident[0], incident[1]
        dot = float(np.clip(np.dot(normals[a], normals[b]), -1.0, 1.0))
        if float(np.arccos(dot)) >= _SHARP_RAD:
            sharp += elen
    if interior <= 1e-12:
        return 0.0
    return float(sharp / interior)


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
    # One incident face per crease sample — not both sides.
    face_nrms = _triangle_normals(welded_v, welded_f)
    xyz = np.empty((count, 3), dtype=np.float32)
    nrm = np.empty((count, 3), dtype=np.float32)
    for k in range(count):
        i0, i1, incident = sharp[int(picks[k])]
        a = welded_v[i0]
        b = welded_v[i1]
        p = a + float(ts[k]) * (b - a)
        fi = incident[int(side[k]) % len(incident)]
        tri = welded_f[fi]
        c_idx = int(tri[0] + tri[1] + tri[2] - i0 - i1)
        q = _offset_into_triangle(p, a, b, welded_v[c_idx], float(dists[k]))
        xyz[k] = q.astype(np.float32)
        nrm[k] = face_nrms[fi].astype(np.float32)
    return _pack_xyz_normal(xyz, nrm)


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
    Sample ``n_surface`` envelope points as ``(N, 6)`` XYZ + unit normal.

    ``mix`` 0 is the original area-weighted cloud (old checkpoints).
    ``mix`` 100 is crease-hugging, then capped by crease length fraction.
    In between, Count is split like the viewer Mix slider, then spilled
    to faces when the mesh has few sharp edges.
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
        # Cap crease dots so a few folds cannot consume the Mix budget.
        frac = crease_length_fraction(verts, tris)
        n_area, n_crease = plan_envelope_counts(count, mix_v, frac)
        chunks: list[np.ndarray] = []
        if n_area > 0:
            chunks.append(_sample_area(verts, tris, n_area, int(seed)))
        if n_crease > 0:
            chunks.append(_sample_crease(verts, tris, n_crease, int(seed) + 1))
        out = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if out.shape != (count, ENVELOPE_FEAT_DIM):
        raise ValueError(
            f"expected envelope shape {(count, ENVELOPE_FEAT_DIM)}, got {tuple(out.shape)}"
        )
    if key is not None:
        _ENVELOPE_CACHE[key] = out
    return out
