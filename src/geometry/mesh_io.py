"""Load OBJ triangle meshes for the NPZ ↔ mesh join.

This module only returns ``vertices (V, 3)`` and ``faces (T, 3)``.
It does **not** sample the envelope (Step 8) or build face tokens (Step 9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

VerticesArray = NDArray[np.float32]
FacesArray = NDArray[np.int32]

# Same resolved OBJ can back several NPZs (``max_files_per_shape``). Cache
# the triangle arrays so catalog load does not re-parse the file.
_TRIANGLE_CACHE: dict[str, tuple[VerticesArray, FacesArray]] = {}


def _as_trimesh(mesh: Any) -> trimesh.Trimesh:
    """Flatten a Trimesh or a Scene of triangle meshes to one Trimesh."""
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("Scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    return mesh


def load_obj_triangles(
    path: Path | str,
    *,
    cache: bool = True,
) -> tuple[VerticesArray, FacesArray]:
    """
    Read one OBJ as triangle vertices and face indices.

    Parameters
    ----------
    path:
        Existing ``.obj`` file.
    cache:
        Reuse arrays for the same resolved path (catalog load).

    Returns
    -------
    vertices:
        ``float32`` array of shape ``(V, 3)``.
    faces:
        ``int32`` array of shape ``(T, 3)`` (0-based vertex indices).
    """
    obj_path = Path(path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError(f"expected .obj, got {obj_path.suffix!r} ({obj_path})")

    cache_key = str(obj_path.resolve())
    if cache and cache_key in _TRIANGLE_CACHE:
        return _TRIANGLE_CACHE[cache_key]

    # process=False keeps the authored vertices; we only need the join.
    loaded = trimesh.load(obj_path, force=None, process=False)
    mesh = _as_trimesh(loaded)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (V, 3), got {tuple(vertices.shape)} in {obj_path}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (T, 3), got {tuple(faces.shape)} in {obj_path}"
        )
    if int(faces.shape[0]) < 1:
        raise ValueError(f"OBJ has no triangles: {obj_path}")

    arrays = (vertices, faces)
    if cache:
        _TRIANGLE_CACHE[cache_key] = arrays
    return arrays
