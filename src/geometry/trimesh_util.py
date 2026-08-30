"""Shared trimesh flatten. No Open3D — safe for train-side mesh_io."""

from __future__ import annotations

from typing import Any

import trimesh


def as_trimesh(mesh: Any) -> trimesh.Trimesh:
    """Flatten a Trimesh or a Scene of triangle meshes to one Trimesh."""
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("Scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    return mesh
