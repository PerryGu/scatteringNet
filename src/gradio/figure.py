"""GLB scene for Gradio ``Model3D`` (Babylon.js orbit, Y-up).

Plotly 3D was the wrong tool: its camera is a unit-cube, not world Y-up, so
the floor never sat on the ground. This writes a real mesh (floor grid +
OBJ + occupancy points) and lets Gradio's 3D viewer frame it.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import trimesh
from trimesh.path.entities import Line
from trimesh.path.path import Path3D
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

# Inspect COLOR_INSIDE is 0xffaa00. Babylon treats GLB vertex colors as
# linear, so that same RGB reads as lemon-yellow. Drop the green channel
# so the on-screen fill matches the inspect orange.
COLOR_INSIDE = np.array([255, 136, 0, 255], dtype=np.uint8)
COLOR_OUTSIDE = np.array([74, 109, 140, 220], dtype=np.uint8)
COLOR_MESH_RGB = (0.60, 0.64, 0.70)
# THREE.GridHelper(span, 20, 0xaabbcc, 0x556677) — center vs cells.
COLOR_GRID = np.array([85, 102, 119, 255], dtype=np.uint8)
COLOR_GRID_CENTER = np.array([170, 187, 204, 255], dtype=np.uint8)
DEFAULT_MESH_OPACITY = 50
# Babylon POINTS are 1 px in the GLB; orbit.js applies this pixel size.
DEFAULT_POINT_SIZE = 8
# Match pipeline.MAX_FILL_POINTS so the GLB is not a random subset of the lattice.
MAX_PLOT_POINTS = 80_000
FLOOR_DIVS = 20


def _floor_bounds(vertices: np.ndarray | None) -> tuple[float, float, float, float, float]:
    """XZ rectangle and Y of the floor (content ymin, else y=0)."""
    if vertices is None or int(np.asarray(vertices).size) == 0:
        return -2.0, 2.0, -2.0, 2.0, 0.0
    verts = np.asarray(vertices, dtype=np.float64)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    radius = 0.5 * float(np.max(vmax - vmin))
    span = max(radius * 4.0, 4.0)
    cx = 0.5 * float(vmin[0] + vmax[0])
    cz = 0.5 * float(vmin[2] + vmax[2])
    y = float(vmin[1])
    half = 0.5 * span
    return cx - half, cx + half, cz - half, cz + half, y


def _clamp_opacity(value: float) -> float:
    """Slider 0–100 → 0–1. Missing / NaN → default 50%."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        t = float(DEFAULT_MESH_OPACITY)
    if not np.isfinite(t):
        t = float(DEFAULT_MESH_OPACITY)
    return min(1.0, max(0.0, t / 100.0))


def _shell_mesh(vertices: np.ndarray, faces: np.ndarray, opacity: float) -> trimesh.Trimesh:
    """
    OBJ shell with a GLTF BLEND material.

    Vertex-color alpha is ignored by Babylon Model3D (that is why the
    torus looked solid). ``alphaMode=BLEND`` + baseColorFactor.a is the
    channel the viewer actually uses.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    alpha = _clamp_opacity(opacity)
    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            name="shell",
            baseColorFactor=[COLOR_MESH_RGB[0], COLOR_MESH_RGB[1], COLOR_MESH_RGB[2], alpha],
            metallicFactor=0.0,
            roughnessFactor=0.85,
            alphaMode="BLEND",
            doubleSided=True,
        )
    )
    mesh.metadata["name"] = "occ_shell"
    return mesh


def _grid_mesh(vertices: np.ndarray | None, *, n: int = FLOOR_DIVS) -> Path3D:
    """
    XZ floor as GL_LINES (1 px in Babylon), not extruded boxes.

    Boxes read as a thick waffle. The inspect viewer uses
    ``THREE.GridHelper(span, 20, 0xaabbcc, 0x556677)`` — same count
    and the lighter centre cross.
    """
    x0, x1, z0, z1, y = _floor_bounds(vertices)
    n = max(2, int(n))
    mid = n // 2
    verts: list[list[float]] = []
    entities: list[Line] = []
    colors: list[np.ndarray] = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        z = z0 + (z1 - z0) * t
        color = COLOR_GRID_CENTER if i == mid else COLOR_GRID
        i0 = len(verts)
        verts.extend(([x, y, z0], [x, y, z1]))
        entities.append(Line(points=[i0, i0 + 1]))
        colors.append(color)
        i1 = len(verts)
        verts.extend(([x0, y, z], [x1, y, z]))
        entities.append(Line(points=[i1, i1 + 1]))
        colors.append(color)
    return Path3D(
        entities=entities,
        vertices=np.asarray(verts, dtype=np.float64),
        colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )


def _axis_mesh(vertices: np.ndarray | None) -> Path3D:
    """
    RGB triad as GL_LINES — same idea as ``THREE.AxesHelper``.

    Length is ``max(radius * 0.45, 1)`` with ``span = 4 * radius``,
    matching the inspect helper. Boxes looked like fat sticks.
    """
    x0, x1, z0, z1, y = _floor_bounds(vertices)
    span = max(x1 - x0, z1 - z0, 1.0)
    length = max(0.45 * (span / 4.0), 1.0)
    origin = np.array([0.5 * (x0 + x1), y, 0.5 * (z0 + z1)], dtype=np.float64)
    # AxesHelper: +X red, +Y green, +Z blue.
    specs = (
        (np.array([length, 0.0, 0.0]), (255, 0, 0, 255)),
        (np.array([0.0, length, 0.0]), (0, 255, 0, 255)),
        (np.array([0.0, 0.0, length]), (0, 0, 255, 255)),
    )
    verts: list[np.ndarray] = []
    entities: list[Line] = []
    colors: list[tuple[int, int, int, int]] = []
    for delta, rgba in specs:
        i0 = len(verts)
        verts.extend((origin, origin + delta))
        entities.append(Line(points=[i0, i0 + 1]))
        colors.append(rgba)
    return Path3D(
        entities=entities,
        vertices=np.asarray(verts, dtype=np.float64),
        colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )


def _points_mesh(xyz: np.ndarray, rgba: np.ndarray) -> trimesh.Trimesh | None:
    """Occupancy dots as a vertex cloud (GLB point primitive)."""
    if xyz.ndim != 2 or xyz.shape[0] < 1:
        return None
    cloud = trimesh.points.PointCloud(xyz.astype(np.float64), colors=rgba)
    # Node name survives GLB import so orbit.js can set pointSize.
    cloud.metadata["name"] = "occ_points"
    return cloud


def _subsample(points: np.ndarray, pred: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(points.shape[0])
    if n <= cap:
        return points, pred
    rng = np.random.default_rng(0)
    inside = np.flatnonzero(pred > 0)
    outside = np.flatnonzero(pred == 0)
    n_in = min(int(inside.shape[0]), cap)
    take_in = rng.choice(inside, size=n_in, replace=False) if n_in else np.empty(0, dtype=int)
    remain = cap - n_in
    n_out = min(int(outside.shape[0]), remain)
    take_out = (
        rng.choice(outside, size=n_out, replace=False) if n_out else np.empty(0, dtype=int)
    )
    idx = np.concatenate([take_in, take_out])
    return points[idx], pred[idx]


def _placeholder_geom() -> trimesh.Trimesh:
    """Tiny triangle so trimesh can export when the draw list is empty."""
    dummy = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], [0.0, 0.0, 1e-4]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    dummy.metadata["name"] = "occ_empty"
    return dummy


def _export_glb(geoms: list) -> Path:
    """Write a unique GLB so the orbit fetch is not served from a stale cache."""
    folder = Path(tempfile.gettempdir()) / "scatteringnet_gradio"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"view_{uuid.uuid4().hex[:10]}.glb"
    scene = trimesh.Scene()
    added = 0
    for i, geom in enumerate(geoms):
        if geom is None:
            continue
        meta = getattr(geom, "metadata", None) or {}
        name = str(meta["name"]) if isinstance(meta, dict) and meta.get("name") else f"g{i}"
        scene.add_geometry(geom, node_name=name)
        added += 1
    if added < 1:
        scene.add_geometry(_placeholder_geom(), node_name="occ_empty")
    scene.export(path)
    return path


def empty_figure(message: str = "", **_kwargs) -> Path:
    """
    Placeholder GLB so the cmd path can change. The visible floor lives
    in orbit.js and is never replaced (a per-mesh grid was the camera jump).
    """
    _ = message
    return _export_glb([])


def occupancy_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    pred: np.ndarray,
    *,
    show_outside: bool = False,
    title: str = "",
    mesh_opacity: float = DEFAULT_MESH_OPACITY,
    **_kwargs,
) -> Path:
    """Mesh + occupancy points as one GLB. Floor stays in orbit.js."""
    _ = title
    verts = np.asarray(vertices, dtype=np.float32) if vertices is not None else np.zeros((0, 3), np.float32)
    tris = np.asarray(faces, dtype=np.int32) if faces is not None else np.zeros((0, 3), np.int32)
    geoms: list = []
    if verts.ndim == 2 and verts.shape[0] > 0:
        if tris.ndim == 2 and tris.shape[0] > 0 and tris.shape[1] == 3:
            geoms.append(_shell_mesh(verts, tris, mesh_opacity))
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim == 2 and xyz.shape[0] > 0:
        labels = np.asarray(pred, dtype=np.uint8).reshape(-1)
        if not show_outside:
            keep = labels > 0
            xyz = xyz[keep]
            labels = labels[keep]
        xyz, labels = _subsample(xyz, labels, MAX_PLOT_POINTS)
        if xyz.shape[0] > 0:
            colors = np.where(labels[:, None] > 0, COLOR_INSIDE, COLOR_OUTSIDE)
            geoms.append(_points_mesh(xyz, colors))
    return _export_glb(geoms)
