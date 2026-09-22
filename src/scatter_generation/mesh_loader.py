"""Load triangle meshes for occupancy NPZ sampling (conda, not Maya)."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
import trimesh
import yaml

from scatteringnet.geometry.trimesh_util import as_trimesh

MESH_EXTENSIONS: frozenset[str] = frozenset(
    {".obj", ".ply", ".stl", ".glb", ".gltf", ".off", ".dae"}
)


def project_root() -> Path:
    """
    Resolve the repository root.

    Same folder as ``config.repo_root`` (this file → parents[2]).
    """
    from scatteringnet.config import repo_root

    return repo_root()


def get_data_dir() -> Path:
    """
    Dataset root from repo ``config.yaml`` (``data_dir``).

    Returns
    -------
    Path
        Absolute folder that contains ``meshes/`` and ``exports/``.
    """
    cfg_path = project_root() / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "data_dir" not in raw:
        raise ValueError(f"config.yaml missing data_dir: {cfg_path}")
    return Path(str(raw["data_dir"]).strip()).expanduser().resolve()


def to_data_relative(path: str | Path, *, data_dir: Path | None = None) -> str:
    """
    Store paths relative to ``data_dir``, using forward slashes.

    Paths outside ``data_dir`` (pytest temp dirs) fall back to an absolute
    POSIX string.

    Parameters
    ----------
    path:
        File or folder to store (absolute or relative).
    data_dir:
        Dataset root. Default is :func:`get_data_dir` from ``config.yaml``.

    Returns
    -------
    str
        Path relative to ``data_dir``, or an absolute POSIX path if ``path``
        is outside that folder.
    """
    from scatteringnet.config import as_data_relative

    root = (data_dir if data_dir is not None else get_data_dir()).resolve()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return as_data_relative(resolved, root)


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Convert a trimesh mesh to Open3D for occupancy / ray queries.

    Parameters
    ----------
    mesh:
        Triangle mesh in world coordinates.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Open3D mesh with vertex normals computed.
    """
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


@dataclass
class MeshInfo:
    """
    Identity of one loaded mesh (used by the batch builder).

    Attributes
    ----------
    path:
        Absolute filesystem path to the mesh file.
    name:
        Filename including extension.
    num_vertices:
        Vertex count after load.
    num_faces:
        Triangle count after load.
    """

    path: str
    name: str
    num_vertices: int
    num_faces: int


def load_mesh(path: str | Path, *, process: bool = False) -> tuple[trimesh.Trimesh, MeshInfo]:
    """
    Load one mesh file.

    Parameters
    ----------
    path:
        Path to an OBJ/PLY/STL (or other supported) file.
    process:
        Forwarded to ``trimesh.load`` (merge vertices, etc.). Default False
        preserves the file as authored.

    Returns
    -------
    mesh:
        Triangle mesh in file coordinates.
    info:
        :class:`MeshInfo` summary for the batch builder.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If the extension is not in :data:`MESH_EXTENSIONS`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")
    if path.suffix.lower() not in MESH_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {path.suffix!r}. "
            f"Supported: {', '.join(sorted(MESH_EXTENSIONS))}"
        )
    loaded = trimesh.load(path, force=None, process=process)
    mesh = as_trimesh(loaded)
    info = MeshInfo(
        path=str(path.resolve()),
        name=path.name,
        num_vertices=int(len(mesh.vertices)),
        num_faces=int(len(mesh.faces)),
    )
    return mesh, info


def iter_mesh_files(
    root: str | Path,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    name_glob: str | None = None,
) -> list[Path]:
    """
    List mesh files under ``root``, sorted by path.

    Parameters
    ----------
    root:
        Folder to scan.
    recursive:
        If True (default), include nested directories.
    extensions:
        Suffixes to keep (with or without a leading dot). Default is
        :data:`MESH_EXTENSIONS`.
    name_glob:
        Optional ``fnmatch`` pattern on the filename only (e.g. ``*_nr5_*.obj``).
        ``None`` keeps every mesh suffix.

    Returns
    -------
    list of Path
        Sorted matching files.

    Raises
    ------
    NotADirectoryError
        If ``root`` is not a directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    exts = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in (extensions or MESH_EXTENSIONS)
    }
    pattern = "**/*" if recursive else "*"
    needle = str(name_glob).strip() if name_glob else ""
    out = []
    for p in sorted(root.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if needle and not fnmatch.fnmatch(p.name, needle):
            continue
        out.append(p)
    return out
