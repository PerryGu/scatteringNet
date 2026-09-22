"""Resolve NPZ ``mesh_path`` for the occupancy viewer helper.

Uses :func:`data_npz.resolve_mesh_path` and then requires the file to sit
under ``data_dir`` (no arbitrary filesystem reads).
"""

from __future__ import annotations

from pathlib import Path

from scatteringnet.data_npz import resolve_mesh_path


def resolve_viewer_mesh(stored: str, data_dir: Path | str) -> Path:
    """
    Return an existing ``.obj`` under ``data_dir`` for a stored mesh_path.

    Parameters
    ----------
    stored:
        Relative or absolute path from the NPZ ``mesh_path`` array.
    data_dir:
        Dataset root (``config.yaml`` ``data_dir``).

    Returns
    -------
    Path
        Resolved OBJ path.

    Raises
    ------
    ValueError
        Empty path or not an OBJ.
    FileNotFoundError
        Mesh does not exist (from :func:`resolve_mesh_path`).
    PermissionError
        Resolved path is outside ``data_dir``.
    """
    text = str(stored).strip()
    if not text:
        raise ValueError("mesh_path is empty")
    root = Path(data_dir).resolve()
    resolved = resolve_mesh_path(text, root)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"mesh is outside data_dir: {resolved} (stored={text!r})"
        ) from exc
    if resolved.suffix.lower() != ".obj":
        raise ValueError(f"mesh is not an OBJ: {resolved}")
    return resolved
