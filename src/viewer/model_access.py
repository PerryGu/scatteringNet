"""List and resolve ``models/<run_id>/best.pt`` for the occupancy viewer.

Read-only: paths must stay under the repo ``models/`` folder.
Does not import torch.
"""

from __future__ import annotations

from pathlib import Path


def list_viewer_models(models_root: Path | str) -> list[dict[str, str | int]]:
    """
    Return ``best.pt`` checkpoints one level under ``models_root``.

    Each item: ``id`` (folder name), ``path`` (repo-relative), ``mtime``.
    Newest first. Missing folder → empty list.
    """
    root = Path(models_root)
    try:
        root = root.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    items: list[dict[str, str | int]] = []
    for best in root.glob("*/best.pt"):
        if not best.is_file():
            continue
        run_id = best.parent.name
        try:
            mtime = int(best.stat().st_mtime)
        except OSError:
            mtime = 0
        items.append(
            {
                "id": run_id,
                "path": "models/" + run_id + "/best.pt",
                "mtime": mtime,
            }
        )
    items.sort(key=lambda row: (-int(row["mtime"]), str(row["id"])))
    return items


def resolve_viewer_checkpoint(run_id: str, models_root: Path | str) -> Path:
    """
    Return ``models_root / <run_id> / best.pt``.

    ``run_id`` is a single folder name (no slashes). Also accepts
    ``models/<run_id>/best.pt`` and strips it down to the folder name.

    Raises
    ------
    ValueError
        Empty or unsafe id.
    FileNotFoundError
        ``best.pt`` is missing.
    PermissionError
        Resolved path is outside ``models_root``.
    """
    raw = str(run_id or "").strip().replace("\\", "/")
    if raw.endswith("/best.pt"):
        raw = raw[: -len("/best.pt")]
    if raw.startswith("models/"):
        raw = raw[len("models/") :]
    name = raw.strip("/")
    if not name or "/" in name or name in (".", "..") or ".." in name:
        raise ValueError("invalid model id")
    root = Path(models_root).resolve()
    best = (root / name / "best.pt").resolve()
    try:
        best.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"checkpoint is outside models/: {best} (id={run_id!r})"
        ) from exc
    if not best.is_file():
        raise FileNotFoundError(f"best.pt not found for {name!r}")
    return best


def match_checkpoint_part(ckpt: dict, npz_name: str, mesh_path: str) -> dict | None:
    """Prefer NPZ filename match, then mesh path, from ``ckpt['parts']``."""
    parts = ckpt.get("parts") or []
    name = Path(str(npz_name or "")).name
    mesh = str(mesh_path or "").replace("\\", "/")
    if name:
        for part in parts:
            stored = str(part.get("npz") or "")
            if stored == name or Path(stored).name == name:
                return part
    mesh_name = Path(mesh).name if mesh else ""
    if mesh:
        for part in parts:
            stored_m = str(part.get("mesh") or "").replace("\\", "/")
            if not stored_m:
                continue
            if stored_m == mesh or Path(stored_m).name == mesh_name:
                return part
    return None
