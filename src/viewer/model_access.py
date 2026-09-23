"""List and resolve ``models/<run_id>/best.pt`` for the occupancy viewer.

Read-only: paths must stay under the repo ``models/`` folder.
Does not import torch.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root: this file lives at src/viewer/model_access.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Pointers (YAML). Occupancy train / infer math does not read these.
INSPECT_POINTER = _REPO_ROOT / "docs" / "inspect_checkpoint.yaml"
HOLDOUT_POINTER = _REPO_ROOT / "docs" / "locked_holdout_objs.yaml"
# Same id as the committed pointer; used if that file is missing.
_INSPECT_FALLBACK = "2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6"


def inspect_run_id() -> str:
    """INSPECT alias: ``run_id`` in ``docs/inspect_checkpoint.yaml``.

    Viewers use this as the default ``models/<id>/best.pt`` when that file
    exists. Occupancy train / infer math is unchanged.
    """
    try:
        raw = yaml.safe_load(INSPECT_POINTER.read_text(encoding="utf-8"))
    except OSError:
        return _INSPECT_FALLBACK
    if not isinstance(raw, dict):
        return _INSPECT_FALLBACK
    token = str(raw.get("run_id") or "").strip()
    return token[:200] if token else _INSPECT_FALLBACK


def locked_holdout_objs() -> list[str]:
    """OBJ basenames from ``docs/locked_holdout_objs.yaml``.

    Train / catalog construction does not consult this list. It is the
    locked inspect set for humans and for tests.
    """
    try:
        raw = yaml.safe_load(HOLDOUT_POINTER.read_text(encoding="utf-8"))
    except OSError:
        return []
    if not isinstance(raw, dict):
        return []
    objs = raw.get("objs") or []
    if not isinstance(objs, list):
        return []
    names: list[str] = []
    for item in objs:
        name = str(item or "").strip()
        if name:
            names.append(name)
    return names


def _shape_encoder_from_run(runs_root: Path | None, run_id: str) -> str:
    """Read ``shape_encoder`` from ``runs/<id>/config.yaml`` (no torch)."""
    if runs_root is None:
        return ""
    path = Path(runs_root) / run_id / "config.yaml"
    try:
        if not path.is_file():
            return ""
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped.startswith("shape_encoder:"):
                continue
            raw = stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'")
            kind = raw.lower()
            if kind in ("surface", "mesh", "none"):
                return kind
            return ""
    except OSError:
        return ""
    return ""


def list_viewer_models(
    models_root: Path | str,
    *,
    runs_root: Path | str | None = None,
) -> list[dict[str, str | int]]:
    """
    Return ``best.pt`` checkpoints one level under ``models_root``.

    Each item: ``id`` (folder name), ``path`` (repo-relative), ``mtime``,
    and ``shape_encoder`` when ``runs/<id>/config.yaml`` is present.
    Newest first. Missing folder → empty list.
    """
    root = Path(models_root)
    try:
        root = root.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    runs = Path(runs_root) if runs_root is not None else None
    items: list[dict[str, str | int]] = []
    for best in root.glob("*/best.pt"):
        if not best.is_file():
            continue
        run_id = best.parent.name
        try:
            mtime = int(best.stat().st_mtime)
        except OSError:
            mtime = 0
        enc = _shape_encoder_from_run(runs, run_id)
        row: dict[str, str | int] = {
            "id": run_id,
            "path": "models/" + run_id + "/best.pt",
            "mtime": mtime,
        }
        if enc:
            row["shape_encoder"] = enc
        items.append(row)
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
    """
    Catalog AABB from ``ckpt['parts']``.

    NPZ: full stored path or the occupancy filename (those stems are unique).
    Mesh: exact stored path only. Basename matches (``cube.obj``) steal a
    catalog box for an OOD upload of the same name — Job B must not do that.
    """
    parts = ckpt.get("parts") or []
    name = Path(str(npz_name or "")).name
    mesh = str(mesh_path or "").replace("\\", "/").strip()
    if name:
        for part in parts:
            stored = str(part.get("npz") or "").replace("\\", "/")
            if stored == name or Path(stored).name == name:
                return part
    if mesh:
        for part in parts:
            stored_m = str(part.get("mesh") or "").replace("\\", "/").strip()
            if stored_m and stored_m == mesh:
                return part
    return None
