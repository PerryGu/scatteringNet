"""List and resolve ``models/<run_id>/best.pt`` for the occupancy viewer.

Read-only: paths must stay under the repo ``models/`` folder.
Does not import torch.
"""

from __future__ import annotations

from pathlib import Path


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
