"""Viewer UI prefs: checkboxes, sliders, selected model.

Written only as ``ui_prefs.json`` next to the viewer page (localhost helper).
Does not import torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFS_NAME = "ui_prefs.json"
MAX_PREFS_BYTES = 32_768
# Viewer-only notes after the run id in the model list (not folder names).
MODEL_LABEL_MAX = 48
MODEL_LABELS_MAX = 80

DEFAULTS: dict[str, Any] = {
    "mesh": True,
    "inside": True,
    "outside": True,
    "wireframe": False,
    "opacity": 100,
    "point_size": 100,
    "draw_cap": 200_000,
    "density": 71,
    "inside_cut": 50,
    "envelope_n": 1024,
    "envelope_mix": 100,
    "model_id": "",
    "model_labels": {},
}

_BOOL_KEYS = ("mesh", "inside", "outside", "wireframe")


def _safe_run_id(raw: object) -> str:
    """Single folder name under ``models/``. Empty if unsafe."""
    name = str(raw or "").strip().replace("\\", "/")
    if (not name) or "/" in name or name in (".", "..") or ".." in name:
        return ""
    return name[:200]


_INT_KEYS = {
    "opacity": (0, 100),
    "point_size": (20, 400),
    "draw_cap": (5000, 200_000),
    "density": (0, 100),
    "inside_cut": (0, 100),
    "envelope_n": (256, 4096),
    "envelope_mix": (0, 100),
}


def prefs_path(viewer_root: Path | str) -> Path:
    """Fixed filename under the viewer folder."""
    return Path(viewer_root) / PREFS_NAME


def clamp_ui_prefs(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Merge unknown-safe keys onto defaults and clamp slider ranges."""
    src = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = dict(DEFAULTS)
    for key in _BOOL_KEYS:
        if key in src:
            out[key] = bool(src[key])
    for key, (lo, hi) in _INT_KEYS.items():
        if key not in src:
            continue
        try:
            val = int(round(float(src[key])))
        except (TypeError, ValueError):
            continue
        out[key] = max(lo, min(hi, val))
    if "model_id" in src:
        out["model_id"] = _safe_run_id(src.get("model_id"))
    labels: dict[str, str] = {}
    raw_labels = src.get("model_labels")
    if isinstance(raw_labels, dict):
        for key, val in raw_labels.items():
            run_id = _safe_run_id(key)
            note = str(val or "").strip()[:MODEL_LABEL_MAX]
            if not run_id or not note:
                continue
            labels[run_id] = note
            if len(labels) >= MODEL_LABELS_MAX:
                break
    out["model_labels"] = labels
    return out


def load_ui_prefs(viewer_root: Path | str) -> dict[str, Any]:
    """Return clamped prefs. Missing or invalid file → defaults."""
    path = prefs_path(viewer_root)
    try:
        if not path.is_file():
            return clamp_ui_prefs({})
        raw = path.read_bytes()
        if len(raw) > MAX_PREFS_BYTES:
            return clamp_ui_prefs({})
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return clamp_ui_prefs({})
    return clamp_ui_prefs(data if isinstance(data, dict) else {})


def save_ui_prefs(viewer_root: Path | str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Write clamped JSON (2-space indent). Same directory only."""
    root = Path(viewer_root).resolve()
    path = (root / PREFS_NAME).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("ui prefs path escaped viewer root") from exc
    if path.name != PREFS_NAME:
        raise PermissionError("ui prefs filename is fixed")
    clamped = clamp_ui_prefs(raw)
    text = json.dumps(clamped, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(PREFS_NAME + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return clamped
