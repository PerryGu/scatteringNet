"""Gradio occupancy loop: fill an OBJ AABB, then classify with ``best.pt``.

Reuses the viewer helper (``obj_fill``, ``infer_job``, ``model_access``).
Does not import Gradio and does not change the Three.js viewer.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np

# Occupancy is the installed ``scatteringnet`` package (``pip install -e .``).
# This folder is not a package so it cannot shadow pip ``gradio``.
_GRADIO_DIR = Path(__file__).resolve().parent
_REPO = _GRADIO_DIR.parents[1]

from scatteringnet.config import load_config
from scatteringnet.viewer.infer_job import infer_uploaded_obj, pred_from_probs  # noqa: E402
from scatteringnet.viewer.model_access import list_viewer_models  # noqa: E402
from scatteringnet.viewer.obj_fill import (  # noqa: E402
    fill_aabb_lattice,
    spacing_from_slider,
    triangles_from_obj_text,
)

# Same inspect checkpoint named in the repo README.
INSPECT_RUN_ID = "2026-09-16_18-09-31_prim_extruded_nr45_knn24_n2048_n6"
# Viewer allows 200k; Plotly + Spaces need a tighter lattice.
MAX_FILL_POINTS = 80_000
MAX_OBJ_BYTES = 32 * 1024 * 1024
# Slider default matches the Three.js Density control (spacing ~0.15).
DEFAULT_DENSITY = 71
DEFAULT_CUT = 0.50


def models_root() -> Path:
    """Repo ``models/`` (``models/<run_id>/best.pt``)."""
    return _REPO / "models"


def list_run_ids(root: Path | None = None) -> list[str]:
    """Newest-first run folder names that have a ``best.pt``."""
    rows = list_viewer_models(root or models_root())
    return [str(row["id"]) for row in rows]


def default_run_id(root: Path | None = None) -> str:
    """Inspect weights if present, else the newest checkpoint, else empty."""
    ids = list_run_ids(root)
    if INSPECT_RUN_ID in ids:
        return INSPECT_RUN_ID
    return ids[0] if ids else ""


def runtime_cfg(cfg: Any | None = None):
    """
    Device / batch from YAML. ``data_dir`` is not required (uploaded OBJ).
    """
    if cfg is not None:
        return cfg
    return load_config(require_existing_data_dir=False)


def apply_cut(probs: np.ndarray, threshold: float) -> tuple[np.ndarray, int, int]:
    """Hard labels from stored sigmoid probs (no extra GPU pass)."""
    pred = pred_from_probs(probs, threshold)
    n_in = int(np.count_nonzero(pred > 0))
    return pred, n_in, int(pred.shape[0]) - n_in


def fill_and_infer(
    obj_text: str,
    *,
    obj_name: str,
    run_id: str,
    density: float,
    models: Path | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """
    Job B for Gradio: AABB lattice + occupancy forward.

    Returns vertices/faces (for the Plotly mesh), query XYZ, sigmoid
    probs, timings, and the run id. The UI recuts ``probs`` locally.
    """
    raw = str(obj_text or "")
    if not raw.strip():
        raise ValueError("OBJ is empty")
    if len(raw.encode("utf-8")) > MAX_OBJ_BYTES:
        raise ValueError(f"OBJ too large (max {MAX_OBJ_BYTES} bytes)")
    name = str(run_id or "").strip()
    if not name:
        raise ValueError("select a checkpoint under models/<run_id>/best.pt")

    vertices, faces = triangles_from_obj_text(raw)
    spacing = spacing_from_slider(density)
    points, used, grid = fill_aabb_lattice(
        vertices, spacing, max_points=MAX_FILL_POINTS
    )
    ckpt_root = Path(models) if models is not None else models_root()
    out = infer_uploaded_obj(
        run_id=name,
        models_root=ckpt_root,
        data_dir=None,
        obj_name=str(obj_name or "upload.obj"),
        obj_text=raw,
        points=points,
        cfg=runtime_cfg(cfg),
    )
    probs = np.frombuffer(base64.b64decode(out["prob_b64"]), dtype=np.float32).reshape(
        -1
    )
    pred, n_in, n_out = apply_cut(probs, DEFAULT_CUT)
    return {
        "vertices": np.ascontiguousarray(vertices, dtype=np.float32),
        "faces": np.ascontiguousarray(faces, dtype=np.int32),
        "points": np.ascontiguousarray(points, dtype=np.float32),
        "probs": np.ascontiguousarray(probs, dtype=np.float32),
        "pred": pred,
        "n": int(out["n"]),
        "n_inside": n_in,
        "n_outside": n_out,
        "used_spacing": float(used),
        "grid": [int(grid[0]), int(grid[1]), int(grid[2])],
        "run_id": str(out["run_id"]),
        "shape_encoder": str(out["shape_encoder"]),
        "timings": dict(out["timings"]),
        "obj_name": str(obj_name or "upload.obj"),
    }
