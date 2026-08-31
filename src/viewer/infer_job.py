"""Job A: classify uploaded NPZ query points with a ``models/*/best.pt``.

Reuses ``load_occupancy_model``, AABB helpers, and envelope sampling.
Does not modify occupancy train/infer modules.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infer_multi_npz import load_occupancy_model  # noqa: E402
from geometry.mesh_io import load_obj_triangles  # noqa: E402
from geometry.surface import sample_surface_points  # noqa: E402
from metrics import occupancy_metrics  # noqa: E402
from normalize import apply_normalization, compute_center_scale  # noqa: E402

from mesh_access import resolve_viewer_mesh  # noqa: E402
from model_access import match_checkpoint_part, resolve_viewer_checkpoint  # noqa: E402
from obj_fill import triangles_from_obj_text  # noqa: E402

MAX_POINTS = 2_000_000


def aabb_for_viewer(
    ckpt: dict[str, Any],
    *,
    npz_name: str,
    mesh_path: str,
    points: np.ndarray,
    data_dir: Path | None,
    vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    """
    Checkpoint AABB when this NPZ/mesh was trained; else mesh vertices; else queries.
    """
    part = match_checkpoint_part(ckpt, npz_name, mesh_path)
    if part is not None:
        center = np.asarray(part["center"], dtype=np.float32).reshape(3)
        scale = float(part["scale"])
        return center, scale, "checkpoint"
    if vertices is not None and int(np.asarray(vertices).shape[0]) > 0:
        center, scale = compute_center_scale(np.asarray(vertices, dtype=np.float32))
        return center, scale, "mesh"
    if mesh_path and data_dir is not None:
        try:
            resolved = resolve_viewer_mesh(mesh_path, data_dir)
            vertices, _faces = load_obj_triangles(resolved)
            center, scale = compute_center_scale(vertices)
            return center, scale, "mesh"
        except (PermissionError, FileNotFoundError, ValueError):
            pass
    center, scale = compute_center_scale(points)
    return center, scale, "points"


def _forward_logits(model, cfg, xyz, envelope, shape_id):
    """Batched occupancy logits on CPU."""
    import torch

    n = int(xyz.shape[0])
    logits_rows: list[Any] = []
    with torch.no_grad():
        for start in range(0, n, int(cfg.batch_size)):
            sl = slice(start, start + int(cfg.batch_size))
            batch_xyz = xyz[sl].to(cfg.device)
            if envelope is None:
                logits_rows.append(model(batch_xyz).cpu())
            else:
                b = int(batch_xyz.shape[0])
                env_b = envelope.expand(b, -1, -1).to(cfg.device)
                sid = shape_id.expand(b).to(cfg.device)
                logits_rows.append(model(batch_xyz, env_b, sid).cpu())
    return torch.cat(logits_rows, dim=0)


def _envelope_from_mesh(ckpt, cfg, vertices, faces, center, scale, cache_key: str):
    import torch

    uses_surface = str(ckpt.get("shape_encoder", "none")).strip().lower() == "surface"
    if not uses_surface:
        return None, None
    n_surface = int(ckpt.get("n_surface") or cfg.n_surface)
    world = sample_surface_points(
        vertices,
        faces,
        n_surface,
        seed=int(cfg.seed),
        cache_key=cache_key,
    )
    env = apply_normalization(world, center, scale)
    envelope = torch.from_numpy(env).unsqueeze(0)
    shape_id = torch.zeros((), dtype=torch.long)
    return envelope, shape_id


def decode_points_b64(text: str) -> np.ndarray:
    """Little-endian float32 XYZ from standard base64."""
    raw = base64.b64decode(text)
    pts = np.frombuffer(raw, dtype=np.float32)
    if pts.size % 3 != 0:
        raise ValueError("points buffer length is not a multiple of 3")
    return np.ascontiguousarray(pts.reshape(-1, 3))


def decode_labels_b64(text: str, n: int) -> np.ndarray:
    """uint8 {0,1} labels, length ``n``."""
    raw = base64.b64decode(text)
    labels = np.frombuffer(raw, dtype=np.uint8)
    if int(labels.shape[0]) != int(n):
        raise ValueError(
            f"labels length {labels.shape[0]} does not match points {n}"
        )
    return np.ascontiguousarray(labels)


def infer_uploaded_npz(
    *,
    run_id: str,
    models_root: Path,
    data_dir: Path | None,
    npz_name: str,
    mesh_path: str,
    points: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """
    Classify ``points`` with ``best.pt``. Returns pred bytes (base64) and scores.
    """
    import torch
    from config import load_config

    n = int(points.shape[0])
    if n < 1:
        raise ValueError("no query points")
    if n > MAX_POINTS:
        raise ValueError(f"too many points ({n}; max {MAX_POINTS})")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    if int(labels.shape[0]) != n:
        raise ValueError("labels length does not match points")

    ckpt_path = resolve_viewer_checkpoint(run_id, models_root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config()
    model = load_occupancy_model(ckpt, cfg.device)
    center, scale, aabb_src = aabb_for_viewer(
        ckpt,
        npz_name=npz_name,
        mesh_path=mesh_path,
        points=points,
        data_dir=data_dir,
    )
    uses_surface = str(ckpt.get("shape_encoder", "none")).strip().lower() == "surface"
    envelope = None
    shape_id = None
    if uses_surface:
        if not mesh_path:
            raise ValueError("this checkpoint needs a mesh_path for the envelope")
        if data_dir is None:
            raise ValueError("helper has no data_dir; cannot sample the envelope")
        mesh_file = resolve_viewer_mesh(mesh_path, data_dir)
        vertices, faces = load_obj_triangles(mesh_file)
        envelope, shape_id = _envelope_from_mesh(
            ckpt, cfg, vertices, faces, center, scale, str(mesh_file.resolve())
        )

    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
    logits = _forward_logits(model, cfg, xyz, envelope, shape_id)
    scores = occupancy_metrics(logits, y)
    pred = (torch.sigmoid(logits.reshape(-1)) >= 0.5).to(torch.uint8).numpy()
    gt = labels > 0
    pred_bool = pred > 0
    n_fn = int(np.count_nonzero(gt & ~pred_bool))
    n_fp = int(np.count_nonzero(~gt & pred_bool))
    return {
        "pred_b64": base64.b64encode(np.ascontiguousarray(pred)).decode("ascii"),
        "n": n,
        "accuracy": float(scores.accuracy),
        "inside_iou": float(scores.inside_iou),
        "inside_f1": float(scores.inside_f1),
        "kind": str(ckpt.get("kind") or ""),
        "aabb": aabb_src,
        "n_error": n_fn + n_fp,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "run_id": Path(ckpt_path).parent.name,
    }


def infer_uploaded_obj(
    *,
    run_id: str,
    models_root: Path,
    data_dir: Path | None,
    obj_name: str,
    obj_text: str,
    points: np.ndarray,
) -> dict[str, Any]:
    """
    Classify fill-lattice XYZ with ``best.pt``. Envelope comes from the uploaded OBJ.
    No file labels (Job B): metrics are omitted.
    """
    import torch
    from config import load_config

    n = int(points.shape[0])
    if n < 1:
        raise ValueError("no query points; Fill points first")
    if n > MAX_POINTS:
        raise ValueError(f"too many points ({n}; max {MAX_POINTS})")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")

    vertices, faces = triangles_from_obj_text(obj_text)
    ckpt_path = resolve_viewer_checkpoint(run_id, models_root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config()
    model = load_occupancy_model(ckpt, cfg.device)
    center, scale, aabb_src = aabb_for_viewer(
        ckpt,
        npz_name="",
        mesh_path=str(obj_name or ""),
        points=points,
        data_dir=data_dir,
        vertices=vertices,
    )
    envelope, shape_id = _envelope_from_mesh(
        ckpt, cfg, vertices, faces, center, scale, "upload:" + str(obj_name or "obj")
    )
    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    logits = _forward_logits(model, cfg, xyz, envelope, shape_id)
    pred = (torch.sigmoid(logits.reshape(-1)) >= 0.5).to(torch.uint8).numpy()
    n_in = int(np.count_nonzero(pred > 0))
    return {
        "pred_b64": base64.b64encode(np.ascontiguousarray(pred)).decode("ascii"),
        "n": n,
        "n_inside": n_in,
        "n_outside": n - n_in,
        "kind": str(ckpt.get("kind") or ""),
        "aabb": aabb_src,
        "run_id": Path(ckpt_path).parent.name,
    }
