"""Job A / B: classify uploaded points with a ``models/*/best.pt``.

Reuses ``load_occupancy_model`` and AABB helpers. Geometry tokens follow
the **checkpoint** (envelope vs face tokens), not live ``config.yaml``.
Does not modify occupancy train/infer modules.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from config import OccupancyConfig

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infer_multi_npz import load_occupancy_model  # noqa: E402
from geometry.face_tokens import apply_face_aabb, face_tokens_from_triangles  # noqa: E402
from geometry.mesh_io import load_obj_triangles  # noqa: E402
from geometry.surface import sample_surface_points  # noqa: E402
from metrics import occupancy_metrics  # noqa: E402
from normalize import apply_normalization, compute_center_scale  # noqa: E402
from occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND  # noqa: E402

from mesh_access import resolve_viewer_mesh  # noqa: E402
from model_access import match_checkpoint_part, resolve_viewer_checkpoint  # noqa: E402
from obj_fill import triangles_from_obj_text  # noqa: E402

MAX_POINTS = 2_000_000


def pred_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard inside labels: 1 iff sigmoid probability is at least ``threshold``."""
    t = float(threshold)
    if not np.isfinite(t):
        t = 0.5
    t = min(1.0, max(0.0, t))
    return (np.asarray(probs, dtype=np.float32).reshape(-1) >= t).astype(np.uint8)


def _sigmoid_probs_and_pred(logits: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Float32 sigmoid of a 1-D logit tensor, plus a 0.5-cut pred for tests/compat.

    The inspect page re-cuts from ``prob_b64``; occupancy train metrics stay at 0.5.
    """
    import torch

    probs = (
        torch.sigmoid(logits.reshape(-1))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return probs, pred_from_probs(probs, 0.5)


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


def _ckpt_shape_encoder(ckpt: dict[str, Any]) -> str:
    """
    Encoder stored on ``best.pt``, not live YAML.

    Missing ``shape_encoder`` on an occupancy-encoder checkpoint is the
    original envelope head.
    """
    raw = str(ckpt.get("shape_encoder") or "").strip().lower()
    if raw in ("surface", "mesh", "none"):
        return raw
    kind = str(ckpt.get("kind") or "")
    if kind == ENCODER_KIND:
        return "surface"
    return "none"


def _runtime_cfg(cfg: OccupancyConfig | None):
    """Use the caller cfg (tests) or load repo YAML for device / batch / seed."""
    if cfg is not None:
        return cfg
    from config import load_config

    return load_config()


def _forward_logits(model, cfg, xyz, geom, shape_id):
    """Batched occupancy logits. ``geom`` is envelope ``(1,N,3)`` or faces ``(1,F,12)``."""
    import torch

    n = int(xyz.shape[0])
    logits_rows: list[Any] = []
    with torch.no_grad():
        for start in range(0, n, int(cfg.batch_size)):
            sl = slice(start, start + int(cfg.batch_size))
            batch_xyz = xyz[sl].to(cfg.device)
            if geom is None:
                logits_rows.append(model(batch_xyz).cpu())
            else:
                b = int(batch_xyz.shape[0])
                geom_b = geom.expand(b, -1, -1).to(cfg.device)
                sid = shape_id.reshape(()).expand(b).to(cfg.device)
                logits_rows.append(model(batch_xyz, geom_b, sid).cpu())
    return torch.cat(logits_rows, dim=0)


def _geom_from_mesh(ckpt, cfg, vertices, faces, center, scale, cache_key: str):
    """
    Build the shape tensor this checkpoint was trained with.

    ``surface`` → ``(1, n_surface, 3)`` envelope XYZ.
    ``mesh`` → ``(1, n_faces, 12)`` triangle tokens.
    ``none`` → no geometry (xyz-only MLP).
    Counts and kind come from the checkpoint so a mesh ``best.pt`` still
    infers after YAML is switched back to envelope (and the reverse).
    """
    import torch

    enc = _ckpt_shape_encoder(ckpt)
    seed = int(ckpt["seed"]) if ckpt.get("seed") is not None else int(cfg.seed)
    if enc == "surface":
        n_surface = (
            int(ckpt["n_surface"]) if ckpt.get("n_surface") is not None else 1024
        )
        mix = int(ckpt["envelope_mix"]) if ckpt.get("envelope_mix") is not None else 0
        world = sample_surface_points(
            vertices,
            faces,
            n_surface,
            seed=seed,
            mix=mix,
            cache_key=cache_key,
        )
        env = apply_normalization(world, center, scale)
        geom = torch.from_numpy(env).unsqueeze(0)
        shape_id = torch.zeros(1, dtype=torch.long)
        return geom, shape_id
    if enc == "mesh":
        n_tok = int(ckpt["n_faces"]) if ckpt.get("n_faces") is not None else 256
        world_tok = face_tokens_from_triangles(
            vertices, faces, n_tok, cache_key=cache_key
        )
        tok = apply_face_aabb(world_tok, center, scale)
        geom = torch.from_numpy(tok).unsqueeze(0)
        shape_id = torch.zeros(1, dtype=torch.long)
        return geom, shape_id
    return None, None


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
    cfg: OccupancyConfig | None = None,
) -> dict[str, Any]:
    """
    Classify ``points`` with ``best.pt``. Returns pred bytes (base64) and scores.

    Envelope vs face tokens follow ``ckpt['shape_encoder']``.
    """
    import torch

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
    cfg = _runtime_cfg(cfg)
    model = load_occupancy_model(ckpt, cfg.device)
    center, scale, aabb_src = aabb_for_viewer(
        ckpt,
        npz_name=npz_name,
        mesh_path=mesh_path,
        points=points,
        data_dir=data_dir,
    )
    enc = _ckpt_shape_encoder(ckpt)
    geom = None
    shape_id = None
    if enc in ("surface", "mesh"):
        if not mesh_path:
            raise ValueError("this checkpoint needs a mesh_path for the geometry encoder")
        if data_dir is None:
            raise ValueError("helper has no data_dir; cannot load the OBJ for the encoder")
        mesh_file = resolve_viewer_mesh(mesh_path, data_dir)
        vertices, faces = load_obj_triangles(mesh_file)
        geom, shape_id = _geom_from_mesh(
            ckpt, cfg, vertices, faces, center, scale, str(mesh_file.resolve())
        )
        if geom is None:
            raise ValueError(f"checkpoint shape_encoder={enc!r} built no geometry tokens")

    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
    logits = _forward_logits(model, cfg, xyz, geom, shape_id)
    scores = occupancy_metrics(logits, y)
    probs, pred = _sigmoid_probs_and_pred(logits)
    gt = labels > 0
    pred_bool = pred > 0
    n_fn = int(np.count_nonzero(gt & ~pred_bool))
    n_fp = int(np.count_nonzero(~gt & pred_bool))
    return {
        "pred_b64": base64.b64encode(np.ascontiguousarray(pred)).decode("ascii"),
        "prob_b64": base64.b64encode(np.ascontiguousarray(probs)).decode("ascii"),
        "n": n,
        "accuracy": float(scores.accuracy),
        "inside_iou": float(scores.inside_iou),
        "inside_f1": float(scores.inside_f1),
        "kind": str(ckpt.get("kind") or ""),
        "shape_encoder": enc,
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
    cfg: OccupancyConfig | None = None,
) -> dict[str, Any]:
    """
    Classify fill-lattice XYZ with ``best.pt``.

    Shape tokens are built from the uploaded OBJ: envelope when the
    checkpoint is ``surface``, face tokens when it is ``mesh``.
    No file labels (Job B): metrics are omitted.
    """
    import torch

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
    cfg = _runtime_cfg(cfg)
    model = load_occupancy_model(ckpt, cfg.device)
    center, scale, aabb_src = aabb_for_viewer(
        ckpt,
        npz_name="",
        mesh_path=str(obj_name or ""),
        points=points,
        data_dir=data_dir,
        vertices=vertices,
    )
    enc = _ckpt_shape_encoder(ckpt)
    geom, shape_id = _geom_from_mesh(
        ckpt, cfg, vertices, faces, center, scale, "upload:" + str(obj_name or "obj")
    )
    if enc in ("surface", "mesh") and geom is None:
        raise ValueError(f"checkpoint shape_encoder={enc!r} built no geometry tokens")
    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    logits = _forward_logits(model, cfg, xyz, geom, shape_id)
    probs, pred = _sigmoid_probs_and_pred(logits)
    n_in = int(np.count_nonzero(pred > 0))
    return {
        "pred_b64": base64.b64encode(np.ascontiguousarray(pred)).decode("ascii"),
        "prob_b64": base64.b64encode(np.ascontiguousarray(probs)).decode("ascii"),
        "n": n,
        "n_inside": n_in,
        "n_outside": n - n_in,
        "kind": str(ckpt.get("kind") or ""),
        "shape_encoder": enc,
        "aabb": aabb_src,
        "run_id": Path(ckpt_path).parent.name,
    }
