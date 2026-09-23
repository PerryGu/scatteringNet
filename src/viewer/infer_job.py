"""Job A / B: classify uploaded points with a ``models/*/best.pt``.

Reuses ``load_occupancy_model`` and AABB helpers. Envelope tokens follow
the **checkpoint**, not live ``config.yaml``. Face-token checkpoints
are rejected. Does not modify occupancy train/infer modules.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scatteringnet.config import OccupancyConfig

import numpy as np


from scatteringnet.infer_multi_npz import load_occupancy_model  # noqa: E402
from scatteringnet.geometry.mesh_io import load_obj_triangles  # noqa: E402
from scatteringnet.geometry.surface import (  # noqa: E402
    apply_envelope_aabb,
    project_envelope_dim,
    sample_surface_points,
)
from scatteringnet.metrics import occupancy_metrics  # noqa: E402
from scatteringnet.normalize import apply_normalization, compute_center_scale  # noqa: E402
from scatteringnet.occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND  # noqa: E402
from scatteringnet.occupancy_encoder import envelope_dim_from_ckpt  # noqa: E402
from scatteringnet.occupancy_encoder import envelope_seed_from_ckpt  # noqa: E402

from scatteringnet.viewer.mesh_access import resolve_viewer_mesh  # noqa: E402
from scatteringnet.viewer.model_access import match_checkpoint_part, resolve_viewer_checkpoint  # noqa: E402
from scatteringnet.viewer.obj_fill import triangles_from_obj_text  # noqa: E402

MAX_POINTS = 2_000_000

# One occupancy head in this process. Same checkpoint + device → skip torch.load.
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: tuple[tuple[str, int, str], Any, dict[str, Any]] | None = None


def clear_model_cache() -> None:
    """Drop the cached head (tests / swapped GPU)."""
    global _MODEL_CACHE
    with _CACHE_LOCK:
        _MODEL_CACHE = None


def _model_cache_key(ckpt_path: Path, device: Any) -> tuple[str, int, str]:
    stat = ckpt_path.stat()
    return (str(ckpt_path.resolve()), int(stat.st_mtime_ns), str(device))


def load_cached_occupancy(
    run_id: str,
    models_root: Path,
    device: Any,
) -> tuple[Any, dict[str, Any], Path, bool]:
    """
    Load ``best.pt`` once per (path, mtime, device).

    Returns ``(model, ckpt, path, cache_hit)``. Switching run_id replaces
    the slot so GPU RAM does not keep every inspect checkpoint.
    """
    import torch

    ckpt_path = resolve_viewer_checkpoint(run_id, models_root)
    key = _model_cache_key(ckpt_path, device)
    global _MODEL_CACHE
    with _CACHE_LOCK:
        slot = _MODEL_CACHE
        if slot is not None and slot[0] == key:
            return slot[1], slot[2], ckpt_path, True
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_occupancy_model(ckpt, device)
    with _CACHE_LOCK:
        _MODEL_CACHE = (key, model, ckpt)
    return model, ckpt, ckpt_path, False


def _sync_if_cuda(device) -> None:
    """Wait for GPU work so lap times are not just the CPU launch."""
    import torch

    dev = device if hasattr(device, "type") else torch.device(str(device))
    if str(dev.type) == "cuda":
        torch.cuda.synchronize()


def _round_s(t0: float) -> float:
    return round(time.perf_counter() - t0, 3)


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
    uploaded_obj: bool = False,
) -> tuple[np.ndarray, float, str]:
    """
    Checkpoint AABB when this catalog NPZ/mesh was trained; else mesh vertices.

    Uploaded OBJ Fill (Job B) always uses this mesh's vertices. Catalog
    ``parts`` matched by filename would apply another cube's box.
    """
    if not uploaded_obj:
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
    if raw in ("surface", "none"):
        return raw
    if raw == "mesh":
        raise ValueError(
            "face-token occupancy checkpoints (shape_encoder='mesh') "
            "are no longer supported"
        )
    kind = str(ckpt.get("kind") or "")
    if kind == ENCODER_KIND:
        return "surface"
    return "none"


def _runtime_cfg(cfg: OccupancyConfig | None):
    """Use the caller cfg (tests) or load repo YAML for device / batch / seed."""
    if cfg is not None:
        return cfg
    from scatteringnet.config import load_config

    return load_config()


def _forward_logits(model, cfg, xyz, geom, shape_id):
    """Batched occupancy logits. ``geom`` is envelope ``(1, N, 3)`` or ``None``."""
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
    Build the envelope this checkpoint was trained with.

    ``surface`` → ``(1, n_surface, C)`` envelope (C from checkpoint).
    ``none`` → no geometry (xyz-only MLP).
    Count comes from the checkpoint. Sampling is always area-weighted
    (old ``envelope_mix`` on ``best.pt`` is ignored).
    """
    import torch

    enc = _ckpt_shape_encoder(ckpt)
    seed = envelope_seed_from_ckpt(ckpt)
    if enc == "surface":
        n_surface = (
            int(ckpt["n_surface"]) if ckpt.get("n_surface") is not None else 1024
        )
        world = sample_surface_points(
            vertices,
            faces,
            n_surface,
            seed=seed,
            cache_key=cache_key,
        )
        env = apply_envelope_aabb(world, center, scale)
        env = project_envelope_dim(env, envelope_dim_from_ckpt(ckpt))
        geom = torch.from_numpy(env).unsqueeze(0)
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

    Envelope rebuild follows ``ckpt['shape_encoder']``.
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

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    cfg = _runtime_cfg(cfg)
    model, ckpt, ckpt_path, cache_hit = load_cached_occupancy(
        run_id, models_root, cfg.device
    )
    _sync_if_cuda(cfg.device)
    load_s = _round_s(t0)
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
    parse_s = 0.0
    envelope_s = 0.0
    if enc in ("surface", "mesh"):
        if not mesh_path:
            raise ValueError("this checkpoint needs a mesh_path for the geometry encoder")
        if data_dir is None:
            raise ValueError("helper has no data_dir; cannot load the OBJ for the encoder")
        t0 = time.perf_counter()
        mesh_file = resolve_viewer_mesh(mesh_path, data_dir)
        vertices, faces = load_obj_triangles(mesh_file)
        parse_s = _round_s(t0)
        t0 = time.perf_counter()
        geom, shape_id = _geom_from_mesh(
            ckpt, cfg, vertices, faces, center, scale, str(mesh_file.resolve())
        )
        envelope_s = _round_s(t0)
        if geom is None:
            raise ValueError(f"checkpoint shape_encoder={enc!r} built no geometry tokens")

    t0 = time.perf_counter()
    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
    logits = _forward_logits(model, cfg, xyz, geom, shape_id)
    _sync_if_cuda(cfg.device)
    forward_s = _round_s(t0)
    scores = occupancy_metrics(logits, y)
    probs, pred = _sigmoid_probs_and_pred(logits)
    timings = {
        "parse_obj": parse_s,
        "load_model": load_s,
        "envelope": envelope_s,
        "forward": forward_s,
        "server_total": _round_s(t_all),
        "load_cached": cache_hit,
        "device": str(cfg.device),
    }
    print(
        "viewer infer-npz timings (s) n=%s parse=%.3f load=%.3f envelope=%.3f "
        "forward=%.3f total=%.3f cached=%s device=%s"
        % (
            n,
            parse_s,
            load_s,
            envelope_s,
            forward_s,
            timings["server_total"],
            cache_hit,
            cfg.device,
        ),
        flush=True,
    )
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
        "timings": timings,
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

    Envelope tokens are built from the uploaded OBJ when the
    checkpoint is ``surface``. No file labels (Job B): metrics are omitted.
    """
    import torch

    n = int(points.shape[0])
    if n < 1:
        raise ValueError("no query points; Fill points first")
    if n > MAX_POINTS:
        raise ValueError(f"too many points ({n}; max {MAX_POINTS})")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    vertices, faces = triangles_from_obj_text(obj_text)
    parse_s = _round_s(t0)
    t0 = time.perf_counter()
    cfg = _runtime_cfg(cfg)
    model, ckpt, ckpt_path, cache_hit = load_cached_occupancy(
        run_id, models_root, cfg.device
    )
    _sync_if_cuda(cfg.device)
    load_s = _round_s(t0)
    center, scale, aabb_src = aabb_for_viewer(
        ckpt,
        npz_name="",
        mesh_path=str(obj_name or ""),
        points=points,
        data_dir=data_dir,
        vertices=vertices,
        uploaded_obj=True,
    )
    enc = _ckpt_shape_encoder(ckpt)
    t0 = time.perf_counter()
    geom, shape_id = _geom_from_mesh(
        ckpt, cfg, vertices, faces, center, scale, "upload:" + str(obj_name or "obj")
    )
    envelope_s = _round_s(t0)
    if enc in ("surface", "mesh") and geom is None:
        raise ValueError(f"checkpoint shape_encoder={enc!r} built no geometry tokens")
    t0 = time.perf_counter()
    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    logits = _forward_logits(model, cfg, xyz, geom, shape_id)
    _sync_if_cuda(cfg.device)
    forward_s = _round_s(t0)
    probs, pred = _sigmoid_probs_and_pred(logits)
    n_in = int(np.count_nonzero(pred > 0))
    timings = {
        "parse_obj": parse_s,
        "load_model": load_s,
        "envelope": envelope_s,
        "forward": forward_s,
        "server_total": _round_s(t_all),
        "load_cached": cache_hit,
        "device": str(cfg.device),
    }
    print(
        "viewer infer-obj timings (s) n=%s parse=%.3f load=%.3f envelope=%.3f "
        "forward=%.3f total=%.3f cached=%s device=%s"
        % (
            n,
            parse_s,
            load_s,
            envelope_s,
            forward_s,
            timings["server_total"],
            cache_hit,
            cfg.device,
        ),
        flush=True,
    )
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
        "timings": timings,
    }


_WARM_OBJ = (
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
    "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
)


def warmup_viewer_helper(models_root: Path) -> None:
    """
    Pay first-click costs before the browser opens: torch, CUDA, trimesh
    fill, load the INSPECT ``best.pt`` (else newest), tiny envelope + forward.
    """
    t0 = time.perf_counter()
    print("warmup: torch / CUDA…", flush=True)
    import torch

    from scatteringnet.config import load_config
    from scatteringnet.viewer.obj_fill import fill_aabb_lattice

    cfg = load_config()
    print("warmup: occupancy device=" + str(cfg.device), flush=True)
    if str(cfg.device.type) == "cuda":
        torch.zeros(1, device=cfg.device)
        torch.cuda.synchronize()
        print("warmup: CUDA " + str(cfg.device), flush=True)
    else:
        print("warmup: CPU only", flush=True)

    print("warmup: tiny fill…", flush=True)
    verts, faces = triangles_from_obj_text(_WARM_OBJ)
    points, _used, _grid = fill_aabb_lattice(verts, 0.40)
    if int(points.shape[0]) < 1:
        points = np.array([[0.2, 0.2, 0.2], [0.8, 0.2, 0.2]], dtype=np.float32)

    from scatteringnet.viewer.model_access import inspect_run_id, list_viewer_models

    rows = list_viewer_models(models_root)
    if not rows:
        print("warmup: no models/; skip infer", flush=True)
        print("warmup: done in %.1fs" % (time.perf_counter() - t0), flush=True)
        return
    ids = [str(row["id"]) for row in rows]
    wanted = inspect_run_id()
    # Warm the inspect alias when that folder exists; otherwise newest.
    run_id = wanted if wanted in ids else ids[0]
    print("warmup: infer " + run_id + "…", flush=True)
    infer_uploaded_obj(
        run_id=run_id,
        models_root=models_root,
        data_dir=None,
        obj_name="warmup.obj",
        obj_text=_WARM_OBJ,
        points=points,
        cfg=cfg,
    )
    print("warmup: done in %.1fs" % (time.perf_counter() - t0), flush=True)
