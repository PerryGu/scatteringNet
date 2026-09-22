"""Classify occupancy NPZs from ``models/<run_id>/best.pt``.

Rebuilds ``OccupancyMLP`` or ``OccupancyEncoder`` from the checkpoint
``kind``. Query XYZ (and envelope, when conditioned) use the stored
per-mesh AABB, not a fresh map. Face-token checkpoints are rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from scatteringnet.config import OccupancyConfig, as_data_relative, load_config, repo_root
from scatteringnet.data_npz import load_points_labels, load_points_labels_mesh
from scatteringnet.geometry.mesh_io import load_obj_triangles
from scatteringnet.geometry.surface import apply_envelope_aabb, project_envelope_dim, sample_surface_points
from scatteringnet.metrics import occupancy_metrics
from scatteringnet.normalize import apply_normalization
from scatteringnet.occupancy_encoder import (
    OccupancyEncoder,
    envelope_dim_from_ckpt,
    envelope_seed_from_ckpt,
)
from scatteringnet.occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND
from scatteringnet.occupancy_mlp import OccupancyMLP
from scatteringnet.occupancy_mlp import CHECKPOINT_KIND as MLP_KIND


def resolve_best_pt(
    *,
    checkpoint: Path | str | None = None,
    run_id: str | None = None,
    root: Path | None = None,
) -> Path:
    """Resolve ``best.pt`` from an explicit path or ``runs/<id>/checkpoint_dir.txt``."""
    base = (root or repo_root()).resolve()
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if not run_id:
        raise ValueError("pass checkpoint= or run_id=")
    pointer = base / "runs" / str(run_id) / "checkpoint_dir.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"run pointer not found: {pointer}")
    rel = pointer.read_text(encoding="utf-8").strip()
    path = Path(rel)
    if not path.is_absolute():
        path = base / path
    best = path / "best.pt" if path.is_dir() else path
    if not best.is_file():
        raise FileNotFoundError(f"best.pt not found: {best}")
    return best


def load_occupancy_model(
    ckpt: dict[str, Any],
    device: torch.device,
) -> OccupancyMLP | OccupancyEncoder:
    """Rebuild the head recorded in ``ckpt['kind']`` and load weights."""
    hidden = int(ckpt["hidden"])
    depth = int(ckpt["depth"])
    kind = str(ckpt["kind"])
    if kind == MLP_KIND:
        model: OccupancyMLP | OccupancyEncoder = OccupancyMLP(hidden=hidden, depth=depth)
    elif kind == ENCODER_KIND:
        latent = int(ckpt["latent_dim"]) if ckpt.get("latent_dim") is not None else hidden
        enc = str(ckpt.get("shape_encoder") or "surface").strip().lower()
        if enc == "mesh":
            raise ValueError(
                "face-token occupancy checkpoints (shape_encoder='mesh') "
                "are no longer supported"
            )
        knn_k = int(ckpt["knn_k"]) if ckpt.get("knn_k") is not None else 0
        knn_local = None
        if knn_k > 0 and ckpt.get("knn_local_dim") is not None:
            knn_local = int(ckpt["knn_local_dim"])
        model = OccupancyEncoder(
            hidden=hidden,
            depth=depth,
            latent_dim=latent,
            shape_encoder=enc,
            knn_k=knn_k,
            knn_local_dim=knn_local,
            envelope_dim=envelope_dim_from_ckpt(ckpt),
        )
    else:
        raise ValueError(f"unsupported checkpoint kind {kind!r}")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def _part_for_npz(ckpt: dict[str, Any], npz_path: Path, data_dir: Path) -> dict[str, Any]:
    rel = as_data_relative(npz_path, data_dir)
    name = npz_path.name
    for part in ckpt.get("parts") or []:
        stored = str(part.get("npz", ""))
        if stored == rel or Path(stored).name == name:
            return part
    raise KeyError(f"no AABB part for {npz_path.name} in checkpoint")


def infer_npz(
    cfg: OccupancyConfig,
    *,
    checkpoint: Path | str | None = None,
    run_id: str | None = None,
    npz_path: Path | str | None = None,
    root: Path | None = None,
) -> dict[str, float]:
    """
    Score one NPZ with ``best.pt``. Returns accuracy / IoU / F1.
    """
    ckpt_path = resolve_best_pt(checkpoint=checkpoint, run_id=run_id, root=root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_occupancy_model(ckpt, cfg.device)
    if npz_path is None:
        rels = ckpt.get("npz_paths") or []
        if not rels:
            raise ValueError("checkpoint has no npz_paths; pass npz_path=")
        npz = cfg.data_dir / str(rels[0])
    else:
        npz = Path(npz_path)
        if not npz.is_absolute():
            npz = cfg.data_dir / npz
    part = _part_for_npz(ckpt, npz, cfg.data_dir)
    center = np.asarray(part["center"], dtype=np.float32)
    scale = float(part["scale"])
    geom = None
    shape_id = None
    enc = str(ckpt.get("shape_encoder", "none")).strip().lower()
    if enc == "mesh":
        raise ValueError(
            "face-token occupancy checkpoints (shape_encoder='mesh') "
            "are no longer supported"
        )
    if enc == "surface":
        points, labels, mesh_path = load_points_labels_mesh(npz, cfg.data_dir)
        vertices, faces = load_obj_triangles(mesh_path)
        cache_key = str(mesh_path.resolve())
        n_surface = int(ckpt.get("n_surface") or cfg.n_surface)
        world = sample_surface_points(
            vertices,
            faces,
            n_surface,
            seed=envelope_seed_from_ckpt(ckpt),
            cache_key=cache_key,
        )
        arr = apply_envelope_aabb(world, center, scale)
        arr = project_envelope_dim(arr, envelope_dim_from_ckpt(ckpt))
        geom = torch.from_numpy(arr).unsqueeze(0)
        shape_id = torch.zeros((), dtype=torch.long)
    else:
        points, labels = load_points_labels(npz)
    xyz = torch.from_numpy(apply_normalization(points, center, scale))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).unsqueeze(1)
    logits_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(xyz.shape[0]), int(cfg.batch_size)):
            sl = slice(start, start + int(cfg.batch_size))
            batch_xyz = xyz[sl].to(cfg.device)
            if geom is None:
                logits_rows.append(model(batch_xyz).cpu())
            else:
                b = int(batch_xyz.shape[0])
                geom_b = geom.expand(b, -1, -1).to(cfg.device)
                sid = shape_id.expand(b).to(cfg.device)
                logits_rows.append(model(batch_xyz, geom_b, sid).cpu())
    logits = torch.cat(logits_rows, dim=0)
    scores = occupancy_metrics(logits, y)
    print(
        f"checkpoint={ckpt_path} npz={npz.name} "
        f"acc={scores.accuracy:.4f} iou={scores.inside_iou:.4f} "
        f"f1={scores.inside_f1:.4f}"
    )
    return {
        "accuracy": scores.accuracy,
        "inside_iou": scores.inside_iou,
        "inside_f1": scores.inside_f1,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infer occupancy from models/<id>/best.pt")
    parser.add_argument("--checkpoint", default=None, help="Path to best.pt")
    parser.add_argument("--run-id", default=None, help="runs/<id> folder name")
    parser.add_argument("--npz", default=None, help="NPZ path (data_dir-relative or absolute)")
    args = parser.parse_args()
    infer_npz(
        load_config(),
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        npz_path=args.npz,
    )
