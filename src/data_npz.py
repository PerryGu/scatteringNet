"""Load occupancy query points and labels from scatter NPZ files.

Phase 1 uses :func:`load_points_labels` on **one** file. Step 3 adds a catalog
resolver so many NPZs can be listed (glob or explicit paths) without training.

Only ``points`` and ``labels`` are used for arrays. Other keys (``mesh_path``,
``tags``, ``method``, …) are ignored on purpose so occupancy stays xyz + label.
"""

from __future__ import annotations

import glob as globlib
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# Labels are converted here (not deferred to the Dataset) so every caller gets
# the same dtypes: float32 XYZ and float32 {0, 1} occupancy.
PointsArray = NDArray[np.float32]
LabelsArray = NDArray[np.float32]


def load_points_labels(path: Path) -> tuple[PointsArray, LabelsArray]:
    """
    Read query coordinates and inside/outside labels from one NPZ file.

    Parameters
    ----------
    path:
        Path to a ``.npz`` with arrays ``points`` ``(N, 3)`` and
        ``labels`` ``(N,)`` (typically uint8 0/1).

    Returns
    -------
    points:
        ``float32`` array of shape ``(N, 3)``.
    labels:
        ``float32`` array of shape ``(N,)`` with values in ``{0.0, 1.0}``
        (0 = outside, 1 = inside).
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # allow_pickle=False: we only need numeric arrays, not object payloads.
    with np.load(npz_path, allow_pickle=False) as raw:
        files = set(raw.files)
        if "points" not in files or "labels" not in files:
            raise KeyError(
                f"NPZ must contain 'points' and 'labels', got {sorted(files)} "
                f"in {npz_path}"
            )
        points = np.asarray(raw["points"])
        labels = np.asarray(raw["labels"])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {tuple(points.shape)} in {npz_path}"
        )
    n = int(points.shape[0])
    if labels.shape != (n,):
        raise ValueError(
            f"labels must have shape (N,) with N={n}, got {tuple(labels.shape)} "
            f"in {npz_path}"
        )

    points_f32 = np.asarray(points, dtype=np.float32)
    labels_f32 = np.asarray(labels, dtype=np.float32)
    unique = np.unique(labels_f32)
    if not np.all((unique == 0.0) | (unique == 1.0)):
        raise ValueError(
            f"labels must be in {{0, 1}}, got unique={unique.tolist()} in {npz_path}"
        )
    return points_f32, labels_f32


def _is_combo_npz(path: Path) -> bool:
    """True when the filename looks like a combo dump (excluded from Step 3)."""
    return "combo" in path.name.lower()


def shape_key(path: Path) -> str:
    """
    Group NPZs that belong to the same mesh.

    Uses the stem before ``__`` (dataset_builder tag), else the full stem.

    Parameters
    ----------
    path:
        NPZ path.

    Returns
    -------
    str
        Stable key for ``max_files_per_shape``.
    """
    stem = Path(path).stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def _cap_per_shape(
    paths: Sequence[Path],
    max_files_per_shape: int | None,
) -> list[Path]:
    """Keep at most ``max_files_per_shape`` files per :func:`shape_key` (sorted order)."""
    if max_files_per_shape is None:
        return list(paths)
    if max_files_per_shape < 1:
        raise ValueError(f"max_files_per_shape must be >= 1 or None, got {max_files_per_shape}")
    counts: dict[str, int] = {}
    out: list[Path] = []
    for path in paths:
        key = shape_key(path)
        taken = counts.get(key, 0)
        if taken >= max_files_per_shape:
            continue
        counts[key] = taken + 1
        out.append(path)
    return out


def resolve_npz_catalog(
    data_dir: Path | str,
    *,
    npz_glob: str = "exports/dataset/*.npz",
    npz_paths: Sequence[str | Path] | None = None,
    max_files_per_shape: int | None = 2,
    exclude_combo: bool = True,
) -> list[Path]:
    """
    Resolve occupancy NPZ paths under ``data_dir`` (no point loading).

    An explicit ``npz_paths`` list wins over ``npz_glob``. Relative entries are
    joined to ``data_dir``. Missing files raise ``FileNotFoundError``.

    Parameters
    ----------
    data_dir:
        Dataset root (``config.yaml`` ``data_dir``).
    npz_glob:
        Glob relative to ``data_dir`` (``*`` and ``**`` allowed).
    npz_paths:
        Explicit relative or absolute NPZ paths. Empty / None → use glob.
    max_files_per_shape:
        Cap per :func:`shape_key` after sort. ``None`` = no cap.
    exclude_combo:
        Drop filenames containing ``combo``.

    Returns
    -------
    list[Path]
        Sorted existing ``.npz`` files.
    """
    root = Path(data_dir)
    chosen: list[Path]
    if npz_paths:
        chosen = []
        for raw in npz_paths:
            item = Path(raw)
            resolved = item if item.is_absolute() else (root / item)
            if not resolved.is_file():
                raise FileNotFoundError(f"NPZ not found: {resolved}")
            chosen.append(resolved.resolve())
    else:
        pattern = str(root / npz_glob)
        recursive = "**" in npz_glob.replace("\\", "/")
        found = globlib.glob(pattern, recursive=recursive)
        chosen = [Path(p).resolve() for p in found if Path(p).is_file()]

    npz_only = [p for p in chosen if p.suffix.lower() == ".npz"]
    if exclude_combo:
        npz_only = [p for p in npz_only if not _is_combo_npz(p)]
    npz_only = sorted(npz_only)
    capped = _cap_per_shape(npz_only, max_files_per_shape)
    if not capped:
        raise FileNotFoundError(
            f"No occupancy NPZ files matched under {root} "
            f"(glob={npz_glob!r}, explicit={bool(npz_paths)})"
        )
    return capped


def _summarize(points: PointsArray, labels: LabelsArray) -> str:
    n = int(points.shape[0])
    inside = float(labels.mean()) if n else float("nan")
    xyz_min = points.min(axis=0) if n else np.full(3, np.nan, dtype=np.float32)
    xyz_max = points.max(axis=0) if n else np.full(3, np.nan, dtype=np.float32)
    return (
        f"N={n}\n"
        f"inside_fraction={inside:.6f}\n"
        f"xyz_min={xyz_min.tolist()}\n"
        f"xyz_max={xyz_max.tolist()}"
    )


# Default smoke-check file from the v2 plan (dataset_test sphere).
_SAMPLE_RELATIVE = Path("exports") / "dataset_test" / "sphere__raycast_z_raut_s0.15_inout.npz"


if __name__ == "__main__":
    import sys

    from config import load_config

    cfg = load_config()
    if "--catalog" in sys.argv:
        paths = resolve_npz_catalog(
            cfg.data_dir,
            npz_glob=cfg.npz_glob,
            npz_paths=cfg.npz_paths or None,
            max_files_per_shape=cfg.max_files_per_shape,
        )
        print(f"data_dir={cfg.data_dir}")
        print(f"npz_glob={cfg.npz_glob}")
        print(f"max_files_per_shape={cfg.max_files_per_shape}")
        print(f"files={len(paths)}")
        # Load a few files only — the full catalog can be thousands of NPZs.
        preview = paths[:3]
        n_all = 0
        n_in = 0
        for path in preview:
            pts, labs = load_points_labels(path)
            n = int(pts.shape[0])
            inside = int((labs == 1.0).sum())
            n_all += n
            n_in += inside
            print(
                f"  {path.name} N={n} inside={inside} outside={n - inside}"
            )
        print(
            f"preview_files={len(preview)} preview_N={n_all} "
            f"preview_inside={n_in} preview_outside={n_all - n_in}"
        )
        from dataset import OccupancyMultiNpzDataset, make_dataloader

        ds = OccupancyMultiNpzDataset(paths)
        loader = make_dataloader(ds, batch_size=8, shuffle=False)
        xyz, y = next(iter(loader))
        print(f"dataset_N={len(ds)} parts={len(ds.parts)}")
        print(f"batch xyz={tuple(xyz.shape)} y={tuple(y.shape)}")
    else:
        sample = cfg.data_dir / _SAMPLE_RELATIVE
        pts, labs = load_points_labels(sample)
        print(f"file={sample}")
        print(_summarize(pts, labs))
