"""Load occupancy query points and labels from scatter NPZ files.

:func:`load_points_labels` reads **one** file. A catalog resolver lists
many NPZs (glob or explicit paths) without training.

``load_points_labels`` still returns only ``points`` and ``labels``.
``load_points_labels_mesh`` also resolves ``mesh_path`` against ``data_dir``.
"""

from __future__ import annotations

import glob as globlib
import random
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


def read_npz_mesh_path(path: Path | str) -> str:
    """
    Read the stored ``mesh_path`` string from one occupancy NPZ.

    Step 2 writes a ``data_dir``-relative POSIX path (for example
    ``meshes/Primitives/Sphere/sphere_r0p5_sa16_sh16.obj``).

    Parameters
    ----------
    path:
        Occupancy ``.npz`` that contains ``mesh_path``.

    Returns
    -------
    str
        Stored path string (relative or absolute). Not resolved here.
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    # allow_pickle=True: some exports store a 0-d string / object array.
    with np.load(npz_path, allow_pickle=True) as raw:
        if "mesh_path" not in raw.files:
            raise KeyError(f"NPZ has no 'mesh_path' in {npz_path}")
        stored = str(np.asarray(raw["mesh_path"]).item()).strip()
    if not stored:
        raise ValueError(f"mesh_path is empty in {npz_path}")
    return stored


def resolve_mesh_path(stored: str, data_dir: Path | str) -> Path:
    """
    Resolve a stored ``mesh_path`` against ``data_dir``.

    Relative entries are joined to ``data_dir``. Absolute entries are
    used as-is. Missing files raise ``FileNotFoundError``.

    Parameters
    ----------
    stored:
        Value from :func:`read_npz_mesh_path`.
    data_dir:
        Dataset root (``config.yaml`` ``data_dir``).

    Returns
    -------
    Path
        Existing resolved mesh file.
    """
    text = str(stored).strip()
    if not text:
        raise ValueError("mesh_path is empty")
    item = Path(text)
    root = Path(data_dir)
    resolved = item if item.is_absolute() else (root / item)
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"mesh not found: {resolved} (stored={text!r})")
    return resolved


def load_points_labels_mesh(
    path: Path | str,
    data_dir: Path | str,
) -> tuple[PointsArray, LabelsArray, Path]:
    """
    Read occupancy arrays and resolve the source OBJ.

    Keeps :func:`load_points_labels` unchanged (xyz + labels only).

    Parameters
    ----------
    path:
        Occupancy ``.npz`` with ``points``, ``labels``, and ``mesh_path``.
    data_dir:
        Root used to resolve a relative ``mesh_path``.

    Returns
    -------
    points, labels, mesh_path:
        Same arrays as :func:`load_points_labels`, plus the existing OBJ.
    """
    npz_path = Path(path)
    points, labels = load_points_labels(npz_path)
    stored = read_npz_mesh_path(npz_path)
    mesh_path = resolve_mesh_path(stored, data_dir)
    return points, labels, mesh_path


def _is_combo_npz(path: Path) -> bool:
    """True when the filename looks like a combo dump (excluded from the catalog)."""
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


def _glob_npz(root: Path, pattern: str) -> list[Path]:
    """Match ``pattern`` under ``root`` (``*`` / ``**``)."""
    full = str(root / pattern)
    recursive = "**" in pattern.replace("\\", "/")
    found = globlib.glob(full, recursive=recursive)
    return [Path(p).resolve() for p in found if Path(p).is_file()]


def _is_parameterized_stem(path: Path) -> bool:
    """
    Maya catalog names are ``family_param_...``. Varied one-off stems
    (``Cone.obj`` → ``Cone__occupancy.npz``) have no ``_`` in the shape key
    and must not ride along when Windows glob is case-insensitive.
    """
    return "_" in shape_key(path)


def _subsample_shapes(
    paths: Sequence[Path],
    max_shapes: int,
    seed: int,
) -> list[Path]:
    """Keep NPZs for at most ``max_shapes`` unique :func:`shape_key` values."""
    if max_shapes < 1:
        raise ValueError(f"max_shapes must be >= 1, got {max_shapes}")
    keys: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = shape_key(path)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    if max_shapes >= len(keys):
        return list(paths)
    # Sort then sample so the same seed always picks the same meshes.
    chosen_keys = set(random.Random(int(seed)).sample(sorted(keys), max_shapes))
    return [path for path in paths if shape_key(path) in chosen_keys]


def resolve_npz_catalog(
    data_dir: Path | str,
    *,
    npz_glob: str = "exports/dataset/*.npz",
    npz_paths: Sequence[str | Path] | None = None,
    npz_catalog: Sequence[tuple[str, int | None]] | None = None,
    max_files_per_shape: int | None = 2,
    exclude_combo: bool = True,
    seed: int = 1,
) -> list[Path]:
    """
    Resolve occupancy NPZ paths under ``data_dir`` (no point loading).

    Priority: explicit ``npz_paths``, else ``npz_catalog`` (union of globs),
    else ``npz_glob``. Relative entries are joined to ``data_dir``.
    Missing files in ``npz_paths`` raise ``FileNotFoundError``.

    Parameters
    ----------
    data_dir:
        Dataset root (``config.yaml`` ``data_dir``).
    npz_glob:
        Single glob relative to ``data_dir`` (``*`` and ``**`` allowed).
    npz_paths:
        Explicit relative or absolute NPZ paths. Empty / None → use glob(s).
    npz_catalog:
        ``(glob, max_shapes)`` rows. ``max_shapes`` is unique meshes after
        the per-shape file cap; ``None`` keeps every mesh the glob hits.
    max_files_per_shape:
        Cap per :func:`shape_key` after sort. ``None`` = no cap.
    exclude_combo:
        Drop filenames containing ``combo``.
    seed:
        RNG for ``max_shapes`` subsampling (YAML ``seed``).

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
    elif npz_catalog:
        # Union in YAML order. Same file from two globs is kept once.
        seen: set[Path] = set()
        chosen = []
        for pattern, max_shapes in npz_catalog:
            hit = _glob_npz(root, str(pattern))
            if exclude_combo:
                hit = [p for p in hit if not _is_combo_npz(p)]
            hit = [
                p
                for p in hit
                if p.suffix.lower() == ".npz" and _is_parameterized_stem(p)
            ]
            hit = sorted(hit)
            hit = _cap_per_shape(hit, max_files_per_shape)
            if max_shapes is not None:
                hit = _subsample_shapes(hit, int(max_shapes), int(seed))
            for path in hit:
                if path in seen:
                    continue
                seen.add(path)
                chosen.append(path)
    else:
        chosen = _glob_npz(root, npz_glob)

    npz_only = [p for p in chosen if p.suffix.lower() == ".npz"]
    if exclude_combo:
        npz_only = [p for p in npz_only if not _is_combo_npz(p)]
    if npz_catalog and not npz_paths:
        # Already capped per glob; keep YAML union order (not a global sort).
        capped = npz_only
    else:
        npz_only = sorted(npz_only)
        capped = _cap_per_shape(npz_only, max_files_per_shape)
    if not capped:
        raise FileNotFoundError(
            f"No occupancy NPZ files matched under {root} "
            f"(glob={npz_glob!r}, catalog={bool(npz_catalog)}, "
            f"explicit={bool(npz_paths)})"
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
            npz_catalog=cfg.npz_catalog or None,
            max_files_per_shape=cfg.max_files_per_shape,
            seed=cfg.seed,
        )
        print(f"data_dir={cfg.data_dir}")
        print(f"npz_glob={cfg.npz_glob}")
        print(f"npz_catalog={list(cfg.npz_catalog)}")
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
        loader = make_dataloader(ds.parts[0], batch_size=8, shuffle=False)
        xyz, y = next(iter(loader))
        print(f"files={len(ds)} n_points={ds.n_points} parts={len(ds.parts)}")
        print(f"batch xyz={tuple(xyz.shape)} y={tuple(y.shape)}")
    else:
        sample = cfg.data_dir / _SAMPLE_RELATIVE
        pts, labs = load_points_labels(sample)
        print(f"file={sample}")
        print(_summarize(pts, labs))
