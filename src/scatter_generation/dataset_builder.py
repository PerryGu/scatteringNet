"""
Batch-build occupancy NPZ files from a folder of meshes (conda, not Maya).

Example (from repo root, after ``pip install -e .``):

  python -m scatteringnet.scatter_generation.dataset_builder E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere \\
    --out E:/Work_stuff/scatteringNet/data/exports/dataset_test \\
    --method occupancy --spacings 0.15 --random-ranges 0 --limit 1 --seed 1

``--random-ranges`` is per-axis travel after the density lattice; labels are
computed on the moved points. ``0`` means no movement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


from scatteringnet.scatter_generation.mesh_loader import (
    iter_mesh_files,
    load_mesh,
    project_root,
    to_data_relative,
)
from scatteringnet.scatter_generation.raycast_scatter import export_scatter_npz, scatter_volume

_AXES = ("x", "y", "z")
_METHODS = ("raycast", "occupancy")
_DEFAULT_OUT = project_root() / "data" / "exports" / "dataset"


@dataclass(frozen=True)
class ScatterParams:
    """
    One scatter configuration (filename stem + sampler knobs).

    Attributes
    ----------
    axis:
        Ray axis ``x|y|z``, or ``n`` for occupancy (no ray).
    rays:
        Fixed ray-grid size; ``0`` when ``auto_rays`` or occupancy.
    spacing:
        Point spacing (density).
    include_outside:
        Keep outside samples.
    method:
        ``occupancy`` or ``raycast``.
    jitter:
        Per-axis random range (world units); ``0`` leaves the lattice.
    auto_rays:
        Derive ray counts from bbox / spacing (raycast only).
    """

    axis: str
    rays: int
    spacing: float
    include_outside: bool = True
    method: str = "occupancy"
    jitter: float = 0.0
    auto_rays: bool = False

    def tag(self) -> str:
        """
        Build the NPZ filename stem for this configuration.

        Returns
        -------
        str
            Token such as ``occupancy_s0.1_inout`` or
            ``raycast_z_r16_s0.1_j0.05_inout``.
        """
        outside = "inout" if self.include_outside else "in"
        spacing_tag = f"{self.spacing:.4f}".rstrip("0").rstrip(".")
        jitter_tag = f"{self.jitter:.4f}".rstrip("0").rstrip(".")
        jpart = f"_j{jitter_tag}" if self.jitter > 0 else ""
        if self.method == "occupancy":
            return f"occupancy_s{spacing_tag}{jpart}_{outside}"
        if self.auto_rays:
            return f"raycast_{self.axis}_raut_s{spacing_tag}{jpart}_{outside}"
        return f"raycast_{self.axis}_r{self.rays}_s{spacing_tag}{jpart}_{outside}"


@dataclass
class SampleRecord:
    """
    One exported NPZ (manifest row).

    Attributes
    ----------
    id:
        Filename stem (mesh + param tag).
    mesh_name, mesh_path, export_path:
        Source mesh identity and written NPZ path (repo-relative when possible).
    method, axis, rays, spacing, jitter, include_outside:
        Sampler knobs used for this file.
    num_inside, num_outside, total:
        Class counts; zeros on dry-run or failure.
    occupancy_verified:
        True when labels came from per-point occupancy.
    error:
        Failure message, or ``None`` if the write succeeded.
    """

    id: str
    mesh_name: str
    mesh_path: str
    export_path: str
    method: str
    axis: str
    rays: int
    spacing: float
    jitter: float
    include_outside: bool
    num_inside: int
    num_outside: int
    total: int
    occupancy_verified: bool = False
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Return this record as a JSON-serializable dict."""
        return asdict(self)


@dataclass
class DatasetBuildResult:
    """
    Summary of a batch dataset build.

    Attributes
    ----------
    root, out_dir:
        Mesh folder and export folder (repo-relative strings).
    created_at:
        UTC ISO timestamp.
    params:
        List of :class:`ScatterParams` as dicts.
    samples:
        One :class:`SampleRecord` per planned or written NPZ.
    skipped_meshes:
        Load failures ``{mesh_path, error}``.
    """

    root: str
    out_dir: str
    created_at: str
    params: list[dict[str, Any]]
    samples: list[SampleRecord] = field(default_factory=list)
    skipped_meshes: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        """Number of samples with ``error is None``."""
        return sum(1 for s in self.samples if s.error is None)

    @property
    def fail_count(self) -> int:
        """Number of samples with a recorded ``error``."""
        return sum(1 for s in self.samples if s.error is not None)

    def to_manifest(self) -> dict[str, Any]:
        """JSON object written to ``manifest.json``."""
        return {
            "root": self.root,
            "out_dir": self.out_dir,
            "created_at": self.created_at,
            "params": self.params,
            "ok_count": self.ok_count,
            "fail_count": self.fail_count,
            "skipped_meshes": self.skipped_meshes,
            "samples": [s.to_row() for s in self.samples],
        }


@dataclass
class _Job:
    """One loaded mesh waiting to be scattered."""
    name: str
    mesh_path: str
    mesh: Any
    parts: list[str]


def expand_param_grid(
    *,
    axes: Sequence[str] = ("z",),
    rays: Sequence[int] = (32,),
    spacings: Sequence[float] = (0.05,),
    jitters: Sequence[float] = (0.0,),
    include_outside: bool = True,
    method: str = "occupancy",
    auto_rays: bool = False,
) -> list[ScatterParams]:
    """
    Cartesian product of scatter parameters (single meshes only).

    Parameters
    ----------
    axes:
        Ray axes (raycast). Ignored for occupancy.
    rays:
        Fixed ray-grid sizes (ignored if ``auto_rays``).
    spacings:
        Point spacings; each must be ``> 0``.
    jitters:
        Random-range values; each must be ``>= 0``.
    include_outside:
        Keep outside samples.
    method:
        ``occupancy`` or ``raycast``.
    auto_rays:
        Derive ray counts from bbox / spacing.

    Returns
    -------
    list of ScatterParams
        One entry per combination.
    """
    method = str(method).lower().strip()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    clean_spacings = sorted({float(s) for s in spacings})
    if any(s <= 0 for s in clean_spacings):
        raise ValueError("Each spacing value must be > 0")
    clean_jitters = sorted({float(j) for j in jitters})
    if any(j < 0 for j in clean_jitters):
        raise ValueError("Each jitter value must be >= 0")

    if method == "occupancy":
        return [
            ScatterParams(
                axis="n",
                rays=0,
                spacing=spacing,
                include_outside=include_outside,
                method=method,
                jitter=jitter,
                auto_rays=False,
            )
            for spacing, jitter in itertools.product(clean_spacings, clean_jitters)
        ]

    clean_axes: list[str] = []
    for a in axes:
        a = str(a).lower().strip()
        if a not in _AXES:
            raise ValueError(f"Invalid axis {a!r}; use x, y, or z")
        if a not in clean_axes:
            clean_axes.append(a)
    if not clean_axes:
        raise ValueError("At least one axis is required")

    if auto_rays:
        return [
            ScatterParams(
                axis=axis,
                rays=0,
                spacing=spacing,
                include_outside=include_outside,
                method=method,
                jitter=jitter,
                auto_rays=True,
            )
            for jitter, spacing, axis in itertools.product(
                clean_jitters, clean_spacings, clean_axes
            )
        ]

    clean_rays = sorted({int(r) for r in rays})
    if any(r < 2 for r in clean_rays):
        raise ValueError("Each rays value must be >= 2")
    return [
        ScatterParams(
            axis=axis,
            rays=r,
            spacing=spacing,
            include_outside=include_outside,
            method=method,
            jitter=jitter,
            auto_rays=False,
        )
        for jitter, spacing, r, axis in itertools.product(
            clean_jitters, clean_spacings, clean_rays, clean_axes
        )
    ]


def _safe_stem(name: str) -> str:
    """Filesystem-safe stem: alphanumerics, hyphen, underscore only."""
    stem = Path(name).stem if "." in name else name
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


def parse_csv_ints(text: str) -> list[int]:
    """
    Parse a comma-separated integer list (CLI ``--rays``).

    Parameters
    ----------
    text:
        E.g. ``16,32``.

    Returns
    -------
    list of int
    """
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return [int(v) for v in values]


def parse_csv_floats(text: str) -> list[float]:
    """
    Parse a comma-separated float list (CLI ``--spacings`` / ``--random-ranges``).

    Parameters
    ----------
    text:
        E.g. ``0.1,0.15``.

    Returns
    -------
    list of float
    """
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return [float(v) for v in values]


def parse_csv_axes(text: str) -> list[str]:
    """
    Parse ray axes from CLI ``--axes``.

    Parameters
    ----------
    text:
        ``x``, ``y,z``, or ``all`` / ``*``.

    Returns
    -------
    list of str
        Lowercase axis names.
    """
    text = text.strip().lower()
    if text in {"all", "*"}:
        return list(_AXES)
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one axis (x,y,z) or 'all'")
    return values


def build_singles_dataset(
    mesh_root: str | Path,
    out_dir: str | Path,
    *,
    axes: Sequence[str] = ("z",),
    rays: Sequence[int] = (16,),
    spacings: Sequence[float] = (0.1,),
    jitters: Sequence[float] = (0.0,),
    include_outside: bool = True,
    method: str = "occupancy",
    auto_rays: bool = False,
    max_points: int = 200_000,
    seed: int | None = None,
    recursive: bool = True,
    dry_run: bool = False,
    write_manifest: bool = True,
    limit: int | None = None,
    skip: int = 0,
    name_glob: str | None = None,
) -> DatasetBuildResult:
    """
    Load every mesh under ``mesh_root`` and export one NPZ per param combo.

    ``skip`` / ``limit`` slice the sorted mesh list. ``max_points`` coarsens
    density rather than emitting multi-million clouds.

    Parameters
    ----------
    mesh_root:
        Folder of OBJ/PLY/STL files.
    out_dir:
        Destination for NPZs and ``manifest.json``.
    axes, rays, spacings, jitters, include_outside, method, auto_rays:
        Forwarded to :func:`expand_param_grid`.
    max_points:
        Per-file sample cap (must be ``>= 8``).
    seed:
        RNG seed for ``jitter`` / ``random_range``.
    recursive:
        Scan nested folders.
    dry_run:
        Plan paths without writing.
    write_manifest:
        Write ``manifest.json`` after a real run.
    limit:
        Max meshes to load (``None`` = all after skip).
    skip:
        Number of sorted mesh files to skip.
    name_glob:
        Optional filename ``fnmatch`` (e.g. ``*_nr5_*.obj``). ``None`` = all meshes.

    Returns
    -------
    DatasetBuildResult
        Manifest payload including successes and failures.
    """
    mesh_root = Path(mesh_root)
    out_dir = Path(out_dir)
    if not mesh_root.is_dir():
        raise NotADirectoryError(f"Mesh root is not a directory: {mesh_root}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when set")
    if skip < 0:
        raise ValueError("skip must be >= 0")
    if max_points < 8:
        raise ValueError("max_points must be >= 8")

    params = expand_param_grid(
        axes=axes,
        rays=rays,
        spacings=spacings,
        jitters=jitters,
        include_outside=include_outside,
        method=method,
        auto_rays=auto_rays,
    )
    result = DatasetBuildResult(
        root=to_data_relative(mesh_root),
        out_dir=to_data_relative(out_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
        params=[asdict(p) for p in params],
    )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_mesh_files(mesh_root, recursive=recursive, name_glob=name_glob)
    total_available = len(files)
    if skip:
        files = files[skip:]
    if limit is not None:
        files = files[:limit]
    print(
        f"Loading {len(files)} mesh file(s) "
        f"(skip={skip}, limit={limit}, available={total_available})...",
        flush=True,
    )

    jobs: list[_Job] = []
    for i, path in enumerate(files, start=1):
        print(f"  [{i}/{len(files)}] load {path.name}", flush=True)
        try:
            mesh, info = load_mesh(path)
        except Exception as exc:  # noqa: BLE001
            result.skipped_meshes.append(
                {"mesh_path": to_data_relative(path), "error": str(exc)}
            )
            continue
        jobs.append(
            _Job(
                name=info.name,
                mesh_path=to_data_relative(info.path),
                mesh=mesh,
                parts=[info.name],
            )
        )

    total = len(jobs) * len(params)
    done = 0
    for job in jobs:
        mesh_stem = _safe_stem(job.name)
        for p in params:
            done += 1
            sample_id = f"{mesh_stem}__{p.tag()}"
            export_path = out_dir / f"{sample_id}.npz"
            if dry_run:
                result.samples.append(
                    SampleRecord(
                        id=sample_id,
                        mesh_name=job.name,
                        mesh_path=to_data_relative(job.mesh_path),
                        export_path=to_data_relative(export_path),
                        method=p.method,
                        axis=p.axis,
                        rays=p.rays,
                        spacing=p.spacing,
                        jitter=p.jitter,
                        include_outside=p.include_outside,
                        num_inside=0,
                        num_outside=0,
                        total=0,
                    )
                )
                print(f"  [{done}/{total}] dry-run {export_path.name}", flush=True)
                continue

            print(
                f"  [{done}/{total}] scatter {job.name} "
                f"axis={p.axis} spacing={p.spacing} jitter={p.jitter}...",
                flush=True,
            )
            try:
                scatter = scatter_volume(
                    job.mesh,
                    method=p.method,
                    axis=p.axis,  # type: ignore[arg-type]
                    ray_grid=(max(p.rays, 2), max(p.rays, 2)),
                    point_spacing=p.spacing,
                    include_outside=p.include_outside,
                    auto_rays=p.auto_rays,
                    jitter=p.jitter,
                    seed=seed,
                    max_points=max_points,
                )
                written = export_scatter_npz(
                    scatter, export_path, mesh_path=to_data_relative(job.mesh_path)
                )
                result.samples.append(
                    SampleRecord(
                        id=sample_id,
                        mesh_name=job.name,
                        mesh_path=to_data_relative(job.mesh_path),
                        export_path=to_data_relative(written),
                        method=p.method,
                        axis=p.axis,
                        rays=p.rays,
                        spacing=p.spacing,
                        jitter=p.jitter,
                        include_outside=p.include_outside,
                        num_inside=scatter.num_inside,
                        num_outside=scatter.num_outside,
                        total=len(scatter.points),
                        occupancy_verified=bool(scatter.occupancy_verified),
                    )
                )
                print(
                    f"      wrote {written.name} "
                    f"inside={scatter.num_inside} outside={scatter.num_outside} "
                    f"total={len(scatter.points)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                result.samples.append(
                    SampleRecord(
                        id=sample_id,
                        mesh_name=job.name,
                        mesh_path=to_data_relative(job.mesh_path),
                        export_path=to_data_relative(export_path),
                        method=p.method,
                        axis=p.axis,
                        rays=p.rays,
                        spacing=p.spacing,
                        jitter=p.jitter,
                        include_outside=p.include_outside,
                        num_inside=0,
                        num_outside=0,
                        total=0,
                        error=str(exc),
                    )
                )
                print(f"      FAIL {export_path.name}: {exc}", flush=True)

    if write_manifest and not dry_run:
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(result.to_manifest(), indent=2),
            encoding="utf-8",
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for :func:`main`."""
    p = argparse.ArgumentParser(
        prog="dataset_builder.py",
        description=(
            "Scan a mesh folder and export occupancy NPZ files "
            "(points, labels, mesh_path). Single meshes only."
        ),
    )
    p.add_argument("mesh_root", type=Path, help="Folder of meshes (OBJ/PLY/STL/...)")
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output folder for NPZs + manifest.json (default: {_DEFAULT_OUT})",
    )
    p.add_argument("--axes", default="z", help="Comma-separated axes, or 'all' (raycast)")
    p.add_argument(
        "--rays",
        default="16",
        help="Fixed ray grid sizes (ignored with --auto-rays)",
    )
    p.add_argument(
        "--auto-rays",
        action="store_true",
        help="Derive ray counts from bbox and spacing (raycast only)",
    )
    p.add_argument(
        "--spacings",
        default="0.1",
        help="Point spacings (density). Smaller = more points.",
    )
    p.add_argument(
        "--random-ranges",
        default=None,
        help=(
            "Per-axis random travel after sampling (world units). "
            "0 = no move; low ≈ 0.25*spacing; high ≈ 1.0*spacing. "
            "Comma-separated. Default: --jitters or 0."
        ),
    )
    p.add_argument(
        "--jitters",
        default="0",
        help="Alias for --random-ranges",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for random_range")
    p.add_argument("--no-outside", action="store_true", help="Keep inside points only")
    p.add_argument(
        "--method",
        choices=("raycast", "occupancy"),
        default="occupancy",
        help="occupancy = 3D grid + per-point test; raycast = bbox-face rays",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Cap samples per NPZ by coarsening density",
    )
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument(
        "--glob",
        default=None,
        help="fnmatch on filename only (e.g. *_nr5_*.obj). Default: all meshes.",
    )
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="At most N meshes")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """
    Batch-build occupancy NPZs from a mesh folder.

    Parameters
    ----------
    argv:
        Argument list; default is ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` invalid args / missing folder, ``2`` if any
        scatter job failed.
    """
    args = build_parser().parse_args(argv)
    mesh_root: Path = args.mesh_root
    if not mesh_root.is_dir():
        print(f"Not a directory: {mesh_root}", file=sys.stderr)
        return 1

    range_csv = args.random_ranges if args.random_ranges is not None else args.jitters
    try:
        axes = parse_csv_axes(args.axes)
        rays = parse_csv_ints(args.rays)
        spacings = parse_csv_floats(args.spacings)
        random_ranges = parse_csv_floats(range_csv)
        params = expand_param_grid(
            axes=axes,
            rays=rays,
            spacings=spacings,
            jitters=random_ranges,
            include_outside=not args.no_outside,
            method=args.method,
            auto_rays=args.auto_rays,
        )
    except ValueError as exc:
        print(f"Invalid parameters: {exc}", file=sys.stderr)
        return 1

    print(f"mesh root:     {mesh_root.resolve()}")
    print(f"out dir:       {args.out.resolve()}")
    print(f"method:        {args.method}")
    print(f"spacings:      {spacings}")
    print(f"random_range:  {random_ranges}")
    print(f"seed:          {args.seed}")
    print(f"max-points:    {args.max_points}")
    print(f"name glob:     {args.glob or '(all)'}")
    print(f"param combos:  {len(params)}")

    result = build_singles_dataset(
        mesh_root,
        args.out,
        axes=axes,
        rays=rays,
        spacings=spacings,
        jitters=random_ranges,
        include_outside=not args.no_outside,
        method=args.method,
        auto_rays=args.auto_rays,
        max_points=args.max_points,
        seed=args.seed,
        recursive=not args.no_recursive,
        dry_run=args.dry_run,
        limit=args.limit,
        skip=args.skip,
        name_glob=args.glob,
    )

    if args.dry_run:
        print(
            f"\ndry-run: would write {len(result.samples)} samples "
            f"(skipped meshes: {len(result.skipped_meshes)})"
        )
        return 0

    print(
        f"\ndone: ok={result.ok_count} fail={result.fail_count} "
        f"skipped_meshes={len(result.skipped_meshes)}"
    )
    print(f"manifest: {Path(result.out_dir) / 'manifest.json'}")
    if result.fail_count:
        for s in result.samples:
            if s.error:
                print(f"  FAIL {s.id}: {s.error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
