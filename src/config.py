"""Hybrid config loader for catalog occupancy training.

Static experiment knobs live in ``config.yaml``. ``device`` is resolved
here from CUDA availability. Training knobs (epochs, lr, batch_size,
optimizer, catalog, val split) are YAML-owned so ``train_multi_npz``
does not hardcode them.

This module does not read ``.env`` and does not open NPZ files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypedDict

import torch
import yaml

# Repo root: src/config.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_YAML = _REPO_ROOT / "config.yaml"

_REQUIRED_YAML_KEYS = (
    "hidden",
    "depth",
    "seed",
    "data_dir",
    "epochs",
    "lr",
)


class YamlKnobs(TypedDict):
    """Subset of OccupancyConfig that is stored in YAML (paths as strings)."""

    hidden: int
    depth: int
    seed: int
    data_dir: str
    epochs: int
    lr: float
    val_fraction: float
    latent_dim: int | None
    npz_glob: str
    npz_paths: tuple[str, ...]
    npz_catalog: tuple[tuple[str, int | None], ...]
    max_files_per_shape: int | None
    run_name: str
    checkpoint_metric: str
    batch_size: int
    optimizer: str
    n_surface: int
    knn_k: int
    knn_local_dim: int | None
    shape_encoder: str


def get_device() -> torch.device:
    """
    CUDA when a GPU is visible; otherwise CPU.

    Hugging Face CPU Spaces have no CUDA: infer still runs (slower).
    ``SCATTERINGNET_DEVICE=cpu`` forces CPU even if a GPU exists.
    ``SCATTERINGNET_DEVICE=cuda`` uses CUDA only when ``is_available()``;
    otherwise it falls back to CPU (no crash).

    Training scripts should still warn on a long catalog run on CPU.
    """
    forced = os.environ.get("SCATTERINGNET_DEVICE", "").strip().lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def repo_root() -> Path:
    """Git / project root (folder that contains ``src/`` and ``config.yaml``)."""
    return _REPO_ROOT


def gpu_name(device: torch.device | None = None) -> str | None:
    """
    Human GPU name for the run snapshot (``None`` on CPU).

    Uses ``cfg.device`` when given so a forced-CPU train does not stamp a
    card that was not used.
    """
    dev = device if device is not None else get_device()
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    index = 0 if dev.index is None else int(dev.index)
    if index < 0 or index >= torch.cuda.device_count():
        return None
    name = str(torch.cuda.get_device_name(index)).strip()
    return name or None


def _as_positive_int(name: str, value: Any) -> int:
    """YAML may yield int or (rarely) str; occupancy dims must be int >= 1."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _as_int_in_range(name: str, value: Any, lo: int, hi: int) -> int:
    """Inclusive integer range (YAML ``knn_k`` is 0–4096)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < lo or parsed > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {parsed}")
    return parsed


def _as_positive_float(name: str, value: Any) -> float:
    """Learning-rate style knobs must be a finite float > 0."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if parsed <= 0.0 or parsed != parsed:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _as_open_unit_interval(name: str, value: Any) -> float:
    """Hold-out fractions must be in (0, 1) so both splits are non-empty."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if parsed != parsed or parsed <= 0.0 or parsed >= 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {parsed}")
    return parsed


def _as_nonempty_path_string(name: str, value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} must be a non-empty path string in config.yaml")
    return str(value).strip()


def _as_data_dir_string(value: Any) -> str:
    return _as_nonempty_path_string("data_dir", value)


def _as_optional_positive_int(name: str, value: Any) -> int | None:
    """YAML null → unlimited catalog cap; otherwise int >= 1."""
    if value is None:
        return None
    return _as_positive_int(name, value)


def _as_run_name(value: Any) -> str:
    """Optional YAML suffix for ``runs/<timestamp>_<name>/``; empty → ``run``."""
    if value is None:
        return "run"
    text = str(value).strip()
    return text if text else "run"


_CHECKPOINT_METRIC_ALIASES = {
    "test_acc": "val_acc",
    "test_iou": "val_iou",
}


def _as_checkpoint_metric(value: Any) -> str:
    """Name of the scalar used to decide ``best.pt`` (strict improve)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("checkpoint_metric must be a non-empty string")
    name = str(value).strip()
    return _CHECKPOINT_METRIC_ALIASES.get(name, name)


def _as_val_fraction(raw: Mapping[str, Any]) -> float:
    """Prefer ``val_fraction``; accept legacy ``test_fraction``."""
    if "val_fraction" in raw:
        return _as_open_unit_interval("val_fraction", raw["val_fraction"])
    if "test_fraction" in raw:
        return _as_open_unit_interval("test_fraction", raw["test_fraction"])
    raise ValueError("Config YAML missing keys: val_fraction")


def _as_optional_latent_dim(raw: Mapping[str, Any]) -> int | None:
    """YAML omit / null → use ``hidden`` at train time."""
    if "latent_dim" not in raw or raw["latent_dim"] is None:
        return None
    return _as_positive_int("latent_dim", raw["latent_dim"])


# Names accepted in config.yaml ``optimizer``. Used by train_multi_npz.
_ALLOWED_OPTIMIZERS = ("adam", "adamw", "sgd")
# ``none`` = OccupancyMLP (xyz only). ``surface`` = envelope encoder.
_ALLOWED_SHAPE_ENCODERS = ("none", "surface")


def _as_optimizer(value: Any) -> str:
    """Optimizer family for multi-NPZ train; default Adam."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return "adam"
    name = str(value).strip().lower()
    if name not in _ALLOWED_OPTIMIZERS:
        allowed = ", ".join(_ALLOWED_OPTIMIZERS)
        raise ValueError(f"optimizer must be one of {allowed}, got {value!r}")
    return name


def _as_shape_encoder(value: Any) -> str:
    """Occupancy head family; default xyz-only so older YAML still loads."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return "none"
    name = str(value).strip().lower()
    if name not in _ALLOWED_SHAPE_ENCODERS:
        allowed = ", ".join(_ALLOWED_SHAPE_ENCODERS)
        raise ValueError(f"shape_encoder must be one of {allowed}, got {value!r}")
    return name


def _as_npz_paths(value: Any) -> tuple[str, ...]:
    """Explicit NPZ list relative to data_dir (empty → use glob)."""
    if value is None:
        return ()
    if isinstance(value, str):
        item = value.strip()
        return (item,) if item else ()
    if not isinstance(value, list):
        raise ValueError(f"npz_paths must be a list of strings or null, got {type(value).__name__}")
    out: list[str] = []
    for i, raw in enumerate(value):
        text = _as_nonempty_path_string(f"npz_paths[{i}]", raw)
        out.append(text)
    return tuple(out)


def _as_npz_catalog(value: Any) -> tuple[tuple[str, int | None], ...]:
    """Union of globs; optional ``max_shapes`` is unique meshes per glob."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"npz_catalog must be a list or null, got {type(value).__name__}"
        )
    out: list[tuple[str, int | None]] = []
    for i, raw in enumerate(value):
        if isinstance(raw, str):
            glob_s = _as_nonempty_path_string(f"npz_catalog[{i}]", raw)
            out.append((glob_s, None))
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"npz_catalog[{i}] must be a glob string or mapping, "
                f"got {type(raw).__name__}"
            )
        if "glob" not in raw:
            raise ValueError(f"npz_catalog[{i}] missing glob")
        glob_s = _as_nonempty_path_string(f"npz_catalog[{i}].glob", raw["glob"])
        max_shapes: int | None = None
        if "max_shapes" in raw and raw["max_shapes"] is not None:
            max_shapes = _as_positive_int(
                f"npz_catalog[{i}].max_shapes", raw["max_shapes"]
            )
        out.append((glob_s, max_shapes))
    return tuple(out)


def as_repo_relative(path: Path | str, *, root: Path | None = None) -> str:
    """
    POSIX string relative to the git repo when ``path`` is inside it.

    Already-relative inputs are returned as POSIX. Absolute paths on another
    drive (the dataset disk) cannot be repo-relative and stay absolute POSIX.
    """
    text = str(path).strip()
    if not text:
        return text
    parsed = Path(text)
    if not parsed.is_absolute():
        return parsed.as_posix()
    base = (root or _REPO_ROOT).resolve()
    try:
        return parsed.resolve().relative_to(base).as_posix()
    except ValueError:
        return parsed.resolve().as_posix()


def as_data_relative(path: Path | str, data_dir: Path | str) -> str:
    """
    POSIX string relative to ``data_dir`` (``exports/...``, not ``E:/...``).

    Already-relative inputs are returned as POSIX. Paths outside ``data_dir``
    (unit-test temp trees) fall back to absolute POSIX.
    """
    text = str(path).strip()
    if not text:
        return text
    parsed = Path(text)
    if not parsed.is_absolute():
        return parsed.as_posix()
    root = Path(data_dir).expanduser().resolve()
    resolved = parsed.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_yaml_knobs(path: Path) -> YamlKnobs:
    """
    Read YAML settings. Does not check that data_dir exists on disk.

    Parameters
    ----------
    path:
        Path to ``config.yaml``.

    Returns
    -------
    YamlKnobs
        Typed dict of experiment knobs (paths still strings).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config YAML must be a mapping, got {type(raw).__name__}")
    missing = [k for k in _REQUIRED_YAML_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Config YAML missing keys: {', '.join(missing)}")
    return {
        "hidden": _as_positive_int("hidden", raw["hidden"]),
        "depth": _as_positive_int("depth", raw["depth"]),
        "seed": _as_positive_int("seed", raw["seed"]),
        "data_dir": _as_data_dir_string(raw["data_dir"]),
        "epochs": _as_positive_int("epochs", raw["epochs"]),
        "lr": _as_positive_float("lr", raw["lr"]),
        "val_fraction": _as_val_fraction(raw),
        "latent_dim": _as_optional_latent_dim(raw),
        "npz_glob": (
            _as_nonempty_path_string("npz_glob", raw["npz_glob"])
            if "npz_glob" in raw
            else "exports/dataset/*.npz"
        ),
        "npz_paths": _as_npz_paths(raw.get("npz_paths")),
        "npz_catalog": _as_npz_catalog(raw.get("npz_catalog")),
        "max_files_per_shape": (
            _as_optional_positive_int("max_files_per_shape", raw["max_files_per_shape"])
            if "max_files_per_shape" in raw
            else 2
        ),
        "run_name": (
            _as_run_name(raw["run_name"]) if "run_name" in raw else "run"
        ),
        "checkpoint_metric": (
            _as_checkpoint_metric(raw["checkpoint_metric"])
            if "checkpoint_metric" in raw
            else "val_acc"
        ),
        "batch_size": (
            _as_positive_int("batch_size", raw["batch_size"])
            if "batch_size" in raw
            else 1024
        ),
        "optimizer": (
            _as_optimizer(raw["optimizer"]) if "optimizer" in raw else "adam"
        ),
        "n_surface": (
            _as_positive_int("n_surface", raw["n_surface"])
            if "n_surface" in raw
            else 1024
        ),
        "knn_k": (
            _as_int_in_range("knn_k", raw["knn_k"], 0, 4096)
            if "knn_k" in raw
            else 0
        ),
        "knn_local_dim": (
            _as_positive_int("knn_local_dim", raw["knn_local_dim"])
            if "knn_local_dim" in raw
            else None
        ),
        "shape_encoder": (
            _as_shape_encoder(raw["shape_encoder"])
            if "shape_encoder" in raw
            else "none"
        ),
    }


def _warn_missing_data_dir(data_dir: Path, yaml_path: Path) -> None:
    """Print a terminal hint so the user can fix config.yaml (no .env involved)."""
    print(
        "\n"
        "Dataset folder not found.\n"
        f"  Looked for: {data_dir}\n"
        "\n"
        "Update `data_dir` in config.yaml to the folder that contains "
        "`exports/` and `meshes/`.\n"
        f"  Config file: {yaml_path}\n"
    )


def require_data_dir(data_dir: Path, *, yaml_path: Path) -> None:
    """Validate the dataset root before training or NPZ loading."""
    if data_dir.is_dir():
        return
    _warn_missing_data_dir(data_dir, yaml_path)
    raise FileNotFoundError(
        f"Dataset directory does not exist: {data_dir}. "
        f"Set data_dir in {yaml_path}."
    )


@dataclass(frozen=True)
class OccupancyConfig:
    """Resolved experiment settings from YAML plus detected device."""

    data_dir: Path
    device: torch.device
    hidden: int
    depth: int
    seed: int
    epochs: int
    lr: float
    # Fraction of catalog **meshes** held out as val (selection split, not a locked test).
    val_fraction: float
    # Catalog knobs (optional in YAML; omitted keys keep these defaults).
    npz_glob: str = "exports/dataset/*.npz"
    npz_paths: tuple[str, ...] = ()
    # ``(glob, max_shapes)`` rows. Empty → use ``npz_glob``. ``max_shapes``
    # None keeps every mesh that glob hits (after ``max_files_per_shape``).
    npz_catalog: tuple[tuple[str, int | None], ...] = ()
    max_files_per_shape: int | None = 2
    # Suffix for runs/<timestamp>_<name>/ (device stays runtime-only).
    run_name: str = "run"
    # Which logged scalar selects best.pt (strict improve).
    checkpoint_metric: str = "val_acc"
    # Encoder latent width; None → use ``hidden`` at train / infer time.
    latent_dim: int | None = None
    # Mini-batch size and optimizer family (train_multi_npz).
    batch_size: int = 1024
    optimizer: str = "adam"
    # Envelope sample count (YAML). Used when ``shape_encoder`` is ``surface``.
    n_surface: int = 1024
    # 0 = global envelope z only. >0 = that many nearest envelope dots per query.
    knn_k: int = 0
    # Width of z_local; None → same as occupancy latent_dim / hidden.
    knn_local_dim: int | None = None
    # ``none`` keeps OccupancyMLP; ``surface`` uses the envelope PointNet.
    shape_encoder: str = "none"


def load_config(
    yaml_path: Path | None = None,
    *,
    require_existing_data_dir: bool = True,
) -> OccupancyConfig:
    """
    Compose OccupancyConfig from ``config.yaml`` (not from ``.env``).

    When ``require_existing_data_dir`` is True (default), a missing folder
    prints a short instruction and then raises FileNotFoundError — used for
    training and data loading.

    Parameters
    ----------
    yaml_path:
        Config file; default is repo-root ``config.yaml``.
    require_existing_data_dir:
        If True, refuse to return a config whose ``data_dir`` is missing.

    Returns
    -------
    OccupancyConfig
        YAML knobs plus detected ``device``.
    """
    cfg_path = yaml_path or _DEFAULT_YAML
    knobs = load_yaml_knobs(cfg_path)
    data_dir = Path(knobs["data_dir"])
    if require_existing_data_dir:
        require_data_dir(data_dir, yaml_path=cfg_path)
    return OccupancyConfig(
        data_dir=data_dir,
        device=get_device(),
        hidden=knobs["hidden"],
        depth=knobs["depth"],
        seed=knobs["seed"],
        epochs=knobs["epochs"],
        lr=knobs["lr"],
        val_fraction=knobs["val_fraction"],
        latent_dim=knobs["latent_dim"],
        npz_glob=knobs["npz_glob"],
        npz_paths=knobs["npz_paths"],
        npz_catalog=knobs["npz_catalog"],
        max_files_per_shape=knobs["max_files_per_shape"],
        run_name=knobs["run_name"],
        checkpoint_metric=knobs["checkpoint_metric"],
        batch_size=knobs["batch_size"],
        optimizer=knobs["optimizer"],
        n_surface=knobs["n_surface"],
        knn_k=knobs["knn_k"],
        knn_local_dim=knobs["knn_local_dim"],
        shape_encoder=knobs["shape_encoder"],
    )


def encoder_latent_dim(cfg: OccupancyConfig) -> int:
    """OccupancyEncoder ``z`` width: YAML ``latent_dim`` or ``hidden``."""
    if cfg.latent_dim is None:
        return int(cfg.hidden)
    return int(cfg.latent_dim)


def encoder_knn_local_dim(cfg: OccupancyConfig) -> int:
    """Local envelope code width: YAML ``knn_local_dim`` or global latent."""
    if cfg.knn_local_dim is None:
        return encoder_latent_dim(cfg)
    return int(cfg.knn_local_dim)


def format_config(cfg: OccupancyConfig) -> str:
    """
    Pretty-print for CLI smoke checks.

    Parameters
    ----------
    cfg:
        Resolved config.

    Returns
    -------
    str
        Multi-line ``OccupancyConfig(...)`` dump.
    """
    return (
        f"OccupancyConfig(\n"
        f"  data_dir={cfg.data_dir}\n"
        f"  device={cfg.device}\n"
        f"  gpu={gpu_name(cfg.device)}\n"
        f"  hidden={cfg.hidden}\n"
        f"  depth={cfg.depth}\n"
        f"  seed={cfg.seed}\n"
        f"  epochs={cfg.epochs}\n"
        f"  lr={cfg.lr}\n"
        f"  val_fraction={cfg.val_fraction}\n"
        f"  latent_dim={cfg.latent_dim}\n"
        f"  npz_glob={cfg.npz_glob}\n"
        f"  npz_paths={list(cfg.npz_paths)}\n"
        f"  npz_catalog={list(cfg.npz_catalog)}\n"
        f"  max_files_per_shape={cfg.max_files_per_shape}\n"
        f"  run_name={cfg.run_name}\n"
        f"  checkpoint_metric={cfg.checkpoint_metric}\n"
        f"  batch_size={cfg.batch_size}\n"
        f"  optimizer={cfg.optimizer}\n"
        f"  n_surface={cfg.n_surface}\n"
        f"  knn_k={cfg.knn_k}\n"
        f"  knn_local_dim={cfg.knn_local_dim}\n"
        f"  shape_encoder={cfg.shape_encoder}\n"
        f")"
    )


if __name__ == "__main__":
    print(format_config(load_config()))
