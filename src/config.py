"""Hybrid config loader for the v2 occupancy MLP MVP.

Static settings (``hidden``, ``depth``, ``seed``, ``data_dir``) live in
``config.yaml``. ``device`` is resolved here from CUDA availability.

This module does not read ``.env`` and does not open NPZ files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypedDict

import torch
import yaml

# Repo root: src/config.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_YAML = _REPO_ROOT / "config.yaml"

_REQUIRED_YAML_KEYS = ("hidden", "depth", "seed", "data_dir")


class YamlKnobs(TypedDict):
    """Subset of OccupancyConfig that is stored in YAML."""

    hidden: int
    depth: int
    seed: int
    data_dir: str


def get_device() -> torch.device:
    """Prefer CUDA when a GPU is visible; otherwise CPU.

    Training scripts (later steps) should still refuse a long run on CPU.
    Config only reports the device that is available right now.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_positive_int(name: str, value: Any) -> int:
    """YAML may yield int or (rarely) str; occupancy dims must be int >= 1."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _as_data_dir_string(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("data_dir must be a non-empty path string in config.yaml")
    return str(value).strip()


def load_yaml_knobs(path: Path) -> YamlKnobs:
    """Read YAML settings. Does not check that data_dir exists on disk."""
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
    """Resolved MVP settings from YAML plus detected device."""

    data_dir: Path
    device: torch.device
    hidden: int
    depth: int
    seed: int


def load_config(
    yaml_path: Path | None = None,
    *,
    require_existing_data_dir: bool = True,
) -> OccupancyConfig:
    """Compose OccupancyConfig from ``config.yaml`` (not from ``.env``).

    When ``require_existing_data_dir`` is True (default), a missing folder
    prints a short instruction and then raises FileNotFoundError — used for
    training and data loading.
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
    )


def format_config(cfg: OccupancyConfig) -> str:
    """Pretty-print for CLI smoke checks."""
    return (
        f"OccupancyConfig(\n"
        f"  data_dir={cfg.data_dir}\n"
        f"  device={cfg.device}\n"
        f"  hidden={cfg.hidden}\n"
        f"  depth={cfg.depth}\n"
        f"  seed={cfg.seed}\n"
        f")"
    )


if __name__ == "__main__":
    print(format_config(load_config()))
