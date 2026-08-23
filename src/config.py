"""Hybrid config loader for the v2 occupancy MLP MVP (Step 2).

Static knobs (``hidden``, ``depth``, ``seed``) live in ``config.yaml``.
Runtime values are resolved here:

- ``DATA_DIR`` from the process environment (``.env`` fill-if-missing)
- ``device`` from CUDA availability

This module does not open NPZ files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

# Repo root: src/config.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _REPO_ROOT / ".env"
_DEFAULT_YAML = _REPO_ROOT / "config.yaml"

_REQUIRED_YAML_KEYS = ("hidden", "depth", "seed")


def _load_dotenv_file(path: Path) -> None:
    """Parse a minimal KEY=VALUE .env into os.environ (fill-if-missing)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


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


def load_yaml_knobs(path: Path) -> dict[str, int]:
    """Read static MLP knobs from YAML. Does not resolve DATA_DIR or device."""
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
    }


@dataclass(frozen=True)
class OccupancyConfig:
    """Resolved MVP settings: YAML knobs + runtime data_dir/device."""

    data_dir: Path
    device: torch.device
    hidden: int
    depth: int
    seed: int


def load_config(yaml_path: Path | None = None) -> OccupancyConfig:
    """Compose OccupancyConfig from YAML knobs and the environment."""
    _load_dotenv_file(_ENV_PATH)
    knobs = load_yaml_knobs(yaml_path or _DEFAULT_YAML)

    raw_data_dir = os.environ.get("DATA_DIR", "").strip()
    if not raw_data_dir:
        raise RuntimeError(
            "DATA_DIR is not set. Add it to the environment or to the repo "
            f".env file ({_ENV_PATH})."
        )
    return OccupancyConfig(
        data_dir=Path(raw_data_dir),
        device=get_device(),
        hidden=knobs["hidden"],
        depth=knobs["depth"],
        seed=knobs["seed"],
    )


def format_config(cfg: OccupancyConfig) -> str:
    """Pretty-print for CLI smoke checks (data_dir and device must be real)."""
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
