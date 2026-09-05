"""Viewer Envelope overlay: same mix sampler as occupancy ``sample_surface_points``.

Does not import torch. Count is clamped to the UI range; Mix 0–100 splits
face-area vs crease samples.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from obj_fill import triangles_from_obj_text

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometry.surface import (  # noqa: E402
    clamp_envelope_mix,
    crease_length_fraction,
    n_sharp_edges,
    plan_envelope_counts,
    sample_surface_points,
    split_envelope_counts,
)

N_SURFACE_MIN = 256
N_SURFACE_MAX = 4096
N_SURFACE_DEFAULT = 1024

PointsArray = NDArray[np.float32]


def clamp_n_surface(n_surface: int) -> int:
    """Keep the overlay count in the UI range."""
    n = int(n_surface)
    if n < N_SURFACE_MIN or n > N_SURFACE_MAX:
        raise ValueError(
            f"n_surface must be between {N_SURFACE_MIN} and {N_SURFACE_MAX}, got {n}"
        )
    return n


def sample_crease_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_surface: int,
    *,
    seed: int = 1,
) -> tuple[PointsArray, NDArray[np.int64], int]:
    """``mix=100`` occupancy envelope plus crease count (overlay tests)."""
    count = int(n_surface)
    points = sample_surface_points(
        vertices, faces, count, seed=int(seed), mix=100
    )
    n_creases = n_sharp_edges(vertices, faces)
    face_idx = np.zeros((int(points.shape[0]),), dtype=np.int64)
    return points, face_idx, n_creases


def envelope_from_obj_text(
    obj_text: str,
    n_surface: int,
    *,
    mix: int = 100,
    seed: int = 1,
) -> dict:
    """World-space overlay cloud. Same mix split as occupancy YAML ``envelope_mix``."""
    count = clamp_n_surface(n_surface)
    mix_v = clamp_envelope_mix(mix)
    vertices, faces = triangles_from_obj_text(obj_text)
    # Same cap-and-spill as occupancy so the status line matches the cloud.
    frac = crease_length_fraction(vertices, faces) if mix_v > 0 else 0.0
    n_area, n_edge = plan_envelope_counts(count, mix_v, frac)
    points = sample_surface_points(
        vertices, faces, count, seed=int(seed), mix=mix_v
    )
    n_creases = n_sharp_edges(vertices, faces) if n_edge > 0 else 0
    return {
        "n": int(points.shape[0]),
        "n_area": int(n_area),
        "n_edge": int(n_edge),
        "n_creases": int(n_creases),
        "mix": int(mix_v),
        "points_b64": base64.b64encode(np.ascontiguousarray(points)).decode("ascii"),
    }
