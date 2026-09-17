"""Viewer Envelope overlay: same area-weighted sampler as occupancy.

Does not import torch. Count is clamped to the UI range.
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

from geometry.surface import sample_surface_points  # noqa: E402

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


def envelope_from_obj_text(
    obj_text: str,
    n_surface: int,
    *,
    seed: int = 1,
) -> dict:
    """World-space overlay cloud. Area-weighted face samples, same as train."""
    count = clamp_n_surface(n_surface)
    vertices, faces = triangles_from_obj_text(obj_text)
    cloud = sample_surface_points(vertices, faces, count, seed=int(seed))
    points = np.ascontiguousarray(cloud[:, :3])
    normals = np.ascontiguousarray(cloud[:, 3:])
    return {
        "n": int(points.shape[0]),
        "points_b64": base64.b64encode(points).decode("ascii"),
        "normals_b64": base64.b64encode(normals).decode("ascii"),
    }
