"""Sample envelope (surface) points from uploaded OBJ text for the viewer overlay.

Uses occupancy ``sample_surface_points`` (area-weighted). Does not import torch.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometry.surface import sample_surface_points  # noqa: E402
from obj_fill import triangles_from_obj_text  # noqa: E402

N_SURFACE_MIN = 256
N_SURFACE_MAX = 4096
N_SURFACE_DEFAULT = 1024


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
    """
    Area-weighted surface samples in world XYZ.

    Same sampler as occupancy envelope training (``n_surface``). Overlay only.
    """
    count = clamp_n_surface(n_surface)
    vertices, faces = triangles_from_obj_text(obj_text)
    points = sample_surface_points(vertices, faces, count, seed=int(seed))
    return {
        "n": int(points.shape[0]),
        "points_b64": base64.b64encode(np.ascontiguousarray(points)).decode("ascii"),
    }
