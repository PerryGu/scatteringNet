"""Face-token overlay: largest-area triangles + normals.

Does not import torch. Tokens stay in world XYZ (no AABB). Overlay only;
not used by occupancy train or infer.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometry.face_tokens import FACE_FEAT_DIM, face_tokens_from_triangles  # noqa: E402
from obj_fill import triangles_from_obj_text  # noqa: E402

N_FACES_MIN = 64
N_FACES_MAX = 1024
N_FACES_DEFAULT = 256


def clamp_n_faces(n_faces: int) -> int:
    """Keep the overlay count in the UI range."""
    n = int(n_faces)
    if n < N_FACES_MIN or n > N_FACES_MAX:
        raise ValueError(
            f"n_faces must be between {N_FACES_MIN} and {N_FACES_MAX}, got {n}"
        )
    return n


def faces_overlay_from_obj_text(obj_text: str, n_faces: int) -> dict:
    """
    World-space ``(n_faces, 12)`` tokens plus a normal-tick length.

    Largest faces by area, then tile if the mesh is short.
    """
    count = clamp_n_faces(n_faces)
    vertices, faces = triangles_from_obj_text(obj_text)
    tokens = face_tokens_from_triangles(vertices, faces, count)
    v0 = tokens[:, 0:3]
    v1 = tokens[:, 3:6]
    v2 = tokens[:, 6:9]
    corners = np.concatenate([v0, v1, v2], axis=0)
    extent = corners.max(axis=0) - corners.min(axis=0)
    diag = float(np.linalg.norm(extent))
    tick = max(diag * 0.05, 1e-4)
    rounded = np.round(tokens, decimals=5)
    n_unique = int(np.unique(rounded, axis=0).shape[0])
    return {
        "n": int(tokens.shape[0]),
        "n_mesh": int(faces.shape[0]),
        "n_unique": n_unique,
        "tick": tick,
        "feat_dim": int(FACE_FEAT_DIM),
        "tokens_b64": base64.b64encode(
            np.ascontiguousarray(tokens, dtype=np.float32)
        ).decode("ascii"),
    }
