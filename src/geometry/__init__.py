"""Training-side geometry: OBJ triangles, envelope samples, and face tokens."""

from geometry.face_tokens import clear_face_token_cache
from geometry.mesh_io import clear_triangle_cache
from geometry.surface import clear_envelope_cache


def clear_geometry_caches() -> None:
    """Drop OBJ triangle, envelope, and face-token process caches."""
    clear_triangle_cache()
    clear_envelope_cache()
    clear_face_token_cache()
