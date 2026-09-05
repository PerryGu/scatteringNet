"""Training-side geometry: OBJ triangles and envelope samples."""

from geometry.mesh_io import clear_triangle_cache
from geometry.surface import clear_envelope_cache


def clear_geometry_caches() -> None:
    """Drop OBJ triangle and envelope process caches."""
    clear_triangle_cache()
    clear_envelope_cache()
