"""Training-side geometry: OBJ triangles and envelope samples."""

from scatteringnet.geometry.mesh_io import clear_triangle_cache
from scatteringnet.geometry.surface import clear_envelope_cache


def clear_geometry_caches() -> None:
    """Drop OBJ triangle and envelope process caches."""
    clear_triangle_cache()
    clear_envelope_cache()
