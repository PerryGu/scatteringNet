"""
Maya exporters (Script Editor only) plus conda-side occupancy NPZ sampling.

Conda imports from this package load Open3D/trimesh. Maya scripts must be
``exec``'d as files; they do not import this ``__init__``.
"""

from scatteringnet.scatter_generation.dataset_builder import build_singles_dataset, expand_param_grid
from scatteringnet.scatter_generation.mesh_loader import load_mesh
from scatteringnet.scatter_generation.raycast_scatter import (
    ScatterResult,
    export_occupancy_npz,
    export_scatter_npz,
    scatter_volume,
)

__all__ = [
    "ScatterResult",
    "build_singles_dataset",
    "expand_param_grid",
    "export_occupancy_npz",
    "export_scatter_npz",
    "load_mesh",
    "scatter_volume",
]
