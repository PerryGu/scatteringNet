"""Surface-conditioned occupancy head. No catalog train."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from occupancy_encoder import OccupancyEncoder, SurfaceEncoder
from geometry.encoder import MeshFaceEncoder


class OccupancyEncoderTests(unittest.TestCase):
    def test_two_shape_batch_logits(self) -> None:
        model = OccupancyEncoder(hidden=16, depth=2, latent_dim=8)
        model.eval()
        xyz = torch.randn(6, 3)
        env_a = torch.randn(32, 3)
        env_b = torch.randn(32, 3)
        envelope = torch.stack(
            [env_a, env_a, env_a, env_b, env_b, env_b], dim=0
        )
        shape_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        with torch.no_grad():
            logits = model(xyz, envelope, shape_id)
        self.assertEqual(tuple(logits.shape), (6, 1))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_unique_encode_once_per_shape(self) -> None:
        model = OccupancyEncoder(hidden=8, depth=1, latent_dim=4)
        seen: list[int] = []
        real = model.surface.forward

        def wrapped(envelope: torch.Tensor) -> torch.Tensor:
            seen.append(int(envelope.shape[0]))
            return real(envelope)

        model.surface.forward = wrapped  # type: ignore[method-assign]
        xyz = torch.randn(5, 3)
        envelope = torch.randn(5, 16, 3)
        shape_id = torch.tensor([2, 2, 7, 7, 2], dtype=torch.long)
        _ = model(xyz, envelope, shape_id)
        self.assertEqual(seen, [2])

    def test_surface_encoder_max_pools(self) -> None:
        enc = SurfaceEncoder(latent_dim=4, hidden=8)
        out = enc(torch.randn(3, 20, 3))
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_mesh_two_shape_batch_logits(self) -> None:
        model = OccupancyEncoder(
            hidden=16,
            depth=2,
            latent_dim=8,
            shape_encoder="mesh",
            encoder_hidden=8,
            encoder_depth=2,
        )
        model.eval()
        xyz = torch.randn(6, 3)
        faces_a = torch.randn(16, 12)
        faces_b = torch.randn(16, 12)
        geom = torch.stack(
            [faces_a, faces_a, faces_a, faces_b, faces_b, faces_b], dim=0
        )
        shape_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        with torch.no_grad():
            logits = model(xyz, geom, shape_id)
        self.assertEqual(tuple(logits.shape), (6, 1))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mesh_unique_encode_once_per_shape(self) -> None:
        model = OccupancyEncoder(
            hidden=8,
            depth=1,
            latent_dim=4,
            shape_encoder="mesh",
            encoder_hidden=8,
            encoder_depth=2,
        )
        seen: list[int] = []
        real = model.mesh.forward

        def wrapped(tokens: torch.Tensor) -> torch.Tensor:
            seen.append(int(tokens.shape[0]))
            return real(tokens)

        model.mesh.forward = wrapped  # type: ignore[method-assign]
        xyz = torch.randn(5, 3)
        geom = torch.randn(5, 10, 12)
        shape_id = torch.tensor([2, 2, 7, 7, 2], dtype=torch.long)
        _ = model(xyz, geom, shape_id)
        self.assertEqual(seen, [2])

    def test_mesh_face_encoder_max_pools(self) -> None:
        enc = MeshFaceEncoder(latent_dim=4, hidden=8, depth=2)
        out = enc(torch.randn(3, 12, 12))
        self.assertEqual(tuple(out.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
