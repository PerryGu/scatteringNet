"""Surface-conditioned occupancy head. No catalog train."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infer_multi_npz import load_occupancy_model
from occupancy_encoder import (
    CHECKPOINT_KIND,
    OccupancyEncoder,
    SurfaceEncoder,
    envelope_dim_from_ckpt,
    envelope_seed_from_ckpt,
    knn_offsets,
)


class OccupancyEncoderTests(unittest.TestCase):
    def test_two_shape_batch_logits(self) -> None:
        model = OccupancyEncoder(hidden=16, depth=2, latent_dim=8)
        model.eval()
        xyz = torch.randn(6, 3)
        env_a = torch.randn(32, 6)
        env_b = torch.randn(32, 6)
        envelope = torch.stack(
            [env_a, env_a, env_a, env_b, env_b, env_b], dim=0
        )
        shape_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        with torch.no_grad():
            logits = model(xyz, envelope, shape_id)
        self.assertEqual(tuple(logits.shape), (6, 1))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(model.envelope_dim, 6)

    def test_xyz_only_envelope_dim(self) -> None:
        model = OccupancyEncoder(hidden=8, depth=1, latent_dim=4, envelope_dim=3)
        xyz = torch.randn(4, 3)
        envelope = torch.randn(4, 10, 3)
        shape_id = torch.zeros(4, dtype=torch.long)
        logits = model(xyz, envelope, shape_id)
        self.assertEqual(tuple(logits.shape), (4, 1))
        self.assertEqual(int(model.surface.point_mlp[0].weight.shape[1]), 3)

    def test_unique_encode_once_per_shape(self) -> None:
        model = OccupancyEncoder(hidden=8, depth=1, latent_dim=4)
        seen: list[int] = []
        real = model.surface.forward

        def wrapped(envelope: torch.Tensor) -> torch.Tensor:
            seen.append(int(envelope.shape[0]))
            return real(envelope)

        model.surface.forward = wrapped  # type: ignore[method-assign]
        xyz = torch.randn(5, 3)
        envelope = torch.randn(5, 16, 6)
        shape_id = torch.tensor([2, 2, 7, 7, 2], dtype=torch.long)
        _ = model(xyz, envelope, shape_id)
        self.assertEqual(seen, [2])

    def test_surface_encoder_max_pools(self) -> None:
        enc = SurfaceEncoder(latent_dim=4, hidden=8, in_dim=6)
        out = enc(torch.randn(3, 20, 6))
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_mesh_encoder_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyEncoder(
                hidden=8,
                depth=1,
                latent_dim=4,
                shape_encoder="mesh",
            )

    def test_knn_offsets_pick_nearest(self) -> None:
        xyz = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float32)
        near = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32)
        far = torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]], dtype=torch.float32)
        env = torch.cat([far, near], dim=0).unsqueeze(0)
        rel = knn_offsets(xyz, env, k=2)
        self.assertEqual(tuple(rel.shape), (1, 2, 3))
        nbrs = rel + xyz.unsqueeze(1)
        self.assertTrue(torch.all(nbrs[0, :, 2] > 0.5))

    def test_knn_head_logits_and_grad(self) -> None:
        model = OccupancyEncoder(hidden=16, depth=2, latent_dim=8, knn_k=4)
        xyz = torch.randn(5, 3, requires_grad=True)
        envelope = torch.randn(5, 20, 6)
        shape_id = torch.zeros(5, dtype=torch.long)
        logits = model(xyz, envelope, shape_id)
        self.assertEqual(tuple(logits.shape), (5, 1))
        logits.sum().backward()
        self.assertIsNotNone(xyz.grad)
        self.assertTrue(xyz.grad.abs().sum().item() > 0.0)

    def test_knn_offsets_six_d_keep_neighbor_normals(self) -> None:
        xyz = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float32)
        near = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.1, 0.0, 1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        far = torch.tensor(
            [[0.0, 0.0, -1.0, 0.0, 1.0, 0.0], [0.1, 0.0, -1.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        env = torch.cat([far, near], dim=0).unsqueeze(0)
        rel = knn_offsets(xyz, env, k=2)
        self.assertEqual(tuple(rel.shape), (1, 2, 6))
        nbrs_xyz = rel[0, :, :3] + xyz
        self.assertTrue(torch.all(nbrs_xyz[:, 2] > 0.5))
        # The two near dots keep the normals stored on those rows.
        got = rel[0, :, 3:]
        expect = near[:, 3:]
        self.assertEqual(set(tuple(row.tolist()) for row in got),
                         set(tuple(row.tolist()) for row in expect))

    def test_envelope_dim_from_weight_shape(self) -> None:
        old = OccupancyEncoder(hidden=8, depth=1, latent_dim=4, envelope_dim=3)
        ckpt = {"state_dict": old.state_dict()}
        self.assertEqual(envelope_dim_from_ckpt(ckpt), 3)
        self.assertEqual(envelope_dim_from_ckpt({"envelope_dim": 6}), 6)

    def test_envelope_seed_from_ckpt_defaults_to_one(self) -> None:
        self.assertEqual(envelope_seed_from_ckpt({}), 1)
        self.assertEqual(envelope_seed_from_ckpt({"seed": 7}), 7)

    def test_load_old_xyz_checkpoint(self) -> None:
        old = OccupancyEncoder(hidden=8, depth=1, latent_dim=4, envelope_dim=3)
        ckpt = {
            "kind": CHECKPOINT_KIND,
            "state_dict": old.state_dict(),
            "hidden": 8,
            "depth": 1,
            "latent_dim": 4,
            "shape_encoder": "surface",
        }
        loaded = load_occupancy_model(ckpt, torch.device("cpu"))
        self.assertEqual(loaded.envelope_dim, 3)
        xyz = torch.randn(2, 3)
        env = torch.randn(2, 8, 3)
        sid = torch.zeros(2, dtype=torch.long)
        out = loaded(xyz, env, sid)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_knn_zero_has_no_local_module(self) -> None:
        model = OccupancyEncoder(hidden=8, depth=1, latent_dim=4, knn_k=0)
        self.assertFalse(hasattr(model, "local"))
        self.assertEqual(model.knn_k, 0)


if __name__ == "__main__":
    unittest.main()
