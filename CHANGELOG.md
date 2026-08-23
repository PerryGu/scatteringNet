# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top.

---

## 2026-08-21 — Step 2: Hybrid configuration

Implemented a split between static experiment knobs and runtime resolution.

- Added `config.yaml` with `hidden: 64`, `depth: 4`, `seed: 1`.
- Updated `src/config.py` to load those YAML knobs, read `DATA_DIR` from the environment / repo `.env` (fill-if-missing, no `python-dotenv`), and set `device` via CUDA detection.
- `data_dir` and `device` are not stored in YAML (machine-specific).
- Verified with `python src/config.py` and `tests/test_config.py`: `data_dir=E:\Work_stuff\scatteringNet\data`, `device=cuda`.

## 2026-08-21 — Step 1: OccupancyMLP

Created the xyz-only occupancy module and smoke-tested it.

- Added `src/occupancy_mlp.py`: `OccupancyMLP(nn.Module)`, input `(B, 3)` → logits `(B, 1)`, defaults `hidden=64`, `depth=4`.
- Added `tests/test_occupancy_mlp.py` (shape, finite values, invalid inputs, CUDA). All four tests passed.
- `src/scattering_net.py` left untouched. No NPZ loading or training.
