# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top.

---

## 2026-08-21 — Step 4: Coordinate normalize

AABB-normalize query XYZ into roughly ``[-1, 1]`` without changing the model.

- Added `src/normalize.py`: `compute_center_scale`, `apply_normalization`.
- Center = AABB midpoint; scale = max half-extent (must be > 0).
- `tests/test_normalize.py`: cube maps to ±1, inverse recovers the first rows, zero-extent rejected.
- Smoke: `python src/normalize.py` on the dataset_test sphere NPZ (normed range ≈ `[-1, 1]`, inverse check passed).

## 2026-08-21 — Config cleanup: `data_dir` in YAML

Dataset path no longer comes from `.env`.

- Added `data_dir` to `config.yaml` (`E:/Work_stuff/scatteringNet/data`).
- `src/config.py` reads `data_dir` from YAML; removed `.env` / `DATA_DIR` environment loading.
- If the folder is missing, `load_config()` prints a short message to set `data_dir` in `config.yaml`, then raises `FileNotFoundError`.
- `device` is still detected at runtime (CUDA vs CPU).
- Updated `tests/test_config.py` (existing path + missing-path case).

## 2026-08-21 — Step 3: NPZ loader

Added a loader that reads only occupancy queries from scatter NPZs.

- Added `src/data_npz.py`: `load_points_labels(path) -> (points, labels)`.
- Uses keys `points` and `labels` only; ignores `mesh_path`, `tags`, and other fields.
- Validates `points` is `(N, 3)` and `labels` is `(N,)`.
- Returns `float32` XYZ and `float32` labels in `{0.0, 1.0}` (conversion happens in the loader).
- Smoke: `python src/data_npz.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, inside fraction ≈ 0.497); `tests/test_data_npz.py` covers dtypes and shape checks.

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
