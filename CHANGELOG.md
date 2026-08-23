# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top.  



## 2026-08-21 18:30 — Step 6: [Train one NPZ / overfit]

Overfit `OccupancyMLP` on one occupancy NPZ (all points used as train; no val split).

- Added `src/train_one_npz.py`: seed, `OccupancyMLP`, Adam (`lr=1e-3`), `BCEWithLogitsLoss`, 30 epochs, prints mean `loss` and train accuracy each epoch.
- Saves `models/one_npz.pt` with `kind`, `state_dict`, AABB `center`/`scale`, `hidden`, `depth`.
- Composes Steps 1–5 (`load_config`, `OccupancyPointDataset`, `make_dataloader`, `OccupancyMLP`); those modules were not refactored.
- Tests: `tests/test_train_one_npz.py` (2 tests OK on CPU synthetic sphere).
- Smoke: `python src/train_one_npz.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, CUDA): epoch 1 `loss=0.693` / `acc=0.502` → epoch 30 `loss=0.329` / `acc=0.889` (peak acc `0.914` at epoch 29). Exit met: loss decreased, train acc ≫ 50%.

## 2026-08-21 17:58 — Step 5: [Dataset + DataLoader]

Wrapped one occupancy NPZ as a PyTorch `Dataset` with AABB normalization applied at construction.

- Added `src/dataset.py`: `OccupancyPointDataset`, `make_dataloader` (default `batch_size=1024`, `shuffle=True`, `num_workers=0`).
- Construction uses `load_points_labels` (`data_npz.py`) then `compute_center_scale` / `apply_normalization` (`normalize.py`); stores `center`, `scale`, and `npz_path` on the dataset.
- `__getitem__` returns float32 tensors: `xyz` shape `(3,)`, `y` shape `(1,)` (label in `{0, 1}`).
- Default collate yields `xyz (B, 3)` and `y (B, 1)` to match `OccupancyMLP` logits for BCE-with-logits.
- Verified: `tests/test_dataset.py` (2 tests OK); `python src/dataset.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, first batch `xyz.shape=(1024, 3)`, `y.shape=(1024, 1)`, `dtype=float32`).
- Previous step modules were not refactored.

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
