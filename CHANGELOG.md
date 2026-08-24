# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top.  



## 2026-08-24 13:00 — Step 9: [Train/val split]

Hold out a random 15% of *points* from the same NPZ (not a new mesh) and report val accuracy.

- `config.yaml` / `OccupancyConfig`: `val_fraction: 0.15`.
- `src/dataset.py`: `split_train_val_indices`; `make_dataloader` accepts a `Subset`.
- `src/train_one_npz.py`: train on the complement, eval val each epoch (`val_acc`), AABB `center`/`scale` still from the full cloud so Step 8 inference stays consistent. `TrainRunResult` now includes `val_accuracies`, `n_train`, `n_val`.
- Tests updated (`test_config`, `test_dataset`, `test_train_one_npz`, `test_infer_one_npz`). Full suite: 28 tests OK. Synthetic sphere: `n_train=218` `n_val=38`, final `train_acc=0.9633` `val_acc=0.8684`.

## 2026-08-24 10:00 — Step 8: [Basic inference]

Classify one occupancy NPZ with a saved checkpoint using the stored AABB map.

- Added `src/infer_one_npz.py`: load `kind=occupancy_mlp` checkpoint, rebuild `OccupancyMLP` from stored `hidden`/`depth`, normalize XYZ with checkpoint `center`/`scale` (not a fresh AABB), `eval` + `no_grad`, sigmoid threshold `0.5` via `occupancy_metrics`.
- Prints `pred_inside` / `pred_outside` counts and accuracy vs NPZ labels. Writes `pred_labels` to `{checkpoint_stem}_pred.npz` next to the checkpoint. Optional NPZ path argument; default is YAML `sample_npz`.
- Tests: `tests/test_infer_one_npz.py` (4 tests OK). Full suite: 27 tests OK.
- Smoke: `python src/infer_one_npz.py` on the train sphere (`N=10661`, CUDA): `pred_inside=5112` `pred_outside=5549` `accuracy=0.9212`. Exit met: overfit accuracy is high.

## 2026-08-23 23:00 — Env: recreate `scatteringNet`

Replaced the broken mixed conda/pip PyTorch install with a single-source CUDA 12.1 stack named after the project.

- Rewrote `environment.yaml`: conda-only `pytorch=2.5.1`, `pytorch-cuda=12.1`, `torchvision`, `torchaudio` (no pip torch wheels).
- Recreated `conda` env `scatteringNet`. Verified `torch 2.5.1`, CUDA 12.1, `torch.cuda.is_available() == True`.
- Full suite in the new env: 23 tests OK (including CUDA).
- `.cursorrules` and `docs/v2_minimal_plan.md` now name `scatteringNet` (not `scatteringNet_v2`) as the project environment.

## 2026-08-23 21:00 — Config: train knobs in YAML

Moved one-NPZ training defaults out of the train script so experiment knobs live in one place.

- `config.yaml` now holds `epochs`, `lr`, `checkpoint_path` (repo-relative), and `sample_npz` (relative to `data_dir`).
- `src/config.py`: `OccupancyConfig` / `load_yaml_knobs` load those fields; relative checkpoints resolve against the repo root; added `sample_npz_path()`.
- `src/train_one_npz.py` reads epochs / lr / checkpoint from `cfg` (no module-level defaults). CLI uses `sample_npz_path(cfg)`. `CHECKPOINT_KIND` stays in code (schema tag, not a train knob).
- Tests updated (`tests/test_config.py`, `tests/test_train_one_npz.py`). Full suite: 23 tests OK.

## 2026-08-23 20:00 — Step 7: [Metrics helper]

Centralized occupancy decision metrics so the train loop no longer computes accuracy inline.

- Added `src/metrics.py`: `occupancy_metrics` / `accuracy_from_logits` from logits vs `{0, 1}` labels (sigmoid + threshold `0.5`). Returns accuracy plus inside-class precision / recall (zero-denominator → `0.0`; no extra deps).
- `src/train_one_npz.py` now prints `train_acc` / `inside_prec` / `inside_rec` via `occupancy_metrics`; removed the private `_batch_accuracy` helper. Checkpoint format and `TrainRunResult` are unchanged.
- Tests: `tests/test_metrics.py` (6 tests OK, including CUDA). Existing `tests/test_train_one_npz.py` still passes (2 tests OK).
- Smoke: `python src/metrics.py` prints `accuracy=1.0000`, `inside_precision=1.0000`, `inside_recall=1.0000` on a perfect 8-point batch.

## 2026-08-21 19:00 — Step 6: [Train one NPZ / overfit]

Overfit `OccupancyMLP` on one occupancy NPZ (all points used as train; no val split).

- Added `src/train_one_npz.py`: seed, `OccupancyMLP`, Adam (`lr=1e-3`), `BCEWithLogitsLoss`, 30 epochs, prints mean `loss` and train accuracy each epoch.
- Saves `models/one_npz.pt` with `kind`, `state_dict`, AABB `center`/`scale`, `hidden`, `depth`.
- Composes Steps 1–5 (`load_config`, `OccupancyPointDataset`, `make_dataloader`, `OccupancyMLP`); those modules were not refactored.
- Tests: `tests/test_train_one_npz.py` (2 tests OK on CPU synthetic sphere).
- Smoke: `python src/train_one_npz.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, CUDA): epoch 1 `loss=0.693` / `acc=0.502` → epoch 30 `loss=0.329` / `acc=0.889` (peak acc `0.914` at epoch 29). Exit met: loss decreased, train acc ≫ 50%.

## 2026-08-21 17:00 — Step 5: [Dataset + DataLoader]

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