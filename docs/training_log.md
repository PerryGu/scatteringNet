# Training log

Write-up for each occupancy **train**. Newest entries at the top.

This file is **not** `CHANGELOG.md`. Changelog records code and knobs that shipped. Train results, conclusions, and what the run actually did live here.

`runs/<id>/` is the machine record (`config.yaml`, `metrics.jsonl`, `catalog.txt`). This file is the human note: the question, a few endpoints, the comparison, and what the numbers must not be used to claim. Do not paste full epoch tables here.

**Headings** use date and hour only (`## YYYY-MM-DD HH:00 — short name`; minutes always `:00`). That is the write-up time. The identifier is always the **run id** under Artifacts.

After every catalog train: append an entry. Point at `runs/<id>/` and `models/<id>/best.pt`.

**Names in older entries.** Before 2026-08-30 hygiene, the selection split was stored as `test_*` (`test_fraction`, `test_acc`, `n_test_files`). That is today's **val** split: scored every epoch to pick `best.pt`, not a locked holdout. Snapshot `split: shape` on those runs means whole **files**, not mesh identity.

| Run id | Name |
|---|---|
| `2026-08-30_22-59-57_extrude_nr1_surface` | Envelope + xyz, `hidden: 128` `depth: 6` (width+depth A/B) |
| `2026-08-30_20-31-00_extrude_nr1_surface` | Envelope + xyz, `hidden: 128` (width A/B) |
| `2026-08-30_18-40-06_extrude_nr1_surface` | Continuation of `15-50-05` (epochs 21–30) |
| `2026-08-30_15-50-05_extrude_nr1_surface` | Envelope + xyz, mesh-identity val |
| `2026-08-29_19-43-15_extrude_nr1_surface` | Envelope + xyz, file holdout |
| `2026-08-29_09-28-54_extrude_nr1` | Xyz-only, pooled point split |

---

## 2026-08-31 09:00 — extrude_nr1_surface hidden 128 depth 6 (width+depth A/B)

Fresh train. Same catalog, seed, mesh val, and envelope as `15-50-05` and `20-31-00`. Knobs: **`hidden: 128`** and **`depth: 6`** (baseline is 64 / 4). `latent_dim` omitted, so encoder `z` is 128. Goal: does extra width **and** depth beat the 20-epoch surface baseline on this holdout?

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py` (from scratch; not a resume) |
| Glob | YAML `exports/dataset/extrude_*_nr1_*.npz` |
| `run_name` | `extrude_nr1_surface` |
| Files | 5000 NPZs, 2500 unique OBJs (`max_files_per_shape: 2`) |
| Envelope | `n_surface=1024`, `shape_encoder=surface` |
| Split | **Mesh** identity (`n_train_meshes=2000` / `n_val_meshes=500`, seed 1) — same groups as `15-50-05` |
| Model | OccupancyEncoder `hidden=128` `depth=6` `latent_dim=128` |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Optimizer | Adam, `lr=0.001` |
| Batch | 1024 |
| Epochs | 20 |
| Seed | 1 |
| Selection | `checkpoint_metric: val_acc` |

**Artifacts**

- Run: `runs/2026-08-30_22-59-57_extrude_nr1_surface/`
- Weights: `models/2026-08-30_22-59-57_extrude_nr1_surface/best.pt`
- Wall: **2h 27m 57.54s** (`started=2026-08-30T22:57:53` → `finished=2026-08-31T01:25:50`)
- `best_epoch: 19`, `best_metric: 0.959084`

### Vs the other 20-epoch surface trains (same split)

Do **not** score this against `18-40-06` epoch 25. That run is 10 extra epochs on `h64/d4`, not a matched architecture A/B.

| | h64 d4 (`15-50-05`) | h128 d4 (`20-31-00`) | **h128 d6 (`22-59-57`)** |
|---|---|---|---|
| best val_acc | 0.9584 @ 20 | 0.9586 @ 18 | **0.9591 @ 19** |
| best val_iou / f1 | 0.756 / 0.861 | 0.757 / 0.861 | 0.754 / 0.860 |
| epoch-20 val_acc | 0.9584 | 0.9567 | 0.9577 |
| epoch-20 val_iou / f1 | 0.756 / 0.861 | 0.759 / 0.863 | **0.768 / 0.869** |
| epoch-20 loss | 0.098 | 0.092 | 0.092 |
| wall | 2h 14m | 2h 17m | 2h 28m |

Loss falls `0.244 → 0.092`. Train and val stay together. It learns interiors. At a matched 20-epoch budget it is **tied** with the small net — a 0.0007 acc bump, not a result. `best.pt` is epoch 19 (val_acc). Epoch 20 is slightly worse on acc but better on IoU (0.768). Selection follows `val_acc`, so 19 is the checkpoint.

**Not shown by this run**

- That more epochs on this recipe would or would not match `18-40-06` (this train stopped at 20)
- A new geometry signal (still the same envelope + extrude family)

**Bottom line:** extra width and depth did **not** move the 20-epoch score. Keep `15-50-05` as the 20-epoch surface baseline. `18-40-06` epoch 25 remains the longest-trained surface checkpoint we have, not proof that `h64/d4` is the better architecture. Do not treat `22-59-57` as the baseline.

---

## 2026-08-30 22:00 — extrude_nr1_surface hidden 128 (width A/B)

Fresh train. Same catalog, seed, mesh val, envelope, and `depth: 4` as `15-50-05`. Only **`hidden: 128`** (was 64). `latent_dim` omitted, so encoder `z` is also 128. Goal: does doubling width beat the narrow surface baseline on this holdout?

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py` (from scratch; not a resume) |
| Glob | YAML `exports/dataset/extrude_*_nr1_*.npz` |
| `run_name` | `extrude_nr1_surface` |
| Files | 5000 NPZs, 2500 unique OBJs (`max_files_per_shape: 2`) |
| Envelope | `n_surface=1024`, `shape_encoder=surface` |
| Split | **Mesh** identity (`n_train_meshes=2000` / `n_val_meshes=500`, seed 1) — same groups as `15-50-05` |
| Model | OccupancyEncoder `hidden=128` `depth=4` `latent_dim=128` |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Optimizer | Adam, `lr=0.001` |
| Batch | 1024 |
| Epochs | 20 |
| Seed | 1 |
| Selection | `checkpoint_metric: val_acc` |

**Artifacts**

- Run: `runs/2026-08-30_20-31-00_extrude_nr1_surface/`
- Weights: `models/2026-08-30_20-31-00_extrude_nr1_surface/best.pt`
- Wall: **2h 16m 57.15s** (`started=2026-08-30T20:30:29` → `finished=2026-08-30T22:47:26`)
- `best_epoch: 18`, `best_metric: 0.958587`

### Vs `hidden: 64` (same split)

| | h64 @ 20 (`15-50-05`) | **h128 @ 20 (`20-31-00`)** |
|---|---|---|
| best val_acc | 0.9584 @ 20 | 0.9586 @ 18 |
| best val_iou / f1 | 0.756 / 0.861 | 0.757 / 0.861 |
| epoch-20 val_acc | 0.9584 | 0.9567 |
| epoch-20 loss | 0.098 | 0.092 |
| wall | 2h 14m | 2h 17m |

Loss falls `0.246 → 0.092`. Train and val stay together. It learns interiors, but it does **not** beat the 64-wide net at the same epoch budget. Best is epoch 18; 19–20 are slightly worse. Do **not** score this against `18-40-06` epoch 25 — that is more epochs on `h64/d4`, not a width A/B.

**Not shown by this run**

- That `hidden: 256` would help (this A/B says width is not the bottleneck at 20 epochs)
- A depth change (`depth` stayed 4; see `22-59-57`)
- That more epochs on `h128` would or would not catch the `h64` continuation

**Bottom line:** at a matched 20-epoch budget, doubling width was **neutral** vs `15-50-05`. Keep `15-50-05` as the 20-epoch surface baseline. `18-40-06` epoch 25 remains the longest-trained surface checkpoint, not the architecture winner. Do not treat `20-31-00` as the baseline.

---

## 2026-08-30 20:00 — extrude_nr1_surface continuation (epochs 21–30)

**Continuation of** [`2026-08-30_15-50-05_extrude_nr1_surface`](#2026-08-30-1800--extrude_nr1_surface-mesh-identity-val-hygiene-replay). Same catalog, seed, mesh val split, and envelope head. Weights loaded from that run’s `best.pt` (epoch 20, val acc 0.9584). Not a new train from scratch. Goal: the parent run was still rising at epoch 20 — do 10 more epochs help?

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py --resume-run-id 2026-08-30_15-50-05_extrude_nr1_surface --epochs 10` |
| Parent | `models/2026-08-30_15-50-05_extrude_nr1_surface/best.pt` (`resume_epoch: 20`) |
| Split | Same mesh holdout as the parent (seed 1, 2000 / 500 meshes) |
| Extra epochs | 10 (printed 021–030). Snapshot `total: 30` |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Note | Parent `best.pt` had no Adam moments; optimizer restarted, weights did not |

**Artifacts**

- Run: `runs/2026-08-30_18-40-06_extrude_nr1_surface/`
- Weights: `models/2026-08-30_18-40-06_extrude_nr1_surface/best.pt`
- Wall: **1h 00m 37.62s** (`started=2026-08-30T18:39:35` → `finished=2026-08-30T19:40:12`)
- `best_epoch: 25`, `best_metric: 0.962418`

### Vs the parent (`15-50-05`)

| | Parent epoch 20 | This run best (25) | This run epoch 30 |
|---|---|---|---|
| val_acc | 0.9584 | **0.9624** | 0.9559 |
| val_iou / val_f1 | 0.756 / 0.861 | **0.775 / 0.873** | 0.762 / 0.865 |
| train_acc | 0.958 | 0.960 | 0.962 |
| loss | 0.098 | 0.093 | 0.089 |

Epoch 21 already beat the parent (`val_acc=0.9592`). Best is epoch **25**. After that, train acc and loss keep improving while val wobbles and finishes **below** 25. That is the start of overfitting, not another climb.

**Not shown by this run**

- A new architecture or split (this is only more epochs on the parent weights)
- That 20 more epochs would help; val already rolled over

**Bottom line:** the extra 10 epochs bought a **small** gain. Use this run’s `best.pt` (epoch 25), not epoch 30 and not the parent epoch-20 file, as the current surface checkpoint. Another resume on this recipe is not justified.

---

## 2026-08-30 18:00 — extrude_nr1_surface (mesh-identity val, hygiene replay)

Same catalog, seed, and envelope head as `2026-08-29_19-43-15_extrude_nr1_surface`. This run is the first full catalog train **after hygiene**: unique-OBJ val, shared per-mesh AABB, point micro-average metrics, `val_*` names. Goal: does the envelope still learn interiors when the same OBJ cannot leak across the split?

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py` |
| Glob | YAML `exports/dataset/extrude_*_nr1_*.npz` |
| `run_name` | `extrude_nr1_surface` |
| Files | 5000 NPZs, 2500 unique OBJs (`max_files_per_shape: 2`) |
| Envelope | `n_surface=1024`, `shape_encoder=surface` (`latent_dim` omitted → 64) |
| Points | `N=11522442` (`n_train=9195763` / `n_val=2326679`) |
| Split | **Mesh** identity (`n_train_meshes=2000` / `n_val_meshes=500`, `val_fraction=0.20`). Both NPZs of one OBJ stay on one side. |
| Model | OccupancyEncoder `hidden=64` `depth=4` `latent_dim=64` |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Optimizer | Adam, `lr=0.001` |
| Batch | 1024 |
| Epochs | 20 |
| Seed | 1 |
| Selection | `checkpoint_metric: val_acc` |

**Artifacts**

- Run: `runs/2026-08-30_15-50-05_extrude_nr1_surface/` (`config.yaml`, `catalog.txt` 5000 lines, `metrics.jsonl`)
- Weights: `models/2026-08-30_15-50-05_extrude_nr1_surface/best.pt`
- Wall: **2h 13m 30.70s** (`started=2026-08-30T15:49:36` → `finished=2026-08-30T18:03:06`)
- `best_epoch: 20`, `best_metric: 0.958357`

### Vs prior catalog trains (same glob, seed, 5000 files)

| | xyz-only (`09-28-54`) | surface file split (`19-43-15`) | surface mesh val (`15-50-05`) |
|---|---|---|---|
| Split | random **points** | whole **files** | whole **meshes** |
| Epoch-1 loss | 0.364 | 0.241 | 0.240 |
| Epoch-20 loss | 0.352 | 0.078 | 0.098 |
| Holdout acc | val 0.853 (flat) | 0.967 (best 0.970 @ 18) | **0.958 @ 20** |
| Inside IoU / F1 | not logged | best-epoch 0.691 / 0.759 | **best-epoch 0.756 / 0.861** |
| Wall | 1h 21m | 1h 59m | 2h 14m |

Loss falls `0.240 → 0.098`. Train and val stay together (`0.958` / `0.958` at epoch 20). That is still occupancy learning, not the 85% outside prior. Val acc is **a bit below** the old 0.970: expected, because this holdout is harder (unseen OBJs, not unseen files of seen OBJs). Do not read the IoU/F1 rise as a pure model win — hygiene also switched eval to a **point micro-average**, so those scalars are not the same recipe as `19-43-15`.

`best.pt` is epoch **20**. The curve was still climbing (epoch-19 acc 0.953 → 0.958). Twenty epochs may be short on this split.

**Not shown by this run**

- A locked test touched only once (val is scored every epoch to pick `best.pt`)
- A new primitive family (val meshes are still level-1 extrudes)
- That more epochs would not help; the last epoch was the best

**Bottom line:** the envelope path **survives** a real mesh holdout. Use `15-50-05` as the 20-epoch surface baseline, not `19-43-15`. **Continued** in `18-40-06` (epochs 21–30); that later `best.pt` (epoch 25) is the current surface checkpoint.

---

## 2026-08-29 21:00 — extrude_nr1_surface (level-1 extrudes, envelope + xyz)

Same catalog as the xyz-only `extrude_nr1` run (`exports/dataset/extrude_*_nr1_*.npz`, seed 1, 5000 files). Each file keeps its NPZ inside/outside labels; the OBJ envelope is encoded (`n_surface=1024`). Holdout is **whole NPZ files** (80/20 of the file list), not unique OBJs. Goal: does the shell plus a file holdout actually learn interiors, or stay on the old **~85% majority-class** ceiling from the xyz-only run?

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py` |
| Glob | YAML `exports/dataset/extrude_*_nr1_*.npz` |
| `run_name` | `extrude_nr1_surface` |
| Files | 5000 NPZs, 2500 unique OBJs (`max_files_per_shape: 2`) |
| Envelope | `n_surface=1024`, `shape_encoder=surface` |
| Points | `N=11522442` (`n_train=9211056` / `n_test=2311386`) |
| Split | Whole **files** (`n_train_files=4000` / `n_test_files=1000`, `test_fraction=0.20`). Not a mesh holdout: lattice and jitter of the same OBJ can sit on both sides. |
| Model | OccupancyEncoder `hidden=64` `depth=4` `latent_dim=64` |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Optimizer | Adam, `lr=0.001` |
| Batch | 1024 |
| Epochs | 20 |
| Seed | 1 |
| Selection | `checkpoint_metric: test_acc` (today: `val_acc`; this split selects `best.pt`) |

**Artifacts**

- Run: `runs/2026-08-29_19-43-15_extrude_nr1_surface/`
- Weights: `models/2026-08-29_19-43-15_extrude_nr1_surface/best.pt`
- Wall: **1h 59m 31.40s** (`started=2026-08-29T19:41:07` → `finished=2026-08-29T21:40:39`)
- `best_epoch: 18`, `best_metric: 0.969705`

### Vs xyz-only extrude_nr1 (same glob, seed, 5000 files)

| | xyz-only (`09-28-54`) | surface (`19-43-15`) |
|---|---|---|
| Split | random **points** of one pooled cloud | whole **files** (4000 / 1000), not meshes |
| Epoch-1 loss | 0.364 | 0.241 |
| Epoch-20 loss | 0.352 | 0.078 |
| Holdout acc | val 0.853 (flat; ~majority outside) | selection-split 0.967 (best 0.970 @ 18) |
| Inside IoU / F1 | not logged | best-epoch IoU 0.691 / F1 0.759 |
| Wall | 1h 21m | 1h 59m |

The xyz-only run sat on “guess outside” (~85% acc, loss almost flat). This run trains: loss keeps falling, holdout acc and inside IoU climb together, train ≈ holdout. That is the first extrude catalog result in this repo that looks like occupancy learning rather than a class prior. Accuracy 0.97 on an outside-heavy cloud still overstates interior quality; IoU **0.69** / F1 **0.76** is the interior score.

**Not shown by this run**

- A locked test touched only once (`test_*` here is today's val: scored every epoch to pick `best.pt`)
- A mesh holdout (file split can leak the same OBJ onto both sides)
- A new primitive family (holdout files are still level-1 extrudes)
- That the envelope **alone** caused the jump (split and head both changed vs xyz-only)

**Bottom line:** pairing NPZ occupancy labels with the OBJ envelope works on this catalog, and it **helped a lot**. Keep the envelope path. Face tokens (Phase 2 Step 9–10) are the next geometry test if we continue the ladder; another xyz-only replay is not needed. A later train under the mesh-identity val split is what would show whether this still holds on unseen OBJs.

---

## 2026-08-29 10:00 — extrude_nr1 (first-level cubes, xyz-only)

Post-cleanup replay of the first-level extrude catalog. Same glob and seed as the earlier `2026-08-27_20-11-07_extrude_nr1` run. Goal: confirm the loop, logs, GPU timer, and `best.pt` still work after removing single-file train/infer and the leftover YAML knobs.

### What ran

| Knob | Value |
|---|---|
| Script | `src/train_multi_npz.py` |
| Glob | `exports/dataset/extrude_*_nr1_*.npz` (override; YAML glob still spheres) |
| `run_name` | `extrude_nr1` |
| Files | 5000 NPZs (`max_files_per_shape: 2` → ~2500 meshes × 2 files) |
| Points | `N=11522442` (`n_train=9217954` / `n_val=2304488`, `val_fraction=0.20`) |
| Split | Random **point** split of the pooled cloud (not held-out files or meshes) |
| Model | OccupancyMLP `hidden=64` `depth=4` (xyz only) |
| Device | `cuda` / NVIDIA GeForce GTX 1080 |
| Optimizer | Adam, `lr=0.001` |
| Batch | 1024 |
| Epochs | 20 |
| Seed | 1 |
| Selection | `checkpoint_metric: val_acc` (strict improve → `best.pt`) |

**Artifacts**

- Run: `runs/2026-08-29_09-28-54_extrude_nr1/`
- Weights: `models/2026-08-29_09-28-54_extrude_nr1/best.pt`
- Wall: **1h 21m 05.86s** (`started=2026-08-29T09:27:32` → `finished=2026-08-29T10:48:38`)
- `best_epoch: 20`, `best_metric: 0.853177`

### Conclusions

The cleanup did not break training. Catalog load, CUDA, run snapshot (`gpu`, `started_at`, `finished_at`, `wall`), and `best.pt` all behaved.

Accuracy is already **~85.2% at epoch 1** and only moves to **85.3%** by epoch 20. Loss creeps `0.364 → 0.352`. Train and val stay locked together.

That is not “the net learned interiors.” It is an xyz-only MLP on a **random point split of the same pooled cloud**. Val is not held-out meshes, so train ≈ val is expected. The 85% level is almost certainly the **majority class** (mostly outside). The model is barely moving off that prior.

Epoch-1 loss `0.364422` matches the previous extrude_nr1 train (same seed and data). That is a reproducibility check, not a new model.

**Not shown by this run**

- Generalization to a new cube (would need a mesh holdout)
- That more epochs would suddenly jump; another 20 would likely shave another tiny bit of loss
- A regression versus the last extrude train — same ceiling, same story

**Bottom line:** the loop works. The occupancy head is still too weak / too class-imbalanced / too xyz-only to beat “guess outside.” Geometry (mesh join, envelope, faces) is the next thing that could change that, not another identical 20-epoch xyz train.
