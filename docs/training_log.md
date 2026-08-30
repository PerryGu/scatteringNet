# Training log

Write-up for each occupancy **train**. Newest entries at the top.

This file is **not** `CHANGELOG.md`. Changelog records code and knobs that shipped. Train results, conclusions, and what the run actually did live here.

`runs/<id>/` is the machine record (`config.yaml`, `metrics.jsonl`, `catalog.txt`). This file is the human note: the question, a few endpoints, the comparison, and what the numbers must not be used to claim. Do not paste full epoch tables here. Do not copy training numbers into `CHANGELOG.md`.

**Headings** use date and hour only (`## YYYY-MM-DD HH:00 — short name`; minutes always `:00`). That is the write-up time. The identifier is always the **run id** under Artifacts.

After every catalog train: append an entry. Point at `runs/<id>/` and `models/<id>/best.pt`.

**Names in older entries.** Before 2026-08-30 hygiene, the selection split was stored as `test_*` (`test_fraction`, `test_acc`, `n_test_files`). That is today's **val** split: scored every epoch to pick `best.pt`, not a locked holdout. Snapshot `split: shape` on those runs means whole **files**, not mesh identity.

| Run id | Name |
|---|---|
| `2026-08-29_19-43-15_extrude_nr1_surface` | Envelope + xyz, file holdout |
| `2026-08-29_09-28-54_extrude_nr1` | Xyz-only, pooled point split |

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
