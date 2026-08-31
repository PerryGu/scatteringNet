# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top. Headings: ``## YYYY-MM-DD HH:00`` (date and hour; no minutes).


## 2026-08-31 20:00 — Viewer: prefs, fill UX, README

Checkboxes, sliders, density, and selected model persist in ``src/viewer/ui_prefs.json``. Fill does not move the camera; density drag refills; Fill checks **Inside** / **Outside**. README matches Job A/B (Gradio skipped). How to run: [`src/viewer/README.md`](src/viewer/README.md). Occupancy train code unchanged.

## 2026-08-31 20:00 — Viewer Step 7: fill OBJ then infer

**Fill points** + density (0 = spacing 0.40, 100 = 0.05, default ≈ 0.15). Unlabeled AABB lattice (no raycast GT). **Run model** classifies those points; envelope from the uploaded OBJ. No Errors view. Occupancy train code unchanged.

## 2026-08-31 19:00 — Viewer Step 6: model on NPZ

``GET /api/models`` lists ``models/<run>/best.pt``. **Run model** classifies NPZ points (envelope if surface-conditioned). **Truth** / **Prediction** / **Errors**. **Select Model**; amber **Run model**. ``open_viewer.bat`` uses conda ``scatteringNet``. Occupancy train code unchanged.

## 2026-08-31 19:00 — Viewer Step 5: inspect UI

Mesh / Inside / Outside / Wireframe (2×2), opacity, point size, **Max points drawn** (display subsample). NPZ **Total points** and counts next to swatches. Equal-width 280px rounded left panels. Occupancy train code unchanged.

## 2026-08-31 18:00 — Viewer: auto-load mesh from NPZ

Opening an NPZ fetches the linked OBJ (``Mesh`` is visibility). ``serve.py`` reads quoted ``config.yaml`` ``data_dir`` and ignores the inline comment. Occupancy train code unchanged.

## 2026-08-31 18:00 — Viewer Step 4: Show object

``GET /api/mesh`` loads the NPZ ``mesh_path`` OBJ from ``data_dir`` (path must stay under ``data_dir``). **Clear view** stays in the recents menu. Occupancy train code unchanged.

## 2026-08-31 16:00 — Viewer: recents and Clear view

Hover Open for last 5 files. A recents click loads the cached file (does not open Explorer). IndexedDB + Chrome file handles. Occupancy train code unchanged.

## 2026-08-31 15:00 — Viewer Step 3: NPZ points

Open/drop an occupancy NPZ shows file-label points (inside orange, outside steel), counts, and stored ``mesh_path``. Occupancy train code unchanged.

## 2026-08-31 15:00 — Viewer Step 2: OBJ load

Open, drop, and recent-click load a Wavefront OBJ (local ``OBJLoader``, camera frames the AABB). Occupancy train code unchanged.

## 2026-08-31 13:00 — Viewer Step 1: page, helper, and plans

``src/viewer/``: dark Three.js page (orbit, grid, Open, recents stub). ``serve.py`` / ``open_viewer.bat`` (local Three.js, free port, ``.js`` as ``text/javascript``, ``no-store``). Plans: Job A = NPZ points; Job B = OBJ fill + envelope; Gradio optional. Occupancy train code unchanged.

## 2026-08-30 18:00 — Resume catalog train from best.pt

``train_multi_npz`` accepts ``--resume-run-id`` / ``--resume``. It loads that ``best.pt``, keeps the stored selection score, and runs YAML ``epochs`` more (printed as 21…). New ``runs/`` + ``models/``; Adam state is restored only if the checkpoint stored it.

## 2026-08-30 15:00 — Collate: pin_memory on expanded envelopes

``occupancy_collate`` materializes the same-``shape_id`` envelope (``contiguous``) so CUDA ``pin_memory`` can pin it. Catalog surface train was crashing on the first batch.

## 2026-08-30 14:00 — Work plans: current rung is Step 9 or stop

Updated Phase 2 headers: Steps 1–8 done; next is face tokens or stop. Hygiene mesh-identity val is not Step 11. Phase 1 plans now point train write-ups at ``training_log.md``.

## 2026-08-30 14:00 — Training log: split and val wording

Clarified ``docs/training_log.md`` only: headings vs run id, historical ``test_*`` = today's val, file split is not a mesh holdout. Numbers unchanged.

## 2026-08-30 14:00 — Hygiene 4–14: val names, infer, shared helpers, encoder extract

Selection split is named **val** (``val_fraction`` / ``val_acc``; legacy ``test_*`` still accepted). ``src/infer_multi_npz.py`` reloads ``best.pt``. ``build_mlp``, shared trimesh/path helpers, envelope collate, point micro-average metrics, cache clear, YAML ``latent_dim``, ``random.seed`` + ``pin_memory``, and ``src/encoder_dataset.py`` land here. Conservative ``pyproject.toml`` keeps flat ``src`` imports.

## 2026-08-30 13:00 — Hygiene 3: shared AABB per mesh

Catalog parts that share a mesh reuse one ``center`` / ``scale`` (OBJ vertices when joined, else the union of that key's query points). Lattice and jitter no longer live in different frames.

## 2026-08-30 13:00 — Hygiene 2: shape_id from mesh identity

Catalog ``shape_id`` is a stable integer per ``mesh_split_key`` (sorted unique OBJs), not the file index. Two NPZs of one mesh share an id so encode-once is per OBJ.

## 2026-08-30 13:00 — Hygiene 1: mesh-identity train/test split

``test_fraction`` now holds out unique OBJs, not NPZ files. ``mesh_split_key`` / ``split_train_test_by_mesh`` keep every file of one mesh on the same side. Snapshot uses ``split: mesh`` plus ``n_train_meshes`` / ``n_test_meshes``.

## 2026-08-29 18:00 — Train/test 80/20 file split

Catalog train holds out whole files as a **test** set (``test_fraction: 0.20``). Terminal and snapshot use ``test_acc`` / ``n_test_files``.

## 2026-08-29 13:00 — Train/val split is by whole files

Catalog train no longer concatenates every NPZ into one point cloud. Each file stays its own shape: mini-batches come from one file, and ``val_fraction`` holds out entire files. Snapshot stamps ``split: shape``, ``n_train_files``, and ``n_val_files``.

## 2026-08-29 12:00 — Step 8: surface envelope occupancy

Added ``n_surface`` and ``shape_encoder`` to ``config.yaml``. ``src/geometry/surface.py`` samples an area-weighted shell cloud; ``OccupancyEncoder`` encodes each ``shape_id`` once per batch and concatenates ``z_surf`` with query xyz. Metrics now include inside IoU / F1. OccupancyMLP is unchanged (``shape_encoder: none``).

## 2026-08-29 12:00 — Step 7: NPZ ↔ OBJ mesh join

Added ``load_points_labels_mesh`` (``load_points_labels`` unchanged) and ``src/geometry/mesh_io.py`` to resolve ``mesh_path`` against ``data_dir`` and load OBJ ``vertices (V, 3)`` / ``faces (T, 3)``. Dataset parts store ``mesh_path``, ``mesh_key``, and triangles. Catalog train prints the join and stamps ``n_meshes``. OccupancyMLP is still xyz-only.

## 2026-08-29 11:00 — Run snapshot records file and point counts

``runs/<id>/config.yaml`` now stamps ``n_files``, ``n_points``, ``n_train``, and ``n_val`` after the catalog is loaded. ``gpu`` (card name) was already in the snapshot next to ``device``.

## 2026-08-29 09:00 — Single-file infer and leftover YAML knobs removed

Deleted ``src/infer_one_npz.py`` and ``tests/test_infer_one_npz.py``. Removed ``checkpoint_path`` and ``sample_npz`` from ``config.yaml`` / ``OccupancyConfig`` / run snapshots. Catalog train writes ``models/<run_id>/best.pt``. ``OccupancyPointDataset`` stays as the per-file reader inside the catalog.

## 2026-08-28 21:00 — Single-file train removed; one ``epochs`` knob

Deleted ``src/train_one_npz.py`` and ``tests/test_train_one_npz.py``. Catalog train is the only train path (``train_multi_npz.py``). Removed ``smoke_epochs``; YAML ``epochs`` is the train length. Infer still exists (``infer_one_npz.py``) and reads ``models/<run_id>/best.pt`` AABB from ``parts``. OccupancyMLP unchanged.

## 2026-08-28 21:00 — YAML ``batch_size`` and ``optimizer``

``config.yaml`` now owns mini-batch size (default 1024) and optimizer family (``adam`` / ``adamw`` / ``sgd``). ``train_multi_npz`` uses both; the run snapshot records them.

## 2026-08-28 19:00 — Run snapshot records GPU name

``runs/<id>/config.yaml`` now stamps ``gpu`` (CUDA card name from PyTorch, or null on CPU) next to ``device``. ``train_multi_npz`` prints the same name. Root ``config.yaml`` is unchanged (runtime-only, like ``device``).

## 2026-08-28 17:00 — Whole-run wall timer

Each ``runs/<id>/`` stamps ``started_at``, ``finished_at``, ``wall_seconds``, and  ``wall`` in ``config.yaml``. ``train_multi_npz`` prints the same duration at the end of the workout. Per-epoch ``wall_seconds`` in ``metrics.jsonl`` is unchanged.


## 2026-08-28 08:00 — Run catalog list moved out of ``config.yaml``

Resolved NPZ paths no longer live in the run snapshot. ``runs/<id>/catalog.txt`` is one data-relative path per line. ``config.yaml`` only records ``catalog_file`` and ``catalog_n``. Empty YAML ``npz_paths`` is omitted from the snapshot (glob was used). Existing sphere and extrude run folders were migrated.

## 2026-08-27 20:00 — Run snapshots store relative paths

``runs/<id>/config.yaml`` and multi-NPZ checkpoint payloads no longer dump absolute ``E:/`` / ``F:/`` paths.

- ``catalog_npz_paths``, ``sample_npz``, ``npz_paths`` → relative to ``data_dir`` (e.g. ``exports/dataset/...``).
- ``checkpoint_path``, ``run_dir`` → relative to the git repo (e.g. ``models/one_npz.pt``).
- ``data_dir`` stays absolute when the dataset disk is not inside the repo (this machine: ``E:`` data vs ``F:`` git).
- ``best.pt`` / ``last.pt`` ``npz`` keys use the same data-relative strings.
- Tests: ``test_as_data_and_repo_relative``, ``test_snapshot_strips_absolute_data_and_repo_paths``. Phase 1 train/infer unchanged.

## 2026-08-27 19:00 — Step 6: [First multi-NPZ occupancy train]

``src/train_multi_npz.py`` trains Phase 1 ``OccupancyMLP`` on the catalog glob. Logs → ``runs/<id>/``; weights → ``models/<id>/last.pt`` + ``best.pt``. No OccupancyMLP edits. ``train_one_npz.py`` unchanged.

- Val: random point split of the pooled queries (loop health).
- YAML additive ``smoke_epochs: 20`` (this CLI). Phase 1 still uses ``epochs``.
- Checkpoint stores ``kind=occupancy_mlp``, NPZ paths, per-mesh AABB.
- Tests: ``tests/test_train_multi_npz.py`` 1 OK (two synthetic NPZs, 2 CPU epochs, ``n=100``, ``best.pt`` + ``metrics.jsonl``, no ``.pt`` under ``runs/``).
- First real smoke: run ``python src/train_multi_npz.py`` on the current glob (2 sphere NPZs). Record ``runs/<id>`` / ``models/<id>`` and metrics after that run.

## 2026-08-27 20:00 — Step 5: [Checkpoints ``last.pt`` + ``best.pt``]

Checkpoint API under ``models/<run_id>/``. No occupancy train. Phase 1 ``train_one_npz.py`` / ``OccupancyMLP`` / ``models/one_npz.pt`` unchanged.

- ``src/checkpointing.py``: always write ``last.pt``; write ``best.pt`` only when the metric strictly improves (apex). Pointer ``runs/<id>/checkpoint_dir.txt``.
- Run snapshot ``runs/<id>/config.yaml`` now stamps ``total`` (planned epochs) and ``checkpoint`` (epoch of the last ``last.pt`` write), plus ``best_epoch`` / ``best_metric``.
- YAML additive ``checkpoint_metric: val_acc``. ``total`` / ``checkpoint`` are runtime snapshot fields, not project knobs.
- Tests: ``tests/test_checkpointing.py`` 1 OK (3 dummy epochs; ``best.pt`` keeps epoch 2 when epoch 3 is worse; no ``.pt`` under ``runs/``).
- Smoke: ``python src/checkpointing.py`` → ``runs/2026-08-27_18-45-58_ckpt_dummy``, ``models/2026-08-27_18-45-58_ckpt_dummy``; ``last_epoch=3`` ``best_epoch=2``; snapshot ``total=3`` ``checkpoint=3``; ``runs_has_pt=False``.

## 2026-08-27 19:00 — Step 4: [Run logs under ``runs/``]

Standalone log API. No occupancy train, no ``.pt``, Phase 1 ``train_one_npz.py`` / ``OccupancyMLP`` unchanged.

- ``src/run_tracking.py``: ``runs/<YYYY-MM-DD_HH-MM-SS>_<name>/`` with config snapshot, ``metrics.jsonl`` (one JSON object per epoch), TensorBoard scalars under ``train/`` and ``val/``.
- ``config.yaml`` additive ``run_name`` (optional in the loader). Device stays runtime-only.
- Gitignore TensorBoard event files only; JSON + YAML under ``runs/`` remain commitable.
- ``environment.yaml``: ``tensorboard>=2.14`` for ``SummaryWriter``.
- Tests: ``tests/test_run_tracking.py`` 2 OK (fake 3-epoch CPU loop; no ``.pt`` under ``runs/``). Phase 1 tests still green.
- Smoke: ``python src/run_tracking.py`` → ``runs/2026-08-27_17-06-04_dummy``; ``metrics.jsonl`` 3 rows; ``has_pt=False``; ``has_tfevents=True``.

## 2026-08-27 18:00 — Step 3: [Multi-NPZ dataset reader]

Catalog loader for many occupancy NPZs. Phase 1 ``load_points_labels`` / ``OccupancyPointDataset`` / ``train_one_npz.py`` unchanged. No occupancy train.

- ``resolve_npz_catalog`` in ``data_npz.py``: glob or explicit list under ``data_dir``, skip ``combo*``, cap ``max_files_per_shape``. Per-mesh key is the stem before ``__``.
- ``OccupancyMultiNpzDataset``: concatenates one ``OccupancyPointDataset`` per file (per-mesh AABB). DataLoader still yields ``xyz (B, 3)``, ``y (B, 1)``.
- ``config.yaml`` additive: ``npz_glob``, optional ``npz_paths``, ``max_files_per_shape``.
- Tests: ``tests/test_multi_npz.py`` 5 OK. Phase 1 loader/train/infer tests still green.
- Smoke: ``python src/data_npz.py --catalog`` → 2 sphere NPZs (lattice + j0.04), ``dataset_N=1729`` (729+1000), inside 270 / outside 1459, batch ``(8, 3)``.

## 2026-08-27 16:00 — Phase 2 ladder: loader + runs before first train

Reordered Phase 2 in [`docs/work_plan_phase2.md`](docs/work_plan_phase2.md). There is no Phase 3. After Step 2, the ladder is: multi-NPZ reader (3) → ``runs/`` + ``best.pt`` (4) → first occupancy train (5). Geometry steps shifted to 6–10. No training code changed.

## 2026-08-27 15:00 — Operator notes for NPZ generation

Operating notes for occupancy NPZ sets: [`docs/npz_dataset_generation.md`](docs/npz_dataset_generation.md). Covers CLI parameters, NPZ keys, a 10-point example from the sphere file, and copy-paste commands. No sampler or training code changed.

## 2026-08-27 15:00 — Full occupancy NPZ batch (r=0 and r=0.04)

Ran ``dataset_builder.py`` over all 3660 OBJs under ``data/meshes/``. Two NPZs per mesh: lattice (``random_range=0``) and low jitter (``0.04``). Spacing ``0.15``, seed ``1``, method occupancy.

- Output: ``E:/Work_stuff/scatteringNet/data/exports/dataset`` (7320 NPZs, ``ok=7320`` ``fail=0`` ``skipped=0``).
- Filenames: ``<stem>__occupancy_s0.15_inout.npz`` and ``<stem>__occupancy_s0.15_j0.04_inout.npz``.
- Spot-check ``TorusX4``: ``mesh_path=meshes/varied/TorusX4.obj``; r=0 N=125000; r=0.04 N=132651.

## 2026-08-27 12:00 — NPZ ``mesh_path`` relative to ``data_dir``

Stored OBJ paths are now relative to ``config.yaml`` ``data_dir`` (e.g. ``meshes/Primitives/Sphere/...obj``). Paths outside that folder still fall back to absolute POSIX. Re-run ``dataset_builder.py`` to refresh existing NPZs.


## 2026-08-27 12:00 — Step 2: [NPZ dataset generation]

Conda-side occupancy sampling now lives in this repo. Phase 1 `OccupancyMLP` / `train_one_npz.py` / `infer_one_npz.py` were not changed.

- Ported `mesh_loader.py`, `raycast_scatter.py`, `dataset_builder.py`, `paths.py`, `near_surface.py` (tagging default off) into `src/scatter_generation/`. Imports use `scatter_generation.*` (no `scattering_net`).
- CLI: `python src/scatter_generation/build_dataset.py <mesh_root> --method occupancy --spacings … --random-ranges … --seed … --limit N`.
- NPZ contract: `points (N,3)`, `labels (N,)`, `mesh_path`, plus `random_range` / `jitter` metadata. Order: lattice → offset → labels on the moved points. `r=0` is a no-op.
- Pinned `open3d>=0.18` and `trimesh>=4.0` in `environment.yaml` (env has Open3D 0.19.0, trimesh 5.0.0).
- Tests: `tests/test_scatter_generation.py` 4 OK. Full suite 32 tests: 31 OK; `test_load_config_resolves_data_dir_and_device` ERROR because `E:/Work_stuff/scatteringNet/data` is missing on this machine (not a Step 2 regression).
- Smoke (synthetic watertight cube OBJ, occupancy, spacing=0.35, random_range=0.04, seed=7): N=216, inside=9, outside=207, method=occupancy, occupancy_verified=True, `mesh_path` set. Maya Step 1 mesh tree not present, so no primitive OBJ batch from `data/meshes`.
- Optional Phase 1 compatibility: `train_one_npz` 2 epochs on a generated box NPZ (N=343, CUDA, hidden=64, depth=4). epoch 001 `loss=0.687535 train_acc=0.6460 val_acc=0.5942`; epoch 002 `loss=0.684318 train_acc=0.6460 val_acc=0.5942`. Loader/train loop accepted the file; this is not an overfit claim.

## 2026-08-27 09:00 — Maya OBJ export: load `objExport` plugin

Sphere batch failed in Maya with `Invalid file type specified: OBJexport` because the OBJ exporter plugin is off by default.

- `maya_batch_primitives.py`, `maya_batch_extrude.py`, `maya_batch_helix.py`: load `objExport` before `cmds.file(..., typ="OBJexport")`.
- Reload the script in Script Editor (`exec(open(...).read())`) then `run(...)` again.

Operating notes for OBJ export: [`docs/maya_batch_scatter_scripts.md`](docs/maya_batch_scatter_scripts.md)

## 2026-08-26 10:00 — Step 1: [Port Maya scatter scripts]

Maya OBJ exporters now live in this repo. Export roots match `config.yaml` `data_dir`. No occupancy model changes. No conda-side NPZ sampler (Step 2).

- Added `src/scatter_generation/maya_batch_primitives.py`, `maya_batch_extrude.py`, `maya_batch_helix.py` (Maya Script Editor only; `maya.cmds` unchanged).
- Export paths: primitives `.../meshes/Primitives`, extrude `.../meshes/Extrude`, helix `.../meshes/Helix`. Loader comments point at `src/scatter_generation/`.
- Dry-read: `run` / `list_families` (primitives) parse cleanly. Conda `unittest discover -s tests`: 28 tests OK.
- Maya export smoke: **not run here** (needs Script Editor). Suggested: `exec(open(...maya_batch_primitives.py).read()); run(families=("sphere",))` → `E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere`.


## ================= END OF PHASE 1 =================

## 2026-08-24 15:00 — Step 10: [Stop and review]

Closed the occupancy MLP MVP with a timestamped review note. No source, tests, or config were changed.

- Added `docs/2026-08-24_14-09_mvp_completion_review.md`: data loop works; xyz MLP fits one field; next product step is a geometry encoder (not in this plan).

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
- `.cursorrules` and `docs/work_plan_phase1_AI.md` now name `scatteringNet` (not `scatteringNet_v2`) as the project environment.

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