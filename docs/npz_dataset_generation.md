# Occupancy NPZ sets — operator notes

Maya gives us **OBJ** meshes. Training needs **NPZ** occupancy files: a cloud of 3D query points, each marked inside or outside the solid.

This note is how to build those files with the conda script. It does **not** run inside Maya. It does **not** train a network. The companion for making the meshes themselves is [`maya_batch_scatter_scripts.md`](maya_batch_scatter_scripts.md).

---

## How to read this

One path in, one path out:

**folder of OBJs → `dataset_builder.py` → folder of NPZs**

You pick density (`--spacings`) and how far each point may wander after the lattice (`--random-ranges`). 
Labels are always computed on the **final** positions.

Progress of what already shipped lives in [`CHANGELOG.md`](../CHANGELOG.md). The Phase 2 story is [`work_plan_phase2.md`](work_plan_phase2.md).

---

## Where things live

| What | Path |
|---|---|
| Git repo (run commands from here) | `F:/Work_stuff/VisualStudio_cursor/scatteringNet` |
| Data root (`config.yaml` `data_dir`) | `E:/Work_stuff/scatteringNet/data` |
| Source meshes | `E:/Work_stuff/scatteringNet/data/meshes/` |
| Script | `src/scatter_generation/dataset_builder.py` |
| Conda env | `scatteringNet` |

Always pass `--out` under the data disk. The script’s built-in default is a `data/exports/dataset` folder **next to the git repo**, which is not where our meshes live.

`mesh_path` stored inside each NPZ is **relative to `data_dir`**, for example `meshes/Primitives/Sphere/sphere_r0p5_sa16_sh16.obj`.

---

## What the sampler does

1. Load each mesh (OBJ, PLY, STL, …). Subfolders are included unless you pass `--no-recursive`.
2. Place query points. Default method `occupancy`: a regular XYZ lattice in a padded axis-aligned box around the mesh. Smaller spacing → more points.
3. If `random_range` is not zero, nudge every point independently on X, Y, and Z by a uniform draw in `[-r, +r]`.
4. Classify each **moved** point: `1` = inside, `0` = outside.
5. Write one NPZ per mesh × parameter combo.

`random_range = 0` is a no-op: the lattice is left as-is, then labeled.

`--spacings` and `--random-ranges` are **lists**. The script takes the cartesian product. Two ranges (`0` and `0.04`) with one spacing means **two NPZs per OBJ**.

---

## CLI parameters

Run from the repo root, env `scatteringNet` active.

Positional:

| Argument | Meaning |
|---|---|
| `mesh_root` | Folder to scan. Example: `E:/Work_stuff/scatteringNet/data/meshes` |

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--out` | repo `data/exports/dataset` | Destination folder for NPZs (and `manifest.json`) |
| `--method` | `occupancy` | `occupancy` = 3D lattice + per-point test. `raycast` = rays from one bbox face (needed for some torus-like solids) |
| `--spacings` | `0.1` | Point spacing in world units. Comma-separated for several densities. Smaller = denser |
| `--random-ranges` | `0` (via `--jitters`) | Per-axis half-width after placement. `0` = lattice. Low ≈ `0.25 * spacing`. High ≈ `1.0 * spacing`. Comma-separated |
| `--jitters` | `0` | Same as `--random-ranges` |
| `--seed` | `1` | RNG seed for the random range (reproducible offsets) |
| `--max-points` | `200000` | If a lattice would exceed this, spacing is coarsened |
| `--no-outside` | off | Keep inside points only |
| `--axes` | `z` | Raycast only. Comma-separated, or `all` |
| `--rays` | `16` | Raycast only. Fixed grid size. Ignored with `--auto-rays` |
| `--auto-rays` | off | Raycast only. Derive ray count from bbox and spacing |
| `--no-recursive` | off | Do not walk subfolders |
| `--skip` | `0` | Skip the first N meshes (resume a long run) |
| `--limit` | none | Process at most N meshes (smoke / test) |
| `--dry-run` | off | Print planned files; still loads meshes |

For the occupancy path we use every day, the knobs that matter are **`--out`**, **`--spacings`**, **`--random-ranges`**, **`--seed`**, and optionally **`--limit`**.

---

## What is inside an NPZ

Each file is an uncompressed NumPy archive (`np.savez`). Training today only needs the first two keys. The rest is provenance for us (and for later mesh join).

| Key | Shape / type | Meaning |
|---|---|---|
| `points` | `(N, 3)` float32 | Query XYZ in **points coordinates** |
| `labels` | `(N,)` uint8 | `1` = inside, `0` = outside |
| `mesh_path` | string | OBJ path relative to `data_dir` |
| `method` | string | `occupancy` or `raycast` |
| `point_spacing` | float32 | Spacing used to place the lattice (may be coarsened by `--max-points`) |
| `random_range` | float32 | Per-axis half-width that was applied |
| `jitter` | float32 | Same value as `random_range` (older name) |
| `occupancy_verified` | bool | `True` when labels came from a per-point occupancy test |
| `axis` | string | `n` for the occupancy lattice; `x` / `y` / `z` for raycast |
| `ray_grid` | `(2,)` int32 | Occupancy: lattice counts on two axes. Raycast: rays on the bbox face |

`N` is not stored as its own key; it is `points.shape[0]`. Class counts are `labels == 1` and `labels == 0`.

### Filename

```text
<mesh_stem>__occupancy_s<spacing>[_j<range>]_inout.npz
```

- Lattice, no wander: `sphere_r0p5_sa16_sh16__occupancy_s0.15_inout.npz`
- Same mesh, `random_range = 0.04`: `sphere_r0p5_sa16_sh16__occupancy_s0.15_j0.04_inout.npz`

The `_j…` token appears only when the range is greater than zero.

A `manifest.json` is also written in `--out`. It is a batch log (counts, failures, parameter grid). Training does not read it.

---

## Example file (ten points)

File:

`E:/Work_stuff/scatteringNet/data/exports/dataset/sphere_r0p5_sa16_sh16__occupancy_s0.15_inout.npz`

This is one Maya sphere (`r = 0.5`), occupancy lattice, spacing `0.15`, **no jitter**.

| Field | Value |
|---|---|
| `mesh_path` | `meshes/Primitives/Sphere/sphere_r0p5_sa16_sh16.obj` |
| `method` | `occupancy` |
| `point_spacing` | `0.15` |
| `random_range` / `jitter` | `0` |
| `occupancy_verified` | `True` |
| `axis` | `n` |
| `N` | `729` (a `9 × 9 × 9` lattice) |
| inside / outside | `123` / `606` |

Ten real rows from that file (five inside, then five outside). Coordinates are world units; label `1` is inside the sphere.

| # | x | y | z | label | class |
|---|---:|---:|---:|:---:|---|
| 121 | −0.4875 | 0.0000 | 0.0000 | 1 | inside |
| 183 | −0.3250 | −0.3250 | −0.1625 | 1 | inside |
| 184 | −0.3250 | −0.3250 | 0.0000 | 1 | inside |
| 185 | −0.3250 | −0.3250 | 0.1625 | 1 | inside |
| 191 | −0.3250 | −0.1625 | −0.3250 | 1 | inside |
| 0 | −0.6500 | −0.6500 | −0.6500 | 0 | outside |
| 1 | −0.6500 | −0.6500 | −0.4875 | 0 | outside |
| 2 | −0.6500 | −0.6500 | −0.3250 | 0 | outside |
| 3 | −0.6500 | −0.6500 | −0.1625 | 0 | outside |
| 4 | −0.6500 | −0.6500 | 0.0000 | 0 | outside |

Index `0` is a box corner — outside, as expected. Index `121` sits near the origin of this sphere — inside.

The jittered twin of the same mesh is `…_s0.15_j0.04_inout.npz`. Same keys; `random_range` is `0.04`; XYZ values are no longer on the lattice.

---

## Command examples

From the repo root:

```text
conda activate scatteringNet
cd F:/Work_stuff/VisualStudio_cursor/scatteringNet
```

### Smoke: one sphere, lattice only

Same settings as the example file above.

```text
python src/scatter_generation/dataset_builder.py E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere --out E:/Work_stuff/scatteringNet/data/exports/dataset_test --method occupancy --spacings 0.15 --random-ranges 0 --limit 1 --seed 1
```

### Smoke: one mesh, lattice and low jitter

Two NPZs from one OBJ.

```text
python src/scatter_generation/dataset_builder.py E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere --out E:/Work_stuff/scatteringNet/data/exports/dataset_test --method occupancy --spacings 0.15 --random-ranges 0,0.04 --limit 1 --seed 1
```

### Full catalog (what we already ran)

All 3660 OBJs under `data/meshes/`, two files each (`r = 0` and `r = 0.04`):

```text
python src/scatter_generation/dataset_builder.py E:/Work_stuff/scatteringNet/data/meshes --out E:/Work_stuff/scatteringNet/data/exports/dataset --method occupancy --spacings 0.15 --random-ranges 0,0.04 --seed 1
```

That run wrote **7320** NPZs (`ok = 7320`, `fail = 0`) into `E:/Work_stuff/scatteringNet/data/exports/dataset`.

### Denser lattice (more points)

```text
python src/scatter_generation/dataset_builder.py E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere --out E:/Work_stuff/scatteringNet/data/exports/dataset_test --method occupancy --spacings 0.08 --random-ranges 0 --limit 1 --seed 1
```

### Peek at a file

```text
python -c "import numpy as np; d=np.load(r'E:/Work_stuff/scatteringNet/data/exports/dataset/sphere_r0p5_sa16_sh16__occupancy_s0.15_inout.npz', allow_pickle=True); print(list(d.files)); p=d['points']; y=d['labels']; print(p.shape, y.shape, int((y==1).sum()), int((y==0).sum())); print(d['mesh_path'])"
```

---

## Practical notes

- **Env:** `open3d` and `trimesh` must be installed in `scatteringNet` (they are listed in `environment.yaml`).
- **Watertight meshes** label cleanly. Open or inverted meshes can mis-count inside/outside.
- **`--max-points`** will coarsen spacing on large solids so a single NPZ does not explode.
- **`--skip` / `--limit`** are counted in mesh-file order, not by existing NPZs. Re-running the same `--out` overwrites matching names.
- Training (`train_multi_npz.py`) reads **`points`** and **`labels` only**. Extra keys are ignored on purpose.
