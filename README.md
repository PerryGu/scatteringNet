# scatteringNet — Occupancy Fill

## TL;DR

- PyTorch occupancy network that labels 3D query points **inside** a mesh volume (not on the surface, not outside).
- Input is an OBJ (or a labeled occupancy NPZ). Output is an inside / outside point cloud. How you display it is up to the host tool.
- Best **out-of-catalog Fill** so far: envelope of **2048** skin dots and **24** nearest neighbors (`2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6`).
- Trained on CAD primitives and extrudes only. Humans, animals, and combo meshes were **not** in the catalog; they are the inspect test.
- This is **not** how you should fill a volume in production. There are plenty of tools and libraries that do this quickly and accurately. This repo exists because I had already built that plugin, and I wanted to see the same job done with a network.

---

## Project Inspiration

I already knew this was a bad production idea.

Filling a mesh with points is a geometry problem. A ray test, or the first-and-last-hit trick in **[scatteringNode](https://github.com/PerryGu/scatteringNode)** (a Maya C++ plugin I wrote **for work**, almost a decade ago), does it **exactly** and **faster** than a network ever will. A model is the wrong tool here: you spend hours of GPU time to approximate a test that the CPU already owns.

I built scatteringNet anyway. scatteringNode came first: ordinary programming on a **production job**. After the AI wave I wanted to close the loop — same task, this time as a network — out of curiosity, and to mark the shift from writing algorithms by hand to training them.

The result is the point of the repo, not a product pitch. **Use a neural net when the hand-written version is ugly or impossible. If the math already works, keep the math.** Volume fill is in the second bucket. This codebase is what it looks like to walk through the first bucket on purpose.

scatteringNet still asks a real research question on top of that: given only a skin sampling and a query point, can a network decide inside vs outside on shapes that were never in the training folder — thin extrudes, then (as a transfer test) humans and animals. The Maya plugin remains the production tool. This is the occupancy experiment next to it.

---

## Overview

scatteringNet is a learned volume-fill pipeline. Maya (or any DCC) still owns the mesh. This repo owns everything after that: occupancy NPZs, catalog training, checkpoints, and a Three.js viewer that fills an unlabeled lattice and runs the network.

The network does **not** emit points the way a particle system does. A regular grid (or a stored NPZ cloud) is placed in the mesh bounding box. Each point is classified: inside the solid or outside.

The stills below are that first step only: query points fill the box. They are not occupancy labels yet.


|                                                                                          |                                                                                              |                                                                                              |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| ![](docs/media/2026-09-15_bbox_fill_dog.png) **Query lattice in the bounding box (dog)** | ![](docs/media/2026-09-15_bbox_fill_human.png) **Query lattice in the bounding box (human)** | ![](docs/media/2026-09-15_bbox_fill_torus.png) **Query lattice in the bounding box (torus)** |


The mesh is also reduced to an **envelope**: `n_surface` samples on the skin. Each sample stores position and a face normal. Early trains used **1024** (the stills below). The best OOD Fill later used **2048**. Mix 75 puts most samples on faces (area-weighted) and the rest on sharp creases. A query does not see the triangles. It sees this cloud: a global PointNet over all envelope dots, plus the `knn_k` nearest neighbors.


|                                                                                      |                                                                                      |                                                                                          |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| ![](docs/media/2026-09-15_envelope_dog.png) **Envelope on the skin (dog, 1024)**     | ![](docs/media/2026-09-15_envelope_human.png) **Envelope on the skin (human, 1024)** | ![](docs/media/2026-09-15_envelope_torus.png) **Envelope on the skin (torus, 1024)**     |
| ![](docs/media/2026-09-15_envelope_helix.png) **Envelope on the skin (helix, 1024)** | ![](docs/media/2026-09-15_envelope_gear.png) **Envelope on the skin (gear, 1024)**   | ![](docs/media/2026-09-15_envelope_extrude.png) **Envelope on the skin (extrude, 1024)** |


The occupancy head is small on purpose (`hidden: 64`, `depth: 4`). What changed the Fill quality was not a wider MLP. It was **how each query reads** that envelope: a PointNet over the whole cloud, then a local k-NN of nearby skin dots.

Catalog val IoU is a useful health check. It is **not** the success bar. Success is viewer **Fill** on shapes the split never saw.

---

## Image Gallery

Fill stills below are the **best current checkpoint** (`knn_k: 24`, `n_surface: 2048`).  
**None of the human / animal / combo stills were in the training catalog.** Gear is in-catalog. Helix is a catalog *family*, but not these bent / FFD meshes. Extrudes are a catalog family (`nr4` / `nr5`).


|                                                                             |                                                                                                  |                                                                                             |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| ![](docs/media/2026-09-14_knn24_n2048_woman.png) **Woman (not in catalog)** | ![](docs/media/2026-09-14_knn24_n2048_man.png) **Man (not in catalog)**                          | ![](docs/media/2026-09-14_knn24_n2048_dog.png) **Dog (not in catalog)**                     |
| ![](docs/media/2026-09-14_knn24_n2048_horse.png) **Horse (not in catalog)** | ![](docs/media/2026-09-14_knn24_n2048_human_stylized.png) **Stylized human (not in catalog)**    | ![](docs/media/2026-09-14_knn24_n2048_combo_animals.png) **Combo animals (not in catalog)** |
| ![](docs/media/2026-09-14_knn24_n2048_gear.png) **Gear (in catalog)**       | ![](docs/media/2026-09-14_knn24_n2048_helix.png) **Helix (family in catalog; this mesh is not)** | ![](docs/media/2026-09-14_knn24_n2048_extrude.png) **Extrude (catalog family)**             |


Earlier n6 Fill (same catalog, `knn_k: 16`, `n_surface: 1024`) already transferred to organics. That was the jump from empty humans to a usable volume. The 2048 / k=24 run is a tighter OOD fill on top of that, not a new architecture.


|                                                                          |                                                                       |                                                                       |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ![](docs/media/2026-09-06_n6_human.png) **n6 human (1080, k=16 / 1024)** | ![](docs/media/2026-09-06_n6_horse.png) **n6 horse (not in catalog)** | ![](docs/media/2026-09-06_n6_shark.png) **n6 shark (not in catalog)** |


---

## Features

- Occupancy classification of lattice / NPZ query points (inside vs outside).
- Envelope-conditioned encoder: PointNet over skin dots, plus per-query k-NN.
- Skin dots are XYZ + face normal (`envelope_dim=6`). Mix 75 splits face-area vs crease samples.
- Catalog training on many NPZs with a **mesh-identity** val split (`best.pt` by `val_iou`).
- Browser occupancy viewer: Open OBJ/NPZ, Fill an AABB lattice, Run model, Inside cut, Truth / Prediction / Errors.
- Local train on a GTX 1080, or the same script on SageMaker `ml.g5.xlarge` (A10G).
- Maya batch OBJ scripts (primitives, extrude, smooth extruded, helix) and a conda NPZ builder.
- Run snapshots under `runs/<id>/` (YAML, metrics, TensorBoard) and weights under `models/<id>/best.pt`.

---

## Occupancy viewer

Training prints accuracy and IoU. Those numbers can look fine while, in reality, the points still aren't positioned correctly: most query points sit **outside** the mesh, so “always guess outside” already scores high. The viewer is the inspect window for that check. It is a Three.js page plus a localhost helper (`src/viewer/serve.py`). It does not train, and it is not a Maya plugin.

**Launch.** Double-click `open_viewer.bat` at the repo root. Close any old helper console first, or the browser may still talk to a Python that has no `torch`.

```text
open_viewer.bat
```

Or: `conda activate scatteringNet` then `python src/viewer/serve.py`. Orbit works on CPU. **Run model** needs the GPU and checkpoints at `models/<run_id>/best.pt`. Control-by-control notes: [`src/viewer/README.md`](src/viewer/README.md).

**What it can do**

- **Open** or drop an `.obj` or `.npz`. Hover Open for the last ten files. An NPZ with `mesh_path` pulls that OBJ from `data_dir`.
- **OBJ Fill** (the gallery path). **Fill points** builds an unlabeled lattice in the bounding box (density slider; cap 200,000). **Envelope** draws purple skin dots. Overlay Mix / Count are display-only. Pick a checkpoint → **Run model** classifies every query. There are no file labels, so no Errors view.
- **NPZ check.** The file already has points and Truth. **Run model** classifies those same coordinates. **Prediction** / **Errors** compare to the labels.
- **Look.** Mesh / wireframe, inside / outside, opacity, point size, **Inside cut** (drag re-cuts the last Run in the browser, no extra GPU pass). Right-click a model row to append a note (`*`); that does not rename `models/`.

Typical inspect path: **Open** an OBJ → **Fill points** → pick a model → **Run model**.

---

## Dataset

There was no large, ready-made mesh collection of any kind — and a network needs volume. The meshes were generated in Maya. Operator notes for those scripts: [`docs/maya_batch_scatter_scripts.md`](docs/maya_batch_scatter_scripts.md).

**Primitives.** Start with Maya’s built-in solids — ten families: sphere, cube, cylinder, cone, torus, pipe, prism, helix, gear, platonic. [`maya_batch_primitives.py`](docs/maya_batch_scatter_scripts.md#maya_batch_primitivespy) sweeps each family’s parameters (radius, subdivisions, teeth, and so on) and writes on the order of **~100 distinct OBJs per type**.

**Extrude.** Primitives alone are too clean. [`maya_batch_extrude.py`](docs/maya_batch_scatter_scripts.md#maya_batch_extrudepy) starts from a subdivided cube, extrudes a few faces (steps / corners; `keepFacesTogether` on), and varies thickness, face count, and how many extrude rounds (`nr1` … `nr5`). Open leftover shells are skipped so occupancy Truth does not leak. The result is non-standard CAD with limbs, holes, and slabs the sphere/cube grid does not cover.

**Smooth extrude.** [`maya_batch_extruded_smooth.py`](docs/maya_batch_scatter_scripts.md#maya_batch_extruded_smoothpy) is a separate recipe: a rectangular box, a few directed extrudes (legs / neck / tail, or a star of sides), then **one** Maya `polySmooth`. The hope was a stand-in for organics, of which there were almost no examples. Viewer Fill later showed that this family was **not** what made humans and animals fill — envelope normals (`n6`) did. Smoothed CAD is optional for that goal.

Primitive stills will go here when they are in `docs/media/`.

---

## Occupancy NPZs

Maya writes **meshes**. Training needs **query points with inside/outside labels**. That second stage is conda, not Maya: [`src/scatter_generation/dataset_builder.py`](src/scatter_generation/dataset_builder.py). Full operator note: [`docs/npz_dataset_generation.md`](docs/npz_dataset_generation.md).

For each OBJ the script places a regular lattice in a padded bounding box (`--spacings`; smaller = denser). Optional `--random-ranges` then nudges every point on X/Y/Z; labels are computed on the **moved** positions (`0` leaves the lattice). The everyday method is `--method occupancy`: each query is classified by a multi-ray vote through Open3D (odd hit counts in enough directions). `--method raycast` exists for some torus-like solids (rays from one bbox face). `--spacings` and `--random-ranges` are lists; their cartesian product is written (two ranges × one spacing = two NPZs per mesh). `--max-points` coarsens density if a lattice would exceed 200,000 samples.

Meshes are loaded with **trimesh**. Occupancy and ray queries go through **Open3D**. NumPy writes the archive.

**Why `.npz`.** It is NumPy’s own archive: a zip of named arrays, loaded with one `np.load`. Training is NumPy / PyTorch, so `points` and `labels` come off disk as arrays with no parser. One file can hold a float cloud, a uint8 label column, a `mesh_path` string, and scalar metadata together. PLY / PCD are point-cloud interchange (geometry, sometimes color) — a poor home for class labels plus provenance. CSV / JSON would be huge and slow. HDF5 could do the same job but adds a dependency this loop does not need. `.npz` is ordinary in ML research for packing arrays; it is not a DCC format, and it is not meant to replace PLY in a film pipeline.

Main flags (occupancy path):


| Flag                            | Default                     | Meaning                                                               |
| ------------------------------- | --------------------------- | --------------------------------------------------------------------- |
| `mesh_root`                     | (required)                  | Folder of OBJs to scan                                                |
| `--out`                         | repo `data/exports/dataset` | Destination for NPZs + `manifest.json`                                |
| `--method`                      | `occupancy`                 | `occupancy` = 3D lattice + per-point test; `raycast` = bbox-face rays |
| `--spacings`                    | `0.1`                       | Lattice step (comma-separated)                                        |
| `--random-ranges` / `--jitters` | `0`                         | Per-axis half-width after the lattice                                 |
| `--seed`                        | `1`                         | RNG for the random range                                              |
| `--max-points`                  | `200000`                    | Coarsen spacing if the lattice would exceed this                      |
| `--limit` / `--skip`            | —                           | Cap / resume a long run                                               |
| `--glob`                        | all meshes                  | Filename filter, e.g. `*_nr5_*.obj`                                   |


Keys in each file (`np.savez`, uncompressed). Training reads `points` and `labels`. `**mesh_path` is stored too**: the OBJ as a path relative to `data_dir` (forward slashes), so the mesh can be loaded again for envelope join without baking the geometry into the NPZ.


| Key             | Shape / type     | Meaning                                |
| --------------- | ---------------- | -------------------------------------- |
| `points`        | `(N, 3)` float32 | Query XYZ                              |
| `labels`        | `(N,)` uint8     | `1` = inside, `0` = outside            |
| `mesh_path`     | string           | OBJ path relative to `data_dir`        |
| `method`        | string           | `occupancy` or `raycast`               |
| `point_spacing` | float32          | Lattice step (may have been coarsened) |
| `random_range`  | float32          | Jitter half-width (`0` = lattice)      |


Example from `sphere_r0p5_sa16_sh16__occupancy_s0.15_inout.npz` (spacing `0.15`, no jitter):


| Key             | Value                                                |
| --------------- | ---------------------------------------------------- |
| `mesh_path`     | `meshes/Primitives/Sphere/sphere_r0p5_sa16_sh16.obj` |
| `method`        | `occupancy`                                          |
| `point_spacing` | `0.15`                                               |
| `random_range`  | `0`                                                  |


Point rows (five inside, then five outside):


| #   | x       | y       | z       | label |
| --- | ------- | ------- | ------- | ----- |
| 121 | −0.4875 | 0.0000  | 0.0000  | 1     |
| 183 | −0.3250 | −0.3250 | −0.1625 | 1     |
| 184 | −0.3250 | −0.3250 | 0.0000  | 1     |
| 185 | −0.3250 | −0.3250 | 0.1625  | 1     |
| 191 | −0.3250 | −0.1625 | −0.3250 | 1     |
| 0   | −0.6500 | −0.6500 | −0.6500 | 0     |
| 1   | −0.6500 | −0.6500 | −0.4875 | 0     |
| 2   | −0.6500 | −0.6500 | −0.3250 | 0     |
| 3   | −0.6500 | −0.6500 | −0.1625 | 0     |
| 4   | −0.6500 | −0.6500 | 0.0000  | 0     |


Index `0` is a bounding-box corner (outside). Index `121` sits near this sphere’s origin (inside).

---

## Algorithm (High-Level)

Training labels come from geometry, not from the network. Each NPZ is a cloud of query points in a padded AABB, marked inside or outside the OBJ (lattice, optional jitter, ray tests). The model never sees those labels at Fill time; Fill is an unlabeled grid plus a forward pass.

At train and infer, the mesh is reduced to an **envelope**: `n_surface` darts on the joined triangles. Mix 75 puts most dots on faces (area-weighted) and the rest on sharp creases. Each dart stores position and a face normal. Catalog trains started at 1024; the best OOD Fill used 2048. Overview has stills of the 1024 envelope.

For one query point the head sees three things:

1. The query XYZ (in the mesh AABB frame).
2. A **global** shape code: a small PointNet (per-point MLP + max-pool) over the whole envelope.
3. A **local** code: the `knn_k` nearest envelope dots. Distance is XYZ only. Each neighbor is the offset `(dx, dy, dz)` plus that neighbor’s normal. Short arrows in several directions → likely inside a limb. Long arrows all one way → likely air.

`knn_k` is neighbors per query, **not** the envelope count. `n_surface` is how many skin dots exist. The winning Fill used **24 neighbors** and **2048 skin dots**.

The occupancy MLP on xyz alone could not fill sleeves. A global envelope code (no k-NN) filled boxes and still left thin arms empty: every query shared the same shape vector. Local neighbors fixed that reading problem. Putting face normals on those neighbors (`n6`) is what transferred Fill to OOD organics. A denser envelope at k=16, on an older XYZ-only catalog, raised val IoU and **did not** change Fill — density only helped later, on the n6 head, together with k=24.

Overlay Mix / Count in the viewer are display-only. **Run model** rebuilds the envelope with the **checkpoint** mix and count.

---

## Results (Fill wrap-up)

Side-by-side viewer Fill on the same OOD set (woman, dog, man, CAD extrudes, combo primitives):


| Checkpoint                             | Box            | `knn_k` | `n_surface` | Catalog val IoU | Fill on this OOD set                                                            |
| -------------------------------------- | -------------- | ------- | ----------- | --------------- | ------------------------------------------------------------------------------- |
| `2026-09-05_12-18-28_…_n6`             | GTX 1080       | 16      | 1024        | 0.965           | Strong baseline. Cleaner woman hands than SageMaker k=16. More dog bleed.       |
| `2026-09-13_11-23-44_…_n6`             | SageMaker A10G | 16      | 1024        | 0.967           | Same YAML as the 1080 n6. Trades leftovers (woman left-hand leak; tighter dog). |
| `2026-09-14_07-43-34_…_knn24_n2048_n6` | SageMaker A10G | **24**  | **2048**    | 0.974           | **Best Fill.** Inside points stay inside the wire on humans, extrudes, and combos.    |


**What that comparison is not.** The two k=16 runs are the same recipe, not the same weights. Seed 1 does not pin cuDNN or TF32. Ampere (A10G) uses TF32 for FP32 matmuls by default; the 1080 does not. Val IoU 0.967 vs 0.965 is noise. Fill leftovers swap (hand vs ear) because catalog val never saw those meshes.

**User conclusion (15 Sep 2026).** The two k=16 models are close: one wins a still, the other wins the next. If you want the **best** result on this inspect set, use 2048 envelope dots and 24 neighbors. Train wall and **Run model** are slower (k-NN scales with envelope `N`; that A10G train was ~9h 29m vs ~5h for SageMaker k=16 / 1024, ~10h on the 1080).

**Catalog caveat (agreed).** None of those inspect shapes were training identities. If humans / animals had been in `npz_catalog`, k=16 / 1024 might have been enough, and this upgrade might not have been justified. Putting them in now would also end the OOD test: the next stills would be in-catalog, not a rerun of this grid.

**What not to do next.** Do not pick a default from val IoU. Do not resume a `best.pt` across a head or catalog change. Raising `knn_k` and `n_surface` **did** help this OOD Fill; more neighbors and a denser envelope (for example 32 and 4096) might help again. That is an open trade against train wall and infer time, not a closed door. A leftover remains: slight dog-ear spray, and fingers that are tighter (sometimes thinner inside) rather than bleeding out.

Full train write-ups: [`docs/training_log.md`](docs/training_log.md). Code / knob history: [`CHANGELOG.md`](CHANGELOG.md).

---

## Architecture

**Data Flow**

The occupancy head never sees triangles. It sees a skin envelope plus one query. Those three clues are glued together and scored: inside the solid, or outside.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"htmlLabels": true, "curve": "basis"}}}%%
flowchart TD
    SETUP["Envelope mix 75<br/>'Skin sampling'<br/>Sprinkle dots on the surface only.<br/>Best Fill: 2048 dots."]
    HUB["OccupancyEncoder<br/>'The occupancy head'<br/>For each test point, gather three clues<br/>then decide: solid or air."]
    Q["Query XYZ<br/>'One test point'<br/>A location in the mesh bounding box.<br/>Question: is this inside the solid."]
    PN["SurfaceEncoder<br/>'Whole-shape summary'<br/>Looks at every skin dot at once.<br/>Same code for all queries on this mesh."]
    KN["k-NN local<br/>'Nearby skin'<br/>The 24 closest surface dots.<br/>Short arrows all around: likely inside.<br/>Long arrows one way: likely air."]
    QO["xyz<br/>'This point's coordinates'<br/>Just where it sits in the box."]
    ZG["z_global<br/>'What the whole mesh is'<br/>One fingerprint for the shape<br/>gear vs dog vs thin extrude."]
    ZL["z_local<br/>'What is right next to it'<br/>Nearby skin plus which way<br/>those faces point."]
    MLP["Occupancy MLP<br/>'Small classifier'<br/>Glue the three clues together.<br/>hidden 64, depth 4."]
    OUT["Inside / Outside<br/>'The Fill label'<br/>Inside stays in the solid.<br/>Air is left empty."]

    SETUP -->|"1. Hand the skin to the head"| HUB
    HUB -->|"2. Where is this point"| Q
    HUB -->|"2. What is the whole shape"| PN
    HUB -->|"2. What skin is nearby"| KN
    Q --> QO
    PN --> ZG
    KN --> ZL
    QO -->|"3. Glue and score"| MLP
    ZG --> MLP
    ZL --> MLP
    MLP --> OUT

    classDef setup fill:#243447,stroke:#6eb5d6,stroke-width:2px,color:#e6edf3
    classDef hub fill:#243447,stroke:#6eb5d6,stroke-width:2px,color:#e6edf3
    classDef proc fill:#1a3c35,stroke:#4caf86,stroke-width:2px,color:#e6edf3
    classDef out fill:#161b22,stroke:#6e7681,stroke-width:1.5px,stroke-dasharray:6 4,color:#9aa3af
    class SETUP setup
    class HUB,MLP hub
    class Q,PN,KN proc
    class QO,ZG,ZL,OUT out
```



k-NN distance is XYZ only. Each neighbor still carries that skin dot’s face normal (`envelope_dim=6`). That facing is what transferred Fill to humans and animals.

**Envelope**

How the mesh is reduced to those skin dots. Mix 75 means most dots sit on faces; the rest hug sharp creases.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"htmlLabels": true, "curve": "basis"}}}%%
flowchart TD
    OBJ["OBJ mesh<br/>'The 3D file'<br/>Triangles that make the surface.<br/>The network never sees these later."]
    MIX["envelope_mix 75<br/>'How to sprinkle the dots'<br/>75% on faces, 25% on sharp folds.<br/>Count is n_surface 1024 or 2048."]
    AREA["Area samples<br/>'Paint the big faces'<br/>Larger triangles get more dots.<br/>This covers the bulk of the skin."]
    CREASE["Crease samples<br/>'Hug the sharp edges'<br/>Extra dots on folds and corners<br/>so thin limbs are not missed."]
    AABB["AABB normalize<br/>'Same frame as the queries'<br/>Join the two clouds, then shift<br/>and scale like the NPZ points."]
    ENV["Envelope<br/>'Skin cloud in query space'<br/>Each dot: position + face direction.<br/>This is all the head will see."]

    OBJ -->|"1. Load the mesh"| MIX
    MIX -->|"most dots"| AREA
    MIX -->|"crease budget"| CREASE
    AREA -->|"2. Join"| AABB
    CREASE -->|"2. Join"| AABB
    AABB -->|"3. Ready for the head"| ENV

    classDef setup fill:#243447,stroke:#6eb5d6,stroke-width:2px,color:#e6edf3
    classDef hub fill:#243447,stroke:#6eb5d6,stroke-width:2px,color:#e6edf3
    classDef proc fill:#1a3c35,stroke:#4caf86,stroke-width:2px,color:#e6edf3
    classDef out fill:#161b22,stroke:#6e7681,stroke-width:1.5px,stroke-dasharray:6 4,color:#9aa3af
    class OBJ setup
    class MIX hub
    class AREA,CREASE,AABB proc
    class ENV out
```



- `OccupancyEncoder` (`src/occupancy_encoder.py`) — `nn.Module`: surface PointNet + optional k-NN + occupancy MLP. Checkpoints store `kind`, `knn_k`, `envelope_dim`.
- `OccupancyMLP` (`src/occupancy_mlp.py`) — xyz-only head; older runs still load.
- `train_multi_npz.py` — catalog loop, mesh split, `val_iou` selection, `models/<run_id>/best.pt`.
- `infer_multi_npz.py` / viewer `infer_job.py` — same AABB and envelope rebuild as train.
- `src/geometry/` — OBJ triangles, envelope sampling, crease mix (no torch in mesh IO).
- `src/scatter_generation/` — Maya OBJ batch scripts + `dataset_builder.py` (NPZ labels).
- `src/viewer/` — Three.js page + localhost helper (`serve.py`). Occupancy Python stays in `src/`.
- `sagemaker/` — job entry and launcher. Same train script; `data_dir` remapped to the training channel.

---

## Project Structure

<details>
<summary><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <strong>scatteringNet/</strong> — click to expand the tree</summary>

<ul>
<li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>config.yaml</code></li>
<li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>environment.yaml</code></li>
<li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>open_viewer.bat</code></li>
<li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>CHANGELOG.md</code></li>
<li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>README.md</code></li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>docs/</code>
  <ul>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>training_log.md</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>sagemaker.md</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>maya_batch_scatter_scripts.md</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>npz_dataset_generation.md</code></li>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>media/</code></li>
  </ul>
</li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>models/</code>
  <ul>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>&lt;run_id&gt;/</code>
    <ul>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>best.pt</code></li>
    </ul>
  </li>
  </ul>
</li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>runs/</code>
  <ul>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>&lt;run_id&gt;/</code>
    <ul>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>config.yaml</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>catalog.txt</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>metrics.jsonl</code></li>
    </ul>
  </li>
  </ul>
</li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>sagemaker/</code>
  <ul>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>entry.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>launch.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>requirements.txt</code></li>
  </ul>
</li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>src/</code>
  <ul>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>occupancy_encoder.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>occupancy_mlp.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>train_multi_npz.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>infer_multi_npz.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>config.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>dataset.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>encoder_dataset.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>data_npz.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>checkpointing.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>metrics.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>normalize.py</code></li>
  <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>run_tracking.py</code></li>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>geometry/</code>
    <ul>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>mesh_io.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>surface.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>trimesh_util.py</code></li>
    </ul>
  </li>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>scatter_generation/</code>
    <ul>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>dataset_builder.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>mesh_loader.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>raycast_scatter.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>maya_batch_primitives.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>maya_batch_extrude.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>maya_batch_extruded_smooth.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>maya_batch_helix.py</code></li>
    </ul>
  </li>
  <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>viewer/</code> — Three.js inspect page + localhost helper
    <ul>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>serve.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>infer_job.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>envelope_job.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>obj_fill.py</code></li>
    <li><img src="docs/media/icons/file.svg" width="16" height="16" alt=""/> <code>index.html</code></li>
    <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>js/</code></li>
    <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>css/</code></li>
    <li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>vendor/</code></li>
    </ul>
  </li>
  </ul>
</li>
<li><img src="docs/media/icons/folder.svg" width="16" height="16" alt=""/> <code>tests/</code> — stdlib <code>unittest</code> modules for the occupancy pipeline</li>
</ul>
</details>

---

## Dependencies

- Python 3.10, PyTorch 2.5.1, CUDA 12.1 (conda; do not pip-install torch on Windows).
- NumPy, PyYAML, TensorBoard.
- Open3D + trimesh (NPZ generation / mesh queries, not the train loop).
- Three.js (vendored under `src/viewer/vendor/`).
- Optional: Amazon SageMaker SDK v3 + AWS CLI for `ml.g5.xlarge` jobs.
- Optional: Autodesk Maya (OBJ batch only; training is conda).

---

## Setup

- Windows 10
- Conda (Miniconda / Anaconda)
- NVIDIA GPU for train and **Run model** (viewer orbit works on CPU; infer does not)

```text
conda env create -f environment.yaml
conda activate scatteringNet
```

Set `data_dir` in `config.yaml` to the folder that contains `exports/` and `meshes/` as siblings. NPZ `mesh_path` values are relative to that root.

Checkpoints the viewer lists are `models/<run_id>/best.pt`. Put a downloaded SageMaker `output.tar.gz` extract at the repo root so `models/` and `runs/` land next to local trains.

---

## Usage

#### Occupancy viewer

Role, launch, and what the page can do: [Occupancy viewer](#occupancy-viewer).

```text
open_viewer.bat
```

#### Train on the YAML catalog

```text
conda activate scatteringNet
python src/train_multi_npz.py
```

Reads `config.yaml`. Fresh train unless you pass `--resume-run-id` / `--resume` (same head and catalog only).

#### Build occupancy NPZs from OBJ folders

```text
python src/scatter_generation/dataset_builder.py E:/path/to/meshes --out E:/path/to/data/exports/dataset --spacings 0.15 --jitters 0,0.04
```

Operator note: [`docs/npz_dataset_generation.md`](docs/npz_dataset_generation.md). Maya mesh scripts: [`docs/maya_batch_scatter_scripts.md`](docs/maya_batch_scatter_scripts.md).

#### SageMaker (same catalog, A10G)

```text
python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE --dry-run
python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE
```

Instance `ml.g5.xlarge`, region `eu-north-1`. Quota, S3 layout, download, CloudWatch: [`docs/sagemaker.md`](docs/sagemaker.md).

#### Unit tests

The `tests/` folder holds stdlib `unittest` modules for the occupancy pipeline.

```text
python -m unittest discover -s tests
```

---

## YAML knobs (train)

```
shape_encoder : surface     # envelope OccupancyEncoder (xyz-only is "none")
n_surface     : 1024|2048   # envelope count (skin dots)
knn_k         : 16|24       # neighbors per query (not envelope count)
envelope_mix  : 75          # 0 = faces, 100 = creases
hidden/depth  : 64 / 4
checkpoint_metric : val_iou
```

Live `config.yaml` is the **next experiment**, not always the best Fill weights. Best Fill as of this README: `models/2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6/best.pt`.

---

## Known Issues

- OOD Fill still leaks or under-fills thin structure (dog ear, some fingers, thin extrude spikes).
- Catalog val can match while Fill leftovers swap across two trains of the same YAML (GPU / TF32 / non-deterministic CUDA).
- Mesh-identity val selects `best.pt`; it is **not** a locked organic holdout (Phase 3 Step 12 still open).
- Viewer Fill lattice is capped at 200,000 points (spacing coarsens if needed).
- Envelope overlay sliders do not change infer; forgetting that looks like a “Mix did nothing” bug.
- Face-token (`shape_encoder: mesh`) checkpoints are no longer loaded.
- SageMaker source upload on Windows must use POSIX S3 keys and Unix LF on `sm_train.sh` (already handled in `launch.py`).

---

## Roadmap (Future Improvements)

- A locked holdout family that never selected `best.pt`.
- Decide whether organics belong in the **train** catalog (that ends the current OOD inspect, on purpose).
- Hosting a SageMaker endpoint is not required for this wrap-up.
- A denser envelope / wider k-NN is still on the table if the leftover leaks are worth the extra wall.

---

## Status

Personal occupancy research repo, wrapped at a Fill result that is willing to be shown: CAD catalog in, OOD volume fill out, with an honest leftover.

Best inspect weights: `2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6` (`knn_k: 24`, `n_surface: 2048`, `envelope_dim=6`).  
The cheaper n6 baseline (`knn_k: 16`, `n_surface: 1024`, `2026-09-05_12-18-28_…` on the 1080) remains the recipe that first made organics fill.

Not a Maya plugin and not a hosted API. The production way to fill a volume is still geometry; this repo is the network detour I chose to take on purpose.

---

## Contact & Author

**Perry Guy** - Programmer  
*Extensive experience in 3D Graphics, Tool Development, and Performance Optimization.*

📧 **Email**: [perryguy2@gmail.com](mailto:perryguy2@gmail.com)

**YouTube Channel**: [@ThePerryGuy](https://www.youtube.com/@ThePerryGuy)