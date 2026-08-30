# Phase 2 — work plan

Now that we've established the first phase—creating a simple pipeline that knows how to read a single NPZ file—we can move on to the second phase: 
adding capabilities and complexity to the existing pipeline so the model can truly "understand" the geometry rather than just memorizing a single file. 
Here, we build the path step by step: from generating OBJ files in Maya and sampling inside/outside points, through a multi-file reader, run logs, and checkpoints, 
then the **first** occupancy train, up to feeding geometric data and faces into the network to test whether we can generalize to entirely new shapes.
The idea is mainly to add one more layer of complexity at a time—points on the envelope, then vertices and normals, 
and so on—and to look at each layer before deciding whether we even need the next.
We might get good enough results without moving on. Some layers may not help at all; some may even make things worse.

This is still **Phase 2**. There is no Phase 3 in this repo. The numbered items below are *steps inside Phase 2*.

**Current rung (2026-08-30):** Steps **1–8 are done**. Next planned rung is **Step 9** (face tokens), or stop if the envelope result is enough. Hygiene already holds out whole meshes as **val** (selection split for `best.pt`). That is not Step 11. Step 11 is still a locked / new-family test. Infer is `src/infer_multi_npz.py` (do not restore `infer_one_npz.py`).

---

## How to read this plan

Work proceeds **one step at a time**. After each step we stop for review. Anything that trains also gets a short smoke run (~20 epochs) so we have numbers, not just code. 
Those numbers go in [`training_log.md`](training_log.md), not in `CHANGELOG.md`. 
Steps 1–2 are generation. Steps 3–5 are training *infrastructure* (many NPZs, then `runs/` logs, then `models/` checkpoints) — they do not start occupancy training. Step 6 is the first occupancy train.

Progress is visible in:

- this file
- `CHANGELOG.md` (what shipped: code and knobs — not train scores)
- [`training_log.md`](training_log.md) (each training: catalog, wall time, conclusions)
- `runs/` (config snapshot + `metrics.jsonl` + TensorBoard)
- `models/<run_id>/` (weights; `best.pt`)

---

## The eleven steps at a glance

| Step | Theme | Status | What we learn |
|---|---|---|---|
| 1 | Maya scatter scripts live in *this* repo | done | We can generate source OBJs without depending on the prototype tree |
| 2 | NPZ inside/outside sampling | done | We can turn those OBJs into occupancy files (points, labels, mesh path) at a density we choose, then optionally nudge each point by a random range (0 / low / high) **before** labeling |
| 3 | Read many NPZs | done | The dataset can load a whole set of occupancy files, not only one |
| 4 | Experiment *logs* | done | A `runs/` folder holds config + JSON (+ TensorBoard). No `.pt` files here |
| 5 | Checkpoints | done | **`best.pt`** lives under `models/<run_id>/`, linked by the same run id |
| 6 | First occupancy train (Phase 1 MLP, many NPZs) | done | Multi-file xyz training uses the reader, the logs, and the checkpoint saver |
| 7 | Join NPZ to OBJ | done | Each occupancy file can load the mesh it came from |
| 8 | Sample the surface envelope | done | Do dense points on the shell help occupancy *before* we use faces? |
| 9 | Describe the mesh as triangles | next | Vertices + normals become a geometry language the net can consume later |
| 10 | Encode faces and fuse with xyz | pending | The occupancy head finally looks at mesh structure |
| 11 | Hold out entire shapes | pending | Locked / new-family test: a mesh that was never a training identity (not the hygiene val split) |

The path in one line:

**Maya OBJ → occupancy NPZ → multi-file NPZ reader → runs/ logs → models/ checkpoints → first xyz train → live mesh load → envelope experiment → face tokens → face encoder → held-out shapes.**

Why logs and weights are two steps: JSON and a config snapshot are small and can live in git. `*.pt` files are large binaries. They share a **run id**, but they do not share a folder.

Why the train is not Step 3: Phase 1 still reads **one** file. The first Phase 2 train should already see many files, write a `runs/` log, and keep the **best** weights under `models/` (last epoch is often worse).

---

## Step 1 — Bring Maya scatter generation home

Later steps need a mesh library that belongs to this project. The prototype already has Maya scripts that emit families of primitives, extrudes, and helices. Those scripts only run **inside Maya**, not in the conda environment.

This step copies them here and points their export folders at our data tree. Success is practical: you can generate an OBJ family (for example a sphere) from Maya into this repo’s mesh folder.

We do **not** build occupancy NPZs here, and we do not change the occupancy model. Point sampling is Step 2. Phase 1 stays as-is.

**Done when:** Maya generation lives in this repository, with export paths that match our data layout.

---

## Step 2 — Fill space with inside and outside points

An OBJ is a surface. Training needs **query points** scattered through the bounding box, each labeled inside or outside, plus a pointer back to the mesh.

This step ports the prototype’s conda-side sampler (Open3D, bounding-box rays or occupancy grid, density knobs) into this repo. From an OBJ it writes NPZ files that Phase 1 already knows how to read: point coordinates, 0/1 labels, and the mesh path.

You choose two things independently: how **dense** the cloud is (spacing, ray count, a cap on total points), and how far each point may **move at random** after it is placed. Random range `0` leaves the lattice as sampled. A **low** range gives each point a small travel on every axis; a **high** range gives a larger travel. Inside vs outside is decided **after** that noise — a point that started inside can land outside, and the other way around. Combo scenes and near-surface extras wait; the job here is a clean, repeatable occupancy export.

Think of this as: “Maya gave us geometry; now we manufacture the occupancy questions the network will answer.”

**Done when:** this repo can emit occupancy NPZs from OBJs, with both classes present, a mesh path stored, and density plus random range under our control. We still have not trained on many files at once.

---

## Step 3 — Read a set of NPZ files

Phase 1 can open **one** occupancy file. The catalog we just built is thousands of files. Before anyone hits train, the data layer must glob or list many NPZs, concatenate their points and labels, and feed batches the trainer will use later.

The network is not trained here. We do not write `runs/` or `models/` checkpoints. We only prove: “give me a folder (or a glob), get one dataset.”

**Done when:** we can load several NPZs in one dataset, count the pooled points, and iterate a DataLoader. Occupancy training has not started.

---

## Step 4 — Run logs (`runs/`, no weights)

If the first multi-file train has no paper trail, we throw away the only numbers that matter. Each later train creates a timestamped folder under `runs/` with **thin** artifacts:

- the config that was actually used
- per-epoch metrics in JSON
- TensorBoard event files (binary; not for git)

No `best.pt` in this folder. Those are Step 5.

This step **builds the log helper** and tests it on a tiny fake loop (dummy metrics only). It does **not** run the occupancy MLP.

JSON + config may be committed. TensorBoard events stay gitignored.

**Done when:** the helper can create `runs/<id>/`, write JSON and TensorBoard scalars, and record the run id. No occupancy smoke yet. No `.pt` files.

---

## Step 5 — Checkpoints (`models/<run_id>/`)

Weights are a different product from logs. `best.pt` (only when the selection metric strictly improves) goes under `models/<same run id>/`.

The log folder stores a pointer to that checkpoint directory so the two stay joined. Inference loads from `models/`, not from `runs/`. We do not overwrite Phase 1’s `models/one_npz.pt` as the Phase 2 default.

This step tests a fake loop: epoch 2 better than epoch 3 keeps epoch-2 weights in `best.pt`. Still no occupancy MLP.

**Done when:** dummy checkpoints land in `models/<id>/`, `best.pt` updates only on improvement, and `runs/<id>/` names that folder. `*.pt` files are gitignored.

---

## Step 6 — First occupancy train (many files, still only xyz)

Now we actually train. The loop uses Step 3’s multi-file dataset, Step 4’s `runs/` logs, and Step 5’s `models/` checkpoints. The network is still the Phase 1 MLP: it sees query points and labels, not the mesh.

Validation is a random split of *points*, which only tells us the loop is healthy. High accuracy here is expected and **does not** mean the model generalizes.

Think of this step as: “can we train on a small catalog of occupancy fields, with curves and a best checkpoint, without breaking anything?”

**Done when:** multi-file xyz training writes `runs/<id>/` logs and `models/<id>/best.pt`. We do not claim generalization.

---

## Step 7 — Connect occupancy files to their meshes

Each NPZ already stores a path to its OBJ (Step 2 writes it; older files already had it). Phase 1 ignored that on purpose. Now we load the mesh: vertices and triangles, resolved against our data directory.

The occupancy head still only sees xyz. This step is a join, not a new model. If we re-train for a short smoke, numbers should look like Step 6. If they do not, the join is wrong and we stop.

**Done when:** every training example can name and load its OBJ. Occupancy is still xyz-only.

**Status:** done. `load_points_labels_mesh` + `src/geometry/mesh_io.py` resolve `mesh_path` and load `vertices (V, 3)` / `faces (T, 3)`. The occupancy head is unchanged.

---

## Step 8 — Surface envelope points (the first geometry experiment)

The simplest picture of a shape is a dense cloud of points sitting on its **skin** — the envelope of the mesh.

We sample those points from the OBJ, keep them in the same coordinate frame as the occupancy queries, and let a small encoder summarize the cloud into a shape vector. The occupancy head then answers “inside or outside?” using both the query point **and** that surface summary.

This is an R&D measurement, not the final architecture. The prototype later preferred triangle faces over surface clouds. We still run this step so we have our own number: did the shell help, sit still, or make training worse?

Compare on the **same** few primitives we used in Step 6. This is not a holdout yet.

**Done when:** envelope clouds are a real batched input, and we have a recorded comparison against xyz-only.

**Status:** done. YAML `n_surface` sets how many shell points are sampled. `shape_encoder: surface` concatenates that cloud's latent with query xyz. OccupancyMLP remains the `shape_encoder: none` path.

---

## Step 9 — Talk about the mesh as triangles

A mesh is not only a point cloud. It is triangles, each with three corners and a facing direction (the normal).

This step builds that description and lines it up with the same bounding-box normalization we already use for query points. We attach it to the data pipeline but **do not yet train** on it. 
The occupancy head stays on the envelope signal from Step 8, so we can see that tokenization did not break training.

**Done when:** each mesh has a stable triangle description, aligned with occupancy queries. The face encoder is not trained yet.

---

## Step 10 — Let the occupancy head see faces

Now the intended geometry encoder comes in: a module that reads the triangle tokens, compresses them into a shape vector, and concatenates that vector with each query point.

Envelope sampling can stay around for comparison, but the trained head uses **faces**. We smoke the same small primitive set and compare to Step 8. If faces are worse, we write that down instead of quietly deleting the envelope path.

We still are not claiming “unseen OBJ.” We are claiming: xyz + mesh faces trains, logs, and keeps a best checkpoint.

**Done when:** occupancy is geometry-conditioned on faces, with a tracked run and a best checkpoint under `models/`.

---

## Step 11 — Hold out whole shapes

Catalog train already holds out **unique OBJs** as val (hygiene, 2026-08-30). That split is scored every epoch to pick `best.pt`. Do not redo “add a mesh split” here.

This step is the **locked** test: a mesh (or a new family) that was **never** a training identity, not used to select the checkpoint every epoch.

Randomly hiding some *points* from a cone you already trained on can look like 99% accuracy and prove nothing. The model already knows that occupancy field.

The real test: train on some meshes, evaluate on a mesh that was **never** a training identity (for example train sphere and cube, hold out cone). Report accuracy,
 but take inside IoU / F1 more seriously — they describe the interior better.

A control xyz-only model trained only on the train shapes should struggle on the new mesh. If geometry does not beat that, we have learned something important.

**Done when:** we can answer, with numbers, whether envelope geometry and/or face geometry beat xyz-only on a held-out shape.

---

## What we are not doing in these eleven steps

Voxel grids, FFT / scattering filters, and Kymatio stay out. We are not shipping a learned “shape name” embedding as the model — that cannot open a new OBJ.
Attention between query points and faces, fancy viewers, combo-scene floods, and helix-heavy mixes are later work, after this sequence.

Phase 2 is a measured scaffold for generalization, not a finished fill-the-volume product. Thin pipes, 
torus holes, and stacked parts will remain hard even if holdout looks promising.

---

## How to follow progress

| You want… | Look at… |
|---|---|
| The current step | this file |
| What actually shipped | `CHANGELOG.md` |
| What a train meant | [`training_log.md`](training_log.md) |
| Curves and JSON | `runs/<id>/` |
| Weights to load | `models/<id>/best.pt` |
| How to build NPZs | [`npz_dataset_generation.md`](npz_dataset_generation.md) |
| Phase 1 | [`work_plan_phase1.md`](work_plan_phase1.md) |

---

## Next action

**Steps 1–8 are done.** Approve or edit **Step 9** (face tokens), or stop here if the envelope result is enough. The Step 8 catalog write-up is in [`training_log.md`](training_log.md) (`extrude_nr1_surface`).
