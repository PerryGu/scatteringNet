# Phase 2 — work plan

Now that we've established the first phase—creating a simple pipeline that knows how to read a single NPZ file—we can move on to the second phase: 
adding capabilities and complexity to the existing pipeline so the model can truly "understand" the geometry rather than just memorizing a single file. 
Here, we build the path step by step: from generating OBJ files in Maya and sampling inside/outside points, through multi-file training and clean run tracking, 
up to feeding geometric data and faces into the network to test whether we can generalize to entirely new shapes.
The idea is mainly to add one more layer of complexity at a time—points on the envelope, then vertices and normals, 
and so on—and to look at each layer before deciding whether we even need the next.
We might get good enough results without moving on. Some layers may not help at all; some may even make things worse.

---

## How to read this plan

Work proceeds **one step at a time**. After each step we stop for review. Anything that trains also gets a short smoke run (~20 epochs) so we have numbers, not just code. 
Steps 1–2 are generation, not occupancy training; they still leave a measurable artifact (exported meshes, NPZ counts).

Progress is visible in three places:

- this file
- `CHANGELOG.md` (what shipped, with date, time, and metrics)
- `runs/` (after Step 4: TensorBoard curves, JSON logs, best checkpoint)

---

## The nine steps at a glance

| Step | Theme | What we learn |
|---|---|---|
| 1 | Maya scatter scripts live in *this* repo | We can generate source OBJs without depending on the prototype tree |
| 2 | NPZ inside/outside sampling | We can turn those OBJs into occupancy files (points, labels, mesh path) at a density we choose, then optionally nudge each point by a random range (0 / low / high) **before** labeling |
| 3 | Train the Phase 1 MLP on many NPZs | Multi-file training works; still no mesh in the model |
| 4 | Experiment tracking | Every run is comparable; we keep the *best* weights, not just the last epoch |
| 5 | Join NPZ to OBJ | Each occupancy file can load the mesh it came from |
| 6 | Sample the surface envelope | Do dense points on the shell help occupancy *before* we use faces? |
| 7 | Describe the mesh as triangles | Vertices + normals become a geometry language the net can consume later |
| 8 | Encode faces and fuse with xyz | The occupancy head finally looks at mesh structure |
| 9 | Hold out entire shapes | The real test: a mesh the model has never trained as an identity |

The path in one line:

**Maya OBJ → occupancy NPZ → multi-file xyz train → tracked runs → live mesh load → envelope experiment → face tokens → face encoder → held-out shapes.**

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

## Step 3 — Train on many files, still only xyz

Before we add geometry into the network, we prove the training loop can swallow **several NPZs at once** — a few primitive shapes from Step 2 (or the existing test set), not combo dumps.

The network is still the Phase 1 MLP: it sees query points and labels, not the mesh. Validation is a random split of *points*, which only tells us the loop is healthy. High accuracy here is expected and **does not** mean the model generalizes.

Think of this step as: “can we train on a small catalog of occupancy fields without breaking anything?”

**Done when:** multi-file xyz training runs and logs. We do not claim generalization.

---

## Step 4 — Make every experiment inspectable

Once multi-file training works, we need a paper trail. Each train creates a timestamped folder under `runs/` with:

- the config that was actually used
- per-epoch metrics in JSON
- TensorBoard curves
- `last.pt` (end of training) and `best.pt` (peak validation — last epoch is often worse)

From here on, smokes are comparable. Inference prefers the best checkpoint, not whichever weights happened to sit at epoch 20.

**Done when:** a train leaves a folder you can open later, TensorBoard shows curves, and peak weights are saved.

---

## Step 5 — Connect occupancy files to their meshes

Each NPZ already stores a path to its OBJ (Step 2 writes it; older files already had it). Phase 1 ignored that on purpose. Now we load the mesh: vertices and triangles, resolved against our data directory.

The occupancy head still only sees xyz. This step is a join, not a new model. If we re-train for a short smoke, numbers should look like Step 4. If they do not, the join is wrong and we stop.

**Done when:** every training example can name and load its OBJ. Occupancy is still xyz-only.

---

## Step 6 — Surface envelope points (the first geometry experiment)

The simplest picture of a shape is a dense cloud of points sitting on its **skin** — the envelope of the mesh.

We sample those points from the OBJ, keep them in the same coordinate frame as the occupancy queries, and let a small encoder summarize the cloud into a shape vector. The occupancy head then answers “inside or outside?” using both the query point **and** that surface summary.

This is an R&D measurement, not the final architecture. The prototype later preferred triangle faces over surface clouds. We still run this step so we have our own number: did the shell help, sit still, or make training worse?

Compare on the **same** few primitives we used in Steps 3–5. This is not a holdout yet.

**Done when:** envelope clouds are a real batched input, and we have a recorded comparison against xyz-only.

---

## Step 7 — Talk about the mesh as triangles

A mesh is not only a point cloud. It is triangles, each with three corners and a facing direction (the normal).

This step builds that description and lines it up with the same bounding-box normalization we already use for query points. We attach it to the data pipeline but **do not yet train** on it. 
The occupancy head stays on the envelope signal from Step 6, so we can see that tokenization did not break training.

**Done when:** each mesh has a stable triangle description, aligned with occupancy queries. The face encoder is not trained yet.

---

## Step 8 — Let the occupancy head see faces

Now the intended geometry encoder comes in: a module that reads the triangle tokens, compresses them into a shape vector, and concatenates that vector with each query point.

Envelope sampling can stay around for comparison, but the trained head uses **faces**. We smoke the same small primitive set and compare to Step 6. If faces are worse, we write that down instead of quietly deleting the envelope path.

We still are not claiming “unseen OBJ.” We are claiming: xyz + mesh faces trains, logs, and keeps a best checkpoint.

**Done when:** occupancy is geometry-conditioned on faces, with a tracked run and a best checkpoint.

---

## Step 9 — Hold out whole shapes

Randomly hiding some *points* from a cone you already trained on can look like 99% accuracy and prove nothing. The model already knows that occupancy field.

The real test: train on some meshes, evaluate on a mesh that was **never** a training identity (for example train sphere and cube, hold out cone). Report accuracy,
 but take inside IoU / F1 more seriously — they describe the interior better.

A control xyz-only model trained only on the train shapes should struggle on the new mesh. If geometry does not beat that, we have learned something important.

**Done when:** we can answer, with numbers, whether envelope geometry and/or face geometry beat xyz-only on a held-out shape.

---

## What we are not doing in these nine steps

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
| Curves and checkpoints (from Step 4) | `runs/` |
| Phase 1 | [`work_plan_phase1.md`](work_plan_phase1.md) |

---

## Next action

**Approve or edit Step 1** (port Maya scatter scripts into this repo) before any prototype files are copied and before any occupancy code changes.
