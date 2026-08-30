# Phase 1 — work plan

Before we could ask a network to understand a mesh, we needed a simple pipeline that knows how to read a **single NPZ file**: points in space, each labeled inside or outside.
Phase 1 is that pipeline. We train a small network on xyz coordinates only, watch loss drop on one occupancy field, then run a thin inference path on the same file.

We knew from the start that this would **memorize** one shape rather than open a new OBJ. That limit was on purpose. The goal was to prove data → model → train → infer, 
one step at a time, without geometry encoders, Maya, or multi-file training. Those layers belong to Phase 2.

---

## How to read this plan

Work proceeded **one step at a time**. After each step we stopped for review. What shipped went in [`CHANGELOG.md`](../CHANGELOG.md). Early MVP smokes also left loss/accuracy in that changelog; the **current** rule is: train write-ups go in [`training_log.md`](training_log.md). Changelog is code and knobs only.

Progress is visible in:

- this file (closed history)
- [`CHANGELOG.md`](../CHANGELOG.md) (what shipped)
- [`training_log.md`](training_log.md) (catalog trains)
- [`work_plan_phase2.md`](work_plan_phase2.md) (what comes next)

---

## The ten steps at a glance

| Step | Theme | What we learn |
|---|---|---|
| 1 | Occupancy MLP | A small network can map a 3D point to “inside or outside” |
| 2 | Config | Hidden size, depth, seed, and data folder live in YAML; the GPU is detected at runtime |
| 3 | NPZ loader | We can read `points` and `labels` from one occupancy file |
| 4 | Normalize | We center and scale the cloud so the network sees a stable coordinate frame |
| 5 | Dataset | Points flow in batches the trainer can consume |
| 6 | Train one file | Loss drops and accuracy rises when we overfit a single NPZ |
| 7 | Metrics | Accuracy (and inside precision/recall) come from one shared helper |
| 8 | Infer | We reload the checkpoint and classify the same points |
| 9 | Train/val split | A random slice of *points* from the same file is held out — loop health, not a new mesh |
| 10 | Review | We write down what the MVP proved, and what it cannot do |

The path in one line:

**one NPZ → load points/labels → normalize → MLP(xyz) → train → infer those same points.**

---

## Step 1 — A small occupancy network

We add a tiny MLP: three numbers in (x, y, z), one number out (a logit for inside vs outside). No data loading yet. No training. Just a module we can call on dummy points.

**Done when:** the module runs, outputs one score per point, and a unit test passes.

---

## Step 2 — Put the knobs in one place

Hidden width, depth, random seed, and the path to the data folder live in `config.yaml`. Code loads that file and picks CUDA if it is there. We do not store the device in YAML — machines differ.

**Done when:** running config prints a resolved data folder and device.

---

## Step 3 — Read one occupancy file

An NPZ is a bag of arrays. For this phase we only need **points** (N×3) and **labels** (inside = 1, outside = 0). Other keys, including the path to the OBJ, are ignored on purpose. Geometry is Phase 2.

**Done when:** we can print how many points we loaded, how many are inside, and the bounding box of the cloud.

---

## Step 4 — Put the cloud in a canonical box

Raw coordinates depend on how the mesh was modeled. We subtract the center of the axis-aligned bounding box and divide by scale so points sit roughly in a unit range. The same center and scale must be stored with the checkpoint so inference does not invent a new frame.

**Done when:** normalized points sit in a stable range, and we can invert the map on a few rows.

---

## Step 5 — Batches the trainer can eat

The dataset applies that normalization once, then yields xyz and labels in batches. Train shuffling is on. Shapes are ordinary: a batch of points, a batch of labels.

**Done when:** one batch from the loader has the shapes we expect.

---

## Step 6 — Train until the file is memorized

We train on **all** points of one NPZ (no val split yet). The loss is binary cross-entropy on the logits. We print loss and accuracy each epoch and save a checkpoint with the weights plus the AABB used to normalize.

High accuracy here means the network fitted this occupancy field. It does **not** mean it understands a sphere in general.

**Done when:** loss decreases and train accuracy is clearly above chance on a simple primitive.

---

## Step 7 — One place for scores

Accuracy (and a little inside precision/recall) live in a small metrics helper so the train script does not invent its own counting.

**Done when:** the trainer prints metrics through that helper.

---

## Step 8 — Infer on the same file

Load the checkpoint, apply the stored center and scale, classify every point, print how many were predicted inside vs outside, and compare to the NPZ labels. Optionally write predicted labels next to the checkpoint.

**Done when:** inference on the train file reports high accuracy. Overfit is enough for this MVP.

---

## Step 9 — Hold out some points, not a new shape

A random fraction of **points** from the same NPZ is kept for validation (about 15%). We still have not seen a second mesh. Val accuracy is a sanity check that the loop reports a number besides train acc. It is not generalization.

**Done when:** val accuracy is reported. On an easy primitive it should still be high.

---

## Step 10 — Write down what we proved

A short review note: the occupancy loop works; an xyz MLP can fit one field; the next product step is geometry (Phase 2), not more of the same.

**Done when:** that note exists and Phase 1 is treated as closed.

---

## What we did not do in these ten steps

No voxel grids, FFT, or scattering filters. No loading the OBJ, no envelope points, no triangle faces. No training on many files, no held-out *shapes*, no experiment `runs/` folder. Those are Phase 2 (and some of them we may never need, if an earlier layer is already good enough).

Phase 1 is a teaching scaffold: one file, xyz only. It is not an interior-population product.

---

## How to follow progress

| You want… | Look at… |
|---|---|
| Phase 1 | this file |
| What actually shipped | [`CHANGELOG.md`](../CHANGELOG.md) |
| Phase 1 wrap-up | [`2026-08-24_14-09_mvp_completion_review.md`](2026-08-24_14-09_mvp_completion_review.md) |
| What comes next | [`work_plan_phase2.md`](work_plan_phase2.md) |

---

## Next action

Phase 1 is finished. Continue in **Phase 2**: approve or edit Step 1 of [`work_plan_phase2.md`](work_plan_phase2.md) (Maya scatter scripts) before any prototype files are copied.
