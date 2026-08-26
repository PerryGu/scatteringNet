# Phase 1 — wrap-up

This note closes Phase 1. It does not add code. It records what the occupancy MVP actually proved.

We set out to build a simple pipeline that can read **one NPZ file**, train a small network on xyz points, and classify those same points as inside or outside. We did that. We also confirmed the limit we accepted on purpose: the network memorizes one occupancy field. It cannot open a new mesh.

---

## What we proved

The data loop works. We load points and labels, put the cloud in a stable coordinate box, train, then infer with the same box we stored in the checkpoint. We never needed to open an OBJ.

The small xyz network can fit one field. On the test-set sphere (about 10,600 points), a 30-epoch CUDA run dropped loss from about 0.69 to about 0.33 and reached train accuracy around 0.89 (peak about 0.91). Inference on that **same** file with the saved checkpoint reported:

- accuracy **0.9212**
- predicted inside **5112**
- predicted outside **5549**

When we held out 15% of the *points* from a synthetic sphere (not a new shape), train accuracy was about 0.96 and val about 0.87: still high on an easy primitive, a bit below train, which is what we expected.

High accuracy on the train sphere is success for this scaffold. It is **not** evidence that we can populate a new interior.

---

## The path we built

**one NPZ → load points/labels → normalize → MLP(xyz) → train → infer those same points.**

Training uses Adam and binary cross-entropy on the raw scores. A random slice of points from the same file is held out for val — loop health, not a new mesh. Inference reloads the checkpoint, applies the stored center and scale (it must not invent a new box), then turns scores into inside/outside at 0.5.

The ten steps in [`work_plan_phase1.md`](work_plan_phase1.md) are all done. Config knobs (epochs, learning rate, checkpoint path, which NPZ, val fraction) live in `config.yaml`. The conda env is `scatteringNet`. At the end of Step 9 the unit suite was 28 tests OK.

---

## What we did not prove

No voxel grids, FFT, or scattering filters. No loading the OBJ, no points on the envelope, no triangle faces. No training on many files, no held-out *shapes*. The stub in `src/scattering_net.py` was left empty on purpose.

Those layers are Phase 2 — and we may stop on an earlier layer if it is already good enough, or if the next one does not help.

---

## What comes next

Phase 2. The occupancy question stays the same (is this point inside or outside?). The missing piece is a picture of the **mesh**, so the model is not a lookup table of one point cloud.

The Phase 2 plan is [`work_plan_phase2.md`](work_plan_phase2.md). Next action: approve or edit **Step 1** (Maya scatter scripts) before any prototype files are copied.

---

## Keep this in view

An xyz-only MLP can look excellent on **one** occupancy field and will fail on a different mesh. Phase 1 reproduced that lesson with a clean train/infer loop. The product we want — filling a new interior — starts only when geometry is in the loop, and even then it is a measurement, not a finished system.
