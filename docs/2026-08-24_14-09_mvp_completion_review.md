# Occupancy MLP MVP — Step 10 review

**Written:** 2026-08-24 14:09  
**Plan:** `docs/work_plan_phase1.md`  
**Environment:** conda `scatteringNet`, Python 3.10, PyTorch 2.5.1, CUDA 12.1

This note closes the occupancy MLP MVP. It does not add product features. No source, tests, or config were changed while writing it.

## What the MVP proved

### The data loop works

NPZ files with `points` `(N, 3)` and `labels` `(N,)` load, AABB-normalize, and batch as `xyz (B, 3)` and `y (B, 1)`. Training uses BCE-with-logits on raw logits. Inference reloads the checkpoint, applies the stored `center` / `scale`, and thresholds `sigmoid(logit)` at `0.5`. The same occupancy query file can be trained, held out, and classified without opening an OBJ.

### An xyz-only MLP fits one occupancy field

`OccupancyMLP` (`hidden=64`, `depth=4`) maps canonical XYZ to one logit. On the `dataset_test` sphere (`N=10661`) a 30-epoch CUDA overfit dropped loss from ~0.693 to ~0.329 and reached train accuracy ~0.89 (peak ~0.914). Inference on that same file with `models/one_npz.pt` reported:

- **accuracy:** 0.9212
- **pred_inside:** 5112
- **pred_outside:** 5549

A 15% point hold-out on a synthetic sphere reported `train_acc=0.9633` and `val_acc=0.8684`: high on an easy primitive, below train, as the plan expected.

### This is a teaching scaffold, not the interior-population product

The net memorizes one field. It cannot open a new mesh. It will fail on a different geometry. That limit was accepted on purpose.

## Loop that now exists

```text
NPZ  →  load points/labels  →  AABB center/scale  →  OccupancyMLP(xyz)  →  logit
         train: Adam + BCEWithLogitsLoss, val split on points from the same file
         infer: eval, no_grad, sigmoid, threshold 0.5, write pred_labels
```

Checkpoint payload: `kind="occupancy_mlp"`, `state_dict`, `center`, `scale`, `hidden`, `depth`. Inference must reuse stored `center` / `scale`; it must not recompute AABB.

## Steps completed (1–9)

1. **OccupancyMLP** — `(B, 3)` → `(B, 1)` logits. No softmax in `forward`.
2. **Hybrid config** — knobs in `config.yaml`; device at runtime.
3. **NPZ loader** — keys `points` and `labels` only; float32 `{0, 1}` labels.
4. **AABB normalize** — center = midpoint, scale = max half-extent.
5. **Dataset / DataLoader** — `xyz (3,)` and `y (1,)`; collate `(B, 3)` and `(B, 1)`.
6. **Train one NPZ** — save `models/one_npz.pt`.
7. **Metrics helper** — accuracy plus inside precision / recall.
8. **Infer one NPZ** — print counts; write `models/one_npz_pred.npz`.
9. **Train/val split** — random 15% point hold-out (same NPZ); report `val_acc` each epoch.

Later knobs (`epochs`, `lr`, `checkpoint_path`, `sample_npz`, `val_fraction`) live in `config.yaml`. The project conda env is `scatteringNet` (not `scatteringNet_v2`). The full unit suite was 28 tests OK at the end of Step 9.

## Explicitly not proven (out of this plan)

- No voxelization, FFT, `build_filters` / `pad` / `unpad`, or Kymatio
- No OBJ loading, envelope sampling, face tokens, or normals
- No PointNet, mesh attention, or shape embeddings
- No multi-mesh generalization, combo datasets, or helix floods
- `src/scattering_net.py` was left as a stub on purpose

## Next product step (not in this plan)

A **geometry encoder**.

The occupancy head can stay (`xyz` + embeddings → inside/outside). The missing piece is a representation of the mesh so the model is not a lookup table of one point cloud. That encoder is the next product design, not a continuation of this MVP.

## Honest limitation (keep visible)

v1 stage 2a already showed: an xyz-only MLP can reach high accuracy on **one** occupancy field and will **fail** on a different mesh. This MVP reproduced that lesson with a clean PyTorch loop. High accuracy on the train sphere is success for the scaffold, not evidence that interior population is solved.
