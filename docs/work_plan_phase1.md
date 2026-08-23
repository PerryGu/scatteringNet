# Phase 1 Work Plan — Occupancy MLP MVP

**Document type:** implementation work plan  

---

## 1. Goal

Learn the core occupancy loop with a lightweight MLP model:
- **Data:** NPZ files containing `points` (N, 3) and `labels` (N,).
- **Model:** `nn.Module` MLP mapping raw XYZ coordinates to a single occupancy logit.
- **Workflow:** Train on a single file until loss drops and accuracy increases, followed by a minimal inference path.

---

## 2. Explicitly out of scope

Do **not** add any of the following until a later plan says so:

- Voxelization, FFT, `build_filters` / `pad` / `unpad`, Kymatio  
- OBJ loading, envelope sampling, face tokens, normals  
- PointNet, mesh attention, shape embeddings  
- Multi-mesh generalization, combo datasets, helix floods  
- Near-surface loss weights, AMP, viewers, PLY galleries  

---

## 3. Architecture & Standards

- **Model:** `OccupancyMLP` with configurable hidden dimensions and depth (defaults: hidden=64, depth=4).
```
  input:  xyz   (B, 3) float32, canonical coords
  hidden: Linear(3 → H) → ReLU → … → Linear(H → H) → ReLU   (depth times)
  output: logits (B, 1)   # not probabilities; use BCEWithLogitsLoss
``` 
- **Normalization:** AABB center subtraction and scale division per NPZ, stored in checkpoints.
- **Loss:** `F.binary_cross_entropy_with_logits`.

---

## 4. Implementation Steps

- **Step 1:** `OccupancyMLP` module only (`src/occupancy_mlp.py`).
- **Step 2:** Hybrid config (`config.yaml` + `src/config.py`).
- **Step 3:** NPZ loader (`src/data_npz.py`).
- **Step 4:** Coordinate normalization (`src/normalize.py`).
- **Step 5:** Dataset & DataLoader (`src/dataset.py`).
- **Step 6:** Train loop for a single NPZ (`src/train_one_npz.py`).
- **Step 7:** Metrics helper (`src/metrics.py`).
- **Step 8:** Basic inference (`src/infer_one_npz.py`).
- **Step 9:** Train/val point split.
- **Step 10:** Review and wrap-up.