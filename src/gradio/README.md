# Occupancy Gradio demo

Thin Hugging Face Gradio UI for occupancy **fill**. The inspect tool is still the Three.js page in [`src/viewer`](../viewer/README.md) (`open_viewer.bat`). This folder does not replace that viewer and does not change occupancy training.

Upload an OBJ → mesh on an XZ **floor** of thin GridHelper-style lines plus RGB AxesHelper lines (Y-up). **Drop the OBJ on the 3D view** (or **Load OBJ**). **Run model** fills with occupancy points. **Mesh opacity** (default 50%) uses a GLTF blend material so inside points show through. **Wireframe** overlays crease edges on the solid shell (inspect-style, not every triangle). **Dot size** (default 8 px) scales occupancy points without remounting the camera. The 3D pane is a persistent Babylon canvas (`orbit.js`). Loading a sample, drop, **Load OBJ**, sliders, and **Run model** keep the orbit. **Reset view** returns to the start camera (45° / 70°, origin). Same Job B path as the helper (`obj_fill` + `infer_uploaded_obj`).

## Launch

Conda env **scatteringNet** (PyTorch already there):

```text
pip install -r src/gradio/requirements.txt
python src/gradio/app.py
```

Or double-click `open_gradio.bat` in this folder. Browser: `http://127.0.0.1:7860`. Sample meshes under `examples/` (cube, horse, Player, dog, Helix_bend, TorusX3_box) appear in the **Sample OBJ** row.

Put the Space checkpoint at `models/<run_id>/best.pt`. The demo loads `default_run_id()` (inspect run `2026-09-16_18-09-31_prim_extruded_nr45_knn24_n2048_n6` when that file exists, else the newest `best.pt`). There is no model picker.

## Hugging Face Space

Create a **Gradio** Space (not Static, not Docker) from **this whole repo**. Set the app file to `src/gradio/app.py`. Install from [`requirements.txt`](requirements.txt) in this folder (includes `torch`). The demo does **not** need Open3D or the training catalog.

**GPU is optional.** Occupancy infer already maps the head with `.to(device)`: CUDA if `torch.cuda.is_available()`, otherwise **CPU**. A CPU Basic Space is the intended free path (example cube is fine; a dense 80k lattice is slower, not broken). Paid GPU / **ZeroGPU** only if you want the inspect-like wait on big meshes.

Force CPU even on a GPU box:

```text
SCATTERINGNET_DEVICE=cpu
```

`SCATTERINGNET_DEVICE=cuda` uses CUDA only when it is actually available; otherwise it falls back to CPU (no crash).

**Weights.** Put `models/<run_id>/best.pt` on the Space (inspect id `2026-09-16_18-09-31_prim_extruded_nr45_knn24_n2048_n6` if you have it). Repo `.gitignore` skips `*.pt` — use Git LFS or the Space files UI; do not expect `git push` of this repo to upload the checkpoint. `config.yaml` `data_dir` can stay as a local path; Gradio infer does not require that folder.

This demo is not the Three.js inspect UI.

## Why this folder is named `gradio`

It sits next to `src/viewer` on purpose. It is **not** a Python package (no `__init__.py`) so `pip install -e .` does not ship a fake `gradio` library. Always `import gradio` from site-packages **before** putting `src/` on `sys.path`. Launch `app.py` as a script.

## Limits (vs the local viewer)

- Lattice cap **80,000** points (viewer 200,000). The GLB draws that same set (not a 25k subsample).
- No NPZ Truth / Errors, no envelope overlay, no recents / IndexedDB.
- Inside cut recuts stored sigmoid probs (no extra GPU pass), same idea as the inspect slider.
