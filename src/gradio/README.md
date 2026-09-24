# Occupancy Gradio demo

A public page that fills a mesh with the points the network labels **inside**. You bring an OBJ (or pick a sample), press **Run model**, and orbit the result. It is the same fill as the inspect viewer — not a second model.

This tool exists out of curiosity. Filling a volume is a geometry problem. A ray test, or the first-and-last-hit trick in **[scatteringNode](https://github.com/PerryGu/scatteringNode)** (a Maya C++ plugin from a production job, almost a decade ago), does it accurately and faster than a network. scatteringNet is the same job done with a trained head, on purpose — not a replacement for that method. If the math already works, keep the math. The longer version is in the root [README](../../README.md#project-inspiration).

The inspect tool stays the Three.js page in [`src/viewer`](../viewer/README.md). This folder is only the Gradio demo. Both load the **INSPECT** weights in [`docs/inspect_checkpoint.yaml`](../../docs/inspect_checkpoint.yaml).

Live: [huggingface.co/spaces/guyPerry/scatteringnet](https://huggingface.co/spaces/guyPerry/scatteringnet)

## Controls

Drop an OBJ on the 3D view, or use **Load OBJ** / a sample. Drag to orbit. There is no model picker and no NPZ / Truth / Errors view — that stays on the inspect page.

| Control | What it does |
|---|---|
| **Load OBJ** | File picker for an `.obj`. Same as dropping a file on the view. |
| **Density** | How tight the query lattice is (0 = coarse, 100 = dense, default 71 ≈ training spacing). **Run model** builds a new lattice at this setting. Cap **80,000** points. |
| **Inside cut** | After a run, a point is inside if its sigmoid *p* is at least this value (default 0.50). Lower counts more points as inside. Dragging re-cuts the last run — no extra forward pass. |
| **Mesh opacity** | Shell transparency (default 50%) so inside points show through. |
| **Dot size** | Occupancy point size in the view (1–24 px, default 8). Display only. |
| **Show outside points** | Draw the points the network labeled outside (off by default). |
| **Wireframe** | Crease-edge overlay on the mesh (not every triangle). |
| **Run model** | Fill the bounding box at the current **Density**, then classify. Needs an OBJ on the floor. |
| **Reset view** | Camera back to the start (45° / 70°, origin). Does not clear the mesh or the fill. |
| **Sample OBJ** | Holdout meshes (Obese, horse, Player, dog, Helix_bend, TorusX3_box). None of these were in the training catalog. |

## Open it locally

Conda env **scatteringNet**:

```text
pip install -e .
pip install -r src/gradio/requirements.txt
python src/gradio/app.py
```

Or double-click `open_gradio.bat` in this folder. Browser: `http://127.0.0.1:7860`.

Put `models/<run_id>/best.pt` in the repo. The demo uses the INSPECT run (`2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6`) when that file is there, otherwise the newest `best.pt`. CUDA if the box has it, otherwise CPU.

## Hugging Face Space

The live page is a slim copy of this demo plus occupancy infer — not the training repo. To refresh it, from the repo root:

```text
python space/push_space.py
```

Then put INSPECT `best.pt` in the Space Files UI (git ignores `*.pt`). [`space/`](../../space/README.md) is that ship kit, not a second app.

A free visitor shares Hugging Face’s ZeroGPU daily allowance. The fill itself is small; CPU is enough.

## Versus the inspect viewer

The lattice caps at **80,000** points (inspect allows 200,000). No envelope overlay, no recents, no file labels. **Inside cut** re-thresholds the last run from stored probabilities — same idea as the inspect slider, no extra forward pass.
