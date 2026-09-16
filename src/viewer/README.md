# Occupancy viewer

Inspect trained occupancy models in the browser. This folder is the viewer only; occupancy train/infer modules under `src/*.py` are not modified.

The page is Three.js. A localhost helper (`serve.py`) lists `models/<run_id>/best.pt`, fetches OBJ files under `config.yaml` `data_dir`, fills an unlabeled AABB lattice, and runs inference (needs conda env **scatteringNet** with PyTorch). The inspect color in this page is display only. The model labels points; another host (for example a Maya plugin) can draw those labels however it wants.

Gradio (viewer Step 8) is skipped; use `open_viewer.bat`.

## How to open

**Double-click** `open_viewer.bat` (repo root or this folder). Close any old black console first, or the browser may still talk to a Python that has no `torch`.

Put checkpoints at `models/<run_id>/best.pt`. The Model list is that folder.

## The page

![Occupancy viewer: cone mesh in the 3D view, controls on the left](../../docs/media/2026-09-16_viewer_main.png)

The 3D view is the right-hand pane: a dark floor grid, RGB world axes, and the loaded mesh. Drag to orbit. This still is an OBJ after **Open** (a cone, mesh + wireframe, 24% opacity). **Inside** / **Outside** are greyed because there is no lattice yet — **Fill points**, then **Run model**. The status line under the sliders repeats that. You can also drop an OBJ or NPZ onto the view (hint at the bottom).

The left column is the whole UI. Top to bottom:

| Control | What it does |
|---|---|
| **Open** | File picker, or hover for recents (see below). |
| **Mesh** / **Wireframe** | Show the OBJ. Opacity is mesh only. |
| **Inside** / **Outside** | Show classified query points after Fill / Run. Off until there is a cloud. |
| **Point size** / **Max points drawn** | Display only. Infer and Fill still use every query; the draw cap is a subsample. Point size also scales envelope dots. |
| **Envelope** | Purple overlay of skin samples. **Count** is how many (256–4096, default 1024). **Mix** splits Faces (area-weighted) vs Edges (creases). Overlay Mix / Count do **not** change infer. |
| **Fill points** + **Density** | Unlabeled AABB lattice (OBJ). Dragging density refills; the camera does not reset. Fill turns Inside and Outside on so the lattice is visible. |
| **Select Model** | Checkpoints under `models/` (see below). |
| **Run model** | Forward pass. Disabled until an NPZ is loaded, or an OBJ has been filled. |
| **Inside cut** | After a Run, inside if sigmoid *p* ≥ this value (default 0.50). Lower counts more points as inside. Dragging re-cuts the last Run in the browser (no extra GPU pass). Status acc/IoU follow this cut; training metrics stay at 0.5. |
| **Truth** / **Prediction** / **Errors** | NPZ views. On an OBJ they stay idle (no file labels). |

## Open / recents

![Open menu: recent files and a picker for OBJ or NPZ](../../docs/media/2026-09-16_viewer_open.png)

- Click **Open** for a picker, or drop an `.obj` / `.npz` on the page.
- Hover **Open** for the last **ten** files (names in `localStorage`; bytes in IndexedDB). **Open an OBJ or NPZ** is the same picker.
- **Clear view** is at the bottom of that recents menu.
- An NPZ with `mesh_path` auto-fetches that OBJ from `data_dir` (`GET /api/mesh`). Paths stay under `data_dir`; checkpoints stay under `models/`.

## Select Model

![Select Model list with envelope checkpoints and user notes](../../docs/media/2026-09-16_viewer_models.png)

This is a custom list (a native `<select>` cannot take a right-click). Each row is `models/<run_id>/best.pt`. Envelope heads are tagged `(envelope)`. Face-token occupancy checkpoints are no longer loaded.

Right-click a row to type a suffix (`*` or a short note) after the existing label — in the still, `_the_best_16` and `sagemaker_the_best`. Stored in `ui_prefs.json` as `model_labels`. Folder names under `models/` are not renamed.

## Job A (NPZ)

Infer on the file’s `points` (the box is not resampled). **Truth** = file labels. **Run model** then **Prediction** / **Errors** (missed inside vs false inside). Envelope checkpoints rebuild the surface cloud from the NPZ `mesh_path` OBJ.

## Job B (OBJ)

Typical inspect path: **Open** an OBJ → **Fill points** → pick a model → **Run model**.

**Create dots** has two rows:

1. **Envelope** — overlay only. **Run model** rebuilds the envelope with the **checkpoint** `envelope_mix` (missing key = 0, all face-area).
2. **Fill points** + **Density** — unlabeled AABB lattice. OBJ-only. Envelope also works when an NPZ has loaded its OBJ.

Then **Run model**. Envelope checkpoints rebuild the surface cloud from the **uploaded** OBJ. No **Errors** view (no file labels). Until you run, fill points are unlabeled and show as outside.

Density slider 0–100 maps to lattice **spacing** (pad = spacing):

| Slider | Spacing | Meaning |
|---|---|---|
| 0 | 0.40 | coarse |
| 71 (default) | ≈ 0.15 | training occupancy default |
| 100 | 0.05 | dense |

Right = denser (smaller step). The helper caps the lattice at **200,000** points and coarsens spacing if needed.

## UI prefs

Checkboxes, sliders, density, **Envelope count**, **Envelope mix**, **Inside cut**, the selected model, and optional **model list notes** are saved to `src/viewer/ui_prefs.json` (helper `GET`/`POST /api/ui-prefs`). Missing file uses defaults; see `ui_prefs.example.json`. Gitignored. Truth / Prediction / Errors are not saved (they depend on the loaded file).

## Tests

From the repo root, conda env **scatteringNet**:

```text
python tests/test_viewer_server.py
```

Do not use `python -m unittest tests.test_viewer_server` (`tests` is not a package).

## Layout

```text
src/viewer/
  serve.py           localhost static + API
  mesh_access.py     OBJ path under data_dir
  model_access.py    models/<run>/best.pt
  infer_job.py       Job A / Job B forward
  envelope_job.py    crease-hugging surface samples for the Envelope overlay
  obj_fill.py        unlabeled AABB lattice
  ui_prefs.py        ui_prefs.json
  js/                page modules (not occupancy Python)
  vendor/            Three.js
```
