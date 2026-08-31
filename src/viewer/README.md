# Occupancy viewer

Inspect trained occupancy models in the browser. This folder is the viewer only; occupancy train/infer modules under `src/*.py` are not modified.

The page is Three.js. A localhost helper (`serve.py`) lists `models/<run_id>/best.pt`, fetches OBJ files under `config.yaml` `data_dir`, fills an unlabeled AABB lattice, and runs inference (needs conda env **scatteringNet** with PyTorch).

Gradio (viewer Step 8) is skipped; use `open_viewer.bat`.

## How to open

**Double-click** `open_viewer.bat` (repo root or this folder). Close any old black console first, or the browser may still talk to a Python that has no `torch`.

Put checkpoints at `models/<run_id>/best.pt`. The Model list is that folder.

## Open / recents

- **Open** or drop an `.obj` or `.npz`
- Hover Open for the last **five** files (names in `localStorage`; bytes in IndexedDB)
- **Clear view** is at the bottom of that recents menu
- An NPZ with `mesh_path` auto-fetches that OBJ from `data_dir` (`GET /api/mesh`). Paths stay under `data_dir`; checkpoints stay under `models/`

## Inspect

- **Mesh / Wireframe** (left), **Inside / Outside** (right)
- **Opacity**, **Point size**, **Max points drawn** (display subsample only)
- Infer and Fill use every query point, not the draw cap

## Job A (NPZ)

Infer on the file’s `points` (the box is not resampled). **Truth** = file labels. **Run model** then **Prediction** / **Errors** (missed inside vs false inside).

## Job B (OBJ)

**Fill points** and **Density** sit on one row. Dragging density refills (short debounce); **Fill points** still works. The camera does not reset. Fill turns **Inside** and **Outside** on so the lattice is not hidden.

Then **Run model**. Envelope is sampled from the **uploaded** OBJ (`n_surface` from the checkpoint). No **Errors** view (no file labels). Until you run, fill points are unlabeled and show as outside.

Density slider 0–100 maps to lattice **spacing** (pad = spacing):

| Slider | Spacing | Meaning |
|---|---|---|
| 0 | 0.40 | coarse |
| 71 (default) | ≈ 0.15 | training occupancy default |
| 100 | 0.05 | dense |

Right = denser (smaller step). The helper caps the lattice at **200,000** points and coarsens spacing if needed.

## UI prefs

Checkboxes, sliders, density, and the selected model are saved to `src/viewer/ui_prefs.json` (helper `GET`/`POST /api/ui-prefs`). Missing file uses defaults; see `ui_prefs.example.json`. Gitignored. Truth / Prediction / Errors are not saved (they depend on the loaded file).

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
  obj_fill.py        unlabeled AABB lattice
  ui_prefs.py        ui_prefs.json
  js/                page modules (not occupancy Python)
  vendor/            Three.js
```
