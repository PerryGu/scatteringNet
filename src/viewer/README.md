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
- **Opacity**, **Point size**, **Max points drawn** (display subsample only). **Point size** scales occupancy dots and envelope (purple) dots, not the Faces triangles
- **Inside cut** (0.00–1.00, default 0.50): after **Run model**, inside if sigmoid *p* ≥ this value. Lower = more orange. Dragging re-cuts the last Run in the browser (no extra GPU pass). Status acc/IoU follow this cut (training metrics stay at 0.5)
- Infer and Fill use every query point, not the draw cap

## Job A (NPZ)

Infer on the file’s `points` (the box is not resampled). **Truth** = file labels. **Run model** then **Prediction** / **Errors** (missed inside vs false inside). Envelope and face-token checkpoints both work; the helper rebuilds the tokens that checkpoint was trained with from the NPZ `mesh_path` OBJ.

## Job B (OBJ)

The lower panel is titled **Create dots**. Three rows, top to bottom:

1. **Envelope** — purple overlay. **Count** is the total (256–4096, default 1024). **Mix** (0–100, default 100) splits that count: left **Faces** = area-weighted darts on triangles (old envelope); right **Edges** = on/near sharp creases. Status reports how many went to each. Overlay Mix is display-only. **Run model** rebuilds the envelope with the **checkpoint** ``envelope_mix`` (missing key = 0, all face-area).
2. **Faces** — teal triangles + cyan normal ticks. These are the occupancy mesh-encoder tokens: largest faces by area, then tiled if the mesh has fewer triangles than Count. Count 64–1024, default 256. Click to show/hide; drag **Count** to refresh while shown. The status line reports token count, unique vs tiled, and mesh triangle count. Envelope and Faces can stay on together.
3. **Fill points** + **Density** — unlabeled AABB lattice. OBJ-only. Dragging density refills (short debounce); **Fill points** still works. The camera does not reset. Fill turns **Inside** and **Outside** on so the lattice is not hidden.

Envelope and Faces also work when an NPZ has loaded its OBJ (not only a direct OBJ open).

Then **Run model**. Shape tokens come from the **uploaded** OBJ, matching the selected checkpoint: envelope cloud if it was trained with ``shape_encoder: surface``, face tokens if ``mesh``. The Model list labels each run ``(envelope)`` or ``(faces)``. No **Errors** view (no file labels). Until you run, fill points are unlabeled and show as outside.

Density slider 0–100 maps to lattice **spacing** (pad = spacing):

| Slider | Spacing | Meaning |
|---|---|---|
| 0 | 0.40 | coarse |
| 71 (default) | ≈ 0.15 | training occupancy default |
| 100 | 0.05 | dense |

Right = denser (smaller step). The helper caps the lattice at **200,000** points and coarsens spacing if needed.

## UI prefs

Checkboxes, sliders, density, **Envelope count**, **Envelope mix**, **Faces count**, **Inside cut**, and the selected model are saved to `src/viewer/ui_prefs.json` (helper `GET`/`POST /api/ui-prefs`). Missing file uses defaults; see `ui_prefs.example.json`. Gitignored. Truth / Prediction / Errors are not saved (they depend on the loaded file).

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
  faces_job.py       triangle + normal overlay for mesh tokens
  obj_fill.py        unlabeled AABB lattice
  ui_prefs.py        ui_prefs.json
  js/                page modules (not occupancy Python)
  vendor/            Three.js
```
