# Changelog

Completed work for the occupancy MLP MVP.  

Format: newest entries at the top. Headings: ``## YYYY-MM-DD HH:00`` (date and hour; no minutes).
Catalog trains get a short ``Train:`` note (run id, score, wall). The full write-up is linked from each note to [docs/training_log.md](docs/training_log.md).


## 2026-09-25 13:00 — Live YAML: 2-epoch max-pool smoke

``config.yaml`` ``epochs: 2`` for a SageMaker pipeline check of ``knn_pool: max``. Set back to 20 after the smoke. Occupancy code unchanged.

## 2026-09-25 10:00 — Train: knn24 n2048 concat pool

Train: ``2026-09-24_13-05-13_prim_extruded_nr45_knn24_n2048_n6_concat`` — val IoU **0.956** · acc **0.993** @ 14, wall **9h 31m** (A10G). Local ``knn_pool: concat``. Viewer Fill vs INSPECT max-pool is slightly worse (more leak on thin OOD parts). Write-up: [docs/training_log.md](docs/training_log.md#2026-09-25-1000--knn24-n2048-concat-pool).

## 2026-09-24 15:00 — Occupancy: ``knn_pool: concat``

Local k-NN can flatten the ``k`` neighbor rows (``LocalConcatEncoder``) instead of max-pooling them, so surround vs one-sided is still in ``z_local``. YAML ``knn_pool`` is ``max`` (omit / older ``best.pt`` / INSPECT) or ``concat``. Do not resume a max-pool ``best.pt`` into concat. Live ``config.yaml`` is the next concat A/B (unweighted BCE). Occupancy INSPECT weights are unchanged.

## 2026-09-23 18:00 — Docs: Gradio link under Video Showcase

Root [`README.md`](README.md) links the Hugging Face Space under Video Showcase (before Image Gallery). Occupancy code unchanged.

## 2026-09-23 18:00 — Docs: Gradio README controls + inspiration

[`src/gradio/README.md`](src/gradio/README.md) adds the curiosity / scatteringNode note and a control table for every button and slider. Occupancy code unchanged.

## 2026-09-23 17:00 — Docs: rewrite Gradio README

[`src/gradio/README.md`](src/gradio/README.md) is now a short description of the demo (what it does, local launch, Space, vs inspect). Occupancy code unchanged.

## 2026-09-23 17:00 — Docs: drop HF YAML from root README

Root [`README.md`](README.md) no longer starts with Space frontmatter (`sdk: gradio`, …). That card belongs on [`space/README.md`](space/README.md) only.


## 2026-09-23 15:00 — Gradio: remove CPU header note

Dropped the “runs on CPU, not GPU” line. Hardware is not customarily advertised in the hero text.


## 2026-09-23 15:00 — Space: do not pin Gradio

HF Gradio Spaces already install Gradio 6. Pinning ``gradio==4.44.1`` made pip fail (6 vs 4.44). Space ``requirements.txt`` now lists only torch / numpy / pyyaml / trimesh.

## 2026-09-23 14:00 — Space: Gradio 6 launch on Hugging Face

Gradio 6 rejects ``js`` / ``css`` on ``Blocks()`` — those go through a wrapped ``demo.launch()`` so orbit JS still loads. HF looks for a module-level ``demo`` and runs ``app.py`` as a script: skip ``launch()`` when ``SPACE_ID`` is set and the process exits 0 (runtime error); a second ``launch()`` double-starts the app (invalid file descriptor). The working shape is ``demo = build_demo()`` at module scope, one ``launch()`` on ``0.0.0.0`` when ``SPACE_ID`` or ``PORT`` is set (not 127.0.0.1).

## 2026-09-23 13:00 — Space: slim Gradio push and HF pip

``python space/push_space.py`` assembles Gradio + occupancy infer (no ``docs/media``, SageMaker, or ``*.pt``) and force-pushes that tree to ``guyPerry/scatteringnet``. Upload INSPECT ``best.pt`` in the Space Files UI. HF ``pip -r`` runs from ``/tmp``, so ``-e .`` and ``-e /home/user/app`` both fail (no ``pyproject.toml`` on that cwd). The slim tree mirrors occupancy modules as ``scatteringnet/`` and ``app.py`` puts the Space root on ``sys.path``.

## 2026-09-23 13:00 — Space: root requirements + README frontmatter

Hugging Face Space ``guyPerry/scatteringnet``: root ``requirements.txt`` (``-e .``) and README ``app_file: src/gradio/app.py``. Occupancy train / infer math is unchanged.

## 2026-09-23 12:00 — Gradio: sample row catalog note

A one-line Markdown above **Sample OBJ** states those meshes were not in the training catalog.


## 2026-09-23 10:00 — Train: knn24 n2048 pos_weight auto

``2026-09-22_10-08-18_prim_extruded_nr45_knn24_n2048_n6_pw`` — val IoU 0.971 @ 14, 9h 37m (G5). ``pos_weight: auto`` (4.280) vs unweighted knn24 / 2048. Viewer Fill is worse than ``2026-09-14_07-43-34_…_n6``. Write-up: [docs/training_log.md](docs/training_log.md#2026-09-23-1000--knn24-n2048-pos_weight-auto).

## 2026-09-22 18:00 — Gradio: Sample OBJ uses Obese instead of cube

The sample row is ``Obese.obj``, horse, Player, dog, Helix_bend, TorusX3_box. ``cube.obj`` stays on disk for tests.

## 2026-09-22 18:00 — Gradio: start camera further back

Default / **Reset view** radius is 60 so a full sample mesh (horse / dog / torus) fits with headroom. Loads still keep the current orbit.

## 2026-09-22 18:00 — Gradio: Sample OBJ keeps Wireframe

Clicking a sample now passes the live Wireframe / opacity / dot-size into ``accept_obj_ui``. A checked Wireframe no longer resets to off on that load.

## 2026-09-22 18:00 — Gradio: large OBJ shells draw on Load

``occupancy_figure`` no longer drops a mesh above 80k vertices (Human2.obj showed the empty-scene error on Load, then only fill points after **Run model**). An empty draw list writes the placeholder GLB instead of crashing. Occupancy training is unchanged.

## 2026-09-22 18:00 — Gradio: larger fixed floor

The world grid is 32 units across (axes 4) so it sits under typical uploaded meshes. Camera and load behavior are unchanged.

## 2026-09-22 17:00 — Gradio: floor is fixed; loads do not move the camera

The XZ grid and RGB axes live in ``orbit.js`` and stay at world origin. A new OBJ only swaps the mesh — the GLB no longer ships a resized floor (that was the grid jump). The camera is frozen across ImportMeshAsync. **Reset view** is still 45° / 70° / radius 7. Occupancy training is unchanged.


## 2026-09-22 12:00 — Train: optional BCE ``pos_weight``

``BCEWithLogitsLoss`` can weight inside queries. YAML ``pos_weight: auto`` uses n_outside / n_inside on the train split; a float is used as-is; omit keeps unweighted BCE. OccupancyEncoder (k-NN max-pool) is unchanged.


## 2026-09-22 09:00 — CI: tests do not require E: catalog, CUDA pin, or libGL

GitHub ``unittest`` no longer errors when live ``data_dir`` is missing, when ``pin_memory`` has no GPU, or when Open3D cannot load ``libGL``. ``load_config()`` still refuses a missing catalog for train. Occupancy math, viewer, Gradio, and SageMaker entry are unchanged.


## 2026-09-21 18:00 — Config: local smoke knn16 / n1024 / 10 epochs

Live ``config.yaml`` is a shorter local catalog train: ``n_surface: 1024``, ``knn_k: 16``, ``epochs: 10``, ``run_name: prim_extruded_nr45_knn16_n1024_n6``. Catalog globs, unweighted BCE, and ``hidden`` / ``depth`` are unchanged. The SageMaker clone ``scatteringnet-n6-20260921164350`` already has its own snapshot and is not affected.


## 2026-09-21 16:00 — Package: ``import scatteringnet``; unittest on push

Occupancy modules install as ``scatteringnet.*`` (``pip install -e .``). Tests, viewer, Gradio pipeline, and SageMaker import that package instead of inserting ``src/`` on ``sys.path``. ``src/gradio`` stays a launch folder so it does not shadow pip Gradio. GitHub Actions runs ``unittest discover``. Train math, viewer Fill, and Gradio draw caps are unchanged.


## 2026-09-21 15:00 — Train: one NPZ in RAM, one envelope per mesh

Catalog construct indexes files and joins OBJs without keeping every lattice in memory. Lattice and jitter of the same OBJ still train as two files; they share one envelope tensor and the mesh AABB. Train/val load that NPZ for the inner loop and drop the query tensors afterward. OccupancyEncoder, viewer, and Gradio are unchanged.


## 2026-09-21 14:00 — Gradio: draw the full 80k lattice

The occupancy GLB used to keep at most 25,000 dots while Fill classified 80,000. Draw cap is now 80,000 so the demo shows the lattice that already ran. Inference, CPU/CUDA, and the Three.js inspect viewer are unchanged.


## 2026-09-21 14:00 — Infer: cache the occupancy head; CPU or CUDA

Viewer and Gradio **Run model** keep one ``best.pt`` in process (same path + mtime + device). Warmup fills that slot so the first inspect click is not a second load. ``get_device()`` still prefers CUDA, uses CPU when there is no GPU, and honors ``SCATTERINGNET_DEVICE=cpu`` (Hugging Face CPU Spaces). Gradio ``requirements.txt`` lists torch / trimesh so a Space can install without the conda CUDA stack.


## 2026-09-21 13:00 — Fill: checkpoint seed and OBJ AABB

New ``best.pt`` files store ``seed``. Envelope resample uses that value; older checkpoints default to ``1``, not live YAML. Uploaded OBJ Fill (Job B) uses this mesh's AABB and no longer matches catalog ``parts`` by filename (``cube.obj``). Catalog NPZ infer still uses the stored box.


## 2026-09-21 09:00 — Gradio: loading a file keeps the orbit again

Sample / drop / **Load OBJ** was snapping the camera back to the start pose on every click — that undid the earlier “keep the orbit when the mesh changes” fix. Loading a file keeps the current camera again, the same way **Run model** does. **Reset view** is still what returns to 45° / 70°. Occupancy training and the Three.js inspect viewer were left unchanged.


## 2026-09-20 19:00 — Gradio: sample load pins the start camera

Clicking a **Sample OBJ** (or dropping / **Load OBJ**) keeps the start camera — 45° / 70°, looking at the origin, same distance as an empty view. The viewer no longer zooms to each mesh. **Run model** and the sliders still keep your orbit. **Reset view** returns to that same start camera. Occupancy training and the Three.js inspect viewer were left unchanged.


## 2026-09-20 17:00 — Gradio: more sample meshes

The **Sample OBJ** row now includes ``horse.obj``, ``Player.obj``, ``dog.obj``, ``Helix_bend.obj``, and ``TorusX3_box.obj`` as well as the cube, so you can try a few shapes without hunting files. Copies live in [`src/gradio/examples/`](src/gradio/README.md). Occupancy training and the Three.js inspect viewer were left unchanged.


## 2026-09-20 16:00 — Gradio: how the 3D view looks

The scene is meant to feel like the inspect tool: mid-gray background, a thin floor grid under the mesh, and a small red / green / blue axis gizmo so “up” is obvious. The camera looks slightly down and to the side, the same framing as the original viewer.

The mesh starts at **50% opacity** so the inside points show through. Those points are **orange**, matching the inspect tool (an earlier yellow look was only a lighting wash). **Dot size** (1–24, default 8) changes only the occupancy dots — not the floor or the axes (those briefly turned into dots by mistake and were fixed).

**Wireframe** draws the sharp crease edges of the solid, the way the original viewer does, instead of outlining every triangle of the surface. You can also show outside points if you want them.


## 2026-09-20 11:00 — Gradio: drop a file on the view

You load a mesh by dropping an OBJ onto the 3D window, or by clicking **Load OBJ**. A hint at the top of the view says “Drop an OBJ file here.” A second file replaces the first. The status line shows the current file name.

The old left-side file box was removed because long names spilled across the header and sat behind the button — even before any file was chosen. **Load OBJ** is a normal button that opens the system file picker. Nothing is drawn in that spot until you have actually picked a mesh.


## 2026-09-20 09:00 — Gradio: the camera stays put while you work

Early versions rebuilt the whole 3D scene every time you moved a slider or pressed **Run model**. That flashed gray and snapped the camera back to the start — it felt like the page was reloading. The first 3D plot also put the floor in the wrong place. A later in-page viewer went blank because the host page strips scripts from embedded HTML.

The pane is now one 3D canvas that stays on the page for the whole session. Changing density, inside cut, opacity, dot size, or wireframe — or running the model — updates the mesh and points **without** resetting your orbit. **Reset view** (or loading a new OBJ) still frames the shape from the usual start angle.


## 2026-09-18 10:00 — Gradio: a public interactive demo

To give this project an interactive view that anyone can open in a browser, I decided to put the trained model on **Hugging Face** and wrap it in a **Gradio** app. Hugging Face is where the checkpoint lives and can be shared; Gradio is the simple web page around it — upload a file, move sliders, press a button — without asking people to install the local inspect tool or run training.

That page lives in ``src/gradio/``, next to the existing inspect viewer. How to run it and what it does not cover are in the [Gradio README](src/gradio/README.md). It is a **thin public demo**, not a copy of that viewer: you bring a 3D mesh (OBJ), press **Run model**, and the network fills the solid with points it judges to be **inside** the shape. The fill-and-classify work is the same path the project already uses. A short line at the top of the page says what the tool does.

Open it locally with ``python src/gradio/app.py`` or by double-clicking ``open_gradio.bat``. An example cube is included. The Space will ship **one** trained model, so there is no model-picker list — **Run model** always uses that default checkpoint. Occupancy training and the Three.js inspect viewer were left unchanged.


## 2026-09-17 17:00 — Docs: README YouTube link and Status wrap-up

[`README.md`](README.md) **Video Showcase** links to [occupancy fill with a neural net](https://www.youtube.com/watch?v=vU45O0Mu0o4). **Status** is the wrap-up (leftover knobs named once; no separate Roadmap). Occupancy code unchanged.

## 2026-09-17 10:00 — Train: prim_extruded_nr45_knn24_n2048_n6 (area-only)

Catalog train ``2026-09-16_18-09-31_prim_extruded_nr45_knn24_n2048_n6`` (from scratch on ``ml.g5.xlarge``). Same knn24 n2048 catalog as mix-75 inspect, area-weighted envelope (no Mix 75).
``best.pt`` epoch 20, val_iou 0.974 / acc 0.996, wall 9h 24m. Val matches mix-75 (0.9737). Fill: slight extrude improvement; organics / CAD leftovers the same class.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-17-1000--knn24-n2048-area-only-drop-mix-75).


## 2026-09-16 19:00 — Envelope area-only; viewer filename and lists

Dropped crease/fold sampling. Envelope is face-area darts; YAML ``envelope_mix``, the Mix slider, and mix keys on new ``best.pt`` / run snapshots are gone. Infer ignores leftover mix on old checkpoints. Historical ``runs/*/`` and ``docs/training_log.md`` unchanged.

**Optimization:** Mix 75 packed extra envelope dots onto folds so sharp corners would not be starved. It did not help the visible result (extrudes looked better without it). That fold hunt was the **Run model** lag (~5.2 s envelope on a dense human vs ~0.00–0.06 s without it). The Fill gain that mattered is face normals on each skin dot (``envelope_dim=6``), not crease mix.

Viewer: opened OBJ/NPZ name in the title bar (selectable) and on the counts row. Right-click model notes no longer shove the left column off-screen. Model and recents menus show the full name (may overlap the 3D view); the closed Select Model button still ellipsizes.

## 2026-09-16 17:00 — Viewer: NPZ Clear points keeps the file job

**Clear points** on an NPZ only hides the cloud; **Run model** still classifies that file’s points. Wiping ``npzState`` left the helper OBJ text, so Run filled the previous Job B lattice. Occupancy train loop unchanged.

## 2026-09-16 16:00 — Docs: viewer README status still

[`src/viewer/README.md`](src/viewer/README.md) has the Prediction status table still, one-click **Run model** fill, and launch warmup. PNG in [`docs/media/`](docs/media/). Occupancy code unchanged.

## 2026-09-16 15:00 — Viewer: status table

After **Run model**, the left panel is a colored key/value table: inside/outside counts plus five times (**Fill points**, **Envelope**, **Load to GPU**, **Run model**, **Total**), each with a short gloss. No Mix row; Inside cut stays on the slider. Occupancy train loop unchanged.

## 2026-09-16 14:00 — Viewer: Windows JS MIME and module load

Helper sends ``.js`` as ``text/javascript`` on Windows (URL and ``F:\\...\\file.js`` paths), loads ``main.js`` as a ``type=module`` tag, and serves scripts on threads so Chrome runs the 3D page. Occupancy train loop unchanged.

## 2026-09-16 13:00 — Viewer: helper GPU warmup at start

``serve.py`` now loads torch/CUDA, a tiny fill, and one dummy infer on the newest ``best.pt`` before the browser opens. Occupancy train loop unchanged.

## 2026-09-16 12:00 — Viewer: Run model fill, Clear points, mix 0

OBJ **Run model** builds the AABB fill at the current Density, then infers (**Fill points** remains an optional preview). **Clear points** drops fill and envelope, keeps the mesh. Infer uses an area-only envelope (``envelope_mix: 0``) even if ``best.pt`` was mix 75; overlay Mix is unchanged. Occupancy train loop unchanged.

## 2026-09-16 10:00 — Docs: README collapsible project tree

[`README.md`](README.md) Project Structure is a closed ``<details>`` block (click ``scatteringNet/`` to expand). Occupancy code unchanged.

## 2026-09-16 10:00 — Docs: viewer README screenshots

[`src/viewer/README.md`](src/viewer/README.md) now has three UI stills (main view, Open recents, Select Model) and maps the left-column buttons to those shots. PNGs in [`docs/media/`](docs/media/). Occupancy code unchanged.

## 2026-09-16 09:00 — Docs: README Fill wording

[`README.md`](README.md) no longer treats inspect-viewer orange as the Fill. Occupancy code unchanged.

## 2026-09-15 10:00 — Viewer: model list notes

The Select Model control is a custom list (native ``<select>`` rows cannot take a right-click). Right-click a row to type a suffix (``*`` or a short note) after the existing label. Stored in ``ui_prefs.json`` as ``model_labels``; ``models/<run_id>/`` is not renamed. Occupancy train loop unchanged.

## 2026-09-15 09:00 — Train: prim_nr45_knn16_n6_nosmooth

Catalog train ``2026-09-14_19-00-37_prim_nr45_knn16_n6_nosmooth`` (from scratch on ``ml.g5.xlarge``). Same n6 head, catalog without smooth ``extruded_*``.
``best.pt`` epoch 17, val_iou 0.978 / acc 0.996, wall 3h 57m. Val is a different split (1077 meshes vs 1227). Fill vs ``12-18-28_…_n6``: organics still fill; smooth ``extruded_*`` optional (tiny extra volume on the older catalog).
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-15-0900--prim_nr45_knn16_n6_nosmooth-drop-smooth-extruded_).


## 2026-09-14 21:00 — Train: prim_extruded_nr45_knn24_n2048_n6

Catalog train ``2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6`` (from scratch on ``ml.g5.xlarge``). Same knn24 n6 recipe, ``n_surface: 2048``.
``best.pt`` epoch 20, val_iou 0.974 / acc 0.996, wall 9h 29m. Val +0.006 vs knn24-1024 (0.967); Fill not judged.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-14-2100--knn24-n2048-envelope-2048).

## 2026-09-14 09:00 — Train: prim_extruded_nr45_knn24_n6

Catalog train ``2026-09-13_19-20-50_prim_extruded_nr45_knn24_n6`` (from scratch on ``ml.g5.xlarge``). Same catalog as n6, ``knn_k: 24``.
``best.pt`` epoch 20, val_iou 0.967 / acc 0.995, wall 5h 01m. Val matches G5 n6 (0.967); +0.0007 is noise. Fill not judged.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-14-0900--prim_extruded_nr45_knn24_n6-k24-ab).

## 2026-09-13 22:00 — Train: prim_extruded_nr45_knn16_n6 (SageMaker A10G)

Catalog train ``2026-09-13_11-23-44_prim_extruded_nr45_knn16_n6`` (from scratch on ``ml.g5.xlarge``). Same n6 recipe as the 1080 inspect default (``knn_k: 16``, mix 75, ``envelope_dim=6``).
``best.pt`` epoch 20, val_iou 0.967 / acc 0.995, wall 4h 58m. Val matches 1080 n6 (0.965); +0.002 is noise. Fill not judged.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-13-2200--prim_extruded_nr45_knn16_n6-sagemaker-a10g).

## 2026-09-13 14:00 — SageMaker: POSIX S3 keys for source upload

Job ``scatteringnet-n6-20260913130138`` got past ``sm_train.sh`` and pip, then died on ``ModuleNotFoundError: No module named 'geometry'``. The four ``src/geometry/*.py`` files were in the code prefix, but Windows ``os.path.relpath`` put ``\\`` in the S3 keys, so Linux never created ``src/geometry/``.
``sagemaker/launch.py`` now normalizes those keys to ``/`` during ``train()``. ``entry.py`` fails fast if ``src/geometry/mesh_io.py`` is missing. Occupancy train loop unchanged.

## 2026-09-13 12:00 — SageMaker: Unix LF for sm_train.sh

Job ``scatteringnet-n6-20260913103358`` failed in ~4 min (``AlgorithmError``, exit 2) before ``entry.py`` ran. CloudWatch showed ``$'\r': command not found`` on the SDK-generated ``sm_train.sh``.
``sagemaker/launch.py`` now wraps ``trainer.train()`` so those ``*.sh`` files are written with Unix LF (``newline='\\n'``) on Windows. Occupancy train loop unchanged; re-launch from this PC after the fix.

## 2026-09-13 10:00 — SageMaker: launch.py for SDK v3

Local ``pip install sagemaker`` pulled SDK 3.x (``3.21``), which dropped ``sagemaker.pytorch.PyTorch`` and ``sagemaker.inputs.TrainingInput``. First launch died on this PC with ``No module named 'sagemaker.inputs'``.
Rewrote ``sagemaker/launch.py`` around ``ModelTrainer``: repo-root upload, ``entry_script=sagemaker/entry.py``, File-mode channel, ``ml.g5.xlarge``, PyTorch 2.5.1 DLC, region ``eu-north-1``. Occupancy train loop unchanged.

## 2026-09-10 15:00 — SageMaker: files under sagemaker/

Job scripts no longer sit under ``src/``. ``entry.py``, job-only ``requirements.txt`` (``trimesh``, ``pyyaml``), and ``.sagemakerignore`` live in ``sagemaker/``.
``launch.py`` still runs on this PC and uploads that folder plus ``src/`` and ``config.yaml`` (not ``models/``, ``runs/``, or the viewer). Occupancy train loop unchanged.

## 2026-09-10 10:00 — SageMaker: job entry + launcher

Added the G5 entry (remap ``data_dir`` to the training channel, run ``train_multi_npz``, copy ``best.pt`` and ``runs/`` to S3 output) and the local launcher (``ml.g5.xlarge``, PyTorch 2.5.1 DLC).
Live YAML is back to the n6 recipe (``knn_k: 16``, ``run_name: prim_extruded_nr45_knn16_n6``); the k=8 A/B is not the default. Occupancy train loop unchanged. Do not submit until G5 training-job quota is 1.

## 2026-09-10 13:00 — Docs: SageMaker S3 data prefix

[`docs/sagemaker.md`](docs/sagemaker.md) now records the uploaded tree: ``s3://scatteringnet-sagemaker-bucket/scatteringNet/data/`` with ``exports/`` and ``meshes/`` as siblings, same as local ``data_dir``.
Job output is ``…/scatteringNet/output/`` (created by the job; do not pre-create ``models/`` or ``runs/``). Occupancy code unchanged.

## 2026-09-09 13:00 — Docs: SageMaker training note

Added [`docs/sagemaker.md`](docs/sagemaker.md) as the human note for the G5 box: same ``train_multi_npz`` / catalog, instance ``ml.g5.xlarge`` (A10G), quota vs endpoint, not a Phase 4 ladder.
Phase 3 next-action and progress table link to it. Occupancy code unchanged.


## 2026-09-07 15:00 — Train: prim_extruded_nr45_knn8_n6

Catalog train ``2026-09-07_15-16-37_prim_extruded_nr45_knn8_n6`` (from scratch). Same catalog as n6, ``knn_k: 8``.
``best.pt`` epoch 20, val_iou 0.961 / acc 0.994, wall 9h 09m. Val just under n6 (0.965). Fill: OOD woman/man leak from the hands (worse than n6); helix/extrudes similar. Keep n6.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-08-0800--prim_extruded_nr45_knn8_n6-k8-ab).

## 2026-09-07 15:00 — Catalog: knn_k 8 A/B

Live YAML ``knn_k: 8`` and ``run_name: prim_extruded_nr45_knn8_n6``. Same catalog, mix 75, ``n_surface: 1024``, ``envelope_dim=6`` as n6. Fresh train only — do not resume the knn16 ``best.pt``.
Occupancy code unchanged. Judge Fill in two-skin gaps vs ``12-18-28_…_n6``, not val IoU.

## 2026-09-05 23:00 — Viewer: envelope normal ticks

``/api/envelope-obj`` now also returns ``normals_b64`` (same unit face normals occupancy uses). Envelope still sends XYZ in ``points_b64``.
The purple overlay draws a short tick along each normal. Run model already fed 6-D envelopes to ``envelope_dim=6`` checkpoints; this is display only.
``main.js?v=37``. Old XYZ ``best.pt`` infer is unchanged.

## 2026-09-05 15:00 — Envelope: XYZ + face normal

Each envelope sample is now ``(x, y, z, nx, ny, nz)``: the unit normal of the triangle the dot sits on. k-NN still picks neighbors by XYZ; each of the ``k`` slots is offset plus that neighbor's normal.
AABB remaps positions only. New ``best.pt`` stores ``envelope_dim=6``. Old XYZ checkpoints still load (first Linear in-features = 3). Viewer overlay still draws XYZ.

## 2026-09-05 14:00 — Viewer: drop Faces overlay

Removed the teal **Faces** button, its Count slider, ``/api/faces-obj``, ``faces.js``, and ``faces_job.py``.
Deleted ``src/geometry/face_tokens.py`` (the ``[v0, v1, v2, n]`` builder). Occupancy train/infer already unused it.
Envelope Mix still has a **Faces / Edges** split; that is area vs crease samples, not the old token overlay.

## 2026-09-05 13:00 — Envelope: cap crease budget

Crease share is now ``2 ×`` (sharp interior-edge length / all interior edges, dihedral ≥ 20°), then never above Mix. Leftover dots go to faces; ``n_surface`` stays 1024.
At mix 75: a cube still gets 768 crease dots; a hull with ~10% folds gets 205 crease / 819 area; no creases → all 1024 area.
Viewer Envelope ``n_area`` / ``n_edge`` follow that plan. Infer already uses it, so an old ``mix=75`` ``best.pt`` sees a different cloud until a retrain.

## 2026-09-05 12:00 — Train: prim_extruded_nr45_knn16_n6

Catalog train ``2026-09-05_12-18-28_prim_extruded_nr45_knn16_n6`` (from scratch). Same catalog as ``15-52-00``, crease cap + ``envelope_dim=6``.
``best.pt`` epoch 20, val_iou 0.965 / acc 0.994, wall 10h 10m. Fill: organics transferred; thin ``nr4``/``nr5`` corridors still leak.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-05-2300--prim_extruded_nr45_knn16_n6-xyz--face-normal).

## 2026-09-05 12:00 — Drop face-token occupancy encoder

Deleted ``src/geometry/encoder.py``. Face tokens never beat the envelope, so train/infer are ``surface`` or ``none`` only; ``shape_encoder: mesh`` raises.
Dataset no longer builds ``(F, 12)`` tokens. YAML dropped ``n_faces`` / ``encoder_hidden`` / ``encoder_depth``. Envelope ``best.pt`` still loads; mesh checkpoints do not.
Viewer **Faces** is display-only (largest triangles + normals) and is not fed to the occupancy head. The failed run stays in ``docs/training_log.md``.

## 2026-09-05 11:00 — Viewer: Run model hides Outside

Clicking **Run model** unchecks Outside, waits 150 ms so the blue cloud drops, then infers. Occupancy train code unchanged.

## 2026-09-05 10:00 — Occupancy labels: multi-ray vote

``_occupancy_labels`` no longer uses Open3D winding number (``compute_occupancy``). Each query casts 24 directions and must get an odd hit count on at least 18 of them. That stops a single hole from painting a ghost inside-slab. Rebuild ``nr4``/``nr5`` NPZs after this change.

## 2026-09-05 8:00 — Maya extrude: do not use polySelectConstraint

The 2016 ``openEdges`` workaround left a selection constraint on, so later extrudes only hit border faces and the skip check reported closed. Open edges are counted with ``polyInfo(edgeToFace=)`` now. Reload the ``.py`` before a new dump.

## 2026-09-04 20:00 — Maya extrude: open-edge check on 2016

``_open_edge_count`` no longer calls ``polyInfo(openEdges=)`` (that flag does not exist in Maya 2016). It uses ``polySelectConstraint`` border edges instead.

## 2026-09-04 18:00 — Deleted leaky nr4/nr5 OBJ and NPZ

Removed all ``extrude_*_nr4_*`` / ``nr5_*`` meshes and occupancy files (open-shell Truth leaks). YAML catalog no longer globs them. Primitives, ``extruded_*`` s0.08, and capped ``nr1`` stay.

## 2026-09-04 17:00 — Maya extrude: close the occupancy leak

``maya_batch_extrude.py`` always uses ``keepFacesTogether=1``, never flats ``sy=1`` on ``nr>=3``, retries ``polyCloseBorder``, and skips export if open edges remain. Reload in Maya before re-dumping ``nr4``/``nr5``.

## 2026-09-04 16:00 — dataset_builder --glob

``dataset_builder.py --glob`` matches filenames only so an Extrude folder dump can emit ``nr5`` NPZs without rewriting the other 2500 meshes.

## 2026-09-04 15:00 — Train: prim_extruded_nr45_knn16

Catalog train ``2026-09-04_15-52-00_prim_extruded_nr45_knn16`` (from scratch). Primitives + smooth ``extruded_*`` + capped ``nr1``/``nr4``/``nr5``, knn16 mix 75.
``best.pt`` epoch 14, val_iou 0.894 / acc 0.982, wall 8h 04m. Fill: easy extrudes + smooth usable; high-round arms and organics still failed.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-05-0800--prim_extruded_nr45_knn16-smooth--high-round-catalog).

## 2026-09-04 14:00 — Maya extruded-smooth: multi-direction recipes

Limb faces come from down / up / forward / back (lying uses +X as forward). Star recipe extrudes several axes separately. Cap scale is ``localScale``, not inset offset. ``run(limit=10)`` interleaves stand and lie.

## 2026-09-04 12:00 — Maya extruded-smooth: clamp cap offset

Mid-step shrink is bounded by the shortest cap edge so limb tips stay a usable width. Reload the Maya script from disk and re-run ``run(limit=10)``.

## 2026-09-04 11:00 — Maya extruded box + two extrudes + smooth

New Script Editor batch [`src/scatter_generation/maya_batch_extruded_smooth.py`](src/scatter_generation/maya_batch_extruded_smooth.py). Rectangular standing/lying box, limb faces, extrude, cap offset, extrude again, ``polySmooth`` once. OBJs go to ``meshes/ExtrudedSmooth/`` as ``extruded_*`` (not ``extrude_*``). Live ``maya_batch_extrude.py`` unchanged.

## 2026-09-04 10:00 — Viewer: recents list of 10

Hover Open keeps the last 10 successful loads (was 5). Occupancy train code unchanged.

## 2026-09-03 16:00 — Train: prim_extrude_knn16_n2048

Catalog train ``2026-09-03_15-59-37_prim_extrude_knn16_n2048`` (from scratch). Same mixed catalog as ``15-25-10``, ``n_surface: 2048``.
``best.pt`` epoch 19, val_iou 0.962 / acc 0.990, wall 4h 58m. Fill did not beat 1024; keep ``15-25-10``.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-03-2100--prim_extrude_knn16_n2048-denser-envelope).

## 2026-09-03 13:00 — Training log Fill gallery

Fill stills in [`docs/training_log.md`](docs/training_log.md) sit in a 3-column grid; click a thumbnail to open the PNG. Occupancy train code unchanged.

## 2026-09-02 15:00 — Train: prim_extrude_knn16

Catalog train ``2026-09-02_15-25-10_prim_extrude_knn16`` (from scratch). Mixed primitives + capped extrude, knn16 mix 75.
``best.pt`` epoch 20, val_iou 0.954 / acc 0.988, wall 4h 32m. Fill won holes + simple CAD; high-round arms later.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-03-1200--prim_extrude_knn16-mixed-primitives--extrude).

## 2026-09-02 15:00 — Catalog union + per-glob mesh cap

YAML ``npz_catalog`` unions several globs. Optional ``max_shapes`` keeps that many unique meshes per glob (sampled with ``seed``) so a mixed primitives + extrude train is not 2500 ``nr1`` files. Explicit ``npz_paths`` still wins. Live YAML is all primitive families plus 700 ``nr1``, 100 ``nr3``, and 80 ``nr4`` meshes (no ``nr2`` NPZs on disk), ``knn_k: 16``, mix 75, ``run_name: prim_extrude_knn16``.

## 2026-09-02 11:00 — Train: extrude_nr1_surface_knn16

Catalog train ``2026-09-02_11-18-13_extrude_nr1_surface_knn16`` (from scratch). ``nr1`` only, local k-NN 16 + mix 75.
``best.pt`` epoch 19, val_iou 0.974 / acc 0.996, wall 2h 17m. Local k-NN won on this val vs 40-epoch global mix 75.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-03-1200--extrude_nr1_surface-knn16-nr1-only).

## 2026-09-02 10:00 — Local k-NN envelope occupancy

Each Fill (and NPZ) point finds the ``knn_k`` closest envelope dots (16 in the occupancy trains; YAML so the count is controllable) instead of one shared summary of the whole skin, so the occupancy decision is local to that point. Surface ``OccupancyEncoder`` concatenates a per-query code from those nearest envelope offsets (``knn_k`` 0 = old global ``z`` only). Stored on ``best.pt``; missing key loads the global head so mix-75 checkpoints still infer. Mesh encoder unchanged. Catalog train is for the user.

## 2026-09-01 10:00 — Phase 3 work plans (local k-NN occupancy)

Occupancy continuation after Phase 2 exhaustion (global envelope could not fill thin arms) is [`docs/work_plan_phase3.md`](docs/work_plan_phase3.md).


## ================= END OF PHASE 2 =================

## 2026-09-01 23:00 — Train: extrude_nr1_surface_mix75 look-see

Resume ``2026-09-01_23-39-04_extrude_nr1_surface_mix75`` (epochs 29–40 from ``22-01-09``).
``best.pt`` epoch 30, val_iou 0.796 / acc 0.965, wall 1h 22m. Nothing after 30 replaced best; do not resume again.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-02-0800--extrude_nr1_surface-mix75-look-see-epochs-2940).

## 2026-09-01 22:00 — Train: extrude_nr1_surface_mix75 continuation

Resume ``2026-09-01_22-01-09_extrude_nr1_surface_mix75`` (epochs 21–30 from ``18-27-19``).
``best.pt`` epoch 28, val_iou 0.780 / acc 0.963, wall 1h 06m. Small real gain vs the 20-epoch mix-75 parent.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-01-2300--extrude_nr1_surface-mix75-continuation-epochs-2130).

## 2026-09-01 18:00 — Train: extrude_nr1_surface_mix75

Catalog train ``2026-09-01_18-27-19_extrude_nr1_surface_mix75`` (from scratch). Crease-weighted envelope mix 75, ``nr1``, ``h64/d4``.
``best.pt`` epoch 20, val_iou 0.771 / acc 0.961, wall 2h 13m. Beat the area-envelope 20-epoch match.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-01-2100--extrude_nr1_surface-mix75-crease-weighted-envelope).

## 2026-09-01 17:00 — Envelope mix in occupancy train

YAML ``envelope_mix`` 0–100 (0 = face-area darts, 100 = creases; same split as the viewer Mix slider). Stored on ``best.pt`` and the run snapshot. Infer rebuilds with the **checkpoint** mix (missing key = 0, so old envelope checkpoints stay area-weighted). OccupancyMLP unchanged.

## 2026-09-01 17:00 — Viewer: Create dots slider column

Create dots uses a two-column grid so Mix lines up with Count and Density. Occupancy train code unchanged.

## 2026-09-01 17:00 — Viewer: envelope Faces/Edges mix

**Envelope Mix** slider (0 = all face-area samples, 100 = all crease samples) splits **Count** between the two overlay methods. Occupancy train and Job A/B infer still use area-weighted ``sample_surface_points``.

## 2026-09-01 15:00 — Viewer: crease-weighted envelope overlay

**Envelope** overlay welds OBJ corners, then puts purple dots **on/near sharp mesh edges** (dihedral ≥ 20°). Density falls off toward face centers. Status reports crease count (0 = area fallback). Occupancy train and Job A/B infer still use area-weighted ``sample_surface_points``.

## 2026-09-01 15:00 — Viewer: face-token overlay

**Faces** in Create dots (between Envelope and Fill). Teal triangles + cyan normal ticks = occupancy ``n_faces`` tokens (largest by area, tiled if short). Count 64–1024, default 256. Occupancy train code unchanged.

## 2026-09-01 14:00 — Viewer: envelope overlay

**Envelope** + **Count** (256–4096, default 1024) in the Fill panel (**Create dots**), above **Fill points**. Purple area-weighted surface samples from the loaded OBJ (same ``sample_surface_points`` as occupancy). Click toggles; slider refreshes while shown. **Point size** scales occupancy and envelope dots. Occupancy train code unchanged.

## 2026-09-01 13:00 — Viewer: envelope and face-token checkpoints

Job A / Job B rebuild shape tokens from the **selected** ``best.pt`` (envelope ``(N,3)`` or faces ``(F,12)``), not from live YAML. Model list labels ``(envelope)`` / ``(faces)``. Occupancy train code unchanged.

## 2026-09-01 10:00 — Train: extrude_nr1_mesh

Catalog train ``2026-09-01_10-38-57_extrude_nr1_mesh`` (from scratch). Face tokens + xyz, ``shape_encoder: mesh``.
``best.pt`` epoch 19, val_iou 0.691 / acc 0.946, wall 2h 11m. Lost to the envelope baseline; do not treat as the new default.
Write-up: [docs/training_log.md](docs/training_log.md#2026-09-01-1400--extrude_nr1_mesh-face-tokens).

## 2026-09-01 09:00 — Step 10: MeshFaceEncoder

``src/geometry/encoder.py`` ``MeshFaceEncoder``: ``(B, F, 12)`` → ``z_face``. YAML ``shape_encoder: mesh``. Occupancy head is ``cat(xyz, z_face)``. Envelope head (``surface``) still loads. Infer rebuilds face tokens with the stored AABB. Catalog train is for the user to run.

## 2026-08-31 19:00 — Step 9: face tokens

``src/geometry/face_tokens.py``: ``(n_faces, 12)`` ``[v0, v1, v2, n]``. YAML ``n_faces: 256``. Encoder dataset stores tokens (AABB on corners); ``OccupancyEncoder`` still uses the envelope only. ``MeshFaceEncoder`` is not trained.

## 2026-08-31 18:00 — Viewer: inside-cut slider

**Inside cut** (0.00–1.00, default 0.50) re-thresholds the last **Run model** from stored sigmoid probabilities (no extra GPU pass). Lower = more inside. Acc/IoU in the status line follow this cut. Occupancy train code and train metrics at 0.5 are unchanged. How to run: [`src/viewer/README.md`](src/viewer/README.md).

## 2026-08-31 20:00 — Viewer: prefs, fill UX, README

Checkboxes, sliders, density, and selected model persist in ``src/viewer/ui_prefs.json``. Fill does not move the camera; density drag refills; Fill checks **Inside** / **Outside**. README matches Job A/B (Gradio skipped). How to run: [`src/viewer/README.md`](src/viewer/README.md). Occupancy train code unchanged.

## 2026-08-31 20:00 — Viewer Step 7: fill OBJ then infer

**Fill points** + density (0 = spacing 0.40, 100 = 0.05, default ≈ 0.15). Unlabeled AABB lattice (no raycast GT). **Run model** classifies those points; envelope from the uploaded OBJ. No Errors view. Occupancy train code unchanged.

## 2026-08-31 19:00 — Viewer Step 6: model on NPZ

``GET /api/models`` lists ``models/<run>/best.pt``. **Run model** classifies NPZ points (envelope if surface-conditioned). **Truth** / **Prediction** / **Errors**. **Select Model**; amber **Run model**. ``open_viewer.bat`` uses conda ``scatteringNet``. Occupancy train code unchanged.

## 2026-08-31 19:00 — Viewer Step 5: inspect UI

Mesh / Inside / Outside / Wireframe (2×2), opacity, point size, **Max points drawn** (display subsample). NPZ **Total points** and counts next to swatches. Equal-width 280px rounded left panels. Occupancy train code unchanged.

## 2026-08-31 18:00 — Viewer: auto-load mesh from NPZ

Opening an NPZ fetches the linked OBJ (``Mesh`` is visibility). ``serve.py`` reads quoted ``config.yaml`` ``data_dir`` and ignores the inline comment. Occupancy train code unchanged.

## 2026-08-31 18:00 — Viewer Step 4: Show object

``GET /api/mesh`` loads the NPZ ``mesh_path`` OBJ from ``data_dir`` (path must stay under ``data_dir``). **Clear view** stays in the recents menu. Occupancy train code unchanged.

## 2026-08-31 16:00 — Viewer: recents and Clear view

Hover Open for last 5 files. A recents click loads the cached file (does not open Explorer). IndexedDB + Chrome file handles. Occupancy train code unchanged.

## 2026-08-31 15:00 — Viewer Step 3: NPZ points

Open/drop an occupancy NPZ shows file-label points (inside orange, outside steel), counts, and stored ``mesh_path``. Occupancy train code unchanged.

## 2026-08-31 15:00 — Viewer Step 2: OBJ load

Open, drop, and recent-click load a Wavefront OBJ (local ``OBJLoader``, camera frames the AABB). Occupancy train code unchanged.

## 2026-08-31 13:00 — Viewer Step 1: page, helper, and plans

``src/viewer/``: dark Three.js page (orbit, grid, Open, recents stub). ``serve.py`` / ``open_viewer.bat`` (local Three.js, free port, ``.js`` as ``text/javascript``, ``no-store``). Plans: Job A = NPZ points; Job B = OBJ fill + envelope; Gradio optional. Occupancy train code unchanged.

## 2026-08-30 23:00 — Train: extrude_nr1_surface hidden 128 depth 6

Catalog train ``2026-08-30_22-59-57_extrude_nr1_surface`` (from scratch). Width+depth A/B: ``hidden: 128`` ``depth: 6``.
``best.pt`` epoch 19, val_acc 0.959 / IoU 0.754, wall 2h 28m. Tied with ``h64/d4``; not the new baseline.
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-31-0900--extrude_nr1_surface-hidden-128-depth-6-widthdepth-ab).

## 2026-08-30 20:00 — Train: extrude_nr1_surface hidden 128

Catalog train ``2026-08-30_20-31-00_extrude_nr1_surface`` (from scratch). Width A/B: ``hidden: 128``.
``best.pt`` epoch 18, val_acc 0.959 / IoU 0.757, wall 2h 17m. Neutral vs ``h64/d4`` at 20 epochs.
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-30-2200--extrude_nr1_surface-hidden-128-width-ab).

## 2026-08-30 19:00 — Train: extrude_nr1_surface continuation

Resume ``2026-08-30_18-40-06_extrude_nr1_surface`` (epochs 21–30 from ``15-50-05``).
``best.pt`` epoch 25, val_acc 0.962 / IoU 0.775, wall 1h 01m. Small gain; another resume not justified.
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-30-2000--extrude_nr1_surface-continuation-epochs-2130).

## 2026-08-30 18:00 — Resume catalog train from best.pt

``train_multi_npz`` accepts ``--resume-run-id`` / ``--resume``. It loads that ``best.pt``, keeps the stored selection score, and runs YAML ``epochs`` more (printed as 21…). New ``runs/`` + ``models/``; Adam state is restored only if the checkpoint stored it.

## 2026-08-30 16:00 — Train: extrude_nr1_surface (mesh-identity val)

Catalog train ``2026-08-30_15-50-05_extrude_nr1_surface`` (from scratch). Envelope + xyz, mesh-identity val.
``best.pt`` epoch 20, val_acc 0.958 / IoU 0.756, wall 2h 14m. Envelope path survived a real mesh holdout.
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-30-1800--extrude_nr1_surface-mesh-identity-val-hygiene-replay).

## 2026-08-30 15:00 — Collate: pin_memory on expanded envelopes

``occupancy_collate`` materializes the same-``shape_id`` envelope (``contiguous``) so CUDA ``pin_memory`` can pin it. Catalog surface train was crashing on the first batch.

## 2026-08-30 14:00 — Work plans: current rung is Step 9 or stop

Updated Phase 2 headers: Steps 1–8 done; next is face tokens or stop. Hygiene mesh-identity val is not Step 11. Phase 1 plans now point train write-ups at ``training_log.md``.

## 2026-08-30 14:00 — Training log: split and val wording

Clarified ``docs/training_log.md`` only: headings vs run id, historical ``test_*`` = today's val, file split is not a mesh holdout. Numbers unchanged.

## 2026-08-30 14:00 — Hygiene 4–14: val names, infer, shared helpers, encoder extract

Selection split is named **val** (``val_fraction`` / ``val_acc``; legacy ``test_*`` still accepted). ``src/infer_multi_npz.py`` reloads ``best.pt``. ``build_mlp``, shared trimesh/path helpers, envelope collate, point micro-average metrics, cache clear, YAML ``latent_dim``, ``random.seed`` + ``pin_memory``, and ``src/encoder_dataset.py`` land here. Conservative ``pyproject.toml`` keeps flat ``src`` imports.

## 2026-08-30 13:00 — Hygiene 3: shared AABB per mesh

Catalog parts that share a mesh reuse one ``center`` / ``scale`` (OBJ vertices when joined, else the union of that key's query points). Lattice and jitter no longer live in different frames.

## 2026-08-30 13:00 — Hygiene 2: shape_id from mesh identity

Catalog ``shape_id`` is a stable integer per ``mesh_split_key`` (sorted unique OBJs), not the file index. Two NPZs of one mesh share an id so encode-once is per OBJ.

## 2026-08-30 13:00 — Hygiene 1: mesh-identity train/test split

``test_fraction`` now holds out unique OBJs, not NPZ files. ``mesh_split_key`` / ``split_train_test_by_mesh`` keep every file of one mesh on the same side. Snapshot uses ``split: mesh`` plus ``n_train_meshes`` / ``n_test_meshes``.

## 2026-08-29 20:00 — Train: extrude_nr1_surface (file holdout)

Catalog train ``2026-08-29_19-43-15_extrude_nr1_surface`` (from scratch). First envelope + xyz on ``nr1``.
``best.pt`` epoch 18, val_acc 0.970 / IoU 0.691, wall 1h 59m. Envelope pairing helped a lot vs xyz-only.
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-29-2100--extrude_nr1_surface-level-1-extrudes-envelope--xyz).

## 2026-08-29 18:00 — Train/test 80/20 file split

Catalog train holds out whole files as a **test** set (``test_fraction: 0.20``). Terminal and snapshot use ``test_acc`` / ``n_test_files``.

## 2026-08-29 13:00 — Train/val split is by whole files

Catalog train no longer concatenates every NPZ into one point cloud. Each file stays its own shape: mini-batches come from one file, and ``val_fraction`` holds out entire files. Snapshot stamps ``split: shape``, ``n_train_files``, and ``n_val_files``.

## 2026-08-29 12:00 — Step 8: surface envelope occupancy

Added ``n_surface`` and ``shape_encoder`` to ``config.yaml``. ``src/geometry/surface.py`` samples an area-weighted shell cloud; ``OccupancyEncoder`` encodes each ``shape_id`` once per batch and concatenates ``z_surf`` with query xyz. Metrics now include inside IoU / F1. OccupancyMLP is unchanged (``shape_encoder: none``).

## 2026-08-29 12:00 — Step 7: NPZ ↔ OBJ mesh join

Added ``load_points_labels_mesh`` (``load_points_labels`` unchanged) and ``src/geometry/mesh_io.py`` to resolve ``mesh_path`` against ``data_dir`` and load OBJ ``vertices (V, 3)`` / ``faces (T, 3)``. Dataset parts store ``mesh_path``, ``mesh_key``, and triangles. Catalog train prints the join and stamps ``n_meshes``. OccupancyMLP is still xyz-only.

## 2026-08-29 11:00 — Run snapshot records file and point counts

``runs/<id>/config.yaml`` now stamps ``n_files``, ``n_points``, ``n_train``, and ``n_val`` after the catalog is loaded. ``gpu`` (card name) was already in the snapshot next to ``device``.

## 2026-08-29 09:00 — Train: extrude_nr1 (xyz-only)

Catalog train ``2026-08-29_09-28-54_extrude_nr1`` (from scratch). Xyz-only OccupancyMLP, pooled point split.
``best.pt`` epoch 20, acc 0.853 (no IoU), wall 1h 21m. Loop works; still too weak / xyz-only to beat “guess outside.”
Write-up: [docs/training_log.md](docs/training_log.md#2026-08-29-1000--extrude_nr1-first-level-cubes-xyz-only).

## 2026-08-29 09:00 — Single-file infer and leftover YAML knobs removed

Deleted ``src/infer_one_npz.py`` and ``tests/test_infer_one_npz.py``. Removed ``checkpoint_path`` and ``sample_npz`` from ``config.yaml`` / ``OccupancyConfig`` / run snapshots. Catalog train writes ``models/<run_id>/best.pt``. ``OccupancyPointDataset`` stays as the per-file reader inside the catalog.

## 2026-08-28 21:00 — Single-file train removed; one ``epochs`` knob

Deleted ``src/train_one_npz.py`` and ``tests/test_train_one_npz.py``. Catalog train is the only train path (``train_multi_npz.py``). Removed ``smoke_epochs``; YAML ``epochs`` is the train length. Infer still exists (``infer_one_npz.py``) and reads ``models/<run_id>/best.pt`` AABB from ``parts``. OccupancyMLP unchanged.

## 2026-08-28 21:00 — YAML ``batch_size`` and ``optimizer``

``config.yaml`` now owns mini-batch size (default 1024) and optimizer family (``adam`` / ``adamw`` / ``sgd``). ``train_multi_npz`` uses both; the run snapshot records them.

## 2026-08-28 19:00 — Run snapshot records GPU name

``runs/<id>/config.yaml`` now stamps ``gpu`` (CUDA card name from PyTorch, or null on CPU) next to ``device``. ``train_multi_npz`` prints the same name. Root ``config.yaml`` is unchanged (runtime-only, like ``device``).

## 2026-08-28 17:00 — Whole-run wall timer

Each ``runs/<id>/`` stamps ``started_at``, ``finished_at``, ``wall_seconds``, and  ``wall`` in ``config.yaml``. ``train_multi_npz`` prints the same duration at the end of the workout. Per-epoch ``wall_seconds`` in ``metrics.jsonl`` is unchanged.


## 2026-08-28 08:00 — Run catalog list moved out of ``config.yaml``

Resolved NPZ paths no longer live in the run snapshot. ``runs/<id>/catalog.txt`` is one data-relative path per line. ``config.yaml`` only records ``catalog_file`` and ``catalog_n``. Empty YAML ``npz_paths`` is omitted from the snapshot (glob was used). Existing sphere and extrude run folders were migrated.

## 2026-08-27 20:00 — Run snapshots store relative paths

``runs/<id>/config.yaml`` and multi-NPZ checkpoint payloads no longer dump absolute ``E:/`` / ``F:/`` paths.

- ``catalog_npz_paths``, ``sample_npz``, ``npz_paths`` → relative to ``data_dir`` (e.g. ``exports/dataset/...``).
- ``checkpoint_path``, ``run_dir`` → relative to the git repo (e.g. ``models/one_npz.pt``).
- ``data_dir`` stays absolute when the dataset disk is not inside the repo (this machine: ``E:`` data vs ``F:`` git).
- ``best.pt`` / ``last.pt`` ``npz`` keys use the same data-relative strings.
- Tests: ``test_as_data_and_repo_relative``, ``test_snapshot_strips_absolute_data_and_repo_paths``. Phase 1 train/infer unchanged.

## 2026-08-27 19:00 — Step 6: [First multi-NPZ occupancy train]

``src/train_multi_npz.py`` trains Phase 1 ``OccupancyMLP`` on the catalog glob. Logs → ``runs/<id>/``; weights → ``models/<id>/last.pt`` + ``best.pt``. No OccupancyMLP edits. ``train_one_npz.py`` unchanged.

- Val: random point split of the pooled queries (loop health).
- YAML additive ``smoke_epochs: 20`` (this CLI). Phase 1 still uses ``epochs``.
- Checkpoint stores ``kind=occupancy_mlp``, NPZ paths, per-mesh AABB.
- Tests: ``tests/test_train_multi_npz.py`` 1 OK (two synthetic NPZs, 2 CPU epochs, ``n=100``, ``best.pt`` + ``metrics.jsonl``, no ``.pt`` under ``runs/``).
- First real smoke: run ``python src/train_multi_npz.py`` on the current glob (2 sphere NPZs). Record ``runs/<id>`` / ``models/<id>`` and metrics after that run.

## 2026-08-27 20:00 — Step 5: [Checkpoints ``last.pt`` + ``best.pt``]

Checkpoint API under ``models/<run_id>/``. No occupancy train. Phase 1 ``train_one_npz.py`` / ``OccupancyMLP`` / ``models/one_npz.pt`` unchanged.

- ``src/checkpointing.py``: always write ``last.pt``; write ``best.pt`` only when the metric strictly improves (apex). Pointer ``runs/<id>/checkpoint_dir.txt``.
- Run snapshot ``runs/<id>/config.yaml`` now stamps ``total`` (planned epochs) and ``checkpoint`` (epoch of the last ``last.pt`` write), plus ``best_epoch`` / ``best_metric``.
- YAML additive ``checkpoint_metric: val_acc``. ``total`` / ``checkpoint`` are runtime snapshot fields, not project knobs.
- Tests: ``tests/test_checkpointing.py`` 1 OK (3 dummy epochs; ``best.pt`` keeps epoch 2 when epoch 3 is worse; no ``.pt`` under ``runs/``).
- Smoke: ``python src/checkpointing.py`` → ``runs/2026-08-27_18-45-58_ckpt_dummy``, ``models/2026-08-27_18-45-58_ckpt_dummy``; ``last_epoch=3`` ``best_epoch=2``; snapshot ``total=3`` ``checkpoint=3``; ``runs_has_pt=False``.

## 2026-08-27 19:00 — Step 4: [Run logs under ``runs/``]

Standalone log API. No occupancy train, no ``.pt``, Phase 1 ``train_one_npz.py`` / ``OccupancyMLP`` unchanged.

- ``src/run_tracking.py``: ``runs/<YYYY-MM-DD_HH-MM-SS>_<name>/`` with config snapshot, ``metrics.jsonl`` (one JSON object per epoch), TensorBoard scalars under ``train/`` and ``val/``.
- ``config.yaml`` additive ``run_name`` (optional in the loader). Device stays runtime-only.
- Gitignore TensorBoard event files only; JSON + YAML under ``runs/`` remain commitable.
- ``environment.yaml``: ``tensorboard>=2.14`` for ``SummaryWriter``.
- Tests: ``tests/test_run_tracking.py`` 2 OK (fake 3-epoch CPU loop; no ``.pt`` under ``runs/``). Phase 1 tests still green.
- Smoke: ``python src/run_tracking.py`` → ``runs/2026-08-27_17-06-04_dummy``; ``metrics.jsonl`` 3 rows; ``has_pt=False``; ``has_tfevents=True``.

## 2026-08-27 18:00 — Step 3: [Multi-NPZ dataset reader]

Catalog loader for many occupancy NPZs. Phase 1 ``load_points_labels`` / ``OccupancyPointDataset`` / ``train_one_npz.py`` unchanged. No occupancy train.

- ``resolve_npz_catalog`` in ``data_npz.py``: glob or explicit list under ``data_dir``, skip ``combo*``, cap ``max_files_per_shape``. Per-mesh key is the stem before ``__``.
- ``OccupancyMultiNpzDataset``: concatenates one ``OccupancyPointDataset`` per file (per-mesh AABB). DataLoader still yields ``xyz (B, 3)``, ``y (B, 1)``.
- ``config.yaml`` additive: ``npz_glob``, optional ``npz_paths``, ``max_files_per_shape``.
- Tests: ``tests/test_multi_npz.py`` 5 OK. Phase 1 loader/train/infer tests still green.
- Smoke: ``python src/data_npz.py --catalog`` → 2 sphere NPZs (lattice + j0.04), ``dataset_N=1729`` (729+1000), inside 270 / outside 1459, batch ``(8, 3)``.

## 2026-08-27 16:00 — Phase 2 ladder: loader + runs before first train

Reordered Phase 2 in [`docs/work_plan_phase2.md`](docs/work_plan_phase2.md). There is no Phase 3. After Step 2, the ladder is: multi-NPZ reader (3) → ``runs/`` + ``best.pt`` (4) → first occupancy train (5). Geometry steps shifted to 6–10. No training code changed.

## 2026-08-27 15:00 — Operator notes for NPZ generation

Operating notes for occupancy NPZ sets: [`docs/npz_dataset_generation.md`](docs/npz_dataset_generation.md). Covers CLI parameters, NPZ keys, a 10-point example from the sphere file, and copy-paste commands. No sampler or training code changed.

## 2026-08-27 15:00 — Full occupancy NPZ batch (r=0 and r=0.04)

Ran ``dataset_builder.py`` over all 3660 OBJs under ``data/meshes/``. Two NPZs per mesh: lattice (``random_range=0``) and low jitter (``0.04``). Spacing ``0.15``, seed ``1``, method occupancy.

- Output: ``E:/Work_stuff/scatteringNet/data/exports/dataset`` (7320 NPZs, ``ok=7320`` ``fail=0`` ``skipped=0``).
- Filenames: ``<stem>__occupancy_s0.15_inout.npz`` and ``<stem>__occupancy_s0.15_j0.04_inout.npz``.
- Spot-check ``TorusX4``: ``mesh_path=meshes/varied/TorusX4.obj``; r=0 N=125000; r=0.04 N=132651.

## 2026-08-27 12:00 — NPZ ``mesh_path`` relative to ``data_dir``

Stored OBJ paths are now relative to ``config.yaml`` ``data_dir`` (e.g. ``meshes/Primitives/Sphere/...obj``). Paths outside that folder still fall back to absolute POSIX. Re-run ``dataset_builder.py`` to refresh existing NPZs.


## 2026-08-27 12:00 — Step 2: [NPZ dataset generation]

Conda-side occupancy sampling now lives in this repo. Phase 1 `OccupancyMLP` / `train_one_npz.py` / `infer_one_npz.py` were not changed.

- Ported `mesh_loader.py`, `raycast_scatter.py`, `dataset_builder.py`, `paths.py`, `near_surface.py` (tagging default off) into `src/scatter_generation/`. Imports use `scatter_generation.*` (no `scattering_net`).
- CLI: `python src/scatter_generation/build_dataset.py <mesh_root> --method occupancy --spacings … --random-ranges … --seed … --limit N`.
- NPZ contract: `points (N,3)`, `labels (N,)`, `mesh_path`, plus `random_range` / `jitter` metadata. Order: lattice → offset → labels on the moved points. `r=0` is a no-op.
- Pinned `open3d>=0.18` and `trimesh>=4.0` in `environment.yaml` (env has Open3D 0.19.0, trimesh 5.0.0).
- Tests: `tests/test_scatter_generation.py` 4 OK. Full suite 32 tests: 31 OK; `test_load_config_resolves_data_dir_and_device` ERROR because `E:/Work_stuff/scatteringNet/data` is missing on this machine (not a Step 2 regression).
- Smoke (synthetic watertight cube OBJ, occupancy, spacing=0.35, random_range=0.04, seed=7): N=216, inside=9, outside=207, method=occupancy, occupancy_verified=True, `mesh_path` set. Maya Step 1 mesh tree not present, so no primitive OBJ batch from `data/meshes`.
- Optional Phase 1 compatibility: `train_one_npz` 2 epochs on a generated box NPZ (N=343, CUDA, hidden=64, depth=4). epoch 001 `loss=0.687535 train_acc=0.6460 val_acc=0.5942`; epoch 002 `loss=0.684318 train_acc=0.6460 val_acc=0.5942`. Loader/train loop accepted the file; this is not an overfit claim.

## 2026-08-27 09:00 — Maya OBJ export: load `objExport` plugin

Sphere batch failed in Maya with `Invalid file type specified: OBJexport` because the OBJ exporter plugin is off by default.

- `maya_batch_primitives.py`, `maya_batch_extrude.py`, `maya_batch_helix.py`: load `objExport` before `cmds.file(..., typ="OBJexport")`.
- Reload the script in Script Editor (`exec(open(...).read())`) then `run(...)` again.

Operating notes for OBJ export: [`docs/maya_batch_scatter_scripts.md`](docs/maya_batch_scatter_scripts.md)

## 2026-08-26 10:00 — Step 1: [Port Maya scatter scripts]

Maya OBJ exporters now live in this repo. Export roots match `config.yaml` `data_dir`. No occupancy model changes. No conda-side NPZ sampler (Step 2).

- Added `src/scatter_generation/maya_batch_primitives.py`, `maya_batch_extrude.py`, `maya_batch_helix.py` (Maya Script Editor only; `maya.cmds` unchanged).
- Export paths: primitives `.../meshes/Primitives`, extrude `.../meshes/Extrude`, helix `.../meshes/Helix`. Loader comments point at `src/scatter_generation/`.
- Dry-read: `run` / `list_families` (primitives) parse cleanly. Conda `unittest discover -s tests`: 28 tests OK.
- Maya export smoke: **not run here** (needs Script Editor). Suggested: `exec(open(...maya_batch_primitives.py).read()); run(families=("sphere",))` → `E:/Work_stuff/scatteringNet/data/meshes/Primitives/Sphere`.


## ================= END OF PHASE 1 =================

## 2026-08-24 15:00 — Step 10: [Stop and review]

Closed the occupancy MLP MVP with a timestamped review note. No source, tests, or config were changed.

- Added `docs/2026-08-24_14-09_mvp_completion_review.md`: data loop works; xyz MLP fits one field; next product step is a geometry encoder (not in this plan).

## 2026-08-24 13:00 — Step 9: [Train/val split]

Hold out a random 15% of *points* from the same NPZ (not a new mesh) and report val accuracy.

- `config.yaml` / `OccupancyConfig`: `val_fraction: 0.15`.
- `src/dataset.py`: `split_train_val_indices`; `make_dataloader` accepts a `Subset`.
- `src/train_one_npz.py`: train on the complement, eval val each epoch (`val_acc`), AABB `center`/`scale` still from the full cloud so Step 8 inference stays consistent. `TrainRunResult` now includes `val_accuracies`, `n_train`, `n_val`.
- Tests updated (`test_config`, `test_dataset`, `test_train_one_npz`, `test_infer_one_npz`). Full suite: 28 tests OK. Synthetic sphere: `n_train=218` `n_val=38`, final `train_acc=0.9633` `val_acc=0.8684`.

## 2026-08-24 10:00 — Step 8: [Basic inference]

Classify one occupancy NPZ with a saved checkpoint using the stored AABB map.

- Added `src/infer_one_npz.py`: load `kind=occupancy_mlp` checkpoint, rebuild `OccupancyMLP` from stored `hidden`/`depth`, normalize XYZ with checkpoint `center`/`scale` (not a fresh AABB), `eval` + `no_grad`, sigmoid threshold `0.5` via `occupancy_metrics`.
- Prints `pred_inside` / `pred_outside` counts and accuracy vs NPZ labels. Writes `pred_labels` to `{checkpoint_stem}_pred.npz` next to the checkpoint. Optional NPZ path argument; default is YAML `sample_npz`.
- Tests: `tests/test_infer_one_npz.py` (4 tests OK). Full suite: 27 tests OK.
- Smoke: `python src/infer_one_npz.py` on the train sphere (`N=10661`, CUDA): `pred_inside=5112` `pred_outside=5549` `accuracy=0.9212`. Exit met: overfit accuracy is high.

## 2026-08-23 23:00 — Env: recreate `scatteringNet`

Replaced the broken mixed conda/pip PyTorch install with a single-source CUDA 12.1 stack named after the project.

- Rewrote `environment.yaml`: conda-only `pytorch=2.5.1`, `pytorch-cuda=12.1`, `torchvision`, `torchaudio` (no pip torch wheels).
- Recreated `conda` env `scatteringNet`. Verified `torch 2.5.1`, CUDA 12.1, `torch.cuda.is_available() == True`.
- Full suite in the new env: 23 tests OK (including CUDA).
- `.cursorrules` and `docs/work_plan_phase1_AI.md` now name `scatteringNet` (not `scatteringNet_v2`) as the project environment.

## 2026-08-23 21:00 — Config: train knobs in YAML

Moved one-NPZ training defaults out of the train script so experiment knobs live in one place.

- `config.yaml` now holds `epochs`, `lr`, `checkpoint_path` (repo-relative), and `sample_npz` (relative to `data_dir`).
- `src/config.py`: `OccupancyConfig` / `load_yaml_knobs` load those fields; relative checkpoints resolve against the repo root; added `sample_npz_path()`.
- `src/train_one_npz.py` reads epochs / lr / checkpoint from `cfg` (no module-level defaults). CLI uses `sample_npz_path(cfg)`. `CHECKPOINT_KIND` stays in code (schema tag, not a train knob).
- Tests updated (`tests/test_config.py`, `tests/test_train_one_npz.py`). Full suite: 23 tests OK.

## 2026-08-23 20:00 — Step 7: [Metrics helper]

Centralized occupancy decision metrics so the train loop no longer computes accuracy inline.

- Added `src/metrics.py`: `occupancy_metrics` / `accuracy_from_logits` from logits vs `{0, 1}` labels (sigmoid + threshold `0.5`). Returns accuracy plus inside-class precision / recall (zero-denominator → `0.0`; no extra deps).
- `src/train_one_npz.py` now prints `train_acc` / `inside_prec` / `inside_rec` via `occupancy_metrics`; removed the private `_batch_accuracy` helper. Checkpoint format and `TrainRunResult` are unchanged.
- Tests: `tests/test_metrics.py` (6 tests OK, including CUDA). Existing `tests/test_train_one_npz.py` still passes (2 tests OK).
- Smoke: `python src/metrics.py` prints `accuracy=1.0000`, `inside_precision=1.0000`, `inside_recall=1.0000` on a perfect 8-point batch.

## 2026-08-21 19:00 — Step 6: [Train one NPZ / overfit]

Overfit `OccupancyMLP` on one occupancy NPZ (all points used as train; no val split).

- Added `src/train_one_npz.py`: seed, `OccupancyMLP`, Adam (`lr=1e-3`), `BCEWithLogitsLoss`, 30 epochs, prints mean `loss` and train accuracy each epoch.
- Saves `models/one_npz.pt` with `kind`, `state_dict`, AABB `center`/`scale`, `hidden`, `depth`.
- Composes Steps 1–5 (`load_config`, `OccupancyPointDataset`, `make_dataloader`, `OccupancyMLP`); those modules were not refactored.
- Tests: `tests/test_train_one_npz.py` (2 tests OK on CPU synthetic sphere).
- Smoke: `python src/train_one_npz.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, CUDA): epoch 1 `loss=0.693` / `acc=0.502` → epoch 30 `loss=0.329` / `acc=0.889` (peak acc `0.914` at epoch 29). Exit met: loss decreased, train acc ≫ 50%.

## 2026-08-21 17:00 — Step 5: [Dataset + DataLoader]

Wrapped one occupancy NPZ as a PyTorch `Dataset` with AABB normalization applied at construction.

- Added `src/dataset.py`: `OccupancyPointDataset`, `make_dataloader` (default `batch_size=1024`, `shuffle=True`, `num_workers=0`).
- Construction uses `load_points_labels` (`data_npz.py`) then `compute_center_scale` / `apply_normalization` (`normalize.py`); stores `center`, `scale`, and `npz_path` on the dataset.
- `__getitem__` returns float32 tensors: `xyz` shape `(3,)`, `y` shape `(1,)` (label in `{0, 1}`).
- Default collate yields `xyz (B, 3)` and `y (B, 1)` to match `OccupancyMLP` logits for BCE-with-logits.
- Verified: `tests/test_dataset.py` (2 tests OK); `python src/dataset.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, first batch `xyz.shape=(1024, 3)`, `y.shape=(1024, 1)`, `dtype=float32`).
- Previous step modules were not refactored.

## 2026-08-21 — Step 4: Coordinate normalize

AABB-normalize query XYZ into roughly ``[-1, 1]`` without changing the model.

- Added `src/normalize.py`: `compute_center_scale`, `apply_normalization`.
- Center = AABB midpoint; scale = max half-extent (must be > 0).
- `tests/test_normalize.py`: cube maps to ±1, inverse recovers the first rows, zero-extent rejected.
- Smoke: `python src/normalize.py` on the dataset_test sphere NPZ (normed range ≈ `[-1, 1]`, inverse check passed).

## 2026-08-21 — Config cleanup: `data_dir` in YAML

Dataset path no longer comes from `.env`.

- Added `data_dir` to `config.yaml` (`E:/Work_stuff/scatteringNet/data`).
- `src/config.py` reads `data_dir` from YAML; removed `.env` / `DATA_DIR` environment loading.
- If the folder is missing, `load_config()` prints a short message to set `data_dir` in `config.yaml`, then raises `FileNotFoundError`.
- `device` is still detected at runtime (CUDA vs CPU).
- Updated `tests/test_config.py` (existing path + missing-path case).

## 2026-08-21 — Step 3: NPZ loader

Added a loader that reads only occupancy queries from scatter NPZs.

- Added `src/data_npz.py`: `load_points_labels(path) -> (points, labels)`.
- Uses keys `points` and `labels` only; ignores `mesh_path`, `tags`, and other fields.
- Validates `points` is `(N, 3)` and `labels` is `(N,)`.
- Returns `float32` XYZ and `float32` labels in `{0.0, 1.0}` (conversion happens in the loader).
- Smoke: `python src/data_npz.py` on `dataset_test/sphere__raycast_z_raut_s0.15_inout.npz` (`N=10661`, inside fraction ≈ 0.497); `tests/test_data_npz.py` covers dtypes and shape checks.

## 2026-08-21 — Step 2: Hybrid configuration

Implemented a split between static experiment knobs and runtime resolution.

- Added `config.yaml` with `hidden: 64`, `depth: 4`, `seed: 1`.
- Updated `src/config.py` to load those YAML knobs, read `DATA_DIR` from the environment / repo `.env` (fill-if-missing, no `python-dotenv`), and set `device` via CUDA detection.
- `data_dir` and `device` are not stored in YAML (machine-specific).
- Verified with `python src/config.py` and `tests/test_config.py`: `data_dir=E:\Work_stuff\scatteringNet\data`, `device=cuda`.

## 2026-08-21 — Step 1: OccupancyMLP

Created the xyz-only occupancy module and smoke-tested it.

- Added `src/occupancy_mlp.py`: `OccupancyMLP(nn.Module)`, input `(B, 3)` → logits `(B, 1)`, defaults `hidden=64`, `depth=4`.
- Added `tests/test_occupancy_mlp.py` (shape, finite values, invalid inputs, CUDA). All four tests passed.
- `src/scattering_net.py` left untouched. No NPZ loading or training.