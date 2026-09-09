# Occupancy viewer — work plan

**Status:** Steps 1–7 **done**. Step 8 (Gradio, optional) was **skipped**. Occupancy training code was not changed. Viewer track is closed unless you reopen Gradio.

**How work will go:** one step at a time. After each step we stop, look at it, and only then go on. Training the occupancy model is a separate track. This viewer does not replace that work; it helps us **see** whether the numbers from training actually mean a filled shape.

---

## Why we need this

Training prints scores: accuracy, and a few other numbers. Those numbers can look very good even when the “inside” of a shape is still wrong. Most of our points sit **outside** the object, so “guess outside” already scores high.

The only honest check is to **look**: the 3D object, the points that are supposed to be inside, the points that are supposed to be outside, and — when we choose a trained model — where that model disagrees with the labels.

That is what this viewer is for. It is an inspection window, not a new training program.

---

## What you will have when it is done

A small **web page** that opens in a normal browser (Chrome, Edge, Firefox). You can:

- spin the view around (click and drag)
- **Open** files with a button, or **drag and drop** them onto the page
- rest the mouse on Open and see the **last five files** you opened, then click one to load it again
- open an **OBJ** on its own (the 3D shape), or an **NPZ** on its own (labeled points) — both stay available; we did not drop OBJ-only load
- from an NPZ, the matching OBJ loads automatically from the path stored in that file (you should not have to hunt for it). **Mesh** turns the shape on and off
- turn the mesh or the points on and off, change point size, read simple counts
- **choose a trained model** and run it: on an NPZ (same points as the file), or on an OBJ after **Fill points**
- mesh **transparency** slider and a **Wireframe** checkbox
- next to **Fill points**, a **density** slider (how many query points in the box)

**Look and feel.** Dark, quiet 3D view (grid on the floor, clear controls), similar to a small demo we already like. We will **not** copy the old heavy desktop tool from the prototype.

---

## Opening files (button, drag-and-drop, recent)

There is always an **Open** button, not only drag-and-drop.

| How | What happens |
|---|---|
| **Open** | A normal file picker (OBJ, NPZ, or later a model file) |
| **Hover on Open** | A small panel lists the **last 5 files** loaded into the tool. Click a name to open it again |
| **Drag and drop** | Drop onto the page or the 3D view; same result as Open |

Recent files are remembered on this computer (the last five successful loads). Missing files are skipped or shown as unavailable.

---

## You can open OBJ, or NPZ, or both

Nothing is “NPZ only.” **Open** and drag-and-drop accept either type.

| You open | What you see at first | What “run the model” can do |
|---|---|---|
| **NPZ** | The file **already contains** the points and their true inside/outside tags. You see that **ground truth** immediately (and can return to it with a **Truth** button). The mesh loads from the path stored in the NPZ. You can then **run the model** on *those same points* — the network tries to classify them; **Prediction** / **Errors** compare to the file. You do **not** fill a new box; the cloud is the NPZ. |
| **OBJ** | You see the shape. **Fill points** (with a **density slider** beside it) builds a cloud in the bounding box and the envelope on the shell, then you **run the model** to paint inside vs outside. There is no file truth, so no Errors vs labels. |

**NPZ → mesh (you should not hunt for the OBJ):**

1. Open the **NPZ**.
2. The helper loads the OBJ from the path stored in that file. Use **Mesh** to hide or show it.

A web page cannot open `E:\…\meshes\….obj` by itself. On your computer a small helper may read our data folder. On a public GitHub site, a missing helper can only offer “choose the OBJ file.”

---

## Running the model — two different jobs

The network does **not invent** points. It answers, for each 3D location we give it: inside or outside?

**Job A — NPZ.**  
Points come from the file. A **Truth** control shows those labels (the usual first view). **Run model** classifies the **same** coordinates. **Prediction** and **Errors** are extra views, not a replacement for the file cloud.

**Job B — OBJ.**  
**Fill points** sits next to a **density** slider (more to the right = denser cloud / more points in the box). That fill plus the envelope is what the model sees. Then **Run model** paints inside vs outside.

1. **Box fill** — a regular cloud of query points filling the geometry’s bounding box (the same kind of volume our occupancy files use, not a hand-placed handful of dots).
2. **Shell / envelope** — points on the outer surface of the mesh. Our current trained head is not xyz-only: it also sees this envelope (`n_surface` samples). Without it, the model is not being used as it was trained.
3. **Normalize** — the same center/scale rule as training (from the mesh box).
4. **Run the selected model** — color predicted inside vs outside.

**Mesh appearance** (any time a shape is on screen): a **transparency** slider, and a **Wireframe** checkbox that turns the cage on and off (solid fill can stay, at the current transparency).

Toggles can hide the envelope vs the volume cloud so the picture stays readable.

We already have envelope sampling in the project. The box-fill for **inference** is new to the viewer; it is **not** the old prototype’s “generate a training NPZ with raycast labels.” We are not rebuilding that heavy scatter UI. We only need unlabeled queries + envelope so the network can classify.

**Model selector** (both jobs): list of saved trainings under `models/`, Browse to a `best.pt`, current model name always visible. Python on this computer runs the existing prediction code. We do not train from this window.

---

## Where the files live

Everything for this tool will sit in one folder: **`src/viewer/`**.

That keeps the viewer separate from the training code. We will not paste an old viewer from another project into this one.

---

## Two kinds of data files

| File | Everyday name | What you should see |
|---|---|---|
| **OBJ** | The 3D shape | The object. Enough, by itself, to later run Job B (box fill + envelope + model) |
| **NPZ** | Occupancy points with inside/outside tags | Colored dots. Stores the path to its OBJ for **Show object** |

A **model file** (`best.pt`) is chosen in the model control, not as a substitute for Open.

---

## GitHub, Gradio, and Python — in plain language

**GitHub as a website (Pages):** only the page. Open, drag-and-drop, recent files, looking at files you provide. No automatic **Show object** from disk, no running the trained model.

**Your computer:** the helper + the same page. Open OBJ or NPZ, **Show object**, **select / run a model** on NPZ labels or on an OBJ (box fill + envelope).

**Gradio:** a convenient way to start that local tool with one command. Same 3D page, not a second viewer. Optional, last.

We will **not** add Node.js just to avoid Python. Node would not run the trained model anyway.

---

## What we are not building

- A copy of the old prototype viewer (Open3D). We take **ideas**, not the code
- Rebuilding the full **training-set** scatter tool (raycast / jitter / save NPZ) inside this viewer
- A browser for thousands of catalog files at once
- Training or editing labels in the viewer
- Changing the occupancy network while we build this
- Publishing our private dataset or model weights on a public site

---

## The steps (in order)

| Step | In one sentence | You can stop and review when… |
|---|---|---|
| **1** | Empty 3D page, **Open** button, hover shows last 5 files (list may still be empty) | You can orbit a dark scene; Open is visible; hover panel exists |
| **2** | Open or drop an **OBJ** (on its own) | The shape appears. No NPZ required |
| **3** | Open or drop an **NPZ** (on its own) | Two-colored points; stored object path is shown |
| **4** | **Show object** from the NPZ path | NPZ + one button shows the matching shape (helper on this computer) |
| **5** | Mesh look + point toggles | Transparency slider, **Wireframe** on/off, hide outside, counts |
| **6** | **Select a model**; run on an **NPZ** | **Truth** (file labels) / **Prediction** / **Errors** — same NPZ points |
| **7** | **Fill points** on an **OBJ** (density slider) then run the model | Predicted fill in the box; no file truth |
| **8** | Gradio launcher (optional) | **Skipped.** `open_viewer.bat` is the launcher |

OBJ load stays in Step 2. NPZ load stays in Step 3. Step 7 is the extra complexity you described; it is not a replacement for opening either file type.

---

## Step 1 — Empty 3D page, Open, recent files

Create the folder and the page: 3D canvas, grid, orbit camera.

**Open** is on screen even before any file works. Resting the mouse on it opens a small window with the **last 5 files** (empty at first). Drag-and-drop area is visible.

**Done when:** a short written command opens the page; you can look around; Open and the hover panel are there.

---

## Step 2 — Load a 3D object (OBJ) — still a first-class path

**Open** and **drag-and-drop** load an OBJ **without** any NPZ. The camera frames the object. Wrong file type: a clear message. Successful loads join the recent-5 list.

This is how you start Job B later (model on a mesh). Until Step 7, you only **look** at the shape.

**Done when:** opening one of our usual shapes shows the geometry, with no points file involved.

---

## Step 3 — Load points (NPZ)

Same Open / drag-and-drop. Inside and outside in two colors. Show how many points, and **show the stored object path** (so you can see what **Show object** will try to load). Do not yet require the mesh.

If there are so many points that the page would freeze, draw a sample and say so.

**Done when:** a real occupancy NPZ shows a colored cloud. **Show object** may still be disabled until Step 4.

---

## Step 4 — Show object (from the NPZ path)

After an NPZ is loaded, a **Show object** button (enabled when the file contains an object path) loads that OBJ.

On this computer the local helper reads our data folder and follows the path — **you do not also open the OBJ**.

If the helper is not running (for example a public static page), the button explains that, shows the path, and lets you pick the OBJ once.

You can still open an OBJ yourself (Step 2) if you want; that is not disabled.

**Done when:** open one NPZ from our dataset, press **Show object**, and see the matching shape with the points. Wrong pairing (dots in empty space) should still be obvious if the path is wrong.

---

## Step 5 — Mesh look and point toggles

- Object on/off, inside on/off, outside on/off  
- Point size; how many points are drawn vs how many exist; counts  
- **Transparency** slider for the solid geometry  
- **Wireframe** checkbox: wires on or off  

**Done when:** you can see through the shell, turn wires on, hide outside points, and look at the interior.

Until Step 6, NPZ colors are still the **file labels** (ground truth).

---

## Step 6 — Select a model; run on an NPZ (Job A)

The cloud stays the NPZ points. **Truth** shows file labels (one click / default after load). **Run model** (after you **select** which `best.pt`) classifies those same points. Then **Prediction** and **Errors**.

**Done when:** NPZ + Show object + chosen model: you can flip Truth vs Errors by eye.

---

## Step 7 — Fill points on an OBJ, then run the model (Job B)

On screen together: **Fill points** and, **beside it**, a **density** slider (amount of query points in the bounding box). Fill also builds the envelope (`n_surface`). Then **Run model** with the selected checkpoint.

No **Errors** view (no labels in the OBJ).

**Done when:** open an OBJ, move the density slider, fill, run the model, see a predicted interior; transparency and wireframe still work on the mesh. **Shipped.**

---

## Step 8 — Gradio (optional)

**Status:** skipped. Use `open_viewer.bat` / `serve.py`. CHANGELOG: Gradio skipped.

If the helper plus the page is already easy to start, we can skip this. Gradio starts the **same** 3D page. We will not build a second viewer inside Gradio.

---

## What success looks like

A non-programmer can:

1. Open the page.
2. Open either an **OBJ** or an **NPZ**.
3. From an NPZ, the mesh loads with the file (or pick the OBJ yourself if the helper cannot). Use **Mesh** to hide or show it.
4. On an NPZ, use **Truth** for file labels; **select a model** and **Run** to see **Prediction** / **Errors**.
5. On an OBJ, set **density**, press **Fill points**, run the model; use **transparency** and **Wireframe** on the mesh.

GitHub-as-a-website may only show files you provide; it cannot run the model or follow disk paths.

---

## How this sits next to the rest of the project

| Question | Where to look |
|---|---|
| What the viewer will do (this file) | [`work_plan_viewer.md`](work_plan_viewer.md) |
| How to open the viewer | [`src/viewer/README.md`](../src/viewer/README.md) |
| Training the occupancy model | [`work_plan_phase2.md`](work_plan_phase2.md) |
| What already shipped in code | [`CHANGELOG.md`](../CHANGELOG.md) |
| What a training run meant | [`training_log.md`](training_log.md) |

Building the viewer does **not** mean we are doing the next geometry step of Phase 2 (face details). Those stay paused until we choose to resume them.

---

## Next action

**Steps 1–7 are done. Step 8 skipped.** Occupancy work is [`work_plan_phase3.md`](work_plan_phase3.md).
