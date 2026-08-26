# Maya batch OBJ scripts

These three scripts create meshes in **Maya** and export **OBJ** files (no `.mtl`). They do not write occupancy NPZ files and they do not run in conda.

Open Maya → Script Editor → **Python** tab. Load a script with `exec(open(...).read())`, then call `run(...)`.

Script folder:

`scatteringNet/src/scatter_generation/`

Mesh root (same as `config.yaml` `data_dir`):

`E:/Work_stuff/scatteringNet/data/meshes/`

| Script | What it builds | Default output folder |
|---|---|---|
| `maya_batch_primitives.py` | Catalog primitives (sphere, cube, cylinder, …) | `.../meshes/Primitives/<Family>/` |
| `maya_batch_extrude.py` | Subdivided cubes with face extrusions | `.../meshes/Extrude/` |
| `maya_batch_helix.py` | Helix tubes only | `.../meshes/Helix/` |

---

## `maya_batch_primitives.py`

Sweeps many catalog primitives: sphere, cube, cylinder, cone, torus, pipe, prism, helix, gear, platonic. Each family is a grid of sizes / subdivisions / options. Aim is roughly 70–100 OBJs per family.

Shape numbers (radii, subdivs, gear teeth, …) are **not** `run()` arguments. They live in the family iterators in the file.

### Arguments (`run` / `run_family`)

| Argument | Default | Meaning |
|---|---|---|
| `families` | all 10 | `None` = everything; a string like `"sphere"`; or a tuple `("cylinder", "torus")` |
| `out_root` | `E:\Work_stuff\scatteringNet\data\meshes\Primitives` | Root folder; each family gets a subfolder |

`list_families()` prints keys and roughly how many variants each family will emit.

### Output

- Files: `{out_root}/{Family}/{name}.obj`  
  Example: `...\Primitives\Sphere\sphere_r1_sa16_sh16.obj`
- Subfolders: `Sphere`, `Cube`, `Cylinder`, `Cone`, `Torus`, `Pipe`, `Prism`, `HelixPrim`, `Gear`, `Platonic`
- Return value: how many OBJs were written. The Script Editor also prints per-family `wrote` / `errors`.

`HelixPrim` here is a smaller helix set. The dedicated helix dump is `maya_batch_helix.py` (different grid, folder `meshes/Helix`).

### Examples

Load once, then run:

```python
exec(open(r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_primitives.py").read())

list_families()

# Smoke: one family
run(families=("sphere",))

# Several families
run(families=("cylinder", "torus", "pipe"))

# Everything (long)
run()

# Same as one family, explicit root
run_family("sphere")
run(families="cube", out_root=r"E:\Work_stuff\scatteringNet\data\meshes\Primitives")
```

---

## `maya_batch_extrude.py`

Starts from a subdivided cube, then extrudes a few faces (steps / corners). Default is **one** extrude round. Thickness, face count, keep-faces-together, and a small offset are chosen at random (reproducible via seed). Before export it tries to merge verts, close borders, and fix normals.

### Arguments (`run`)

| Argument | Default | Meaning |
|---|---|---|
| `out_dir` | `E:\Work_stuff\scatteringNet\data\meshes\Extrude` | Where OBJs go (one flat folder) |
| `subdivs` | `(2, 3, 4, 5)` | Cube subdivisions |
| `variants_per_subdiv` | `32` | Random variants per subdiv value (~4×32 = 128 files) |
| `aspect_flat_frac` | `0.35` | Chance of a flat slab (`sy=1`) instead of a cube |
| `max_rounds` | `1` | Max extrude rounds per mesh |
| `round_weights` | `(1.0,)` | Relative chance of 1, 2, … rounds |
| `base_seed` | `1` | RNG seed |
| `variant_offset` | `0` | Shift filename `v` so a second batch does not overwrite the first |
| `limit` | `0` | If `> 0`, stop after that many files |

Cube size and extrude thickness lists are constants at the top of the file unless you edit them.

### Output

- Files: `{out_dir}/extrude_sx{N}_sy{N}_sz{N}_nr{N}_v{N}.obj`  
  `sx/sy/sz` = subdivs, `nr` = extrude rounds, `v` = variant id.
- Return value: number of OBJs written.

### Examples

```python
_p = r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_extrude.py"
exec(compile(open(_p, "r").read(), _p, "exec"))

# Smoke: 10 files
run(limit=10)

# Default full batch (~128) into meshes/Extrude
run()

# Another batch without overwriting v0.. (shift ids)
run(
    limit=250,
    max_rounds=4,
    round_weights=(0.0, 0.0, 0.5, 0.5),
    variants_per_subdiv=64,
    variant_offset=500,
    base_seed=2,
)
```

---

## `maya_batch_helix.py`

Only Maya `polyHelix`. Walks coils × height × width × tube radius, skips combos that are too tightly packed (low pitch, or width too small vs radius), and exports the rest. Tube subdivs and round cap are fixed in the file.

Full grid is 3×5×4×4 = 240 planned; filters drop some of those.

### Arguments (`run`)

| Argument | Default | Meaning |
|---|---|---|
| `out_dir` | `E:\Work_stuff\scatteringNet\data\meshes\Helix` | Output folder |
| `coils` | `(2, 3, 4)` | Number of turns |
| `heights` | `(2.5, 3.0, 3.5, 4.0, 4.5)` | Total height |
| `widths` | `(1.5, 2.0, 2.5, 3.0)` | How wide the spiral is |
| `radii` | `(0.3, 0.4, 0.5, 0.55)` | Tube thickness |
| `dry_run` | `False` | If `True`, print names only; write no files |

### Output

- Files: `{out_dir}/helix_c{coils}_h{height}_w{width}_r{radius}.obj`  
  Dots in numbers become `p` (for example `h3p5`).
- Return value: number written. Script Editor also prints `skipped`. With `dry_run=True` it counts planned names, not disk writes.

### Examples

```python
exec(open(r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_helix.py").read())

# List names only (no files)
run(dry_run=True)

# Default grid into meshes/Helix
run()

# Smaller custom grid
run(coils=(2,), heights=(3.5,), widths=(2.0,), radii=(0.4,))
```

---

## Quick comparison

| Script | Typical first run | Where files land |
|---|---|---|
| primitives | `run(families=("sphere",))` | `data/meshes/Primitives/Sphere/` |
| extrude | `run(limit=10)` | `data/meshes/Extrude/` |
| helix | `run(dry_run=True)` then `run()` | `data/meshes/Helix/` |
