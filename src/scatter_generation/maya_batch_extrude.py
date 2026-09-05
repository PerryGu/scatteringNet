# Batch-create subdivided cubes with random face extrusions (steps / corners).
# Run INSIDE Maya (Script Editor, Python tab). Do not run in conda.
#
# Export dir (config.yaml data_dir + meshes/Extrude):
#   E:/Work_stuff/scatteringNet/data/meshes/Extrude
#
# 1) polyCube with subdivision ranges (sx/sy/sz)
# 2) polyExtrudeFacet rounds (default one; run() can request nr4 / nr5)
#    - keepFacesTogether always ON (kft=0 left open shells; GWN occupancy
#      then filled ghost slabs — the nr4/nr5 Truth leaks)
#    - offset: sometimes 0, sometimes random in [0, OFFSET_MAX]
#    - skip export if open edges remain after finalize
#
# LOADER (Maya Script Editor -> PYTHON tab, then Execute):
#   _p = r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_extrude.py"
#   exec(compile(open(_p, "r").read(), _p, "exec"))
#   run(limit=10)

import os
import random
from itertools import product

import maya.cmds as cmds

# Must match config.yaml data_dir / meshes / Extrude (Maya has no conda YAML).
OUT_DIR = r"E:\Work_stuff\scatteringNet\data\meshes\Extrude"

# Cube size (Maya units). Subdivs vary; size stays modest for scatter.
WIDTH = 1.0
HEIGHT = 1.0
DEPTH = 1.0

# Subdivision ranges. Same value for sx/sy/sz unless flat-aspect kicks in.
# Flat slabs (sy=1) only on nr1–nr2. High-round flats were the leak set.
SUBDIVS = (2, 3, 4, 5)
ASPECT_FLAT_FRAC = 0.35  # nr1/nr2 only: chance of a flat slab (sy=1)
FLAT_MAX_ROUNDS = 2  # n_rounds above this never gets sy=1

# Faces extruded per round (clamped to available count).
N_FACES = (2, 3)

# Extrude thickness per round (world units along face normal).
THICKNESSES = (0.4, 0.7, 1.0, 1.2)

# keepFacesTogether always on. Off produced non-manifold open borders.
KEEP_TOGETHER = (True,)

# Exactly one extrude round per cube (simpler shapes for learning corners/steps).
MAX_ROUNDS = 1
ROUND_WEIGHTS = (1.0,)  # 100% → 1 round

# Offset: with OFFSET_USE_FRAC chance use uniform[0, OFFSET_MAX], else 0.
OFFSET_MAX = 0.1
OFFSET_USE_FRAC = 0.5

# Extra variants per base subdiv combo (different seeds / round counts).
# Grid ≈ len(SUBDIVS) * VARIANTS_PER_SUBDIV  (default 4*32 = 128).
VARIANTS_PER_SUBDIV = 32

CREATE_UVS = 2
OBJ_OPTIONS = "groups=1;ptgroups=1;materials=0;smoothing=1;normals=1"


def _fmt(x):
    return ("%g" % x).replace(".", "p").replace("-", "m")


def _mesh_name(sx, sy, sz, n_rounds, variant):
    return "extrude_sx%d_sy%d_sz%d_nr%d_v%d" % (
        int(sx),
        int(sy),
        int(sz),
        int(n_rounds),
        int(variant),
    )


def _ensure_obj_export_plugin():
    # Type "OBJexport" exists only after plugin objExport is loaded (off by default).
    if not cmds.pluginInfo("objExport", query=True, loaded=True):
        cmds.loadPlugin("objExport")


def _export_selected(path):
    _ensure_obj_export_plugin()
    cmds.file(
        path,
        force=True,
        options=OBJ_OPTIONS,
        typ="OBJexport",
        pr=True,
        es=True,
    )


def _delete_nodes(nodes):
    if not nodes:
        return
    existing = [n for n in nodes if cmds.objExists(n)]
    if existing:
        cmds.delete(existing)


def _created_transform(created):
    return created[0] if isinstance(created, (list, tuple)) else created


def _list_faces(transform):
    return cmds.ls("%s.f[*]" % transform, flatten=True) or []


def _open_edge_count(transform):
    """
    Count edges that touch fewer than two faces.

    Do not use ``polySelectConstraint`` (it stays on in Maya 2016 and
    poisons later extrudes) or ``polyInfo(openEdges=)`` (flag missing).
    ``edgeToFace`` exists in 2016.
    """
    if not transform or not cmds.objExists(transform):
        return 0
    edges = cmds.ls("%s.e[*]" % transform, flatten=True) or []
    n_open = 0
    for edge in edges:
        raw = cmds.polyInfo(edge, edgeToFace=True)
        if not raw:
            n_open += 1
            continue
        text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
        tail = text.split(":", 1)[-1]
        n_faces = 0
        for tok in tail.split():
            try:
                int(tok)
                n_faces += 1
            except ValueError:
                continue
        # Closed edge: two face ids. Border: one. Degenerate: zero.
        if n_faces < 2:
            n_open += 1
    return int(n_open)


def _finalize_solid(transform):
    """
    Merge, close borders (retry), conform normals, bake history.

    Occupancy uses Open3D winding number. An open shell gets a ghost
    inside-volume across the hole — that was the Truth leak on nr4/nr5.
    """
    cmds.select(transform, replace=True)
    try:
        cmds.polyMergeVertex(transform, d=1e-4, am=True, ch=False)
    except Exception:
        pass
    # Close, then merge again, then close once more. One pass left holes
    # on stacked extrudes.
    for _try in range(3):
        try:
            cmds.polyCloseBorder(transform, ch=False)
        except Exception:
            pass
        try:
            cmds.polyMergeVertex(transform, d=1e-4, am=True, ch=False)
        except Exception:
            pass
        if _open_edge_count(transform) == 0:
            break
    try:
        # 2 = conform / average normals outward
        cmds.polyNormal(transform, normalMode=2, userNormalMode=0, ch=False)
    except Exception:
        pass
    try:
        cmds.delete(transform, constructionHistory=True)
    except Exception:
        pass


def _pick_offset(rng, offset_max=OFFSET_MAX, offset_use_frac=OFFSET_USE_FRAC):
    if rng.random() < float(offset_use_frac):
        return float(rng.uniform(0.0, float(offset_max)))
    return 0.0


def _pick_n_rounds(rng, max_rounds=MAX_ROUNDS, weights=ROUND_WEIGHTS):
    """Weighted round count: default 50% / 30% / 15% / 5% for 1..4."""
    n_max = int(max_rounds)
    w = list(weights)[:n_max]
    if len(w) < n_max:
        w.extend([0.0] * (n_max - len(w)))
    total = float(sum(w))
    if total <= 0:
        return 1
    r = rng.random() * total
    acc = 0.0
    for i, wi in enumerate(w):
        acc += float(wi)
        if r <= acc:
            return i + 1
    return n_max


def _extrude_round(transform, rng, n_faces_choices, thicknesses, keep_together_list):
    faces = _list_faces(transform)
    if not faces:
        return False
    n_faces = int(rng.choice(list(n_faces_choices)))
    take = min(n_faces, len(faces))
    if take < 1:
        return False
    chosen = rng.sample(faces, take)
    thickness = float(rng.choice(list(thicknesses)))
    # Always together, even if a caller passes a list that includes False.
    kft = 1
    offset = _pick_offset(rng)

    # keepFacesTogether must be 0/1 (Maya rejects values > 1).
    cmds.polyExtrudeFacet(
        chosen,
        constructionHistory=True,
        keepFacesTogether=kft,
        divisions=1,
        twist=0,
        taper=1,
        offset=float(offset),
        thickness=thickness,
    )
    return True


def _make_one(job, width=WIDTH, height=HEIGHT, depth=DEPTH):
    sx = job["sx"]
    sy = job["sy"]
    sz = job["sz"]
    n_rounds = job["n_rounds"]
    variant = job["variant"]
    seed = job["seed"]
    n_faces_choices = job.get("n_faces_choices", N_FACES)
    thicknesses = job.get("thicknesses", THICKNESSES)
    keep_together_list = job.get("keep_together_list", KEEP_TOGETHER)

    name = _mesh_name(sx, sy, sz, n_rounds, variant)
    transform = _created_transform(
        cmds.polyCube(
            w=float(width),
            h=float(height),
            d=float(depth),
            sx=int(sx),
            sy=int(sy),
            sz=int(sz),
            cuv=CREATE_UVS,
            name=name,
        )
    )
    if not _list_faces(transform):
        _delete_nodes([transform])
        raise RuntimeError("polyCube produced no faces: %s" % name)

    rng = random.Random(int(seed))
    for _round in range(int(n_rounds)):
        ok = _extrude_round(
            transform,
            rng,
            n_faces_choices,
            thicknesses,
            keep_together_list,
        )
        if not ok:
            break
    _finalize_solid(transform)
    n_open = _open_edge_count(transform)
    if n_open > 0:
        # Do not write a leaky solid. Occupancy would invent a ghost slab.
        _delete_nodes([transform])
        return None, name, n_open
    return transform, name, 0


def _iter_jobs(
    subdivs=SUBDIVS,
    variants_per_subdiv=VARIANTS_PER_SUBDIV,
    aspect_flat_frac=ASPECT_FLAT_FRAC,
    max_rounds=MAX_ROUNDS,
    round_weights=ROUND_WEIGHTS,
    base_seed=1,
    variant_offset=0,
):
    """Yield job dicts (reproducible).

    ``variant_offset`` is added to the ``v`` in the filename so a second batch
    (e.g. offset=100) does not overwrite an earlier set.
    """
    idx = 0
    v0 = int(variant_offset)
    for sx in subdivs:
        for variant in range(int(variants_per_subdiv)):
            v_id = v0 + int(variant)
            rng = random.Random(base_seed * 10007 + idx * 13 + v_id)
            n_rounds = _pick_n_rounds(
                rng, max_rounds=max_rounds, weights=round_weights
            )
            # Slabs only on short stacks. nr>=3 + sy=1 was the leak set.
            if int(n_rounds) > int(FLAT_MAX_ROUNDS) or rng.random() >= float(
                aspect_flat_frac
            ):
                sy = int(sx)
                sz = int(sx)
            else:
                sy = 1
                sz = int(sx)
            seed = base_seed + idx * 17 + v_id * 101
            yield {
                "sx": int(sx),
                "sy": int(sy),
                "sz": int(sz),
                "n_rounds": n_rounds,
                "variant": int(v_id),
                "seed": int(seed),
            }
            idx += 1


def run(
    out_dir=OUT_DIR,
    subdivs=SUBDIVS,
    variants_per_subdiv=VARIANTS_PER_SUBDIV,
    aspect_flat_frac=ASPECT_FLAT_FRAC,
    max_rounds=MAX_ROUNDS,
    round_weights=ROUND_WEIGHTS,
    base_seed=1,
    variant_offset=0,
    limit=0,
):
    """
    Create extruded-cube OBJs under ``out_dir``.

    Parameters
    ----------
    out_dir:
        Export folder. Default :data:`OUT_DIR`.
    subdivs, variants_per_subdiv, aspect_flat_frac:
        Sweep knobs for the job iterator.
    max_rounds:
        Maximum extrusion rounds (1..4).
    round_weights:
        Relative weights for 1..max_rounds (e.g. ``(0, 0, 0.5, 0.5)``
        keeps only 3–4 rounds).
    base_seed:
        RNG seed for the batch.
    variant_offset:
        Added to filename ``v`` ids (use 100+ to add a new batch).
    limit:
        If ``> 0``, stop after that many exports.

    Returns
    -------
    int
        Number of OBJ files written.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # A leftover constraint from an older open-edge check stays on for the
    # whole Maya session and only lets later extrudes pick border faces.
    try:
        cmds.polySelectConstraint(disable=True)
    except Exception:
        pass

    jobs = list(
        _iter_jobs(
            subdivs=subdivs,
            variants_per_subdiv=variants_per_subdiv,
            aspect_flat_frac=aspect_flat_frac,
            max_rounds=max_rounds,
            round_weights=round_weights,
            base_seed=base_seed,
            variant_offset=variant_offset,
        )
    )
    if limit and int(limit) > 0:
        jobs = jobs[: int(limit)]

    written = 0
    skipped = 0
    print("Extrude batch -> %s" % out_dir)
    print(
        "Writing %d OBJs (max_rounds=%d, weights=%s, kft=1, skip if open)"
        % (len(jobs), int(max_rounds), list(round_weights))
    )

    for i, job in enumerate(jobs, start=1):
        name = _mesh_name(
            job["sx"],
            job["sy"],
            job["sz"],
            job["n_rounds"],
            job["variant"],
        )
        out_path = os.path.join(out_dir, name + ".obj")

        transform, name, n_open = _make_one(job)
        if transform is None:
            skipped += 1
            print("  [%d/%d] skip %s open_edges=%d" % (i, len(jobs), name, n_open))
            continue
        cmds.select(transform, replace=True)
        _export_selected(out_path.replace("\\", "/"))
        _delete_nodes([transform])
        written += 1
        if i == 1 or i % 25 == 0 or i == len(jobs):
            print("  [%d/%d] wrote %s.obj" % (i, len(jobs), name))

    print("done: wrote=%d skipped=%d out=%s" % (written, skipped, out_dir))
    return written


# Maya Script Editor -> PYTHON tab, then Execute:
# _p = r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_extrude.py"
# exec(compile(open(_p, "r").read(), _p, "exec"))
# run(limit=10)
#
# Complex batch (3–4 rounds, 250 shapes; does not overwrite v0..499 simple set):
# run(
#     limit=250,
#     max_rounds=4,
#     round_weights=(0.0, 0.0, 0.5, 0.5),
#     variants_per_subdiv=64,
#     variant_offset=500,
#     base_seed=2,
# )
