# Rectangular box + two extrude passes + one smooth. Organic-like CAD bank.
# Run INSIDE Maya (Script Editor, Python tab). Do not run in conda.
#
# Separate from maya_batch_extrude.py (cubes / nr rounds / meshes/Extrude).
# Filenames are extruded_* so occupancy globs for extrude_* do not swallow them.
#
# Export dir (config.yaml data_dir + meshes/ExtrudedSmooth):
#   E:/Work_stuff/scatteringNet/data/meshes/ExtrudedSmooth
#
# Per mesh:
#   1) polyCube as a standing (1 x 2 x 1) or lying (2 x 1 x 1) rectangle, 4^3 faces
#   2) Pick limb faces from several directions (not only -Y):
#        animal recipes: legs down, neck up or forward, optional tail back
#        star recipe: 3-5 cardinal sides, each extruded on its own
#   3) Two extrude passes (animal) or one pass per direction (star)
#   4) Cap scale is Maya localScale (flare > 1, mild taper >= 0.88), offset 0
#   5) Close the solid, then polySmooth once on the whole mesh
#
# LOADER (Maya Script Editor -> PYTHON tab, then Execute):
#   _p = r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_extruded_smooth.py"
#   exec(compile(open(_p, "r").read(), _p, "exec"))
#   run(limit=10)

import os
import random

import maya.cmds as cmds

# Must match config.yaml data_dir / meshes / ExtrudedSmooth (Maya has no conda YAML).
OUT_DIR = r"E:\Work_stuff\scatteringNet\data\meshes\ExtrudedSmooth"

# Standing box: tall. Lying box: long. Matches the Maya Channel Box examples.
STAND_SIZE = (1.0, 2.0, 1.0)
LIE_SIZE = (2.0, 1.0, 1.0)

# Face grid on the box. 4 matches the manual prototype (16 faces per side).
SUBDIVS = 4

# First and second extrude length (world units along the face normal).
THICKNESSES = (0.35, 0.45, 0.55, 0.70)

# Maya Channel Box Local Scale on the cap (1.0 = unchanged). Values below ~0.8
# crush a 4-subdiv face; 1.3–1.5 is the flare used on the manual prototypes.
LOCAL_SCALES = (0.88, 0.95, 1.0, 1.15, 1.3, 1.45)

# Face-normal threshold for bucketing a face onto +X/-X/+Y/-Y/+Z/-Z.
NORMAL_THRESH = 0.65

# Animal recipes are listed twice so they show up more often than the star.
RECIPES = ("quadruped", "quadruped", "tripod", "biped", "star")

# polySmooth: exponential, one division (manual prototype).
SMOOTH_DIVISIONS = 1
SMOOTH_CONTINUITY = 1.0

VARIANTS_PER_POSE = 8
CREATE_UVS = 2
OBJ_OPTIONS = "groups=1;ptgroups=1;materials=0;smoothing=1;normals=1"


def _fmt(x):
    return ("%g" % x).replace(".", "p").replace("-", "m")


def _mesh_name(pose, sx, sy, sz, variant):
    return "extruded_%s_sx%d_sy%d_sz%d_v%d" % (
        str(pose),
        int(sx),
        int(sy),
        int(sz),
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


def _face_centroid(face):
    """World-space average of the face vertices."""
    verts = cmds.ls(
        cmds.polyListComponentConversion(face, toVertex=True),
        flatten=True,
    ) or []
    if not verts:
        return (0.0, 0.0, 0.0)
    acc = [0.0, 0.0, 0.0]
    for vtx in verts:
        p = cmds.pointPosition(vtx, world=True)
        acc[0] += float(p[0])
        acc[1] += float(p[1])
        acc[2] += float(p[2])
    n = float(len(verts))
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def _face_normal(face):
    """Unit-ish face normal from polyInfo (Maya 2016 string format)."""
    raw = cmds.polyInfo(face, faceNormals=True)
    if not raw:
        return (0.0, 1.0, 0.0)
    text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
    parts = text.replace(":", " ").replace(",", " ").split()
    nums = []
    for token in parts:
        try:
            nums.append(float(token))
        except ValueError:
            continue
    if len(nums) < 3:
        return (0.0, 1.0, 0.0)
    # Last three floats are nx ny nz (leading ints are face ids).
    nx, ny, nz = nums[-3], nums[-2], nums[-1]
    mag = (nx * nx + ny * ny + nz * nz) ** 0.5
    if mag < 1e-8:
        return (0.0, 1.0, 0.0)
    return (nx / mag, ny / mag, nz / mag)


def _finalize_solid(transform):
    """
    Extrude stacks often leave open borders / non-manifold junk so raycast
    occupancy fails. Merge, close borders, conform normals, bake history.
    """
    cmds.select(transform, replace=True)
    try:
        cmds.polyMergeVertex(transform, d=1e-4, am=True, ch=False)
    except Exception:
        pass
    try:
        cmds.polyCloseBorder(transform, ch=False)
    except Exception:
        pass
    try:
        cmds.polyNormal(transform, normalMode=2, userNormalMode=0, ch=False)
    except Exception:
        pass
    try:
        cmds.delete(transform, constructionHistory=True)
    except Exception:
        pass


def _smooth_mesh(transform):
    """One exponential smooth pass on the whole solid (prototype: divisions 1)."""
    cmds.polySmooth(
        transform,
        method=0,
        divisions=int(SMOOTH_DIVISIONS),
        continuity=float(SMOOTH_CONTINUITY),
        keepBorder=True,
        keepHardEdge=False,
        constructionHistory=True,
    )
    try:
        cmds.delete(transform, constructionHistory=True)
    except Exception:
        pass


def _bucket_faces(transform):
    """Split box faces into the six cardinal sides by normal."""
    buckets = {"down": [], "up": [], "px": [], "nx": [], "pz": [], "nz": []}
    thresh = float(NORMAL_THRESH)
    for face in _list_faces(transform):
        nx, ny, nz = _face_normal(face)
        if ny < -thresh:
            buckets["down"].append(face)
        elif ny > thresh:
            buckets["up"].append(face)
        elif nx > thresh:
            buckets["px"].append(face)
        elif nx < -thresh:
            buckets["nx"].append(face)
        elif nz > thresh:
            buckets["pz"].append(face)
        elif nz < -thresh:
            buckets["nz"].append(face)
    return buckets


def _corner_faces(faces, n):
    """Bottom (or any) faces farthest from the Y axis in XZ — leg corners."""
    scored = []
    for face in faces:
        cx, _cy, cz = _face_centroid(face)
        scored.append((abs(cx) + abs(cz), face))
    scored.sort(reverse=True)
    return [f for _s, f in scored[: int(n)]]


def _faces_toward_axis(faces, axis, n, reverse=True):
    """Faces whose centroid is most + or - along world axis 0/1/2."""
    scored = [(_face_centroid(face)[int(axis)], face) for face in faces]
    scored.sort(reverse=bool(reverse))
    return [f for _s, f in scored[: int(n)]]


def _central_faces(faces, n):
    """Faces closest to the mean centroid of the bucket (middle of a side)."""
    if not faces:
        return []
    acc = [0.0, 0.0, 0.0]
    for face in faces:
        c = _face_centroid(face)
        acc[0] += c[0]
        acc[1] += c[1]
        acc[2] += c[2]
    k = float(len(faces))
    mx, my, mz = acc[0] / k, acc[1] / k, acc[2] / k
    scored = []
    for face in faces:
        c = _face_centroid(face)
        dx = c[0] - mx
        dy = c[1] - my
        dz = c[2] - mz
        scored.append((dx * dx + dy * dy + dz * dz, face))
    scored.sort()
    return [f for _s, f in scored[: int(n)]]


def _unique_faces(groups):
    """Flatten face groups, drop duplicates, skip empty groups."""
    seen = set()
    out = []
    for group in groups:
        uniq = []
        for face in group:
            if face in seen:
                continue
            seen.add(face)
            uniq.append(face)
        if uniq:
            out.append(uniq)
    return out


def _pick_limb_groups(transform, pose, rng):
    """
    Return (recipe_name, list_of_face_groups).

    Standing: long axis is Y, forward is +Z.
    Lying: long axis is X, forward is +X so a neck can rise from the head end
    and a tail from the opposite end — not another copy of four -Y legs.
    """
    buckets = _bucket_faces(transform)
    recipe = rng.choice(list(RECIPES))
    # Longitudinal forward / back for this pose.
    if pose == "stand":
        fwd, back, fwd_axis = "pz", "nz", 2
        lateral = ("px", "nx")
    else:
        fwd, back, fwd_axis = "px", "nx", 0
        lateral = ("pz", "nz")

    if recipe == "star":
        # Jack / cross: several sides, one or two faces each, extruded separately.
        dirs = [d for d in ("down", "up", "px", "nx", "pz", "nz") if buckets[d]]
        n_dir = min(int(rng.choice((3, 4, 5))), len(dirs))
        groups = []
        for d in rng.sample(dirs, n_dir):
            n_f = int(rng.choice((1, 1, 2)))
            picked = _central_faces(buckets[d], n_f)
            if picked:
                groups.append(picked)
        groups = _unique_faces(groups)
        if not groups:
            recipe = "quadruped"
        else:
            return recipe, groups

    n_down = {"quadruped": int(rng.choice((3, 4))), "tripod": 3, "biped": 2}[recipe]
    groups = []
    down = _corner_faces(buckets["down"], n_down)
    if down:
        groups.append(down)

    # Neck: usually the TOP of the box, biased to the forward end (lying head,
    # standing front-top). Sometimes the forward SIDE instead (snout / chest).
    n_neck = int(rng.choice((1, 2)))
    if rng.random() < 0.7 and buckets["up"]:
        neck = _faces_toward_axis(buckets["up"], fwd_axis, n_neck, reverse=True)
    else:
        # Upper faces of the forward cap so a lying snout is not on the floor.
        neck = _faces_toward_axis(buckets[fwd], 1, n_neck, reverse=True)
    if neck:
        groups.append(neck)

    # Tail on the opposite longitudinal side (skip often on biped).
    if recipe != "biped" or rng.random() < 0.45:
        n_tail = int(rng.choice((1, 2)))
        tail = _central_faces(buckets[back], n_tail)
        if not tail:
            tail = _faces_toward_axis(buckets[back], 1, n_tail, reverse=True)
        if tail:
            groups.append(tail)

    # Occasional extra arm on a remaining side so not every animal is planar.
    if recipe in ("tripod", "biped") and rng.random() < 0.55:
        sides = list(lateral)
        rng.shuffle(sides)
        for d in sides:
            extra = _central_faces(buckets[d], 1)
            if extra:
                groups.append(extra)
                break

    groups = _unique_faces(groups)
    if not groups:
        take = min(4, len(_list_faces(transform)))
        fallback = rng.sample(_list_faces(transform), take) if take else []
        groups = [fallback] if fallback else []
    return recipe, groups


def _selected_faces():
    sel = cmds.ls(selection=True, flatten=True) or []
    return [s for s in sel if ".f[" in s]


def _extrude_caps(faces, thickness, local_scale=1.0):
    """
    Extrude the given faces together. Maya then selects the new caps.

    ``local_scale`` is Channel Box Local Scale X/Y/Z (offset stays 0 so faces
    are not inset into needles).
    """
    if not faces:
        return []
    ls = float(local_scale)
    cmds.polyExtrudeFacet(
        faces,
        constructionHistory=True,
        keepFacesTogether=1,
        divisions=1,
        twist=0,
        taper=1,
        offset=0.0,
        thickness=float(thickness),
        lsx=ls,
        lsy=ls,
        lsz=ls,
    )
    return _selected_faces() or list(faces)


def _two_pass_extrude(faces, rng):
    """Length, then a second extrude that flares or mildly tapers the cap."""
    t1 = float(rng.choice(list(THICKNESSES)))
    t2 = float(rng.choice(list(THICKNESSES)))
    sc = float(rng.choice(list(LOCAL_SCALES)))
    caps = _extrude_caps(faces, thickness=t1, local_scale=1.0)
    _extrude_caps(caps, thickness=t2, local_scale=sc)


def _make_one(job):
    pose = job["pose"]
    sx = int(job["sx"])
    sy = int(job["sy"])
    sz = int(job["sz"])
    variant = int(job["variant"])
    seed = int(job["seed"])
    width, height, depth = STAND_SIZE if pose == "stand" else LIE_SIZE

    name = _mesh_name(pose, sx, sy, sz, variant)
    transform = _created_transform(
        cmds.polyCube(
            w=float(width),
            h=float(height),
            d=float(depth),
            sx=sx,
            sy=sy,
            sz=sz,
            cuv=CREATE_UVS,
            name=name,
        )
    )
    if not _list_faces(transform):
        _delete_nodes([transform])
        raise RuntimeError("polyCube produced no faces: %s" % name)

    rng = random.Random(seed)
    recipe, groups = _pick_limb_groups(transform, pose, rng)
    if not groups:
        _delete_nodes([transform])
        raise RuntimeError("no limb faces: %s" % name)

    if recipe == "star":
        # One extrude per direction (manual jack: several polyExtrudeFace nodes).
        for faces in groups:
            t = float(rng.choice(list(THICKNESSES)))
            sc = float(rng.choice(list(LOCAL_SCALES)))
            _extrude_caps(faces, thickness=t, local_scale=sc)
    else:
        # Legs + neck + tail in one two-pass so keepFacesTogether can merge
        # adjacent neck faces into a single wider limb.
        limbs = []
        for faces in groups:
            limbs.extend(faces)
        _two_pass_extrude(limbs, rng)

    # Solid first (occupancy needs a closed mesh), then one global smooth.
    _finalize_solid(transform)
    _smooth_mesh(transform)
    return transform, name, recipe


def _iter_jobs(
    subdivs=SUBDIVS,
    variants_per_pose=VARIANTS_PER_POSE,
    base_seed=1,
    variant_offset=0,
):
    """Yield standing and lying variants interleaved (so limit=10 is not 8 stand)."""
    idx = 0
    v0 = int(variant_offset)
    n_sub = int(subdivs)
    n_var = int(variants_per_pose)
    for variant in range(n_var):
        for pose in ("stand", "lie"):
            v_id = v0 + int(variant)
            seed = int(base_seed) + idx * 19 + v_id * 101
            yield {
                "pose": pose,
                "sx": n_sub,
                "sy": n_sub,
                "sz": n_sub,
                "variant": int(v_id),
                "seed": seed,
            }
            idx += 1


def run(
    out_dir=OUT_DIR,
    subdivs=SUBDIVS,
    variants_per_pose=VARIANTS_PER_POSE,
    base_seed=1,
    variant_offset=0,
    limit=0,
):
    """
    Create smoothed extruded-box OBJs under ``out_dir``.

    Parameters
    ----------
    out_dir:
        Export folder. Default :data:`OUT_DIR`.
    subdivs:
        Cube subdivisions on each axis (prototype used 4).
    variants_per_pose:
        Random variants for standing and for lying.
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

    jobs = list(
        _iter_jobs(
            subdivs=subdivs,
            variants_per_pose=variants_per_pose,
            base_seed=base_seed,
            variant_offset=variant_offset,
        )
    )
    if limit and int(limit) > 0:
        jobs = jobs[: int(limit)]

    written = 0
    print("Extruded-smooth batch -> %s" % out_dir)
    print(
        "Writing %d OBJs (subdivs=%d, recipes + localScale + smooth dv=%d)"
        % (len(jobs), int(subdivs), int(SMOOTH_DIVISIONS))
    )

    for i, job in enumerate(jobs, start=1):
        name = _mesh_name(
            job["pose"],
            job["sx"],
            job["sy"],
            job["sz"],
            job["variant"],
        )
        out_path = os.path.join(out_dir, name + ".obj")

        transform, name, recipe = _make_one(job)
        cmds.select(transform, replace=True)
        _export_selected(out_path.replace("\\", "/"))
        _delete_nodes([transform])
        written += 1
        print("  [%d/%d] wrote %s.obj [%s]" % (i, len(jobs), name, recipe))

    print("done: wrote=%d out=%s" % (written, out_dir))
    return written


# Maya Script Editor -> PYTHON tab, then Execute:
# _p = r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_extruded_smooth.py"
# exec(compile(open(_p, "r").read(), _p, "exec"))
# run(limit=10)
