# Batch-create Maya polyHelix variants and export OBJs.
# Run INSIDE Maya (Script Editor, Python tab). Do not run in conda.
#
# Export dir (config.yaml data_dir + meshes/Helix):
#   E:/Work_stuff/scatteringNet/data/meshes/Helix
#
# Easiest: in Maya Script Editor, paste ONLY the 3 lines under "LOADER"
# at the bottom of this file (or use File > Open Script on this .py, then run).

import os
from itertools import product

import maya.cmds as cmds

# Must match config.yaml data_dir / meshes / Helix (Maya has no conda YAML).
OUT_DIR = r"E:\Work_stuff\scatteringNet\data\meshes\Helix"

SUBDIV_AXIS = 20
SUBDIV_COIL = 50
# Round Cap only shows with enough Subdivisions Caps (sc=0 + rcp=on looks flat).
SUBDIV_CAPS = 3
CREATE_UVS = 3
ROUND_CAP = True

# Open/tall helices (pitch ~= height/coils). Target look: c=2,h=3.5,w=2,r=0.4
# Grid: 3*5*4*4 = 240 planned; pitch + width filters drop the packed ones.
COILS = (2, 3, 4)
HEIGHTS = (2.5, 3.0, 3.5, 4.0, 4.5)
WIDTHS = (1.5, 2.0, 2.5, 3.0)
RADII = (0.3, 0.4, 0.5, 0.55)

MIN_WIDTH_OVER_RADIUS = 2.2
MIN_PITCH = 1.0  # skip if height/coils is below this (keeps rings spaced)
# materials=0 → OBJ only (no .mtl sidecar; unused by ScatterNet).
OBJ_OPTIONS = "groups=1;ptgroups=1;materials=0;smoothing=1;normals=1"


def _fmt(x):
    return ("%g" % x).replace(".", "p")


def _mesh_name(coils, height, width, radius):
    return "helix_c%d_h%s_w%s_r%s" % (
        int(coils),
        _fmt(height),
        _fmt(width),
        _fmt(radius),
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


def run(out_dir=OUT_DIR, coils=COILS, heights=HEIGHTS, widths=WIDTHS, radii=RADII, dry_run=False):
    """
    Create one OBJ per (coils, height, width, radius) combo.

    Parameters
    ----------
    out_dir:
        Export folder. Default :data:`OUT_DIR`.
    coils, heights, widths, radii:
        Sweep lists (Cartesian product). Defaults are module constants.
    dry_run:
        If True, print planned files without exporting.

    Returns
    -------
    int
        Number of OBJ files written (0 when ``dry_run``).
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    combos = list(product(coils, heights, widths, radii))
    written = 0
    skipped = 0

    print("Helix batch -> %s" % out_dir)
    print("Planned combos: %d (dry_run=%s)" % (len(combos), dry_run))

    for i, (c, h, w, r) in enumerate(combos, start=1):
        pitch = float(h) / float(c)
        if pitch < MIN_PITCH:
            skipped += 1
            print(
                "  [%d/%d] skip c=%s h=%s w=%s r=%s (pitch=%.2f < %.2f)"
                % (i, len(combos), c, h, w, r, pitch, MIN_PITCH)
            )
            continue
        if float(w) < float(r) * MIN_WIDTH_OVER_RADIUS:
            skipped += 1
            print(
                "  [%d/%d] skip c=%s h=%s w=%s r=%s (width too small vs radius)"
                % (i, len(combos), c, h, w, r)
            )
            continue

        name = _mesh_name(c, h, w, r)
        out_path = os.path.join(out_dir, name + ".obj")

        if dry_run:
            print("  [%d/%d] dry-run %s.obj" % (i, len(combos), name))
            written += 1
            continue

        created = cmds.polyHelix(
            c=int(c),
            h=float(h),
            w=float(w),
            r=float(r),
            sa=SUBDIV_AXIS,
            sco=SUBDIV_COIL,
            sc=SUBDIV_CAPS,
            cuv=CREATE_UVS,
            rcp=ROUND_CAP,
            name=name,
        )
        transform = created[0] if isinstance(created, (list, tuple)) else created
        cmds.select(transform, replace=True)
        _export_selected(out_path.replace("\\", "/"))
        _delete_nodes([transform])
        written += 1
        if i == 1 or i % 25 == 0 or i == len(combos):
            print("  [%d/%d] wrote %s.obj" % (i, len(combos), name))

    print("done: wrote=%d skipped=%d out=%s" % (written, skipped, out_dir))
    return written


# LOADER (paste these 3 lines in Maya Script Editor -> Python, then Execute):
# exec(open(r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_helix.py").read())
# run(dry_run=True)   # optional: list files only
# run()
