# Batch-create Maya primitive variants and export OBJs.
# Run INSIDE Maya (Script Editor, Python tab). Do not run in conda.
#
# Export root (config.yaml data_dir + meshes/Primitives):
#   E:/Work_stuff/scatteringNet/data/meshes/Primitives
# Families land in subfolders (Sphere, Cube, ...).
#
# Covers: sphere, cube, cylinder, cone, torus, pipe, prism, helix, gear, platonic.
# Target ~70-100 OBJs per family (similar counts across families).
#
# LOADER (Maya Script Editor -> Python, then Execute):
#   exec(open(r"F:/Work_stuff/VisualStudio_cursor/scatteringNet/src/scatter_generation/maya_batch_primitives.py").read())
#   run()                      # all families
#   run(families=("sphere",))  # one family smoke
#   run(families=("cylinder", "torus", "pipe"))
#   list_families()

import os
from itertools import product

import maya.cmds as cmds
import maya.mel as mel

# Must match config.yaml data_dir / meshes / Primitives (Maya has no conda YAML).
OUT_ROOT = r"E:\Work_stuff\scatteringNet\data\meshes\Primitives"

CREATE_UVS = 2  # normalize (int families)
# materials=0 → OBJ only (no .mtl sidecar; unused by ScatterNet).
OBJ_OPTIONS = "groups=1;ptgroups=1;materials=0;smoothing=1;normals=1"

# Fixed mesh density (vary shape params, not poly count chaos).
SA = 20  # subdivisions axis
SH = 1  # subdivisions height (body)

# Interdependent params (Maya):
# - Round Cap (rcp) only looks rounded if Subdivisions Caps (sc) is high enough.
# - Varying sc while rcp=off barely changes the silhouette -> couple them.


def _fmt(x):
    return ("%g" % x).replace(".", "p").replace("-", "m")


def _tag(**kwargs):
    parts = []
    for key in sorted(kwargs.keys()):
        val = kwargs[key]
        if isinstance(val, bool):
            parts.append("%s%s" % (key, "1" if val else "0"))
        elif isinstance(val, float):
            parts.append("%s%s" % (key, _fmt(val)))
        else:
            parts.append("%s%s" % (key, val))
    return "_".join(parts)


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


# ---------------------------------------------------------------------------
# Per-family builders: each yields (name, create_callable) or None to skip.
# create_callable() -> transform node name
# ---------------------------------------------------------------------------


def _iter_sphere():
    # ~4*4*5 = 80 (include low-poly / faceted spheres)
    for r, sa, sh in product(
        (0.5, 1.0, 1.5, 2.0),
        (4, 8, 16, 24),
        (4, 8, 16, 24, 32),
    ):
        # Skip a few near-identical ultra-dense tiny combos to stay ~80.
        if sa >= 24 and sh >= 32 and float(r) != 1.0:
            continue
        name = "sphere_" + _tag(r=r, sa=sa, sh=sh)

        def _make(r=r, sa=sa, sh=sh, name=name):
            return _created_transform(
                cmds.polySphere(r=float(r), sa=int(sa), sh=int(sh), cuv=CREATE_UVS, name=name)
            )

        yield name, _make


def _iter_cube():
    # Boxes + cubes + subdiv variation. Aim ~80-90.
    sizes = (0.5, 1.0, 1.5, 2.0, 2.5)
    count = 0
    for w, h, d in product(sizes, sizes, sizes):
        if max(w, h, d) / max(min(w, h, d), 1e-6) > 3.5:
            continue
        # Keep cubes + varied proportions; drop some near-cubes at extreme sizes.
        if w == h == d and w not in (0.5, 1.0, 1.5, 2.0, 2.5):
            continue
        name = "cube_" + _tag(w=w, h=h, d=d, sx=1, sy=1, sz=1)

        def _make(w=w, h=h, d=d, name=name):
            return _created_transform(
                cmds.polyCube(
                    w=float(w),
                    h=float(h),
                    d=float(d),
                    sx=1,
                    sy=1,
                    sz=1,
                    cuv=CREATE_UVS,
                    name=name,
                )
            )

        yield name, _make
        count += 1
        if count >= 72:
            break

    for w, h, d, sx, sy, sz in (
        (1.0, 1.0, 1.0, 2, 2, 2),
        (1.0, 1.0, 1.0, 3, 3, 3),
        (2.0, 1.0, 1.0, 4, 1, 1),
        (1.0, 2.0, 1.0, 1, 4, 1),
        (1.0, 1.0, 2.0, 1, 1, 4),
        (2.0, 1.5, 1.0, 3, 2, 1),
        (1.5, 2.5, 1.0, 2, 4, 1),
        (2.5, 0.8, 1.2, 4, 1, 2),
    ):
        name = "cube_" + _tag(w=w, h=h, d=d, sx=sx, sy=sy, sz=sz)

        def _make(w=w, h=h, d=d, sx=sx, sy=sy, sz=sz, name=name):
            return _created_transform(
                cmds.polyCube(
                    w=float(w),
                    h=float(h),
                    d=float(d),
                    sx=int(sx),
                    sy=int(sy),
                    sz=int(sz),
                    cuv=CREATE_UVS,
                    name=name,
                )
            )

        yield name, _make


def _iter_cylinder():
    # Flat caps: rcp=off, sc=0/1 (sc alone does little).
    # Round caps: rcp=on AND sc>=3 (otherwise roundCap is barely visible).
    for r, h, sa in product((0.4, 0.7, 1.0, 1.4), (0.8, 1.5, 2.5, 3.5), (8, 20)):
        # Flat
        name = "cyl_" + _tag(r=r, h=h, rcp=False, sa=sa, sc=0)

        def _make_flat(r=r, h=h, sa=sa, name=name):
            return _created_transform(
                cmds.polyCylinder(
                    r=float(r),
                    h=float(h),
                    sa=int(sa),
                    sh=SH,
                    sc=0,
                    cuv=CREATE_UVS,
                    rcp=False,
                    name=name,
                )
            )

        yield name, _make_flat

        # Rounded / capsule-like
        if float(h) < float(r) * 1.2:
            continue
        for sc in (3, 5):
            name_r = "cyl_" + _tag(r=r, h=h, rcp=True, sa=sa, sc=sc)

            def _make_round(r=r, h=h, sa=sa, sc=sc, name=name_r):
                return _created_transform(
                    cmds.polyCylinder(
                        r=float(r),
                        h=float(h),
                        sa=int(sa),
                        sh=SH,
                        sc=int(sc),
                        cuv=CREATE_UVS,
                        rcp=True,
                        name=name,
                    )
                )

            yield name_r, _make_round


def _iter_cone():
    # Flat base (rcp=off). Round Cap alone is useless without sc>=2:
    # sc=2 -> diamond / double-cone look; sc=4 -> teardrop / rounded base.
    for r, h in product((0.5, 1.0, 1.5, 2.0), (1.0, 1.5, 2.5, 3.5)):
        for sa in (8, 20):
            name = "cone_" + _tag(r=r, h=h, sa=sa, rcp=False, sc=0)

            def _make_flat(r=r, h=h, sa=sa, name=name):
                return _created_transform(
                    cmds.polyCone(
                        r=float(r),
                        h=float(h),
                        sa=int(sa),
                        sh=SH,
                        sc=0,
                        cuv=CREATE_UVS,
                        rcp=False,
                        name=name,
                    )
                )

            yield name, _make_flat

        for sa, sc in product((8, 17), (2, 4)):
            name_r = "cone_" + _tag(r=r, h=h, sa=sa, rcp=True, sc=sc)

            def _make_round(r=r, h=h, sa=sa, sc=sc, name=name_r):
                return _created_transform(
                    cmds.polyCone(
                        r=float(r),
                        h=float(h),
                        sa=int(sa),
                        sh=SH,
                        sc=int(sc),
                        cuv=CREATE_UVS,
                        rcp=True,
                        name=name,
                    )
                )

            yield name_r, _make_round


def _iter_torus():
    # sa = ring polygonality (3=triangle, 4=square, ...), sh = tube cross-section.
    # Fixed sa=sh=20 misses most of the interesting family (tri/square rings, faceted tubes).

    def _emit(R, sr, tw, sa, sh):
        name = "torus_" + _tag(R=R, sr=sr, tw=tw, sa=sa, sh=sh)

        def _make(R=R, sr=sr, tw=tw, sa=sa, sh=sh, name=name):
            return _created_transform(
                cmds.polyTorus(
                    r=float(R),
                    sr=float(sr),
                    tw=float(tw),
                    sa=int(sa),
                    sh=int(sh),
                    cuv=True,
                    name=name,
                )
            )

        return name, _make

    # Smooth-ish tori: size + twist, high subdiv
    for R, sr, tw in product(
        (0.8, 1.2, 1.6, 2.0),
        (0.15, 0.25, 0.35, 0.45),
        (0, 45),
    ):
        if float(sr) >= float(R) * 0.55:
            continue
        yield _emit(R, sr, tw, sa=20, sh=16)

    # Faceted family: vary sa/sh (the shapes you showed). Fewer sizes, no twist.
    for R, sr in ((1.0, 0.25), (1.2, 0.4), (1.6, 0.3)):
        for sa in (3, 4, 5, 6, 8):
            for sh in (3, 4, 6, 12, 20):
                yield _emit(R, sr, tw=0, sa=sa, sh=sh)


def _iter_pipe():
    # Same coupling as cylinder: Round Cap needs Subdivisions Caps >= ~3.
    for r, h, t in product((0.6, 1.0, 1.4), (1.0, 2.0, 3.0), (0.15, 0.3, 0.45)):
        if float(t) >= float(r) * 0.85:
            continue
        # Flat ends
        name = "pipe_" + _tag(r=r, h=h, t=t, rcp=False, sc=0)

        def _make_flat(r=r, h=h, t=t, name=name):
            return _created_transform(
                cmds.polyPipe(
                    r=float(r),
                    h=float(h),
                    t=float(t),
                    sa=SA,
                    sh=SH,
                    sc=0,
                    cuv=True,
                    rcp=False,
                    name=name,
                )
            )

        yield name, _make_flat

        # Rounded ends (vary sc so roundCap actually changes the mesh)
        for sc in (3, 5):
            name_r = "pipe_" + _tag(r=r, h=h, t=t, rcp=True, sc=sc)

            def _make_round(r=r, h=h, t=t, sc=sc, name=name_r):
                return _created_transform(
                    cmds.polyPipe(
                        r=float(r),
                        h=float(h),
                        t=float(t),
                        sa=SA,
                        sh=SH,
                        sc=int(sc),
                        cuv=True,
                        rcp=True,
                        name=name,
                    )
                )

            yield name_r, _make_round


def _iter_prism():
    # sides 3..8 changes look a lot (tri prism -> almost cylinder). ~75-90.
    for sides, length, side in product(
        (3, 4, 5, 6, 7, 8),
        (0.8, 1.5, 2.5, 3.5),
        (0.5, 0.8, 1.1, 1.5),
    ):
        name = "prism_" + _tag(n=sides, L=length, s=side)

        def _make(sides=sides, length=length, side=side, name=name):
            return _created_transform(
                cmds.polyPrism(
                    l=float(length),
                    w=float(side),
                    ns=int(sides),
                    sh=SH,
                    sc=0,
                    cuv=CREATE_UVS,
                    name=name,
                )
            )

        yield name, _make


def _iter_helix():
    # Size family (smooth tube) + subdiv family (square / low-poly coils).
    def _emit(c, h, w, r, sa, sco, sc, rcp):
        name = "helix_" + _tag(c=c, h=h, w=w, r=r, sa=sa, sco=sco, sc=sc, rcp=rcp)

        def _make(c=c, h=h, w=w, r=r, sa=sa, sco=sco, sc=sc, rcp=rcp, name=name):
            return _created_transform(
                cmds.polyHelix(
                    c=int(c),
                    h=float(h),
                    w=float(w),
                    r=float(r),
                    sa=int(sa),
                    sco=int(sco),
                    sc=int(sc),
                    cuv=CREATE_UVS,
                    rcp=bool(rcp),
                    name=name,
                )
            )

        return name, _make

    # Open helices, high subdiv
    for c, h, w, r in product((2, 3, 4), (2.5, 3.5, 4.5, 6.0), (1.5, 2.0, 2.5), (0.3, 0.45)):
        pitch = float(h) / float(c)
        if pitch < 1.0:
            continue
        if float(w) < float(r) * 2.2:
            continue
        yield _emit(c, h, w, r, sa=20, sco=50, sc=3, rcp=True)

    # Play with sa / sco (sa=4 -> square tube helix, low sco -> angular coil)
    for c, h, w, r in ((2, 3.5, 2.0, 0.4), (3, 7.0, 2.0, 0.4), (4, 6.0, 2.5, 0.35)):
        for sa in (3, 4, 6, 8, 12):
            for sco in (8, 16, 32):
                yield _emit(c, h, w, r, sa=sa, sco=sco, sc=0, rcp=False)


def _set_poly_gear_attrs(transform, **attrs):
    """Set polyGear history attrs by long name (works across Maya versions)."""
    hist = cmds.listHistory(transform) or []
    nodes = [n for n in hist if cmds.nodeType(n) == "polyGear"]
    if not nodes:
        return
    node = nodes[0]
    for key, val in attrs.items():
        if cmds.attributeQuery(key, node=node, exists=True):
            cmds.setAttr("%s.%s" % (node, key), val)


def _create_poly_gear(
    name,
    sides,
    r,
    ir,
    h,
    gs=0.4,
    go=0.5,
    gt=0.5,
    gm=1.0,
    hd=1,
    tw=0.0,
    tp=1.0,
):
    """
    Maya's cog primitive is MEL `polyGear` (often missing from maya.cmds).
    Tooth silhouette needs gearMiddle/gearTip/gearOffset, not just sides.
    Height is the chalk thickness — must span thin disks through tall columns.
    """
    transform = None
    if hasattr(cmds, "polyGear"):
        try:
            transform = _created_transform(
                cmds.polyGear(
                    name=name,
                    sides=int(sides),
                    radius=float(r),
                    internalRadius=float(ir),
                    height=float(h),
                    subdivisions=int(hd),
                    gearSpacing=float(gs),
                    gearOffset=float(go),
                    gearTip=float(gt),
                )
            )
        except (TypeError, RuntimeError):
            transform = None

    if transform is None:
        cmds.select(clear=True)
        mel.eval(
            "polyGear -sides %d -r %g -ir %g -h %g -hd %d -gs %g -go %g -gt %g;"
            % (
                int(sides),
                float(r),
                float(ir),
                float(h),
                int(hd),
                float(gs),
                float(go),
                float(gt),
            )
        )
        sel = cmds.ls(selection=True, type="transform") or []
        if not sel:
            gears = cmds.ls("pGear*", type="transform") or []
            if not gears:
                raise RuntimeError("polyGear MEL created nothing")
            sel = [gears[-1]]
        transform = sel[0]
        if transform != name and not cmds.objExists(name):
            transform = cmds.rename(transform, name)

    # Apply middle / twist / taper (and reinforce tip/offset) on the history node.
    _set_poly_gear_attrs(
        transform,
        sides=int(sides),
        radius=float(r),
        internalRadius=float(ir),
        height=float(h),
        subdivisions=int(hd),
        gearSpacing=float(gs),
        gearOffset=float(go),
        gearTip=float(gt),
        gearMiddle=float(gm),
        twist=float(tw),
        taper=float(tp),
    )
    return transform


def _iter_gear():
    # Aim ~90-110. Previous grid kept h~0.3-1.2 and gm fixed -> all chalks looked alike.

    def _emit(sides, r, ir, h, gs, go, gt, gm, hd=1, tw=0.0, tp=1.0):
        name = "gear_" + _tag(
            n=sides, r=r, ir=ir, h=h, gs=gs, go=go, gt=gt, gm=gm, hd=hd, tw=tw, tp=tp
        )

        def _make(
            sides=sides,
            r=r,
            ir=ir,
            h=h,
            gs=gs,
            go=go,
            gt=gt,
            gm=gm,
            hd=hd,
            tw=tw,
            tp=tp,
            name=name,
        ):
            return _create_poly_gear(
                name, sides, r, ir, h, gs=gs, go=go, gt=gt, gm=gm, hd=hd, tw=tw, tp=tp
            )

        return name, _make

    # 1) Thickness / chalk height (thin disk -> tall column). hd rises with height.
    for h, hd in ((0.25, 1), (0.6, 2), (1.0, 4), (2.0, 6), (5.0, 8), (10.0, 10)):
        for sides, r in product((8, 16), (0.8, 1.2)):
            yield _emit(sides, r, ir=0.3, h=h, gs=0.5, go=0.3, gt=0.5, gm=1.0, hd=hd)

    # 2) Blade count (sides) — include low tooth counts like 5
    for sides in (3, 4, 5, 6, 8, 10, 12, 16, 20, 24):
        for r, h in ((1.0, 0.8), (1.4, 1.5)):
            yield _emit(sides, r, ir=0.35, h=h, gs=0.55, go=0.25, gt=0.5, gm=1.1, hd=4)

    # 3) Tooth / blade profile (spacing, offset, tip, middle)
    for gs, go in product((0.2, 0.45, 0.7), (0.1, 0.35, 0.6)):
        for gt, gm in ((0.15, 0.6), (0.5, 1.0), (0.9, 1.4)):
            yield _emit(12, 1.2, 0.35, 1.0, gs=gs, go=go, gt=gt, gm=gm, hd=4)

    # 4) Twist / taper on a few tall chalks
    for tw, tp, h in product((0, 20, 45), (0.5, 1.0, 1.5), (2.0, 8.0)):
        yield _emit(16, 1.0, 0.3, h, gs=0.55, go=0.25, gt=0.5, gm=1.2, hd=8, tw=tw, tp=tp)


def _make_platonic(pt, r, name):
    if hasattr(cmds, "polyPrimitive"):
        return _created_transform(
            cmds.polyPrimitive(pt=int(pt), r=float(r), cuv=CREATE_UVS, name=name)
        )
    cmds.select(clear=True)
    mel.eval("polyPrimitive -pt %d -r %g;" % (int(pt), float(r)))
    sel = cmds.ls(selection=True, type="transform") or []
    if not sel:
        raise RuntimeError("polyPrimitive MEL created nothing")
    transform = sel[0]
    if transform != name and not cmds.objExists(name):
        transform = cmds.rename(transform, name)
    return transform


def _iter_platonic():
    # polyPrimitive polyType: 0 Tetrahedron, 1 Cube, 2 Octahedron,
    # 3 Dodecahedron, 4 Icosahedron (Maya docs). 5 * 16 = 80.
    types = (
        (0, "tetra"),
        (1, "cube"),
        (2, "octa"),
        (3, "dodeca"),
        (4, "icosa"),
    )
    radii = (
        0.5,
        0.75,
        0.9,
        1.0,
        1.1,
        1.25,
        1.4,
        1.5,
        1.75,
        2.0,
        2.2,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
    )
    for (pt, label), r in product(types, radii):
        name = "platonic_" + _tag(t=label, r=r)

        def _make(pt=pt, r=r, name=name):
            return _make_platonic(pt, r, name)

        yield name, _make


FAMILIES = {
    "sphere": ("Sphere", _iter_sphere),
    "cube": ("Cube", _iter_cube),
    "cylinder": ("Cylinder", _iter_cylinder),
    "cone": ("Cone", _iter_cone),
    "torus": ("Torus", _iter_torus),
    "pipe": ("Pipe", _iter_pipe),
    "prism": ("Prism", _iter_prism),
    "helix": ("HelixPrim", _iter_helix),
    "gear": ("Gear", _iter_gear),
    "platonic": ("Platonic", _iter_platonic),
}


def list_families():
    """
    Print family keys, export subfolders, and planned variant counts.

    Returns
    -------
    None
        Writes to stdout for Script Editor use.
    """
    print("Available families:")
    for key, (subdir, it_fn) in sorted(FAMILIES.items()):
        n = sum(1 for _ in it_fn())
        print("  %-10s -> %s/  (~%d variants)" % (key, subdir, n))


def run_family(family, out_root=OUT_ROOT):
    """
    Export one primitive family to ``out_root/<Family>/``.

    Parameters
    ----------
    family:
        Key from :data:`FAMILIES` (e.g. ``sphere``).
    out_root:
        Parent of family subfolders. Default :data:`OUT_ROOT`.

    Returns
    -------
    written, errors:
        Export success count and exception count.
    """
    if family not in FAMILIES:
        raise ValueError("Unknown family %r. Call list_families()." % family)
    subdir, it_fn = FAMILIES[family]
    out_dir = os.path.join(out_root, subdir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    items = list(it_fn())
    written = 0
    errors = 0
    print("Family %s -> %s  (%d planned)" % (family, out_dir, len(items)))

    for i, (name, make) in enumerate(items, start=1):
        out_path = os.path.join(out_dir, name + ".obj").replace("\\", "/")
        try:
            transform = make()
            cmds.select(transform, replace=True)
            _export_selected(out_path)
            _delete_nodes([transform])
            written += 1
            if i == 1 or i % 20 == 0 or i == len(items):
                print("  [%d/%d] wrote %s.obj" % (i, len(items), name))
        except Exception as exc:  # noqa: BLE001 — Maya versions differ
            errors += 1
            print("  [%d/%d] FAIL %s: %s" % (i, len(items), name, exc))
            # Best-effort cleanup of any leftover selection.
            try:
                sel = cmds.ls(selection=True) or []
                _delete_nodes(sel)
            except Exception:  # noqa: BLE001
                pass

    print("  done %s: wrote=%d errors=%d" % (family, written, errors))
    return written, errors


def run(families=None, out_root=OUT_ROOT):
    """
    Export selected families (default: all).

    Parameters
    ----------
    families:
        ``None`` (all keys), one family name, or a sequence of keys from
        :data:`FAMILIES`.
    out_root:
        Parent of family subfolders. Default :data:`OUT_ROOT`.

    Returns
    -------
    None
        Prints per-family write/error counts.
    """
    if families is None:
        keys = list(FAMILIES.keys())
    elif isinstance(families, str):
        keys = [families]
    else:
        keys = list(families)

    total_w = 0
    total_e = 0
    print("Primitives batch -> %s" % out_root)
    print("Families: %s" % ", ".join(keys))
    for key in keys:
        w, e = run_family(key, out_root=out_root)
        total_w += w
        total_e += e
    print("ALL DONE: wrote=%d errors=%d root=%s" % (total_w, total_e, out_root))
    return total_w
