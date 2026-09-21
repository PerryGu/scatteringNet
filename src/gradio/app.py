"""Gradio occupancy demo: upload an OBJ, fill the AABB, classify with best.pt.

The 3D pane is a persistent Babylon canvas (``orbit.js`` via ``launch(js=)``).
Gradio ``Model3D`` remounts WebGL on every new GLB — that is the gray flash
and the camera snap. ``gr.HTML`` scripts are stripped, so the viewer JS is
not put in the HTML component. The inspect tool remains ``src/viewer``.

Launch (conda env scatteringNet):

    python src/gradio/app.py
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Pip package first — this directory must not shadow it.
import gradio as gr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from figure import (  # noqa: E402
    DEFAULT_MESH_OPACITY,
    DEFAULT_POINT_SIZE,
    empty_figure,
    occupancy_figure,
)
from pipeline import (  # noqa: E402
    DEFAULT_CUT,
    DEFAULT_DENSITY,
    apply_cut,
    default_run_id,
    fill_and_infer,
)
from obj_fill import triangles_from_obj_text  # noqa: E402

# Shipped sample meshes in examples/ (cube plus a few catalog shapes).
_EXAMPLE_OBJS = (
    "cube.obj",
    "horse.obj",
    "Player.obj",
    "dog.obj",
    "Helix_bend.obj",
    "TorusX3_box.obj",
)

# Inspect viewer: scene.background = 0x2a2a32 (not near-black).
_BG_HEX = "#2a2a32"
_ORBIT_JS = _HERE / "orbit.js"
# Host never goes in event outputs — remounting it would flash again.
_ORBIT_HOST = (
    f'<div id="sn-orbit-host" style="position:relative;width:100%;height:640px;'
    f'background:{_BG_HEX};border-radius:8px;overflow:hidden;">'
    '<canvas id="sn-orbit" style="width:100%;height:100%;display:block;"></canvas>'
    '<div id="sn-drop-hint">Drop an OBJ file here</div>'
    "</div>"
)


def _glb_dir() -> Path:
    """Same folder ``figure._export_glb`` writes; ``allowed_paths`` serves it."""
    folder = Path(tempfile.gettempdir()) / "scatteringnet_gradio"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _cmd_html(
    glb: str,
    reset_n: int,
    point_size: float = DEFAULT_POINT_SIZE,
    wire: bool = False,
    force_default: bool = False,
) -> str:
    """Tiny span orbit.js polls. Remounting this does not remount the canvas.

    ``force_default`` (last field) is **Reset view** only. Sample / drop /
    Load OBJ / Run keep the current orbit — same as the original load fix.
    """
    path = html.escape(str(Path(glb).resolve()))
    px = max(1, min(24, int(round(float(point_size)))))
    w = 1 if wire else 0
    d = 1 if force_default else 0
    return f'<span id="sn-cmd">{path}|{int(reset_n)}|{px}|{w}|{d}</span>'


def _held_path(file_obj) -> str | None:
    """File widget / UploadButton → filepath string for later Run / restyle."""
    if file_obj is None:
        return None
    return str(getattr(file_obj, "name", None) or file_obj)


def _read_upload(file_obj) -> tuple[str, str]:
    """Gradio File (filepath) → (basename, OBJ text)."""
    if file_obj is None:
        raise ValueError("upload an .obj")
    path = Path(getattr(file_obj, "name", None) or str(file_obj))
    if path.suffix.lower() != ".obj":
        raise ValueError("file must be an .obj")
    return path.name, path.read_text(encoding="utf-8", errors="replace")


def _status_md(result: dict, *, cut: float, shown_out: bool) -> str:
    """Short inspect-style summary under the view."""
    t = result.get("timings") or {}
    extra = " (outside hidden)" if not shown_out else ""
    return (
        f"**{result.get('obj_name', 'OBJ')}** · `{result.get('run_id', '')}`  \n"
        f"Inside **{result['n_inside']}** / outside **{result['n_outside']}** "
        f"(cut {cut:.2f}){extra}  \n"
        f"Lattice `{result['n']}` pts · spacing `{result['used_spacing']:.3f}` · "
        f"grid `{result['grid']}`  \n"
        f"Times s: fill+forward total `{t.get('server_total', 0):.3f}` · "
        f"envelope `{t.get('envelope', 0):.3f}` · run `{t.get('forward', 0):.3f}` · "
        f"device `{t.get('device', '?')}`"
    )


def _view(result: dict | None, cut: float, show_outside: bool, mesh_opacity: float):
    """Rebuild the occupancy GLB. The canvas stays; only the cmd span changes."""
    if not result:
        return str(empty_figure()), "Upload an OBJ, then **Run model**."
    pred, n_in, n_out = apply_cut(result["probs"], float(cut))
    view = dict(result)
    view["pred"] = pred
    view["n_inside"] = n_in
    view["n_outside"] = n_out
    glb = occupancy_figure(
        view["vertices"],
        view["faces"],
        view["points"],
        pred,
        show_outside=bool(show_outside),
        title=str(view.get("obj_name") or "occupancy fill"),
        mesh_opacity=float(mesh_opacity),
    )
    return str(glb), _status_md(view, cut=float(cut), shown_out=bool(show_outside))


def run_model(file_obj, obj_held, density, cut, show_outside, mesh_opacity):
    """Fill + infer with the single default checkpoint (no model picker)."""
    src = file_obj or obj_held
    held = _held_path(src)
    try:
        obj_name, obj_text = _read_upload(src)
        result = fill_and_infer(
            obj_text,
            obj_name=obj_name,
            run_id=default_run_id(),
            density=float(density),
        )
        glb, md = _view(result, float(cut), bool(show_outside), float(mesh_opacity))
        return result, glb, md, None, held
    except Exception as exc:
        return None, str(empty_figure(str(exc))), f"**Error:** {exc}", None, held


def preview_upload(file_obj, mesh_opacity=DEFAULT_MESH_OPACITY):
    """Show the uploaded mesh on the floor immediately (no occupancy yet)."""
    if file_obj is None:
        return None, str(empty_figure()), "Upload an OBJ, then click **Run model**."
    try:
        obj_name, obj_text = _read_upload(file_obj)
        vertices, faces = triangles_from_obj_text(obj_text)
        glb = occupancy_figure(
            vertices,
            faces,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            title=obj_name,
            mesh_opacity=float(mesh_opacity),
        )
        return (
            None,
            str(glb),
            f"**{obj_name}** is on the floor. Click **Run model** to classify the fill.",
        )
    except Exception as exc:
        return None, str(empty_figure(str(exc))), f"**Error:** {exc}"


def accept_obj(file_obj, mesh_opacity=DEFAULT_MESH_OPACITY, reset_n=0):
    """
    Load an OBJ, then clear the File box so the next drop can fire change.

    Keep ``reset_n`` so orbit.js does not reframe (examples / drop / Load
    OBJ stay on the default camera). **Reset view** is the only bump.
    Do not bind File.clear — emptying the box must not wipe the view.
    """
    keep = int(reset_n or 0)
    if file_obj is None:
        yield None, str(empty_figure()), keep, "Upload an OBJ, then click **Run model**.", None, None
        return
    held = _held_path(file_obj)
    state, glb, md = preview_upload(file_obj, mesh_opacity)
    yield state, str(glb), keep, md, None, held


def restyle(result, file_obj, obj_held, cut, show_outside, mesh_opacity):
    """Rebuild GLB after cut / opacity / outside. Reset counter is not touched."""
    try:
        if result:
            return _view(result, float(cut), bool(show_outside), float(mesh_opacity))
        src = file_obj or obj_held
        if src is None:
            return str(empty_figure()), "Upload an OBJ, then **Run model**."
        _, glb, md = preview_upload(src, mesh_opacity)
        return str(glb), md
    except Exception as exc:
        return str(empty_figure(str(exc))), f"**Error:** {exc}"


def bump_reset(glb_held, reset_n, point_size, wire):
    """Reset view: same GLB, increment token so JS frames 45° / 70°."""
    nxt = int(reset_n or 0) + 1
    path = glb_held or str(empty_figure())
    return nxt, _cmd_html(path, nxt, point_size, wire, force_default=True)


def accept_obj_ui(
    file_obj,
    mesh_opacity=DEFAULT_MESH_OPACITY,
    reset_n=0,
    point_size=DEFAULT_POINT_SIZE,
    wire=False,
):
    """accept_obj plus the #sn-cmd span orbit.js reads."""
    for state, glb, nxt, md, cleared, held in accept_obj(file_obj, mesh_opacity, reset_n):
        yield state, glb, nxt, _cmd_html(glb, nxt, point_size, wire), md, cleared, held


def run_model_ui(
    file_obj, obj_held, density, cut, show_outside, mesh_opacity, reset_n, point_size, wire
):
    """Run: new GLB, same reset token, so the canvas keeps the orbit."""
    result, glb, md, cleared, held = run_model(
        file_obj, obj_held, density, cut, show_outside, mesh_opacity
    )
    return result, glb, _cmd_html(glb, int(reset_n or 0), point_size, wire), md, cleared, held


def restyle_ui(
    result, file_obj, obj_held, cut, show_outside, mesh_opacity, reset_n, point_size, wire
):
    """Restyle: new GLB, same reset token."""
    glb, md = restyle(result, file_obj, obj_held, cut, show_outside, mesh_opacity)
    return glb, _cmd_html(glb, int(reset_n or 0), point_size, wire), md


def set_view_flags(glb_held, reset_n, point_size, wire):
    """Dot size / wireframe — no new GLB, orbit.js applies both."""
    path = glb_held or str(empty_figure())
    return _cmd_html(path, int(reset_n or 0), point_size, wire)


def build_demo() -> gr.Blocks:
    """One-page Gradio occupancy fill."""
    # Only list files that are actually on disk (partial checkout still launches).
    example_list = [
        str(path)
        for name in _EXAMPLE_OBJS
        if (path := _HERE / "examples" / name).is_file()
    ] or None
    empty_glb = empty_figure()

    with gr.Blocks(title="scatteringNet occupancy") as demo:
        gr.Markdown(
            """
# scatteringNet — occupancy fill

A trained occupancy network that fills a 3D mesh with points it labels
**inside** the solid (not outside). Drop an OBJ on the view (or **Load OBJ**),
then **Run model**.
            """.strip()
        )
        state = gr.State(None)
        obj_held = gr.State(None)
        glb_held = gr.State(str(Path(empty_glb).resolve()))
        reset_n = gr.State(0)
        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("Load OBJ", elem_id="sn-load-obj")
                density_in = gr.Slider(
                    0,
                    100,
                    value=DEFAULT_DENSITY,
                    step=1,
                    label="Density (higher = denser lattice)",
                )
                cut_in = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_CUT,
                    step=0.01,
                    label="Inside cut",
                )
                opacity_in = gr.Slider(
                    0,
                    100,
                    value=DEFAULT_MESH_OPACITY,
                    step=1,
                    label="Mesh opacity",
                    info="Default 50% so occupancy points show through the shell.",
                )
                psize_in = gr.Slider(
                    1,
                    24,
                    value=DEFAULT_POINT_SIZE,
                    step=1,
                    label="Dot size",
                    info="Occupancy points in the 3D view (pixels).",
                )
                with gr.Row():
                    show_out = gr.Checkbox(label="Show outside points", value=False)
                    wire_in = gr.Checkbox(label="Wireframe", value=False)
                with gr.Row():
                    run_btn = gr.Button("Run model", variant="primary")
                    reset_btn = gr.Button("Reset view")
                status = gr.Markdown(
                    "Drop an OBJ on the **3D view**, or click **Load OBJ**. "
                    "**Run model** fills the volume."
                )
            with gr.Column(scale=2):
                # Never list this HTML in outputs — a new value remounts the canvas.
                gr.HTML(value=_ORBIT_HOST, elem_id="sn-orbit-wrap")
        # Plumbing only. The cmd span is a temp GLB path; if it sits in the
        # layout it paints through Load OBJ (even with height:0).
        with gr.Column(elem_id="sn-obj-file-slot"):
            cmd = gr.HTML(
                value=_cmd_html(str(empty_glb), 0, DEFAULT_POINT_SIZE),
                elem_id="sn-cmd-wrap",
                visible="hidden",
                container=False,
            )
            obj_in = gr.File(
                label="OBJ",
                file_types=[".obj"],
                type="filepath",
                show_label=False,
                container=False,
                elem_id="sn-obj-file",
            )
        _accept_out = [state, glb_held, reset_n, cmd, status, obj_in, obj_held]
        _run_out = [state, glb_held, cmd, status, obj_in, obj_held]
        _style_out = [glb_held, cmd, status]
        if example_list:
            gr.Examples(
                examples=example_list,
                inputs=obj_in,
                outputs=_accept_out,
                fn=accept_obj_ui,
                run_on_click=True,
                cache_examples=False,
                label="Sample OBJ",
            )
        # Picker is a plain Button: UploadButton paints the path over the label.
        # orbit.js clicks #sn-obj-file's input; File.upload runs accept.
        # Do not bind File.clear — we empty the box on purpose so DND stays.
        obj_in.upload(
            accept_obj_ui,
            inputs=[obj_in, opacity_in, reset_n, psize_in, wire_in],
            outputs=_accept_out,
        )
        run_btn.click(
            run_model_ui,
            inputs=[
                obj_in,
                obj_held,
                density_in,
                cut_in,
                show_out,
                opacity_in,
                reset_n,
                psize_in,
                wire_in,
            ],
            outputs=_run_out,
        )
        reset_btn.click(
            bump_reset,
            inputs=[glb_held, reset_n, psize_in, wire_in],
            outputs=[reset_n, cmd],
        )
        _style_in = [
            state,
            obj_in,
            obj_held,
            cut_in,
            show_out,
            opacity_in,
            reset_n,
            psize_in,
            wire_in,
        ]
        cut_in.release(restyle_ui, inputs=_style_in, outputs=_style_out)
        opacity_in.release(restyle_ui, inputs=_style_in, outputs=_style_out)
        show_out.change(restyle_ui, inputs=_style_in, outputs=_style_out)
        _flag_in = [glb_held, reset_n, psize_in, wire_in]
        psize_in.release(set_view_flags, inputs=_flag_in, outputs=cmd)
        wire_in.change(set_view_flags, inputs=_flag_in, outputs=cmd)
    return demo


def main() -> None:
    """Local binds localhost; Spaces set ``PORT`` and need ``0.0.0.0``."""
    demo = build_demo()
    demo.queue()
    # Gradio 6: js/head belong on launch(), not Blocks(). Scripts in gr.HTML
    # are stripped; this is the documented way to run page JS.
    orbit_js = _ORBIT_JS.read_text(encoding="utf-8")
    kwargs = {
        "js": orbit_js,
        # Off-screen File slot: display:none can block input.click() for Load OBJ.
        "css": (
            "#sn-cmd-wrap { display: none !important; }"
            "#sn-obj-file-slot {"
            " position: fixed !important; left: -100vw !important; top: 0 !important;"
            " width: 8px !important; height: 8px !important; overflow: hidden !important;"
            " opacity: 0 !important; pointer-events: none !important;"
            "}"
            "#sn-obj-file-slot .file-preview-holder, #sn-obj-file-slot table,"
            " #sn-obj-file-slot .filename { display: none !important; }"
            "#sn-orbit-host.sn-drop-over { outline: 2px solid #f5a623; outline-offset: -2px; }"
            "#sn-drop-hint {"
            " position: absolute; left: 0; right: 0; top: 12px;"
            " text-align: center; pointer-events: none; z-index: 2;"
            " font-size: 13px; line-height: 1.3; color: #aabbcc;"
            " text-shadow: 0 1px 2px #1a1a20;"
            "}"
        ),
        "allowed_paths": [str(_glb_dir())],
    }
    port_env = os.environ.get("PORT")
    if port_env:
        demo.launch(server_name="0.0.0.0", server_port=int(port_env), **kwargs)
        return
    demo.launch(server_name="127.0.0.1", server_port=7860, **kwargs)


if __name__ == "__main__":
    main()
