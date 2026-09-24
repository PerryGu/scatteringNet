"""Assemble a slim Gradio tree and force-push it to the Hugging Face Space.

Copies only infer + Gradio text files. No docs/media, no SageMaker, no .pt.
Run from the repo (conda env scatteringNet):

    python space/push_space.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPACE = Path(__file__).resolve().parent
_REMOTE = "https://huggingface.co/spaces/guyPerry/scatteringnet"

# Text / code the Space needs to import scatteringnet and run app.py.
_FILES = (
    "config.yaml",
    "docs/inspect_checkpoint.yaml",
    "src/__init__.py",
    "src/config.py",
    "src/data_npz.py",
    "src/infer_multi_npz.py",
    "src/metrics.py",
    "src/normalize.py",
    "src/occupancy_encoder.py",
    "src/occupancy_mlp.py",
    "src/geometry/__init__.py",
    "src/geometry/mesh_io.py",
    "src/geometry/surface.py",
    "src/geometry/trimesh_util.py",
    "src/viewer/__init__.py",
    "src/viewer/infer_job.py",
    "src/viewer/mesh_access.py",
    "src/viewer/model_access.py",
    "src/viewer/obj_fill.py",
    "src/gradio/app.py",
    "src/gradio/figure.py",
    "src/gradio/pipeline.py",
    "src/gradio/orbit.js",
    "src/gradio/examples/Obese.obj",
    "src/gradio/examples/horse.obj",
    "src/gradio/examples/Player.obj",
    "src/gradio/examples/dog.obj",
    "src/gradio/examples/Helix_bend.obj",
    "src/gradio/examples/TorusX3_box.obj",
)


def _run(args: list[str], cwd: Path) -> None:
    """Run a git command; raise if it fails."""
    subprocess.run(args, cwd=cwd, check=True)


def assemble(dest: Path) -> None:
    """Copy the allowlist plus Space metadata into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)

    for rel in _FILES:
        src = _REPO / rel
        if not src.is_file():
            raise FileNotFoundError(rel)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        # HF pip runs from /tmp and cannot ``-e``-install this tree.
        # Mirror occupancy modules so ``import scatteringnet`` works from repo root.
        if rel.startswith("src/") and not rel.startswith("src/gradio/"):
            pkg = dest / "scatteringnet" / rel[len("src/") :]
            pkg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, pkg)

    shutil.copy2(_SPACE / "README.md", dest / "README.md")
    shutil.copy2(_SPACE / "requirements.txt", dest / "requirements.txt")
    shutil.copy2(_SPACE / "pyproject.toml", dest / "pyproject.toml")

    models = dest / "models"
    models.mkdir()
    (models / ".gitkeep").write_text("", encoding="utf-8")
    (dest / ".gitignore").write_text("*.pt\n", encoding="utf-8")


def push(dest: Path) -> None:
    """New git repo in ``dest``, force-push to the Space only."""
    _run(["git", "init", "-b", "main"], dest)
    _run(["git", "add", "-A"], dest)
    _run(
        [
            "git",
            "-c",
            "user.email=space@local",
            "-c",
            "user.name=scatteringnet-space",
            "commit",
            "-m",
            "Slim Gradio Space: infer + demo only",
        ],
        dest,
    )
    _run(["git", "remote", "add", "origin", _REMOTE], dest)
    _run(["git", "push", "-u", "origin", "main", "--force"], dest)


def main() -> None:
    dest = Path(tempfile.mkdtemp(prefix="scatteringnet_hf_space_"))
    print("assemble", dest)
    assemble(dest)
    print("push", _REMOTE)
    push(dest)
    print("done. Upload models/2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6/best.pt in the Space Files UI.")


if __name__ == "__main__":
    main()
