"""SageMaker training entry: same catalog train, data_dir remapped to the channel.

Uploaded from ``sagemaker/`` plus ``src/`` and ``config.yaml``. YAML still
has the local ``E:/`` path; we override ``data_dir`` from
``SM_CHANNEL_TRAINING`` so NPZ ``mesh_path`` values resolve under
``exports/`` + ``meshes/``. After ``train_multi_npz``, copy ``best.pt``
and the run snapshot into SageMaker output so they land in S3 ``output/``.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

# SDK v3 uploads the repo (entry at sagemaker/entry.py). Older launch
# merged ``src/`` next to this file. Accept both layouts.
_HERE = Path(__file__).resolve().parent
if (_HERE / "src").is_dir():
    _CODE = _HERE
elif (_HERE.parent / "src").is_dir():
    _CODE = _HERE.parent
else:
    raise RuntimeError(f"cannot find src/ next to or above {_HERE}")
_SRC = _CODE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Occupancy imports ``geometry.mesh_io``. That directory must exist on this
# box (not an S3 key with a Windows backslash, which Linux will not unpack
# as ``src/geometry/``).
if not (_SRC / "geometry" / "mesh_io.py").is_file():
    names = sorted(p.name for p in _SRC.iterdir()) if _SRC.is_dir() else []
    raise RuntimeError(
        f"src/geometry/mesh_io.py missing under {_SRC}. "
        f"src entries={names}"
    )

from config import load_config, require_data_dir
from train_multi_npz import train_multi_npz


def _training_channel() -> Path:
    """Folder SageMaker mounted for the ``training`` input channel."""
    raw = os.environ.get("SM_CHANNEL_TRAINING") or os.environ.get("SM_CHANNEL_TRAIN")
    if not raw:
        raise RuntimeError(
            "SM_CHANNEL_TRAINING is not set. The launcher must fit() with "
            "channel name 'training'."
        )
    path = Path(raw)
    if not (path / "exports").is_dir() or not (path / "meshes").is_dir():
        raise FileNotFoundError(
            f"training channel must contain exports/ and meshes/: {path}"
        )
    return path


def _copy_job_artifacts(run_dir: Path, model_dir: Path) -> None:
    """Put weights and the run folder where SageMaker packs them to S3."""
    sm_model = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    sm_out = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    sm_model.mkdir(parents=True, exist_ok=True)
    sm_out.mkdir(parents=True, exist_ok=True)

    best = model_dir / "best.pt"
    if best.is_file():
        shutil.copy2(best, sm_model / "best.pt")

    # Viewer / training_log expect models/<run_id>/best.pt and runs/<run_id>/.
    shutil.copytree(model_dir, sm_out / "models" / model_dir.name, dirs_exist_ok=True)
    shutil.copytree(run_dir, sm_out / "runs" / run_dir.name, dirs_exist_ok=True)
    print(f"copied_model_dir={sm_model}")
    print(f"copied_output_data={sm_out}")


def main() -> None:
    data_dir = _training_channel()
    yaml_path = _CODE / "config.yaml"
    # YAML data_dir is the desk path; do not require it on this box.
    cfg = load_config(yaml_path, require_existing_data_dir=False)
    cfg = replace(cfg, data_dir=data_dir)
    require_data_dir(data_dir, yaml_path=yaml_path)
    print(f"data_dir={data_dir}")
    result = train_multi_npz(cfg)
    _copy_job_artifacts(result.run_dir, result.model_dir)


if __name__ == "__main__":
    main()
