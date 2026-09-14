"""Submit one occupancy catalog train as a SageMaker training job.

Run from the repo root on this PC (AWS CLI already configured). Does not
train locally. Quota for ``ml.g5.xlarge`` training-job usage must be >= 1
or ``train()`` fails at launch.

Uses SageMaker Python SDK v3 (``ModelTrainer``). The pip package
``sagemaker>=3`` no longer has ``sagemaker.pytorch.PyTorch``.

Example::

    python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE --dry-run
    python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# Repo root: sagemaker/launch.py → parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGION = "eu-north-1"
_DEFAULT_BUCKET_DATA = (
    "s3://scatteringnet-sagemaker-bucket/scatteringNet/data/"
)
_DEFAULT_BUCKET_OUT = (
    "s3://scatteringnet-sagemaker-bucket/scatteringNet/output"
)
_INSTANCE = "ml.g5.xlarge"
# 1080 catalog was ~10h; A10G should be less. Leave headroom for S3 download.
_MAX_RUN_SECONDS = 18 * 60 * 60

# Do not upload local weights, logs, or the viewer with the job.
_IGNORE = [
    ".git",
    ".cursor",
    ".vscode",
    "models",
    "runs",
    "docs",
    "tests",
    "src/viewer",
    "src/scatter_generation",
    "__pycache__",
    "*.pt",
    "*.pyc",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch occupancy train_multi_npz on SageMaker G5"
    )
    parser.add_argument(
        "--role",
        default=None,
        help="IAM execution role ARN (or set SAGEMAKER_EXECUTION_ROLE)",
    )
    parser.add_argument("--region", default=_DEFAULT_REGION)
    parser.add_argument("--data-s3", default=_DEFAULT_BUCKET_DATA)
    parser.add_argument("--output-s3", default=_DEFAULT_BUCKET_OUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job settings and exit without calling train()",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job finishes (hours). Default: submit and exit.",
    )
    return parser.parse_args()


@contextmanager
def _unix_newlines_for_shell_scripts() -> Iterator[None]:
    """Force LF when the SDK writes ``sm_train.sh`` (Windows ``open('w')`` is CRLF).

    That wrapper runs on Linux. CRLF produced ``$'\r': command not found``
    and exit 2 on ``scatteringnet-n6-20260913103358``. Local occupancy
    train does not use this helper.
    """
    original = builtins.open

    def open_lf(file, mode="r", *args, **kwargs):
        path = str(file)
        text_write = "w" in str(mode) and "b" not in str(mode)
        if text_write and path.endswith(".sh"):
            kwargs.setdefault("newline", "\n")
        return original(file, mode, *args, **kwargs)

    builtins.open = open_lf
    try:
        yield
    finally:
        builtins.open = original


@contextmanager
def _posix_s3_upload_keys() -> Iterator[None]:
    """Force ``/`` in S3 object keys that the SDK builds with ``os.path.relpath``.

    On Windows that relpath is ``src\\geometry``. The object key then contains a
    backslash, so Linux File-mode download never creates ``src/geometry/``.
    Job ``scatteringnet-n6-20260913130138`` died with
    ``ModuleNotFoundError: No module named 'geometry'`` even though the four
    ``geometry/*.py`` files were in the prefix (keys ``src\\geometry/...``).
    Local occupancy train does not use this helper.
    """
    original = os.path.relpath

    def relpath_posix(path, start=os.curdir):
        return original(path, start).replace("\\", "/")

    os.path.relpath = relpath_posix
    try:
        yield
    finally:
        os.path.relpath = original


def _execution_role(cli_role: str | None) -> str:

    role = (cli_role or os.environ.get("SAGEMAKER_EXECUTION_ROLE") or "").strip()
    if not role:
        raise SystemExit(
            "Pass --role arn:aws:iam::ACCOUNT:role/YourSageMakerRole "
            "or set SAGEMAKER_EXECUTION_ROLE. "
            "IAM → Roles → a role SageMaker can assume "
            "(AmazonSageMakerFullAccess + this bucket). "
            "Your user also needs iam:PassRole on that role."
        )
    return role


def main() -> None:
    args = _parse_args()
    try:
        import boto3
        from sagemaker.core import image_uris
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.core.training.configs import (
            Compute,
            InputData,
            OutputDataConfig,
            SourceCode,
            StoppingCondition,
        )
        from sagemaker.train.model_trainer import ModelTrainer
    except ImportError as exc:
        raise SystemExit(
            "Need SageMaker SDK v3 in this env (not torch via pip):\n"
            "  pip install 'sagemaker>=3' boto3\n"
            f"({exc})"
        ) from exc

    role = _execution_role(args.role)
    boto_sess = boto3.Session(region_name=args.region)
    session = Session(boto_session=boto_sess)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"scatteringnet-n6-{stamp}"[:63]

    training_image = image_uris.retrieve(
        framework="pytorch",
        region=args.region,
        version="2.5.1",
        py_version="py311",
        instance_type=_INSTANCE,
        image_scope="training",
    )

    source_code = SourceCode(
        source_dir=str(_REPO_ROOT),
        entry_script="sagemaker/entry.py",
        requirements="sagemaker/requirements.txt",
        ignore_patterns=_IGNORE,
    )
    compute = Compute(
        instance_type=_INSTANCE,
        instance_count=1,
        volume_size_in_gb=50,
    )
    output = OutputDataConfig(s3_output_path=args.output_s3.rstrip("/"))
    stopping = StoppingCondition(max_runtime_in_seconds=_MAX_RUN_SECONDS)
    train_data = InputData(channel_name="training", data_source=args.data_s3)

    trainer = ModelTrainer(
        training_image=training_image,
        role=role,
        sagemaker_session=session,
        source_code=source_code,
        compute=compute,
        output_data_config=output,
        stopping_condition=stopping,
        input_data_config=[train_data],
        training_input_mode="File",
        base_job_name="scatteringnet-n6",
    )

    print(f"job_name={job_name}")
    print(f"instance={_INSTANCE}")
    print(f"region={args.region}")
    print(f"data={args.data_s3}")
    print(f"output={args.output_s3}")
    print(f"entry=sagemaker/entry.py")
    print(f"image={training_image}")
    print("Wait for ml.g5.xlarge training-job quota = 1 before train().")

    if args.dry_run:
        print("dry-run: not submitting")
        return

    with _unix_newlines_for_shell_scripts(), _posix_s3_upload_keys():
        trainer.train(
            input_data_config=[train_data],
            wait=bool(args.wait),
            logs=bool(args.wait),
        )
    print("submitted. SageMaker → Training → Training jobs.")
    if not args.wait:
        print("Re-run with --wait to block on completion, or watch the console.")


if __name__ == "__main__":
    sys.exit(main() or 0)
