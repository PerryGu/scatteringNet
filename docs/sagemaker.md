# SageMaker

The occupancy trains so far ran on the machine under the desk: a GTX 1080. A full catalog pass on that card takes about nine or ten hours. That loop works — the numbers and Fill stills are in [`training_log.md`](training_log.md) — but it is slow, and the card is old.

The next trains run on **Amazon SageMaker**. Same script (`src/train_multi_npz.py`), same `config.yaml` catalog, same occupancy head. Only the GPU changes. This is not a new phase of the model; the occupancy story stays in [`work_plan_phase3.md`](work_plan_phase3.md).

---

## Which box

SageMaker offers a long menu of instances. Most of them are the wrong size for this project.

We use **`ml.g5.xlarge`**: one NVIDIA A10G with 24 GB. That is a clear step up from the 1080 (newer architecture, more memory, more throughput) without jumping to an A100 or H100 cluster we would not fill. The occupancy network is small (`hidden: 64`, `depth: 4`, batch 1024). One GPU is enough.

Wall time will come down. It will not drop to a tenth of ten hours. Loading the catalog and scoring the whole val set still take real time; the card is the part we are upgrading.

---

## Quota

SageMaker will not let you start an instance until the account quota for that type is at least 1. The default for `ml.g5.xlarge` is 0.

The quota to raise is:

**`ml.g5.xlarge for training job usage`** → **1**

That is permission to run a **training job**. There is a similar line for **endpoint usage**; that one is for hosting a live API, which we are not doing yet. Asking for the endpoint quota by mistake does not help training.

Raising the quota does not start a machine and does not cost anything. The hourly bill starts only while a job is actually running.

A request for this quota is in (**Case Opened**, 9 Sep 2026). The applied value is still 0 until AWS closes that case. Do not ``fit()`` a G5 job until Service Quotas shows **1**.

---

## Data on S3

The job cannot see `E:/Work_stuff/scatteringNet/data`. The training set lives in the bucket, with the same layout as local `data_dir`: `exports/` and `meshes/` as siblings, so each NPZ `mesh_path` still resolves.

Prefix:

`s3://scatteringnet-sagemaker-bucket/scatteringNet/data/`

That copy is **up** (full local `data/` tree, including Combos / organics that are not in the YAML catalog). Later jobs reuse it. Re-sync only when NPZs or meshes change:

```text
aws s3 sync "E:\Work_stuff\scatteringNet\data" "s3://scatteringnet-sagemaker-bucket/scatteringNet/data/" --region eu-north-1
```

Use the CLI for this many files, not the console. Checkpoints and run snapshots come **back** under `s3://scatteringnet-sagemaker-bucket/scatteringNet/output/`. Code stays with the job; it does not live under `data/`.

On the box, `data_dir` must be the downloaded channel root that still contains both `exports/` and `meshes/`. The job entry (`sagemaker/entry.py`) sets that from the SageMaker channel; do not leave `config.yaml` pointing at `E:/` on the box.

---

## Output from S3

When a job Completes, SageMaker writes two tarballs under:

`s3://scatteringnet-sagemaker-bucket/scatteringNet/output/<job-name>/output/`

- `output.tar.gz` — `models/<run_id>/best.pt` and `runs/<run_id>/` (viewer and `training_log.md` need this one)
- `model.tar.gz` — a copy of `best.pt` at the SageMaker model dir (optional)

The console cannot download a **folder** prefix. Open the `output/` object prefix and grab the `.tar.gz` files, or use the CLI.

From the repo root (PowerShell). `<job-name>` is the Training job name (`scatteringnet-n6-…`). Unpack into this repo so `models/` and `runs/` land next to the local trains:

```text
aws s3 cp "s3://scatteringnet-sagemaker-bucket/scatteringNet/output/<job-name>/output/output.tar.gz" "$env:TEMP\<name>.tar.gz" --region eu-north-1
tar -xf "$env:TEMP\<name>.tar.gz" -C .
```

Example (k=24 job):

```text
aws s3 cp "s3://scatteringnet-sagemaker-bucket/scatteringNet/output/scatteringnet-n6-20260913221313/output/output.tar.gz" "$env:TEMP\sm_knn24_output.tar.gz" --region eu-north-1
tar -xf "$env:TEMP\sm_knn24_output.tar.gz" -C .
```

---

## How to launch

The training loop is still `src/train_multi_npz.py`. SageMaker job files live in **`sagemaker/`**:

- `sagemaker/entry.py` — runs **on the G5** (remap `data_dir`, train, copy `best.pt` + `runs/` to S3 output).
- `sagemaker/launch.py` — run **on this PC** after `pip install sagemaker boto3` in the conda env (SDK **v3**, `ModelTrainer`). On Windows it writes the job’s `sm_train.sh` with Unix line endings, and it uploads source with POSIX S3 keys (`src/geometry/…`, not `src\geometry/…`) so Linux can import `geometry`.
- `sagemaker/requirements.txt` — extra pip packages on the training image (not the local conda env; that is still `environment.yaml`).

You need an IAM **execution role** SageMaker can assume (often `AmazonSageMaker-ExecutionRole-…`), with access to this bucket. Pass its ARN. Your user needs `iam:PassRole` on that role.

Dry run (no instance, no bill):

```text
python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE --dry-run
```

Submit (only after training-job quota is 1):

```text
python sagemaker/launch.py --role arn:aws:iam::ACCOUNT:role/ROLE
```

That does not open a Studio notebook. Watch **SageMaker → Training → Training jobs**. Add `--wait` only if you want the terminal to block until the job ends.

The launcher prints `Training Job Name: scatteringnet-n6-…` (no extra dashes in the timestamp). Follow CloudWatch from this PC. `--format short` is print style only (timestamp + message). Log streams appear after the container starts; if the first tail is empty, wait a minute and run it again:

```text
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job-name> --region eu-north-1 --since 1h --follow --format short
```

Example (nosmooth job):

```text
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix scatteringnet-n6-20260914215452 --region eu-north-1 --since 1h --follow --format short
```

---

## First train on SageMaker

When the quota is approved, the first job should match the inspect checkpoint we already trust, unless we decide on a new A/B first:

`2026-09-05_12-18-28_prim_extruded_nr45_knn16_n6`

That means envelope + local k-NN, `knn_k: 16`, six-D skin dots (position and face normal), 1024 envelope samples, mix 75. Fresh train if the head width changed; do not paste old weights into a different first layer.

When it finishes, write it up in [`training_log.md`](training_log.md) like any other catalog train. The run snapshot (`runs/<id>/config.yaml`) should name the A10G under `gpu`, the same way local runs name the 1080.

---

## Where things stand

Local 1080 trains are done. First SageMaker n6 catalog train finished: job `scatteringnet-n6-20260913141750`, run `2026-09-13_11-23-44_prim_extruded_nr45_knn16_n6` (A10G, wall ~5 h, val IoU 0.967 — same band as the 1080 n6). Write-up in [`training_log.md`](training_log.md). Inspect default stays `12-18-28_…_n6` until Fill is judged. Hosting an endpoint can wait.
