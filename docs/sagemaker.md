# SageMaker

The occupancy trains so far ran on the machine under the desk: a GTX 1080. A full catalog pass on that card takes about nine or ten hours. That loop works — the numbers and Fill stills are in [`training_log.md`](training_log.md) — but it is slow, and the card is old.

The next trains run on **Amazon SageMaker**. Same occupancy problem, same catalog script (`src/train_multi_npz.py`), same write-up in [`training_log.md`](training_log.md). This is not a new phase of the model; the occupancy story stays in [`work_plan_phase3.md`](work_plan_phase3.md).

It will not be a single SageMaker job. It is a series of catalog trains, the same loop as on the 1080: train, look at Fill and the numbers, then change something and train again. Between jobs the code, the YAML, or the training setup may move — that is how we try to improve the model. The faster box is there so those iterations do not each cost a full day.

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

A request for this quota is already in. The first SageMaker catalog train has not run yet.

---

## First train, then the rest

When the quota is approved, the **first** job should match the inspect checkpoint we already trust, unless we decide on a new A/B before that:

`2026-09-05_12-18-28_prim_extruded_nr45_knn16_n6`

That means envelope + local k-NN, `knn_k: 16`, six-D skin dots (position and face normal), 1024 envelope samples, mix 75. Fresh train if the head width changed; do not paste old weights into a different first layer.

Jobs after that are not copies of this recipe. If Fill still leaks, or val looks fine but the stills do not, we change the setup and run another catalog train. Each finished job still gets a [`training_log.md`](training_log.md) entry. The run snapshot (`runs/<id>/config.yaml`) should name the A10G under `gpu`, the same way local runs name the 1080.

---

## Where things stand

Local 1080 trains are done. SageMaker quota for `ml.g5.xlarge` training-job usage is requested (that number is how many of those boxes can run at once, not how many trains we will ever do). The first SageMaker catalog train has not run yet. Hosting an endpoint can wait.
