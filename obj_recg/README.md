# uplevels_cv

A benchmark harness for comparing computer-vision model families on a sports
training task: **detect** the athlete and equipment (bounding boxes) and **trace
the limb lines** of a golf swing (17 pose keypoints → skeleton edges). Same data,
same metrics, three families side by side.

| Family | Model(s) | Style | Role in the comparison |
|---|---|---|---|
| `yolo` | YOLO11-nano detect + pose | one-stage, anchor-free | speed/size leader |
| `resnet` | Faster R-CNN + Keypoint R-CNN, ResNet-50 FPN | two-stage | accuracy leaning |
| `cnn` | SSDlite MobileNetV3 | one-stage, lightweight | the cheap baseline to beat |

The `pose` models produce the golf-swing limb overlay; the `detect` models locate
the golfer/club/ball. Every model reports the same columns — detection mAP, pose
OKS, p50/p95 latency, parameter count, model size — into one gold table so you can
answer "which wins on accuracy vs. speed vs. footprint?".

## Target

- **Workspace:** SLED (`SLED_fe_demo` → `adb-3438839487639471...`)
- **UC namespace:** `rpeng_upleveling.object_and_vision`
- **Tables:** prefixed `00_seeing_models_` (the leading number sorts this project
  first in the catalog explorer). Names lead with a digit, so SQL must backtick them:
  `` `rpeng_upleveling`.`object_and_vision`.`00_seeing_models_comparison` ``.
- **Compute:** serverless GPU (AI Runtime, env v5 / Runtime 17.3). Training runs
  inside each notebook via the `@distributed` API — no cluster to manage.

## Layout

```
databricks.yml                 bundle config, variables, dev/prod targets
resources/
  schema.yml                   object_and_vision schema
  volume.yml                   cv_data managed volume (images, labels, artifacts)
  pipeline.yml                 medallion LDP (bronze → silver → gold)
  train_compare.job.yml        setup → medallion → [yolo | resnet | cnn] → compare
transformations/               LDP source (one dataset per file)
  bronze.py                    Auto Loader ingest: raw COCO labels + frame metadata
  silver.py                    exploded, validated instances + image index
  gold.py                      per-image training manifest
src/uplevels_cv/
  config.py                    model zoo + Paths (add a model = one line here)
  skeleton.py                  COCO-17 keypoints, limb edges, golf joint angles
  data.py                      UC Volume access, COCO→YOLO export, manifest read
  models.py                    ModelSpec → concrete model
  train.py                     dispatch + MLflow + serverless-GPU @distributed
  evaluate.py                  COCO mAP / OKS, latency, footprint
  compare.py                   aggregate runs → gold model-comparison table
notebooks/                     00_setup … 04_compare (Databricks py-source)
fixtures/golf_swing_skeleton.json   the limb-line drawing spec
tests/unit/                    zoo + skeleton tests (no GPU needed)
```

## Data shape (medallion)

A serverless Lakeflow Declarative Pipeline builds three layers in
`rpeng_upleveling.object_and_vision`, all prefixed `00_seeing_models_`:

| Layer | Table | What |
|---|---|---|
| bronze | `..._bronze_raw_labels` | raw COCO JSON, ingested as-is (Auto Loader) |
| bronze | `..._bronze_frames` | frame image metadata (path/size/split) |
| silver | `..._silver_images` | one row per image (id → file, size, split) |
| silver | `..._silver_keypoints` | one validated instance per row (bbox + 17 kpts), expectations drop degenerate rows |
| gold | `..._gold_training_manifest` | per-image manifest with split + counts — trainers read this |
| gold | `..._gold_model_comparison` | the leaderboard, written from MLflow runs by `compare.py` |

End-to-end flow the benchmark job runs:

```
00_setup_data ─> medallion pipeline ─┬─> train_yolo   ─┐
   (stage data)   (bronze→silver→     ├─> train_resnet ─┼─> 04_compare
                   gold manifest)     └─> train_cnn    ─┘   (gold leaderboard)
```

## Quickstart

```bash
# 0. One-time: local package + tests
uv venv --python 3.12.3 && uv sync --extra dev
uv run pytest tests/unit -q

# 1. Validate and deploy the bundle to dev
databricks bundle validate -t dev -p SLED_fe_demo
databricks bundle deploy   -t dev -p SLED_fe_demo   # creates schema + volume

# 2. Stage data into the volume (see notebook 00 — TODO: drop labeled data)

# 3. Run the whole benchmark (setup → medallion → train 3 families → compare)
databricks bundle run train_and_compare -t dev -p SLED_fe_demo
```

The leaderboard lands at
`` `rpeng_upleveling`.`object_and_vision`.`00_seeing_models_gold_model_comparison` ``.

## What's wired vs. what's a TODO

**Wired and runnable:** the bundle (validates + deploys the schema/volume/pipeline/job),
the medallion pipeline structure (Auto Loader bronze, silver with expectations,
gold manifest), the model zoo and dispatch, MLflow logging + UC registration with
a `@prod` alias, the shared metrics (latency, footprint, and the YOLO mAP/OKS path
via Ultralytics' own validator), and the comparison-table writer.

**TODO (needs your labeled data):**
- Stage COCO-format annotations + frames in the volume (notebook `00`).
- In `transformations/silver.py`, confirm the COCO field paths match your export.
- The torchvision train loop in `train._torchvision_train` and the COCOeval loop
  in `evaluate.evaluate_torchvision` — both marked inline. YOLO trains end-to-end
  once the `export_yolo_dataset` per-image label conversion in `data.py` is filled in.
- Optional: enrich the gold manifest with swing joint angles (`gold.py` TODO).

## Serverless GPU notes

- `train.run_on_gpu` uses `serverless_gpu.api.distributed`; if that import fails
  (feature not enabled) it falls back to in-process, so the code path is portable.
- Verify serverless GPU is enabled on the workspace. If not, use the **classic GPU
  cluster** fallback — a ready-to-uncomment `job_clusters` block (Azure
  `Standard_NC*` node types) is at the bottom of `resources/train_compare.job.yml`.
- `gpu_type` is a bundle variable (`A10` default, `H100` for the heavier runs).

## Extending

Add a model: append one `ModelSpec` to `_ZOO` in `config.py`. Dispatch,
evaluation, and the comparison table pick it up automatically. Add a training
notebook only if it's a new family; existing notebooks train every spec in their
family via `specs_for(family=...)`.
