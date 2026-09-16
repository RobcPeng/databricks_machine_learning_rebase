# uplevels_cv

A Databricks Asset Bundle for sports computer vision and team-performance
prediction. It has two layers:

1. **CV benchmark**: trains and evaluates multiple model families on object
   detection (bounding boxes) and pose estimation (keypoints → skeleton edges),
   per sport, and writes one gold table ranking them on shared metrics.
2. **Team-performance prediction**: aggregates the CV-derived features to
   team-game and per-athlete level and trains tabular models for game outcome,
   score, efficiency, player index, and scholarship ROI.

Both layers are parameterized by sport (golf, football, basketball, baseball,
hockey, tennis) and by a registry, so adding a sport, a CV model, or a
prediction target is a one-line change.

This README documents every file, the data model, the commands, and the
extension points, for use by a person or an agent.

## Contents

1. [Target environment](#1-target-environment)
2. [Sports](#2-sports)
3. [CV model registry](#3-cv-model-registry)
4. [Team-performance targets](#4-team-performance-targets)
5. [Repository layout](#5-repository-layout)
6. [Data architecture (medallion)](#6-data-architecture-medallion)
7. [Metrics](#7-metrics)
8. [Setup](#8-setup)
9. [Deploy and run](#9-deploy-and-run)
10. [Execution flow](#10-execution-flow)
11. [How a model runs](#11-how-a-model-runs)
12. [Configuration reference](#12-configuration-reference)
13. [Extending the project](#13-extending-the-project)
14. [Implemented vs. TODO](#14-implemented-vs-todo)
15. [Framework install notes](#15-framework-install-notes)
16. [Troubleshooting](#16-troubleshooting)
17. [Next steps](#17-next-steps)

## 1. Target environment

| Setting | Value |
|---|---|
| Workspace | SLED — `adb-3438839487639471.11.azuredatabricks.net` (Azure) |
| CLI profile | `SLED_fe_demo` |
| Catalog | `rpeng_upleveling` (must already exist) |
| Schemas | one per medallion layer: `seeing_models_00_landing`, `seeing_models_01_bronze`, `seeing_models_02_silver`, `seeing_models_03_gold` (created by the bundle) |
| Volume | `cv_data`, managed, in the landing schema (created by the bundle) |
| Project | `seeing_models` (prod) / `seeing_models_dev` (dev): schema = `<project>_<numbered_layer>`; table names are bare |
| Compute | serverless GPU (AI Runtime, env v5 / Runtime 17.3); MediaPipe and the tabular models run on CPU |

Schema names start with the project (`seeing_models_…`), so they are plain
identifiers that need no backticking, e.g.
`rpeng_upleveling.seeing_models_03_gold.model_comparison`.

The dev and prod targets write to separate namespaces, so a routine
`bundle deploy` (which defaults to `dev`) can't overwrite prod. Dev uses the
`seeing_models_dev_*` schemas and the `uplevels_cv_dev` MLflow experiment; prod
uses `seeing_models_*` and `uplevels_cv`. The `seeing_models_*` names in the
examples below are the prod ones; on the default dev target, read
`seeing_models_dev_*`.

## 2. Sports

Defined in `src/uplevels_cv/sports.py` (`SPORTS`). Each sport sets the detection
classes, signature events, biomechanics angles, and scoring unit. Data is
namespaced by sport under the volume, and every gold table carries a `sport`
column.

| Sport | Detection classes | Score unit |
|---|---|---|
| golf | golfer, club, ball | strokes |
| football | player, football, goalpost, pylon | points |
| basketball | player, basketball, rim, backboard | points |
| baseball | player, baseball, bat, glove, base | runs |
| hockey | skater, puck, stick, goal_net | goals |
| tennis | player, tennis_ball, racket, net | points |

The 17-keypoint COCO person layout is shared across sports (`skeleton.py`). Each
sport defines its own biomechanics angles (e.g. golf `lead_arm`, basketball
`shooting_elbow`) as triples of COCO keypoints, measured by `skeleton.joint_angle`.
Golf and tennis are individual sports; their team performance follows NCAA team
scoring (aggregate strokes; dual-match points).

## 3. CV model registry

Defined in `src/uplevels_cv/config.py` (`MODEL_ZOO`). Nine specs across six
families; `specs_for(family=...)` / `specs_for(task=...)` filter the zoo.

| key | family | task | framework | weights | trainable |
|---|---|---|---|---|---|
| `yolo_detect` | yolo | detect | Ultralytics | `yolo11n.pt` | yes |
| `yolo_pose` | yolo | pose | Ultralytics | `yolo11n-pose.pt` | yes |
| `resnet_detect` | resnet | detect | torchvision | `fasterrcnn_resnet50_fpn_v2` | yes |
| `resnet_pose` | resnet | pose | torchvision | `keypointrcnn_resnet50_fpn` | yes |
| `fast_rcnn_detect` | resnet | detect | torchvision | `fasterrcnn_resnet50_fpn` (external proposals) | yes |
| `cnn_detect` | cnn | detect | torchvision | `ssdlite320_mobilenet_v3_large` | yes |
| `rtdetr_detect` | detr | detect | Ultralytics | `rtdetr-l.pt` | yes |
| `rtmpose_pose` | mmpose | pose | MMPose | `rtmpose-m` | yes |
| `blazepose_pose3d` | mediapipe | pose3d | MediaPipe | `pose_landmarker_full.task` | no |

Tasks: `detect` = bounding boxes; `pose` = 17 COCO keypoints; `pose3d` = 33
BlazePose landmarks with depth. `trainable: no` = pretrained, evaluated only.

Notes:
- **Fast R-CNN**: torchvision ships no native Fast R-CNN. `fast_rcnn_detect` runs
  the Faster R-CNN architecture fed external Selective Search proposals
  (`models.selective_search`, needs `opencv-contrib`) with the RPN bypassed.
- **RTMPose** is top-down and needs person boxes from a detector (e.g. `rtdetr_detect`).
- **BlazePose** is scored on a COCO-17 projection for 2D OKS; its 3D metric
  (`mpjpe_mm`) needs 3D ground truth and is left blank on 2D COCO data.

## 4. Team-performance targets

Defined in `src/uplevels_cv/team_performance.py` (`TARGETS`); `targets_for(scope=...)`
filters by grain.

| key | type | scope | label | feature table |
|---|---|---|---|---|
| `win_probability` | classification | team_game | `win` | `team_game_features` |
| `points_for` | regression | team_game | `points_for` | `team_game_features` |
| `points_against` | regression | team_game | `points_against` | `team_game_features` |
| `off_efficiency` | regression | team_game | `off_efficiency` | `team_game_features` |
| `def_efficiency` | regression | team_game | `def_efficiency` | `team_game_features` |
| `player_index` | regression | athlete | `player_rating` | `athlete_metrics` |
| `scholarship_roi` | regression | recruit | `roi` | `athlete_metrics` |

Each target trains one XGBoost model (classifier or regressor), registered to
Unity Catalog as `seeing_models_03_gold.<sport>_teamperf_<key>` with a `@prod` alias, then
batch-scored into `seeing_models_03_gold.team_predictions`.

## 5. Repository layout

```
databricks.yml                       bundle config: variables, dev/prod targets, sync
pyproject.toml                       package + optional deps (train, mmpose, mediapipe, dev)
resources/
  schema.yml                         four medallion-layer schemas (landing/bronze/silver/gold)
  volume.yml                         cv_data managed volume
  pipeline.yml                       medallion LDP definition
  train_compare.job.yml              multi-task job (setup → medallion → train → compare)
transformations/                     LDP source (one dataset per file)
  landing.py                         Auto Loader raw ingest: labels, frames, game results
  bronze.py                          structured records: images, keypoints, games (exploded/typed)
  silver.py                          validated records (quality expectations)
  gold.py                            training manifest, athlete metrics, team-game features
src/uplevels_cv/
  __init__.py                        package overview + public exports
  config.py                          Family/Task enums, ModelSpec, MODEL_ZOO, Paths
  sports.py                          Sport registry: classes, events, angles, score unit
  skeleton.py                        COCO-17 keypoints, limb edges, joint-angle math
  skeleton_blazepose.py              BlazePose 33 landmarks + COCO-17 projection
  data.py                            sport-namespaced volume paths, COCO loader, YOLO export
  models.py                          build_model(spec) factory + selective_search
  train.py                           train_spec dispatch, serverless-GPU wrapper, MLflow
  evaluate.py                        mAP / OKS / MPJPE, latency, footprint
  compare.py                         aggregate CV runs → gold model-comparison table
  team_performance.py                PredictionTarget registry, train + batch-score
notebooks/
  00_setup_data.py                   volume layout + per-sport data staging
  01_train_yolo.py                   yolo family
  02_train_resnet.py                 resnet family (Faster / Keypoint / Fast R-CNN)
  03_train_cnn_baseline.py           cnn family (SSDlite)
  04_compare_models.py               write + display the CV leaderboard (per sport)
  05_train_rtdetr.py                 detr family (RT-DETR)
  06_train_rtmpose.py                mmpose family (RTMPose)
  07_eval_mediapipe.py               mediapipe family (BlazePose, inference-only)
  08_train_team_performance.py       train + score the prediction targets
fixtures/golf_swing_skeleton.json    COCO-17 skeleton drawing spec
tests/unit/
  test_config.py                     zoo, Paths, trainable flag
  test_skeleton.py                   COCO-17 + BlazePose skeletons, angle math
  test_sports.py                     sport registry
  test_team_performance.py           prediction-target registry
```

## 6. Data architecture (medallion)

A serverless Lakeflow Declarative Pipeline (`resources/pipeline.yml`, source in
`transformations/`) builds landing → bronze → silver → gold. Each layer is its
own schema, `<numbered_layer>_seeing_models`, so schemas sort in medallion order;
table names are bare and every table carries a `sport` column. The pipeline
publishes to all four schemas via fully-qualified dataset names.

| Schema | Table | Type | Contents |
|---|---|---|---|
| `seeing_models_00_landing` | `raw_labels` | streaming (Auto Loader) | raw COCO JSON, one row per file |
| `seeing_models_00_landing` | `frames` | streaming (Auto Loader) | frame metadata (path, size, sport, split) |
| `seeing_models_00_landing` | `games` | streaming (Auto Loader) | game-result CSV rows |
| `seeing_models_01_bronze` | `images` | streaming | one row per image (id, file_name, size, sport, split) |
| `seeing_models_01_bronze` | `keypoints` | streaming | one instance per row (bbox + 17 keypoints, sport), unfiltered |
| `seeing_models_01_bronze` | `games` | streaming | one team-game per row, typed |
| `seeing_models_02_silver` | `images` | streaming | images with a known split |
| `seeing_models_02_silver` | `keypoints` | streaming | instances with a real box and ≥5 visible keypoints (expectations drop the rest) |
| `seeing_models_02_silver` | `games` | streaming | games with a non-negative score |
| `seeing_models_03_gold` | `training_manifest` | materialized view | per-image manifest (sport, split, counts); CV trainers read it for the split assignment |
| `seeing_models_03_gold` | `athlete_metrics` | materialized view | per-instance CV features; feeds `player_index` and `scholarship_roi` |
| `seeing_models_03_gold` | `team_game_features` | materialized view | per team-game features (rolling form) + labels; feeds the team-game targets |
| `seeing_models_03_gold` | `model_comparison` | Delta table | CV leaderboard, written by `compare.py` |
| `seeing_models_03_gold` | `team_predictions` | Delta table | team-performance predictions, written by `team_performance.py` |

Landing ingests files verbatim; bronze structures them (explodes the COCO arrays,
types the game rows) without filtering; silver applies quality expectations; gold
aggregates. Registered models also live in the gold schema
(`seeing_models_03_gold.<sport>_<model>`).

Both gold Delta tables are partitioned by `sport` and written with `replaceWhere`
on `sport`, so per-sport runs update only their own rows and the tables
accumulate every sport.

Volume layout under `/Volumes/rpeng_upleveling/seeing_models_00_landing/cv_data/`:

```
raw/<sport>/            original videos / frame dumps
images/<sport>/<split>/ extracted frames
labels/<sport>/         COCO json — instances_<split>.json, person_keypoints_<split>.json
games/<sport>/          game results CSV (team-performance labels)
splits/<sport>/         YOLO-format export written by data.export_yolo_dataset()
artifacts/              per-run model files, plots, downloaded checkpoints
predictions/            scored overlays
```

## 7. Metrics

CV metrics (`compare.METRIC_COLS`, computed in `evaluate.py`):

| Column | Meaning | Applies to |
|---|---|---|
| `detect_map` / `detect_map50` | COCO bbox mAP@[.5:.95] and @.5 | detect models |
| `pose_map` / `pose_oks` | COCO keypoint OKS mAP and @.5 | 2D pose models |
| `mpjpe_mm` | mean per-joint position error | 3D pose (needs 3D ground truth) |
| `latency_ms_p50` / `latency_ms_p95` | single-image inference latency | all |
| `params_millions` / `size_mb` | model footprint | torch models |
| `train_seconds` | wall-clock training time | trainable models |

Detection mAP and keypoint OKS use `pycocotools` COCOeval. Team-performance
targets report `roc_auc` (classification) or `rmse` (regression).

## 8. Setup

1. Confirm the catalog exists and you are authenticated:
   ```bash
   databricks catalogs get rpeng_upleveling --profile SLED_fe_demo
   databricks current-user me --profile SLED_fe_demo
   ```
2. Create the local environment and run the tests:
   ```bash
   cd obj_recg
   uv venv --python 3.12.3
   uv sync --extra dev
   uv run pytest tests/unit -q
   ```

## 9. Deploy and run

1. Validate and deploy (creates the schema, volume, pipeline, and job):
   ```bash
   databricks bundle validate -t dev -p SLED_fe_demo
   databricks bundle deploy   -t dev -p SLED_fe_demo
   ```
2. Stage labeled data into the volume for each sport. Notebook `00_setup_data.py`
   documents the expected COCO files, game CSVs, and directory layout.
3. Run the benchmark for one sport (`sport` defaults to `golf`):
   ```bash
   databricks bundle run train_and_compare -t dev -p SLED_fe_demo
   ```
4. Run another sport by overriding the job parameter:
   ```bash
   databricks bundle run train_and_compare -t dev -p SLED_fe_demo --params sport=basketball
   ```
   Repeat for `football`, `baseball`, `hockey`, `tennis`. The gold tables
   accumulate all sports.
5. Read the results:
   ```sql
   SELECT * FROM rpeng_upleveling.seeing_models_03_gold.model_comparison
   ORDER BY sport, task, pose_map DESC, detect_map DESC;

   SELECT * FROM rpeng_upleveling.seeing_models_03_gold.team_predictions
   WHERE sport = 'basketball';
   ```

To run one family or the predictive layer alone, open its notebook (Section 5)
and run it against the same widgets.

## 10. Execution flow

The `train_and_compare` job (sport is a job parameter):

```
00_setup_data ─> run_medallion ─┬─> train_yolo     (01) ─┐
   (stage data)   (LDP pipeline) ├─> train_resnet   (02) ─┤
                                 ├─> train_cnn      (03) ─┼─> compare (04)
                                 ├─> train_rtdetr   (05) ─┤   → gold.model_comparison
                                 ├─> train_rtmpose  (06) ─┤
                                 ├─> eval_mediapipe (07) ─┘
                                 └─> train_team_performance (08) → gold.team_predictions
```

Task → notebook → environment:

| Task | Notebook | Family / layer | Environment |
|---|---|---|---|
| `setup_data` | `00_setup_data.py` | — | `cv_env` |
| `run_medallion` | pipeline | — | serverless (pipeline) |
| `train_yolo` | `01_train_yolo.py` | yolo | `cv_env` |
| `train_resnet` | `02_train_resnet.py` | resnet | `cv_env` |
| `train_cnn` | `03_train_cnn_baseline.py` | cnn | `cv_env` |
| `train_rtdetr` | `05_train_rtdetr.py` | detr | `cv_env` |
| `train_rtmpose` | `06_train_rtmpose.py` | mmpose | `mmpose_env` |
| `eval_mediapipe` | `07_eval_mediapipe.py` | mediapipe | `mediapipe_env` |
| `train_team_performance` | `08_train_team_performance.py` | predictive | `teamperf_env` |
| `compare` | `04_compare_models.py` | — | `cv_env` |

## 11. How a model runs

### CV models (`train.train_spec`)

Each CV notebook calls `train_spec(spec, data, ...)` for every spec in its family:

1. Merge hyperparameters (spec defaults + per-run overrides).
2. Point MLflow at the Unity Catalog registry and the family experiment.
3. Open a run named `<sport>_<model_key>`; log the spec, sport, and hyperparameters.
4. Select the trainer and wrap it for serverless GPU unless the spec is
   inference-only: `yolo`/`detr` → `_ultralytics_train`; `mmpose` → `_mmpose_train`;
   `resnet`/`cnn` → `_torchvision_train`; `trainable == False` → `_inference_only`.
5. Run it → `(model, metrics, artifact_path)`; record train time.
6. Log numeric metrics.
7. Log the artifact and register the model to UC as `seeing_models_03_gold.<sport>_<model_key>`
   (best-effort); on success, move the `@prod` alias.
8. Return the metrics dict.

Serverless GPU is requested via `run_on_gpu`, which applies the
`serverless_gpu.api.distributed` decorator; if unavailable it runs in-process, so
the code path works on a classic GPU cluster or locally too.

### Team-performance models (`team_performance.train_target`)

Notebook `08` calls `train_target` for every target, then `score_targets`:

1. Load features + label for the sport from the gold feature table.
2. Train/test split (stratified for classification).
3. Train XGBoost (classifier or regressor by target type) with MLflow autolog.
4. Score the primary metric (`roc_auc` / `rmse`) on the test split.
5. Register the model to UC as `seeing_models_03_gold.<sport>_teamperf_<key>` and move `@prod`.

`score_targets` loads each `@prod` model, scores the latest features, and writes
long-format rows (sport, scope, target, entity_id, prediction) to
`seeing_models_03_gold.team_predictions`.

## 12. Configuration reference

Bundle variables (`databricks.yml`), overridable with `--var name=value`:

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `rpeng_upleveling` | Unity Catalog for the schemas, volume, models |
| `project` | `seeing_models` (dev overrides to `seeing_models_dev`) | schema per layer = `<project>_<numbered_layer>` (e.g. `seeing_models_03_gold`) |
| `volume` | `cv_data` | managed volume name (in the landing schema) |
| `experiment_root` | `/Users/rob.peng@databricks.com/uplevels_cv` (dev: `…/uplevels_cv_dev`) | parent folder for MLflow experiments |
| `serverless_env_version` | `5` | serverless environment version (job `client`) |
| `gpu_type` | `A10` | serverless GPU accelerator |

Job parameter `sport` (default `golf`) is set per run with
`--params sport=<sport>`. Targets: `dev` (default; its own `seeing_models_dev_*`
schemas and `uplevels_cv_dev` experiment, no name prefix, schedules paused) and
`prod` (`seeing_models_*`, runs as the owner).

## 13. Extending the project

- **Add a CV model**: append one `ModelSpec` to `_ZOO` in `config.py`. The
  family's notebook trains it and `compare.py` includes it.
- **Add a sport**: append one `SportSpec` to `_SPORTS` in `sports.py` (detection
  classes, events, angles, score unit). Stage data under the new sport's volume
  paths and run the job with `--params sport=<new>`.
- **Add a prediction target**: append one `PredictionTarget` to `_TARGETS` in
  `team_performance.py`. Notebook `08` trains and scores it.
- **Add a CV family**: add the `Family` value, its spec(s), a `build_model`
  branch, a `train._select_trainer` branch and trainer, an evaluator, a notebook,
  and a job task with the right environment.

## 14. Implemented vs. TODO

Implemented and runnable:
- Bundle: validates and deploys the schema, volume, pipeline, and job.
- Sport registry and prediction-target registry, threaded through config, data,
  training, comparison, and the medallion.
- Medallion: Auto Loader bronze (labels, frames, games), silver with expectations,
  gold manifest / athlete metrics / team-game features (leakage-safe rolling form).
- CV model zoo, `build_model` factory, and `train_spec` dispatch for six families.
- Team-performance training (XGBoost + autolog + UC registration) and batch scoring.
- MLflow logging, best-effort UC registration with a `@prod` alias.
- CV latency/footprint metrics; the Ultralytics mAP/OKS path via `model.val()`.
- Comparison-table and predictions-table writers (per-sport `replaceWhere`).

TODO (each marked inline in the source):
- Stage COCO annotations, frames, and game CSVs per sport (`00_setup_data.py`).
- Confirm the COCO and game-result field paths in `transformations/silver.py`.
- torchvision training loop (`train._torchvision_train`) and COCOeval loop
  (`evaluate.evaluate_torchvision`).
- YOLO/RT-DETR per-image label conversion in `data.export_yolo_dataset`.
- Fast R-CNN external-proposal feeding (`train._torchvision_train`).
- MMPose training/eval (`train._mmpose_train`, `evaluate.evaluate_mmpose`) and
  MediaPipe eval (`evaluate.evaluate_mediapipe`).
- `seeing_models_03_gold.athlete_metrics`: biomechanics angles, `athlete_id` roster linkage, and a
  `player_rating` label (`transformations/gold.py`); these unlock `player_index`
  and `scholarship_roi`.
- `seeing_models_03_gold.team_game_features`: join aggregated athlete metrics for CV-derived
  team-strength features (`transformations/gold.py`).

## 15. Framework install notes

- **Serverless GPU** must be enabled for `@distributed`. If it is not, use the
  classic GPU cluster fallback: uncomment the `job_clusters` block (Azure
  `Standard_NC*` node types) at the bottom of `resources/train_compare.job.yml`
  and replace each training task's `environment_key` with a `job_cluster_key`.
- **MMPose** (`mmpose_env`): `mmcv` must match the runtime's torch and CUDA. If
  pip cannot resolve a wheel, install via OpenMIM
  (`mim install mmengine mmcv mmdet mmpose`); notebook `06` has the cell.
- **MediaPipe** (`mediapipe_env`): CPU only; notebook `07` downloads the `.task`
  bundle into the volume.
- **Selective Search** for Fast R-CNN needs `opencv-contrib` (`cv2.ximgproc`),
  which the default `opencv-python-headless` does not include. Swap to
  `opencv-contrib-python-headless`.
- **Team-performance** (`teamperf_env`): `xgboost` and `scikit-learn`.

## 16. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Catalog 'rpeng_upleveling' does not exist` | Create it or point `--var catalog=<name>` at an existing one. |
| `No finished runs for sport '<x>'` at compare | The sport's training tasks did not produce finished runs; check the training task logs. |
| `mmcv` install fails on the RTMPose task | Install via OpenMIM in notebook `06`; align `mmcv` with torch/CUDA. |
| `cv2.ximgproc` not found | Install `opencv-contrib-python-headless` for Selective Search. |
| `@distributed` import fails | Serverless GPU not enabled; use the classic GPU cluster fallback. |
| `register_skipped` tag on a run | The model has no wired MLflow flavor (MMPose/MediaPipe); metrics still reach the leaderboard. |
| Pipeline stuck `INITIALIZING` | Serverless cold start; wait a few minutes. |

## 17. Next steps

Current state: the bundle is deployed to `dev` on SLED. The four schemas and the
`cv_data` volume exist and are empty. No tables or MLflow experiments exist until
the pipeline and jobs run.

### 1. Stage data (per sport)

Upload labeled data into the landing volume under each sport, then the pipeline
has files to ingest and infer schema from. Expected paths for a sport (see
notebook `00`):

```
/Volumes/rpeng_upleveling/seeing_models_00_landing/cv_data/
  images/<sport>/{train,val,test}/*.jpg
  labels/<sport>/person_keypoints_{train,val}.json    # COCO keypoints
  labels/<sport>/instances_{train,val}.json           # COCO boxes
  games/<sport>/*.csv                                  # team-performance labels
```

Upload with the CLI (the `dbfs:` scheme routes to the volume):

```bash
databricks fs cp -r ./local_data \
  dbfs:/Volumes/rpeng_upleveling/seeing_models_00_landing/cv_data/ \
  --recursive --profile SLED_fe_demo
```

Auto Loader needs at least one file per source to infer schema, so a zero-file
run fails at landing. To smoke-test the medallion before real data, stage a
minimal sample for one sport: a small `person_keypoints_train.json` +
`instances_train.json` (a handful of COCO images/annotations), one or two frames
under `images/<sport>/train/`, and a `games/<sport>/games.csv` with the columns
`game_id, team, opponent, game_date, points_for, points_against, off_efficiency,
def_efficiency`.

### 2. Run

Per sport (the job's `sport` parameter drives everything; tables carry a `sport`
column so runs accumulate):

```bash
databricks bundle run train_and_compare -t dev -p SLED_fe_demo --params sport=golf
# repeat for football, basketball, baseball, hockey, tennis
```

To run only the medallion (to verify landing → bronze → silver → gold):

```bash
databricks bundle run medallion -t dev -p SLED_fe_demo
```

### 3. Read the results

```sql
SELECT * FROM rpeng_upleveling.seeing_models_03_gold.model_comparison
ORDER BY sport, task, pose_map DESC, detect_map DESC;

SELECT * FROM rpeng_upleveling.seeing_models_03_gold.team_predictions
WHERE sport = 'basketball';
```

A Databricks AI/BI (Lakeview) dashboard over these two gold tables is a natural
add for a shared leaderboard.

### 4. Make training real (see Section 14)

The training and evaluation loops are wired but stubbed where they depend on the
labeled data. Fill in the inline TODOs in this order:

1. `transformations/silver.py` / `bronze.py`: confirm the COCO and game-result
   field paths match your export.
2. `data.export_yolo_dataset`: the per-image YOLO label conversion (unblocks
   YOLO and RT-DETR end-to-end).
3. `train._torchvision_train` + `evaluate.evaluate_torchvision`: the torchvision
   train and COCOeval loops (R-CNN and SSDlite).
4. `train._mmpose_train` / `evaluate.evaluate_mmpose` and
   `evaluate.evaluate_mediapipe`: MMPose and MediaPipe scoring.
5. `transformations/gold.py`: `athlete_metrics` biomechanics angles + roster
   linkage + `player_rating`, which unblock `player_index` and `scholarship_roi`.

### 5. Build a dashboard (optional)

Once the gold tables have data, add a Databricks AI/BI (Lakeview) dashboard over
`seeing_models_03_gold.model_comparison` and `seeing_models_03_gold.team_predictions`
for a shared, per-sport leaderboard.
