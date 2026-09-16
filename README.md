# levelup

Two self-contained Databricks Asset Bundles, each a benchmark harness that trains
a grid of models on one task and rolls the runs up into a single gold leaderboard.
The shape is the same both times: deploy the bundle, run one job, read the
comparison table. Use them to find which modeling approach wins on a task before
committing to one.

Both bundles deploy to the SLED workspace (CLI profile `SLED_fe_demo`,
`adb-3438839487639471.11.azuredatabricks.net`) and write to the `rpeng_upleveling`
catalog. They share no code: each has its own `databricks.yml`, package, and job,
so you deploy and run them independently from their own directory.

## The bundles

| Directory | Bundle | Task | Compares |
|---|---|---|---|
| `obj_recg/` | `uplevels_cv` | Detect the golfer/club/ball and trace golf-swing pose keypoints | YOLO11 vs. ResNet R-CNN vs. lightweight CNN |
| `pca_ica/` | `dr_bench` | Predict graduate placement and salary (binary classification) | classic models × PCA/ICA/random-projection × clustering |

`obj_recg` is computer vision on serverless GPU. `pca_ica` is scikit-learn on
serverless CPU and also runs off-platform for fast local iteration.

## Shared conventions

- **Databricks Asset Bundles**: each bundle declares its schema, volume, pipeline,
  and job under `resources/`; you deploy them with the Databricks CLI.
- **Medallion + MLflow**: a bronze → silver → gold flow feeds training, every run
  logs to MLflow, and the final task writes a gold comparison table.
- **Catalog**: `rpeng_upleveling`, one schema per bundle (`object_and_vision`,
  `dimensionality_reduction`).
- **Table prefix `00_...`**: sorts each project first in the catalog explorer.
  Because the name leads with a digit, SQL must backtick it:
  `` `rpeng_upleveling`.`object_and_vision`.`00_seeing_models_...` ``.
- **Registry-driven**: add a model, dataset, or reducer by appending one entry;
  dispatch, evaluation, and the leaderboard pick it up.

## Prerequisites

- A Databricks CLI profile `SLED_fe_demo` targeting
  `adb-3438839487639471.11.azuredatabricks.net`.
- Write access to the `rpeng_upleveling` catalog.
- `uv` for `obj_recg`, `pip` for `pca_ica` (local package and tests).

## Use it

Each bundle is standalone. `cd` into it, deploy, run the job, read the gold table.

**obj_recg** (target `dev`, also `prod`):

```bash
cd obj_recg
databricks bundle validate -t dev -p SLED_fe_demo
databricks bundle deploy   -t dev -p SLED_fe_demo   # schema + volume + pipeline + job
databricks bundle run train_and_compare -t dev -p SLED_fe_demo
```

**pca_ica** (target `sled`):

```bash
cd pca_ica
databricks bundle deploy -t sled -p SLED_fe_demo
databricks bundle run dr_benchmark -t sled -p SLED_fe_demo
```

Each bundle's own `README.md` has the full picture — data staging, the model grid,
open TODOs, serverless-GPU notes, and how to extend it. Read it before changing a
bundle.

## Docs site

`docs/` is a static, GitHub Pages–ready site that explains both bundles: the
datasets and their domain, what each model and reducer does, how PCA/ICA/DR reads
SLED outcomes, what to build next, and where the pattern reuses. Enable it under
**Settings → Pages → Deploy from a branch → `/docs`**. See `docs/README.md`.
