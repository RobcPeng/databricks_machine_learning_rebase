# Dimensionality-Reduction Benchmark

A config-driven harness for comparing classic ML models across datasets and
dimensionality-reduction techniques. One config expands into a grid of
`dataset × reducer × clustering × model` cells; each cell is trained with
cross-validation, logged as an MLflow run, and rolled up into a gold leaderboard.

Packaged as a Databricks Asset Bundle, runs on serverless, targets the SLED
workspace.

## The question

*Given what we know about a graduate, can we predict their placement outcome —
and do PCA/ICA/clustering features help the classic models get there?*

Two public, no-auth datasets, both framed as one binary classification task so a
single leaderboard is comparable across them:

| Dataset | Rows | Features | Label |
|---|---|---|---|
| `campus_placement` — Kaggle Campus Recruitment (MBA grads) | 215 | 12 | `status` → Placed vs Not Placed |
| `engineering_salary` — Aspiring Minds AMEO 2015 (engineering grads) | 2,998 | 13 | `Salary` → above vs below median |

The two differ deliberately: one is small and class-imbalanced, the other is
larger with more numeric signal — so you can see which techniques transfer.

## Medallion layout

| Layer | Table | Contents |
|---|---|---|
| Bronze | `00_experiment_type_<dataset>_bronze` | Raw CSV, landed untouched |
| Silver | `00_experiment_type_<dataset>_silver` | Cleaned features + binary `label` |
| Gold | `00_experiment_type_leaderboard_gold` | Every grid cell's CV + test metrics |
| Gold | `00_experiment_type_model_comparison_gold` | Best score per dataset × model × reducer |

All in `rpeng_upleveling.dimensionality_reduction`. Names are backtick-quoted
everywhere because the prefix starts with a digit.

## Run it

```bash
# from this directory
databricks bundle deploy -t sled -p SLED_fe_demo
databricks bundle run dr_benchmark -t sled -p SLED_fe_demo
```

`deploy` syncs the files and creates the job; `run` executes the five tasks in
order: setup → bronze → silver → experiments → leaderboard. Results land in the
gold tables and in an MLflow experiment at
`/Users/<you>/dimensionality_reduction/00_experiment_type`.

To validate config changes without deploying: `databricks bundle validate --strict -t sled -p SLED_fe_demo`.

## The grid

Edit `conf/experiments.yml`. The defaults give **96 cells**:

```
2 datasets × (none + 3 reducers × 1 n_components) × 2 clustering × 6 models
```

- `reducers`: `none`, `pca`, `ica`, `random_projection`
- `clustering`: `none`, `kmeans` (appends one-hot cluster membership as features; `gmm` also available)
- `models`: `logistic_regression`, `svm_rbf`, `knn`, `random_forest`, `xgboost`, `mlp` (`svm_linear` also available)
- `n_components`: list — add values like `[5, 10, 15]` to sweep reduction depth

The `none` reducer ignores `n_components`, so it stays one cell instead of one
per value. On these dataset sizes the full grid runs in a few minutes.

## Extend it

Everything is a registry — add one entry, nothing else changes:

- **Dataset** → append a `DatasetSpec` in `src/dr_bench/datasets.py`
- **Reducer** → add a branch in `src/dr_bench/reducers.py` + its name to `REDUCERS`
- **Model** → add a branch in `src/dr_bench/models.py` + its name to `MODELS`

Set `register_best: true` in the config to also register the top pipeline per
dataset to Unity Catalog (`rpeng_upleveling.dimensionality_reduction.dr_best_<dataset>`).

## Local development

The `dr_bench` package is pure scikit-learn, so it runs off-platform for fast
iteration:

```bash
pip install -r requirements.txt
pytest tests/          # synthetic-data smoke tests, no network
```

Off Databricks, `load_datasets` reads the CSVs directly from their source URLs;
on Databricks it reads the `_silver` tables.

## Layout

```
databricks.yml                 bundle + sled target (profile: SLED_fe_demo)
conf/experiments.yml           the grid
resources/dr_benchmark.job.yml serverless job, 5 medallion tasks
notebooks/                     00 setup · 01 bronze · 02 silver · 03 experiments · 04 leaderboard
src/dr_bench/                  the reusable harness
  datasets.py  reducers.py  clustering.py  models.py  pipeline.py  grid.py  experiment.py  config.py
tests/test_smoke.py            local tests
```

## Notes

- Serverless environment is pinned to client `"3"` with scikit-learn / xgboost /
  mlflow in `resources/dr_benchmark.job.yml`. Bump the client version there if
  the workspace defaults move on.
- `campus_placement` drops `salary` as a feature — it only exists for placed
  students, so it would leak the label.
- Sources: [Campus Recruitment](https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Placement_Data_Full_Class.csv),
  [AMEO Engineering Graduate Salary](https://raw.githubusercontent.com/anuragsingh2207/dsba-jupyter-notebook/master/lpm/Revision/Dataset/Engineering_graduate_salary.csv).
```
