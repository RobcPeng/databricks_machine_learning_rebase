# Dimensionality-Reduction Benchmark

A config-driven harness for comparing classic ML models across datasets and
dimensionality-reduction techniques. One config expands into a grid of
`dataset × reducer × clustering × model` cells; each cell is trained with
cross-validation, logged as an MLflow run, and rolled up into a gold leaderboard.

Packaged as a Databricks Asset Bundle, runs on serverless, targets the SLED
workspace (`SLED_fe_demo`).

Reviewing results, the open College Scorecard egress item, and extensions are in
[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Question

*Given what we know about a graduate, an institution, or a student, can we predict
an above-median earnings/attainment outcome — and do PCA, ICA, random projection,
or clustering-derived features help the classic models get there?*

## Datasets

Four public, US-anchored datasets, each a different "track," all reduced to one
binary classification task so a single leaderboard is comparable across them.
Full data dictionary in [docs/DATASETS.md](docs/DATASETS.md).

| Key | Grain | Rows (usable) | Cycle | Target (label = 1) |
|---|---|---|---|---|
| `nscg_2023` | graduate | ~79,800 | 2023 | Above-median salary among the employed |
| `scorecard_2026` | institution | ~4,817 | 2025–26 | Above-median graduate earnings 10 yrs out |
| `acs_co_2024` | person (CO) | ~10–15k | 2024 | Above-median wage income among employed degree holders |
| `pisa_2022` | student | ~600k | 2022 | Above-median expected occupational status (`BSMJ`) |

Recency note: per-student, multi-institution microdata is published with a 1–2
year lag, so `nscg_2023` is the newest available at the graduate grain.
`scorecard_2026` reaches 2025–26 because it aggregates to the institution.

`pisa_2022` downloads a ~600 MB SPSS file and requires `pyreadstat`; a dataset
that fails to load (network, schema, missing dependency) is skipped, and the rest
of the sweep proceeds.

## Medallion layout

Three schemas under `rpeng_upleveling`, ordered by the number in the schema name:

| Schema | Layer | Tables |
|---|---|---|
| `dimensionality_reduction_00_landing` | bronze | `<dataset>` — raw source, verbatim |
| `dimensionality_reduction_01_cleansed` | silver | `<dataset>` — filtered, cleaned, binary `label` |
| `dimensionality_reduction_02_curated` | gold | `leaderboard`, `model_comparison` |

Example: `rpeng_upleveling.dimensionality_reduction_00_landing.nscg_2023`. The
schema prefix is set by the `schema_prefix` bundle variable.

## Run

```bash
databricks bundle deploy -t sled -p SLED_fe_demo
databricks bundle run dr_benchmark -t sled -p SLED_fe_demo
```

`deploy` syncs files and creates the job. `run` executes the five tasks in order:
setup → bronze → silver → experiments → leaderboard. Results land in the gold
tables and in an MLflow experiment at
`/Users/<you>/dimensionality_reduction/dr_benchmark`.

Validate config changes without deploying:

```bash
databricks bundle validate --strict -t sled -p SLED_fe_demo
```

## Grid and configuration

Edit `conf/experiments.yml`. Fields:

| Field | Meaning |
|---|---|
| `datasets` | Dataset keys to benchmark (registered in `src/dr_bench/datasets.py`) |
| `reducers` | `none`, `pca`, `ica`, `random_projection` |
| `n_components` | List of target dimensions for reducers (the `none` reducer ignores it) |
| `clustering` | `none`, `kmeans` (appends one-hot cluster membership); `gmm` also available |
| `models` | `logistic_regression`, `svm_rbf`, `knn`, `random_forest`, `xgboost`, `mlp`; `svm_linear` also available |
| `cv_folds` | Cross-validation folds |
| `test_size` | Held-out test fraction |
| `sample_rows` | Stratified per-dataset row cap (0 disables); bounds runtime for RBF SVM and MLP on the large files |
| `register_best` | Register the top pipeline per dataset to Unity Catalog |

Grid size = `datasets × [none + reducers × n_components] × clustering × models`.
Defaults (4 × 4 × 2 × 6) = 192 cells.

## Extending

Each axis is a registry — add one entry:

- Dataset → append a `DatasetSpec` in `src/dr_bench/datasets.py`; add a loader in
  `src/dr_bench/loaders.py` if the source needs a new fetch pattern
- Reducer → add a branch in `src/dr_bench/reducers.py` and its name to `REDUCERS`
- Model → add a branch in `src/dr_bench/models.py` and its name to `MODELS`

`register_best: true` registers the top pipeline per dataset to Unity Catalog as
`rpeng_upleveling.dimensionality_reduction_02_curated.dr_best_<dataset>`.

## Local development

`dr_bench` is plain scikit-learn and runs off-platform:

```bash
pip install -r requirements.txt
pytest tests/          # synthetic-data smoke tests, no network
```

Off Databricks, `load_datasets` fetches from the public sources; on Databricks it
reads the `_silver` tables.

## Repository layout

```
databricks.yml                  bundle + sled target (profile: SLED_fe_demo)
conf/experiments.yml            the grid
resources/dr_benchmark.job.yml  serverless job, 5 medallion tasks
notebooks/                      00 setup · 01 bronze · 02 silver · 03 experiments · 04 leaderboard
src/dr_bench/
  config.py       ExperimentConfig + YAML loading
  loaders.py      public-source download helpers (zip/csv/sav)
  datasets.py     dataset registry, target/feature specs, bronze->silver
  reducers.py     none / PCA / ICA / random projection
  clustering.py   KMeans / GMM cluster-membership features
  models.py       classifier registry
  pipeline.py     preprocess -> reduce -> cluster -> model
  grid.py         config -> list of experiment cells
  experiment.py   run a cell, log to MLflow, sampling
tests/test_smoke.py             local tests
docs/DATASETS.md                data dictionary
```

## Notes

- Serverless environment is pinned to client `"3"` with scikit-learn, xgboost,
  mlflow, matplotlib, and pyreadstat in `resources/dr_benchmark.job.yml`.
- Targets are balanced by construction (median split), so accuracy, F1, and
  ROC-AUC are all interpretable.
- One-hot encoding caps at 50 categories per feature (`max_categories`), which
  bounds dimensionality on high-cardinality codes (field of study, occupation).
- Numeric values `>= 9,999,990` are treated as missing (survey sentinel codes).
- `nscg_2023` restricts to employed respondents (`LFSTAT == 1`); the salary
  sentinel `9999998` maps exactly to the non-employed and is excluded.
- RBF SVM does not scale to the full graduate/person files, which is why
  `sample_rows` defaults to 6,000.
```
