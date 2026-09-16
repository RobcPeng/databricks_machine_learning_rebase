# Next Steps

Status at last deploy: the benchmark job runs on the SLED workspace
(`SLED_fe_demo`, job `dimensionality-reduction-benchmark`). NSCG, ACS, and PISA
land through bronze → silver → gold. College Scorecard does not yet land (see #2).

## 1. Review results

After a run completes, results are in two places.

Curated (gold) tables in `rpeng_upleveling.dimensionality_reduction_02_curated`:

```sql
-- Every grid cell's cross-validation and held-out test metrics
SELECT * FROM `rpeng_upleveling`.`dimensionality_reduction_02_curated`.`leaderboard`
ORDER BY dataset, test_roc_auc DESC;

-- Best score per dataset x model x reducer
SELECT * FROM `rpeng_upleveling`.`dimensionality_reduction_02_curated`.`model_comparison`
ORDER BY dataset, test_roc_auc DESC;
```

MLflow experiment: `/Users/rob.peng@databricks.com/dimensionality_reduction/dr_benchmark`
— one nested run per cell, filterable by the `dataset`, `reducer`, and `model` tags.

What to look for: whether PCA/ICA/random projection beats `reducer = none` for each
model, whether KMeans cluster features add lift, and how `pca_explained_var`
tracks with accuracy.

## 2. Restore the College Scorecard dataset

Symptom: no `00_scorecard_2026_*` tables; the other three datasets land. The
bronze step skips a dataset whose source fails to download and continues.

Likely cause: serverless egress reaches `ncses.nsf.gov`, `census.gov`, and
`webfs.oecd.org` but not `ed-public-download.scorecard.network`.

Diagnose — run in a serverless notebook and read the status/exception:

```python
import urllib.request
u = "https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-All-Data-Elements.csv"
print(urllib.request.urlopen(u, timeout=60).status)
```

Fixes, in order of preference:

1. Add the host to the workspace's serverless egress allowlist, then rerun.
2. Download the CSV once, upload to a UC Volume
   (`/Volumes/rpeng_upleveling/dimensionality_reduction_00_landing/<vol>/scorecard.csv`), and
   point the `scorecard_2026` spec's `source_url` at the volume path (extend the
   `scorecard_csv` loader to read local/volume paths).
3. Use the College Scorecard API instead of the bulk file.

The spec, target (`MD_EARN_WNE_P10`), and features are already validated against
the real file (see docs/DATASETS.md); only the fetch path needs to change.

## 3. Remove the superseded single-schema tables

Earlier runs wrote to one schema, `rpeng_upleveling.dimensionality_reduction`
(tables prefixed `00_` and `00_experiment_type_`). The medallion is now split
across `_00_landing` / `_01_cleansed` / `_02_curated`, so that original schema is
legacy. Dropping it requires explicit authorization:

```bash
databricks schemas delete rpeng_upleveling.dimensionality_reduction --force -p SLED_fe_demo
```

## 4. Optional extensions

- **Register the winner.** Set `register_best: true` in `conf/experiments.yml` to
  register the top pipeline per dataset to Unity Catalog as
  `rpeng_upleveling.dimensionality_reduction_02_curated.dr_best_<dataset>`.
- **Dashboard.** Build an AI/BI dashboard over
  `dimensionality_reduction_02_curated.model_comparison` (model × reducer heatmap
  per dataset). Genie can also query the curated tables directly.
- **Widen the grid.** Add `n_components` values (e.g. `[10, 30, 50]`) and models
  (`svm_linear`); raise `sample_rows` for higher-fidelity metrics at more runtime.
- **More ACS coverage.** Swap or add state files in the `acs_co_2024` spec
  (`csv_p<state>.zip`) to compare geographies, or point at the national person file.
- **Schedule.** Add a `schedule` block to `resources/dr_benchmark.job.yml` to
  refresh the leaderboard on a cadence.

## Operational notes

- Serverless environment is pinned to client `"3"`. Bump it in
  `resources/dr_benchmark.job.yml` if the workspace default moves on.
- `sample_rows` (default 6,000) caps rows per dataset so RBF SVM and MLP stay
  tractable; raise it for final numbers.
- `pisa_2022` is the largest download (~600 MB) and needs `pyreadstat`; remove it
  from `conf/experiments.yml` to skip that cost.
