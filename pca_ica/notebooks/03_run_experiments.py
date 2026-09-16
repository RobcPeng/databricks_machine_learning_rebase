# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Run the grid → MLflow + gold leaderboard
# MAGIC
# MAGIC Expand the config into `dataset × reducer × clustering × model` cells,
# MAGIC train each with cross-validation, log every cell as a nested MLflow run,
# MAGIC and write all results to the `*_leaderboard_gold` table.

# COMMAND ----------

import os
import sys

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
FILES_ROOT = os.path.dirname(os.path.dirname("/Workspace" + _nb_path))
sys.path.insert(0, os.path.join(FILES_ROOT, "src"))
CONF_PATH = os.path.join(FILES_ROOT, "conf", "experiments.yml")

dbutils.widgets.text("catalog", "rpeng_upleveling")
dbutils.widgets.text("schema", "dimensionality_reduction")
dbutils.widgets.text("table_prefix", "00_experiment_type")

from dr_bench import expand_grid, load_config, load_datasets, run_one

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    table_prefix=dbutils.widgets.get("table_prefix"),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Point MLflow at an experiment under the user's home

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")
_user = WorkspaceClient().current_user.me().user_name
_exp_dir = f"/Users/{_user}/dimensionality_reduction"
WorkspaceClient().workspace.mkdirs(_exp_dir)  # set_experiment does NOT create parents
EXPERIMENT = f"{_exp_dir}/{cfg.experiment_leaf()}"
mlflow.set_experiment(EXPERIMENT)
print("MLflow experiment:", EXPERIMENT)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Sweep

# COMMAND ----------

import pandas as pd

specs = expand_grid(cfg)
print(f"Grid cells: {len(specs)}")
cache = load_datasets(cfg, spark=spark)  # reads the *_silver tables

rows = []
with mlflow.start_run(run_name="grid_sweep"):
    mlflow.log_params(
        {
            "n_cells": len(specs),
            "datasets": ",".join(cfg.datasets),
            "reducers": ",".join(cfg.reducers),
            "clustering": ",".join(cfg.clustering),
            "models": ",".join(cfg.models),
            "cv_folds": cfg.cv_folds,
        }
    )
    for i, s in enumerate(specs, 1):
        try:
            rows.append(run_one(s, cfg, cache, mlflow=mlflow))
        except Exception as e:  # keep the sweep going if one cell blows up
            print(f"  [skip] {s} -> {type(e).__name__}: {str(e)[:160]}")
        if i % 10 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} cells done")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write the gold leaderboard

# COMMAND ----------

leaderboard = pd.DataFrame(rows)
spark.createDataFrame(leaderboard).createOrReplaceTempView("gold_tmp")
spark.sql(
    f"CREATE OR REPLACE TABLE {cfg.table('leaderboard_gold')} AS SELECT * FROM gold_tmp"
)
print(f"Wrote {len(leaderboard)} rows -> {cfg.table('leaderboard_gold')}")

top = leaderboard.sort_values("test_roc_auc", ascending=False).head(10)
print("\nTop 10 by test ROC-AUC:")
print(top[["dataset", "model", "reducer", "clustering", "test_roc_auc", "test_f1"]].to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## (Optional) register the best pipeline per dataset to Unity Catalog
# MAGIC Enabled by `register_best: true` in `conf/experiments.yml`.

# COMMAND ----------

if cfg.register_best:
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split

    from dr_bench import build_pipeline
    from dr_bench.datasets import DATASETS

    for ds in cfg.datasets:
        sub = leaderboard[leaderboard.dataset == ds]
        if sub.empty:
            continue
        best = sub.sort_values("test_roc_auc", ascending=False).iloc[0]
        dspec, X, y = cache[ds]
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
        ncomp = None if int(best["n_components"]) < 0 else int(best["n_components"])
        pipe = build_pipeline(
            dspec, best["reducer"], ncomp, best["clustering"], best["model"], cfg.random_state
        ).fit(X_tr, y_tr)
        # UC model names cannot start with a digit, so use a safe prefix.
        model_name = f"{cfg.catalog}.{cfg.schema}.dr_best_{ds}"
        with mlflow.start_run(run_name=f"register_{ds}"):
            mlflow.sklearn.log_model(pipe, name="model", registered_model_name=model_name)
        print(f"Registered {model_name} ({best['model']} / {best['reducer']})")
else:
    print("register_best is false — skipping UC registration.")
