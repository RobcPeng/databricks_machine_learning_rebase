# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — clean + label
# MAGIC
# MAGIC Read each `*_bronze` table, drop id/leakage columns, coerce types, and
# MAGIC derive the binary `label` (Placed / above-median salary). The resulting
# MAGIC `*_silver` table is the contract the modeling step trains on.

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

from dr_bench import DATASETS, load_config, to_silver

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    table_prefix=dbutils.widgets.get("table_prefix"),
)

# COMMAND ----------

for name in cfg.datasets:
    spec = DATASETS[name]
    bronze = spark.sql(f"SELECT * FROM {cfg.table(f'{name}_bronze')}").toPandas()
    silver = to_silver(spec, bronze)
    sdf = spark.createDataFrame(silver)
    sdf.createOrReplaceTempView("silver_tmp")
    spark.sql(f"CREATE OR REPLACE TABLE {cfg.table(f'{name}_silver')} AS SELECT * FROM silver_tmp")
    pos_rate = round(float(silver["label"].mean()), 3)
    print(
        f"{name}: {silver.shape[0]} rows, {silver.shape[1] - 1} features, "
        f"positive rate {pos_rate} -> {cfg.table(f'{name}_silver')}"
    )
