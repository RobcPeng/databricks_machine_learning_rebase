# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Cleansed (silver) — clean + label
# MAGIC
# MAGIC Read each landing table, apply filters, derive the binary `label`, keep the
# MAGIC spec's feature columns, and write to the `_01_cleansed` schema. Cleansed is
# MAGIC the contract the modeling step trains on.

# COMMAND ----------

import os
import sys

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
FILES_ROOT = os.path.dirname(os.path.dirname("/Workspace" + _nb_path))
sys.path.insert(0, os.path.join(FILES_ROOT, "src"))
CONF_PATH = os.path.join(FILES_ROOT, "conf", "experiments.yml")

dbutils.widgets.text("catalog", "rpeng_upleveling")
dbutils.widgets.text("schema_prefix", "dimensionality_reduction")

from dr_bench import DATASETS, load_config, to_silver

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema_prefix=dbutils.widgets.get("schema_prefix"),
)

# COMMAND ----------

for name in cfg.datasets:
    spec = DATASETS[name]
    try:
        landing = spark.sql(f"SELECT * FROM {cfg.table('landing', name)}").toPandas()
    except Exception as e:
        print(f"[skip] {name}: no landing table ({type(e).__name__})")
        continue
    silver = to_silver(spec, landing)
    spark.createDataFrame(silver).createOrReplaceTempView("cleansed_tmp")
    spark.sql(f"CREATE OR REPLACE TABLE {cfg.table('cleansed', name)} AS SELECT * FROM cleansed_tmp")
    pos_rate = round(float(silver["label"].mean()), 3)
    print(
        f"{name}: {silver.shape[0]} rows, {silver.shape[1] - 1} features, "
        f"positive rate {pos_rate} -> {cfg.table('cleansed', name)}"
    )
