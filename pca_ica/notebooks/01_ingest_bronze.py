# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Landing (bronze) — land raw datasets
# MAGIC
# MAGIC Fetch each public source and write it verbatim into the `_00_landing` schema.
# MAGIC A source that fails to download is skipped so the others still land.

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

from dr_bench import DATASETS, load_config
from dr_bench.datasets import load_raw

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema_prefix=dbutils.widgets.get("schema_prefix"),
)

# COMMAND ----------

for name in cfg.datasets:
    spec = DATASETS[name]
    try:
        pdf = load_raw(spec)
    except Exception as e:
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:200]}")
        continue
    obj_cols = pdf.select_dtypes(include="object").columns
    pdf[obj_cols] = pdf[obj_cols].fillna("")  # avoid all-null column type-inference errors
    spark.createDataFrame(pdf).createOrReplaceTempView("landing_tmp")
    spark.sql(f"CREATE OR REPLACE TABLE {cfg.table('landing', name)} AS SELECT * FROM landing_tmp")
    print(f"{name}: {pdf.shape[0]} rows x {pdf.shape[1]} cols -> {cfg.table('landing', name)}")
