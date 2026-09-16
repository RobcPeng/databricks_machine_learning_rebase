# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — land raw datasets
# MAGIC
# MAGIC Pull each public CSV straight from its source URL and write it, untouched,
# MAGIC to a `*_bronze` Delta table. No cleaning here — bronze is the faithful copy.

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

from dr_bench import DATASETS, load_config

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    table_prefix=dbutils.widgets.get("table_prefix"),
)

# COMMAND ----------

import pandas as pd

for name in cfg.datasets:
    spec = DATASETS[name]
    pdf = pd.read_csv(spec.source_url)
    sdf = spark.createDataFrame(pdf)
    sdf.createOrReplaceTempView("bronze_tmp")
    spark.sql(f"CREATE OR REPLACE TABLE {cfg.table(f'{name}_bronze')} AS SELECT * FROM bronze_tmp")
    print(f"{name}: {pdf.shape[0]} rows, {pdf.shape[1]} cols -> {cfg.table(f'{name}_bronze')}")
    print(f"   {spec.description}")
