# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC Create the target schema. Catalog `rpeng_upleveling` already exists; this
# MAGIC notebook only ensures `dimensionality_reduction` is present. Idempotent.

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

from dr_bench import load_config

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    table_prefix=dbutils.widgets.get("table_prefix"),
)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cfg.catalog}`.`{cfg.schema}`")
print(f"Schema ready: {cfg.catalog}.{cfg.schema}")
print(f"Tables will be prefixed: {cfg.table_prefix}_*")
