# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC Create the three medallion schemas under the catalog. Idempotent.

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

from dr_bench import load_config

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema_prefix=dbutils.widgets.get("schema_prefix"),
)

# COMMAND ----------

for schema in cfg.all_schemas():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cfg.catalog}`.`{schema}`")
    print(f"schema ready: {cfg.catalog}.{schema}")
