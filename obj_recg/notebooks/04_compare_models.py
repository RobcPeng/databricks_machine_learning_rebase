# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Compare models
# MAGIC
# MAGIC Pulls the latest run per model from the family experiments and writes the
# MAGIC gold leaderboard
# MAGIC `rpeng_upleveling.object_and_vision.00_seeing_models_gold_model_comparison`.
# MAGIC One row per model: detection mAP, pose OKS, latency, params, size.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "schema": "object_and_vision",
             "volume": "cv_data", "table_prefix": "00_seeing_models",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.compare import write_comparison
from uplevels_cv.config import Paths

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "schema", "volume", "table_prefix")})
experiment_root = dbutils.widgets.get("experiment_root")

# COMMAND ----------

table = write_comparison(spark, paths, experiment_root)
print("Wrote", table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard

# COMMAND ----------

display(
    spark.table(table.replace("`", ""))
    .orderBy("task", "pose_map", "detect_map", ascending=False)
)

# COMMAND ----------

dbutils.notebook.exit(table)
