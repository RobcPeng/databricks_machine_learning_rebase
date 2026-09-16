# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Compare models
# MAGIC
# MAGIC Pulls the latest run per model from the family experiments and writes the
# MAGIC gold leaderboard
# MAGIC `rpeng_upleveling.seeing_models_03_gold.model_comparison`.
# MAGIC One row per model: detection mAP, pose OKS, latency, params, size.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "project": "seeing_models",
             "volume": "cv_data",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "sport": "golf"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.compare import write_comparison
from uplevels_cv.config import Paths

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "project", "volume")})
experiment_root = dbutils.widgets.get("experiment_root")
sport = dbutils.widgets.get("sport")

# COMMAND ----------

table = write_comparison(spark, paths, experiment_root, sport)
print("Wrote", table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard

# COMMAND ----------

display(
    spark.table(table).where(f"sport = '{sport}'")
    .orderBy("task", "pose_map", "detect_map", ascending=False)
)

# COMMAND ----------

dbutils.notebook.exit(table)
