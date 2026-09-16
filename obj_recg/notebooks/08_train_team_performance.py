# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Train team-performance models
# MAGIC
# MAGIC Trains one tabular model per prediction target (win probability, score,
# MAGIC efficiency, player index, scholarship ROI) for the selected sport, over the
# MAGIC gold feature tables the medallion built, then batch-scores them into
# MAGIC `00_seeing_models_03_gold_team_predictions`.
# MAGIC
# MAGIC Targets are the `team_performance.TARGETS` registry; add one there to add a
# MAGIC use case. Feature and label columns are read from the gold tables — populate
# MAGIC those (see `transformations/gold.py` and notebook `00`) before running.

# COMMAND ----------

# %pip install "xgboost>=2.1" "scikit-learn>=1.5" "mlflow>=3.1.0"
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "project": "seeing_models",
             "volume": "cv_data",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "sport": "golf"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.config import Paths
from uplevels_cv.team_performance import (
    TARGETS, missing_columns, score_targets, train_target,
)

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "project", "volume")})
experiment_root = dbutils.widgets.get("experiment_root")
sport = dbutils.widgets.get("sport")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train one model per target

# COMMAND ----------

results = {}
skipped = {}
for key, target in TARGETS.items():
    # Skip targets whose gold feature table isn't populated yet (e.g. the
    # athlete/recruit targets before athlete_metrics grows player_rating / roi /
    # athlete_id), rather than crashing the whole run on a missing column.
    missing = missing_columns(spark, paths, target, sport)
    if missing:
        skipped[key] = missing
        print(f"Skipping {key}: {target.feature_table} missing {missing}")
        continue
    print(f"Training {key} ({target.scope.value}, {target.target_type.value})")
    results[key] = train_target(spark, paths, target, sport=sport,
                                experiment_root=experiment_root)
    print(results[key])

if skipped:
    print(f"Skipped {len(skipped)} target(s) with unpopulated features: {skipped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch-score into the gold predictions table

# COMMAND ----------

table = score_targets(spark, paths, sport=sport)
if table:
    print("Wrote", table)
    display(spark.table(table).where(f"sport = '{sport}'"))
else:
    print("No ready targets to score — populate the gold feature tables first "
          "(see transformations/gold.py).")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
