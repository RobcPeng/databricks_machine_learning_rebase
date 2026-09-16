# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train SSDlite (CNN)
# MAGIC
# MAGIC `cnn_detect` = SSDlite, MobileNetV3-Large backbone. One-stage detector.
# MAGIC Serverless GPU. Detection only.

# COMMAND ----------

# Interactive runs only:
# %pip install "torch>=2.4" "torchvision>=0.19" "pycocotools>=2.0.7" "mlflow>=3.1.0"
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "project": "seeing_models",
             "volume": "cv_data",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "gpu_type": "A10", "model_family": "cnn", "sport": "golf"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.config import Family, Paths, specs_for
from uplevels_cv.data import DataConfig
from uplevels_cv.train import train_spec

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "project", "volume")})
data = DataConfig(paths=paths, sport=dbutils.widgets.get("sport"))
gpu_type = dbutils.widgets.get("gpu_type")
experiment_root = dbutils.widgets.get("experiment_root")

# COMMAND ----------

results = {}
for spec in specs_for(family=Family.CNN):
    print(f"Training {spec.key}: {spec.description}")
    results[spec.key] = train_spec(spec, data, experiment_root=experiment_root,
                                   gpu_type=gpu_type)
    print(results[spec.key])

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
