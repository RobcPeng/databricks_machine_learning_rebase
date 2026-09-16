# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train ResNet-50 FPN R-CNN (detect + pose)
# MAGIC
# MAGIC Two-stage family. `resnet_detect` = Faster R-CNN, `resnet_pose` = Keypoint
# MAGIC R-CNN (17 keypoints → limb lines). Same ResNet-50 FPN backbone, so this is
# MAGIC the accuracy-leaning counterweight to YOLO's speed. Serverless GPU.

# COMMAND ----------

# Interactive runs only:
# %pip install "torch>=2.4" "torchvision>=0.19" "pycocotools>=2.0.7" "mlflow>=3.1.0"
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "schema": "object_and_vision",
             "volume": "cv_data", "table_prefix": "00_seeing_models",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "gpu_type": "A10", "model_family": "resnet"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.config import Family, Paths, specs_for
from uplevels_cv.data import DataConfig
from uplevels_cv.train import train_spec

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "schema", "volume", "table_prefix")})
data = DataConfig(paths=paths)
gpu_type = dbutils.widgets.get("gpu_type")
experiment_root = dbutils.widgets.get("experiment_root")

# COMMAND ----------

results = {}
for spec in specs_for(family=Family.RESNET):
    print(f"Training {spec.key}: {spec.description}")
    results[spec.key] = train_spec(spec, data, experiment_root=experiment_root,
                                   gpu_type=gpu_type)
    print(results[spec.key])

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
