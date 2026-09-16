# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Train / evaluate RTMPose (MMPose)
# MAGIC
# MAGIC Top-down 2D pose from OpenMMLab. Top-down means it needs person boxes
# MAGIC first; supply them from a detector in the zoo (e.g. rtdetr_detect). COCO-17
# MAGIC keypoints, scored on the same OKS as the other 2D-pose models. Serverless GPU.
# MAGIC
# MAGIC **Install note:** MMPose's `mmcv` must match the runtime's torch + CUDA.
# MAGIC Plain `pip install mmcv` often fails to find a wheel — use OpenMIM.

# COMMAND ----------

# %pip install -U openmim "mlflow>=3.1.0" "pycocotools>=2.0.7"
# import subprocess; subprocess.run(["mim", "install", "mmengine", "mmcv", "mmdet", "mmpose"], check=True)
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "project": "seeing_models",
             "volume": "cv_data",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "gpu_type": "A10", "model_family": "mmpose", "sport": "golf"}.items():
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
for spec in specs_for(family=Family.MMPOSE):
    print(f"Training {spec.key}: {spec.description}")
    results[spec.key] = train_spec(spec, data, experiment_root=experiment_root,
                                   gpu_type=gpu_type)
    print(results[spec.key])

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
