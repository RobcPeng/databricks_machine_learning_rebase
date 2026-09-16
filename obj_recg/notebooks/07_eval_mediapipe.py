# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Evaluate MediaPipe BlazePose (3D pose)
# MAGIC
# MAGIC BlazePose predicts **33 landmarks with 3D (x, y, z)**. It is **pretrained:
# MAGIC evaluated, not trained**, and runs on CPU, so no GPU is requested.
# MAGIC
# MAGIC 2D accuracy is scored on the COCO-17 projection (`skeleton_blazepose`). The
# MAGIC 3D metric (MPJPE) needs 3D ground truth and is left blank on 2D COCO data.

# COMMAND ----------

# %pip install "mediapipe>=0.10" "mlflow>=3.1.0" "pycocotools>=2.0.7"
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

for k, v in {"catalog": "rpeng_upleveling", "project": "seeing_models",
             "volume": "cv_data",
             "experiment_root": "/Users/rob.peng@databricks.com/uplevels_cv",
             "sport": "golf"}.items():
    dbutils.widgets.text(k, v)

from uplevels_cv.config import Family, Paths, specs_for
from uplevels_cv.data import DataConfig
from uplevels_cv.train import train_spec

paths = Paths.from_params({k: dbutils.widgets.get(k)
                           for k in ("catalog", "project", "volume")})
data = DataConfig(paths=paths, sport=dbutils.widgets.get("sport"))
experiment_root = dbutils.widgets.get("experiment_root")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch the pretrained BlazePose task bundle into the volume

# COMMAND ----------

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
model_path = os.path.join(paths.volume_root, "artifacts", "pose_landmarker_full.task")
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
print("Model at", model_path)

# COMMAND ----------

# The spec's `weights` is the task filename; point it at the downloaded path.
results = {}
for spec in specs_for(family=Family.MEDIAPIPE):
    # rebind weights to the concrete volume path for this run
    from dataclasses import replace

    spec = replace(spec, weights=model_path)
    print(f"Evaluating {spec.key}: {spec.description}")
    results[spec.key] = train_spec(spec, data, experiment_root=experiment_root)
    print(results[spec.key])

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
