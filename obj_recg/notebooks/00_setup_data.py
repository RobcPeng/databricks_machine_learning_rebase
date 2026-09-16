# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup data
# MAGIC
# MAGIC Creates the UC Volume layout for the benchmark and stages the golf-swing
# MAGIC dataset (COCO bbox + 17-keypoint annotations). The pose keypoints are what
# MAGIC become the **limb lines** drawn on the swing.
# MAGIC
# MAGIC Ground truth lives under the volume as COCO json so torchvision reads it
# MAGIC natively and Ultralytics gets a YOLO-format export.

# COMMAND ----------

import json
import os
import sys

# Make the bundle's src/ package importable (notebook runs from notebooks/).
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

dbutils.widgets.text("catalog", "rpeng_upleveling")
dbutils.widgets.text("schema", "object_and_vision")
dbutils.widgets.text("volume", "cv_data")

from uplevels_cv.config import Paths
from uplevels_cv.data import ensure_volume_layout
from uplevels_cv.skeleton import KEYPOINT_NAMES, SKELETON

paths = Paths(
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    volume=dbutils.widgets.get("volume"),
    table_prefix="00_seeing_models",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume layout

# COMMAND ----------

ensure_volume_layout(paths)
print("Volume root:", paths.volume_root)
for sub in sorted(os.listdir(paths.volume_root)):
    print(" ", sub)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage the dataset
# MAGIC
# MAGIC TODO: drop your labeled data into the volume. Two supported shapes:
# MAGIC - **COCO subset** (person keypoints) as a quick start — the 17-keypoint
# MAGIC   person layout already gives limb lines for a golfer.
# MAGIC - **Golf-specific labels** in the same COCO format (add club/ball classes).
# MAGIC
# MAGIC Expected files:
# MAGIC `labels/person_keypoints_train.json`, `labels/instances_train.json`, and the
# MAGIC matching `val` splits, with frames under `images/train` and `images/val`.

# COMMAND ----------

print(f"{len(KEYPOINT_NAMES)} keypoints, {len(SKELETON)} limb edges to draw:")
for a, b in SKELETON:
    print(f"  {KEYPOINT_NAMES[a]:>16}  ──  {KEYPOINT_NAMES[b]}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"volume_root": paths.volume_root, "status": "ready"}))
