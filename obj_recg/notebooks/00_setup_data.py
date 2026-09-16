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
dbutils.widgets.text("project", "seeing_models")
dbutils.widgets.text("volume", "cv_data")
dbutils.widgets.text("sport", "golf")

from uplevels_cv.config import Paths
from uplevels_cv.data import ensure_volume_layout
from uplevels_cv.skeleton import KEYPOINT_NAMES, SKELETON
from uplevels_cv.sports import sport_spec

sport = dbutils.widgets.get("sport")
paths = Paths(
    catalog=dbutils.widgets.get("catalog"),
    project=dbutils.widgets.get("project"),
    volume=dbutils.widgets.get("volume"),
)
spec = sport_spec(sport)

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
# MAGIC ## Stage the dataset (per sport)
# MAGIC
# MAGIC TODO: drop labeled data into the volume, namespaced by sport. Expected
# MAGIC files for the selected sport:
# MAGIC - `labels/<sport>/person_keypoints_<split>.json` and `instances_<split>.json`
# MAGIC - frames under `images/<sport>/<split>/`
# MAGIC - game results under `games/<sport>/*.csv` (team-performance labels)
# MAGIC
# MAGIC The 17-keypoint COCO person layout is shared across sports; detection
# MAGIC classes and biomechanics angles are per-sport (see below).

# COMMAND ----------

import os

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(paths.volume_root, "images", sport, split), exist_ok=True)
for sub in ("labels", "games", "splits"):
    os.makedirs(os.path.join(paths.volume_root, sub, sport), exist_ok=True)

# COMMAND ----------

print(f"Sport: {sport}")
print(f"Detection classes: {spec.detect_classes}")
print(f"Events: {spec.events}")
print(f"Score unit: {spec.score_unit}")
print(f"\nBiomechanics angles ({len(spec.angles)}):")
for angle_name, (vertex, a, b) in spec.angles.items():
    print(f"  {angle_name:>18}: angle at {vertex} between {a} and {b}")

print(f"\n{len(KEYPOINT_NAMES)} keypoints, {len(SKELETON)} limb edges to draw:")
for a, b in SKELETON:
    print(f"  {KEYPOINT_NAMES[a]:>16}  --  {KEYPOINT_NAMES[b]}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(
    {"volume_root": paths.volume_root, "sport": sport, "status": "ready"}))
