"""Gold — the training manifest the model trainers read.

  <prefix>_gold_training_manifest : one row per image, split-labeled, with
  instance counts and keypoint coverage. Dimensions the trainers/dashboards
  slice by (split, file_name, image_id, dimensions) are preserved — aggregate
  further in queries, never lose them here.

The model-comparison leaderboard (<prefix>_gold_model_comparison) is the other
gold table; it is written from MLflow runs by src/uplevels_cv/compare.py after
training, not by this pipeline.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

PREFIX = spark.conf.get("table_prefix")  # noqa: F821


def name(layer: str, n: str) -> str:
    return f"{PREFIX}_{layer}_{n}"


def ref(layer: str, n: str) -> str:
    return f"`{name(layer, n)}`"


@dp.materialized_view(
    name=name("gold", "training_manifest"),
    comment="Per-image training manifest: split, instance counts, keypoint coverage.",
)
def gold_training_manifest():
    per_image = (
        spark.read.table(ref("silver", "keypoints"))  # noqa: F821
        .groupBy("image_id")
        .agg(
            F.count("*").alias("num_instances"),
            F.round(F.avg("num_visible_keypoints"), 2).alias("avg_visible_keypoints"),
            F.round(F.sum("bbox_area"), 1).alias("total_bbox_area"),
        )
    )
    images = spark.read.table(ref("silver", "images"))  # noqa: F821

    # Left join keeps images with no valid instances (they're training-relevant
    # as negatives). Preserve split/file_name/dimensions for downstream slicing.
    return (
        images.join(per_image, "image_id", "left")
        .na.fill(0, ["num_instances", "avg_visible_keypoints", "total_bbox_area"])
        .withColumn("has_pose", F.col("avg_visible_keypoints") > 0)
    )

    # TODO: enrich with swing joint angles (skeleton.GOLF_ANGLES) once the
    #       keypoints array schema is confirmed — add a pandas_udf over
    #       silver_keypoints.keypoints computing lead_arm / spine_tilt / etc.
