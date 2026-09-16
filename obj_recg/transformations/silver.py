"""Silver (02) — validated, conformed records from bronze.

Expectations drop degenerate rows so the gold features downstream are
trustworthy. Structure is unchanged from bronze; this layer is about quality.
Publishes to the silver schema (<project>_02_silver).

  <catalog>.<project>_02_silver.images    : images with a known split
  <catalog>.<project>_02_silver.keypoints : instances with a real box and enough keypoints
  <catalog>.<project>_02_silver.games     : games with a non-negative score
"""

from pyspark import pipelines as dp

from _layers import make_fqn

CATALOG = spark.conf.get("catalog")  # noqa: F821
PROJECT = spark.conf.get("project")  # noqa: F821

fqn = make_fqn(CATALOG, PROJECT)


@dp.table(name=fqn("silver", "images"),
          comment="Images with a known split.")
@dp.expect_or_drop("known_split", "split IN ('train', 'val', 'test')")
def silver_images():
    return spark.readStream.table(fqn("bronze", "images"))  # noqa: F821


@dp.table(name=fqn("silver", "keypoints"),
          comment="Instances with a real box and enough visible keypoints.")
@dp.expect_or_drop("has_bbox_area", "bbox_area > 0")
@dp.expect_or_drop("enough_visible_keypoints", "num_visible_keypoints >= 5")
def silver_keypoints():
    return spark.readStream.table(fqn("bronze", "keypoints"))  # noqa: F821


@dp.table(name=fqn("silver", "games"),
          comment="Games with a non-negative score.")
@dp.expect_or_drop("non_negative_score", "points_for >= 0 AND points_against >= 0")
def silver_games():
    return spark.readStream.table(fqn("bronze", "games"))  # noqa: F821
