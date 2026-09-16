"""Silver — explode the COCO arrays into clean, validated rows.

  <prefix>_silver_images    : one row per image  (id -> file_name, size, split)
  <prefix>_silver_keypoints : one row per labeled instance (bbox + 17 keypoints)

Expectations drop degenerate instances (zero-area boxes, too few visible
keypoints) so the training manifest downstream is trustworthy.

NOTE: the exact COCO field paths (a.bbox, a.keypoints, a.num_keypoints, i.id,
i.file_name) assume standard COCO. If your export nests differently, adjust the
`select`s here — this is the one place the schema is pinned.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

PREFIX = spark.conf.get("table_prefix")  # noqa: F821


def name(layer: str, n: str) -> str:
    return f"{PREFIX}_{layer}_{n}"


def ref(layer: str, n: str) -> str:
    # Backtick sibling names — they lead with a digit.
    return f"`{name(layer, n)}`"


@dp.table(name=name("silver", "images"),
          comment="One row per image: id, file name, dimensions, split.")
def silver_images():
    raw = spark.readStream.table(ref("bronze", "raw_labels"))  # noqa: F821
    img = raw.select(F.explode("images").alias("i"), "source_file")
    return img.select(
        F.col("i.id").alias("image_id"),
        F.col("i.file_name").alias("file_name"),
        F.col("i.width").alias("width"),
        F.col("i.height").alias("height"),
        # person_keypoints_<split>.json / instances_<split>.json -> split
        F.regexp_extract("source_file", r"_(train|val|test)\.json$", 1).alias("split"),
    )


@dp.table(name=name("silver", "keypoints"),
          comment="One validated instance per row: bbox + 17 keypoints.")
@dp.expect_or_drop("has_bbox_area", "bbox_area > 0")
@dp.expect_or_drop("enough_visible_keypoints", "num_visible_keypoints >= 5")
def silver_keypoints():
    raw = spark.readStream.table(ref("bronze", "raw_labels"))  # noqa: F821
    ann = raw.select(F.explode("annotations").alias("a"))
    return ann.select(
        F.col("a.image_id").alias("image_id"),
        F.col("a.category_id").alias("category_id"),
        F.col("a.bbox").alias("bbox"),                       # [x, y, w, h]
        (F.col("a.bbox")[2] * F.col("a.bbox")[3]).alias("bbox_area"),
        F.col("a.keypoints").alias("keypoints"),             # [x1,y1,v1, ... x17,y17,v17]
        F.coalesce(F.col("a.num_keypoints"), F.lit(0)).alias("num_visible_keypoints"),
    )
