"""Bronze (01) — structured records parsed from the landing zone, unfiltered.

Explodes the raw COCO arrays into typed per-record rows and types the game
rows. No quality filtering yet — silver does that. Publishes to the bronze
schema (<project>_01_bronze).

  <catalog>.<project>_01_bronze.images    : one row per image (id, file, size, sport, split)
  <catalog>.<project>_01_bronze.keypoints : one instance per row (bbox + 17 keypoints, sport)
  <catalog>.<project>_01_bronze.games     : one team-game per row, typed

NOTE: the COCO field paths (a.bbox, a.keypoints, a.num_keypoints, i.id,
i.file_name) and the game-result columns assume a standard schema. Adjust the
`select`s here if the export differs — this is where the schema is pinned.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

from _layers import make_fqn

CATALOG = spark.conf.get("catalog")  # noqa: F821
PROJECT = spark.conf.get("project")  # noqa: F821

fqn = make_fqn(CATALOG, PROJECT)

# Pull sport from the labels sub-path and split from the COCO file suffix.
_SPORT_RE = r"/labels/([^/]+)/"
_SPLIT_RE = r"_(train|val|test)\.json$"


@dp.table(name=fqn("bronze", "images"),
          comment="One row per image: id, file name, dimensions, sport, split.")
def bronze_images():
    raw = spark.readStream.table(fqn("landing", "raw_labels"))  # noqa: F821
    img = raw.select(F.explode("images").alias("i"), "source_file")
    return img.select(
        F.col("i.id").alias("image_id"),
        F.col("i.file_name").alias("file_name"),
        F.col("i.width").alias("width"),
        F.col("i.height").alias("height"),
        F.regexp_extract("source_file", _SPORT_RE, 1).alias("sport"),
        F.regexp_extract("source_file", _SPLIT_RE, 1).alias("split"),
    )


@dp.table(name=fqn("bronze", "keypoints"),
          comment="One instance per row: bbox + 17 keypoints, sport-tagged (unfiltered).")
def bronze_keypoints():
    raw = spark.readStream.table(fqn("landing", "raw_labels"))  # noqa: F821
    ann = raw.select(F.explode("annotations").alias("a"), "source_file")
    return ann.select(
        F.col("a.image_id").alias("image_id"),
        F.col("a.category_id").alias("category_id"),
        F.col("a.bbox").alias("bbox"),                       # [x, y, w, h]
        (F.col("a.bbox")[2] * F.col("a.bbox")[3]).alias("bbox_area"),
        F.col("a.keypoints").alias("keypoints"),             # [x1,y1,v1, ... x17,y17,v17]
        F.coalesce(F.col("a.num_keypoints"), F.lit(0)).alias("num_visible_keypoints"),
        F.regexp_extract("source_file", _SPORT_RE, 1).alias("sport"),
    )


@dp.table(name=fqn("bronze", "games"),
          comment="One team-game per row, typed (unfiltered).")
def bronze_games():
    games = spark.readStream.table(fqn("landing", "games"))  # noqa: F821
    # TODO: align these columns with the games CSV schema. points_for/against are
    #       the score; win derives from them; efficiency columns are optional
    #       labels for the efficiency targets.
    return games.select(
        F.col("sport"),
        F.col("game_id").cast("string").alias("game_id"),
        F.col("team").alias("team"),
        F.col("opponent").alias("opponent"),
        F.to_date("game_date").alias("game_date"),
        F.col("points_for").cast("double").alias("points_for"),
        F.col("points_against").cast("double").alias("points_against"),
        (F.col("points_for") > F.col("points_against")).cast("int").alias("win"),
        F.col("off_efficiency").cast("double").alias("off_efficiency"),
        F.col("def_efficiency").cast("double").alias("def_efficiency"),
    )
