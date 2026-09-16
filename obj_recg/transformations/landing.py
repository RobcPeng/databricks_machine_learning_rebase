"""Landing (00) — raw file ingest, as-is, from the UC Volume via Auto Loader.

Each layer publishes to its own schema, <project>_<numbered_layer>, via
fully-qualified dataset names. The landing zone captures files verbatim (one row
per file), with source path and ingest timestamp and no parsing.

  <catalog>.<project>_00_landing.raw_labels : COCO annotation JSON files
  <catalog>.<project>_00_landing.frames     : frame image metadata (no pixels)
  <catalog>.<project>_00_landing.games      : game-result CSV rows

Data is namespaced by sport under the volume (images/<sport>/<split>/,
labels/<sport>/, games/<sport>/). Config comes from the pipeline settings
(resources/pipeline.yml): catalog, project, volume_root.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

from _layers import make_fqn

VOLUME_ROOT = spark.conf.get("volume_root")  # noqa: F821 (spark is injected by LDP)
CATALOG = spark.conf.get("catalog")           # noqa: F821
PROJECT = spark.conf.get("project")           # noqa: F821

fqn = make_fqn(CATALOG, PROJECT)  # `<catalog>`.`<project>_<layer>`.`<table>`


@dp.table(name=fqn("landing", "raw_labels"),
          comment="Raw COCO annotation JSON files ingested as-is (one row per file).")
def landing_raw_labels():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .option("multiLine", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_ROOT}/labels")
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("ingested_at", F.current_timestamp())
    )


@dp.table(name=fqn("landing", "frames"),
          comment="Frame image metadata (path, size, sport, split). Pixels stay in the volume.")
def landing_frames():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .load(f"{VOLUME_ROOT}/images")
        .select(
            F.col("path"),
            F.col("length").alias("bytes"),
            F.col("modificationTime").alias("modified_at"),
            F.regexp_extract("path", r"/images/([^/]+)/", 1).alias("sport"),
            F.regexp_extract("path", r"/images/[^/]+/([^/]+)/", 1).alias("split"),
        )
    )


@dp.table(name=fqn("landing", "games"),
          comment="Game-result CSV rows ingested from games/<sport>/*.csv.")
def landing_games():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_ROOT}/games")
        .withColumn("sport", F.regexp_extract(F.col("_metadata.file_path"),
                                              r"/games/([^/]+)/", 1))
        .withColumn("ingested_at", F.current_timestamp())
    )
