"""Bronze — raw ingest, as-is, from the UC Volume via Auto Loader.

Two streams land here:
  <prefix>_bronze_raw_labels : COCO annotation JSON files (nested arrays intact)
  <prefix>_bronze_frames     : frame image metadata (path/size/split; no pixels)

Config comes from the pipeline settings (resources/pipeline.yml):
  volume_root  -> /Volumes/<catalog>/<schema>/<volume>
  table_prefix -> 00_seeing_models  (leading digit sorts this project first)

Table names lead with a digit, so sibling reads in silver/gold backtick them.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

VOLUME_ROOT = spark.conf.get("volume_root")  # noqa: F821 (spark is injected by LDP)
PREFIX = spark.conf.get("table_prefix")       # noqa: F821


def name(layer: str, n: str) -> str:
    return f"{PREFIX}_{layer}_{n}"


@dp.table(
    name=name("bronze", "raw_labels"),
    comment="Raw COCO annotation JSON files ingested as-is (one row per file).",
)
def bronze_raw_labels():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .option("multiLine", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_ROOT}/labels")
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("ingested_at", F.current_timestamp())
    )


@dp.table(
    name=name("bronze", "frames"),
    comment="Frame image metadata (path, size, split) — pixels stay in the volume.",
)
def bronze_frames():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .load(f"{VOLUME_ROOT}/images")
        .select(
            F.col("path"),
            F.col("length").alias("bytes"),
            F.col("modificationTime").alias("modified_at"),
            # images/<split>/<file>.jpg -> split
            F.regexp_extract("path", r"/images/([^/]+)/", 1).alias("split"),
        )
    )
