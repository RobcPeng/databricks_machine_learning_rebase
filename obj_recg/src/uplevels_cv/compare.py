"""Aggregate the per-model MLflow runs into one gold comparison table.

Reads the latest run per model spec from the family experiments, flattens the
shared metric columns, and writes them to the gold-schema table
``<catalog>.03_gold_<project>.model_comparison`` — the single table a dashboard
or notebook reads to answer "which model wins on accuracy vs. speed vs. size?".
"""

from __future__ import annotations

from uplevels_cv.config import MODEL_ZOO, Paths

# Columns pulled from each run's metrics, in display order.
METRIC_COLS = [
    "detect_map", "detect_map50", "pose_map", "pose_oks", "mpjpe_mm",
    "latency_ms_p50", "latency_ms_p95", "params_millions", "size_mb",
    "train_seconds",
]

# Identifier columns, in the order collect_rows() builds them; the rest of each
# row is METRIC_COLS. Used to pin the write schema (see write_comparison).
ID_COLS = ["sport", "model_key", "family", "task", "description", "run_id"]


def collect_rows(experiment_root: str, sport: str) -> list[dict]:
    """One row per model spec for a sport, using its most recent finished run.

    Runs are named ``<sport>_<model_key>`` (see train.train_spec), so the search
    is scoped to the given sport.
    """
    import mlflow

    rows = []
    for key, spec in MODEL_ZOO.items():
        exp = mlflow.get_experiment_by_name(f"{experiment_root}/{spec.family.value}")
        if exp is None:
            continue
        runs = mlflow.search_runs(
            [exp.experiment_id],
            filter_string=(f"tags.mlflow.runName = '{sport}_{key}' "
                           "and attributes.status = 'FINISHED'"),
            order_by=["start_time DESC"], max_results=1,
        )
        if runs.empty:
            continue
        r = runs.iloc[0]
        row = {"sport": sport, "model_key": key, "family": spec.family.value,
               "task": spec.task.value, "description": spec.description,
               "run_id": r["run_id"]}
        for col in METRIC_COLS:
            mcol = f"metrics.{col}"
            row[col] = float(r[mcol]) if mcol in r and r[mcol] == r[mcol] else None
        rows.append(row)
    return rows


def write_comparison(spark, paths: Paths, experiment_root: str, sport: str) -> str:
    """Write the comparison rows for one sport. Returns the table's full name.

    Uses replaceWhere on ``sport`` so each per-sport run updates only its own rows
    and the leaderboard accumulates every sport.
    """
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    rows = collect_rows(experiment_root, sport)
    if not rows:
        raise RuntimeError(f"No finished runs for sport '{sport}' — did training succeed?")

    # Pin the schema instead of inferring it. A metric no model populated yet
    # (mpjpe_mm needs 3D ground truth; the torchvision detect/pose columns wait
    # on that eval loop) is None in every row, and inferring from all-None values
    # yields a void column that Delta rejects on write. Force those to nullable
    # doubles. Positional records keep column order aligned with the schema.
    schema = StructType(
        [StructField(c, StringType(), True) for c in ID_COLS]
        + [StructField(c, DoubleType(), True) for c in METRIC_COLS]
    )
    records = [[row[c] for c in ID_COLS] + [row[c] for c in METRIC_COLS] for row in rows]

    # Backtick-quoted (the gold schema name leads with a digit).
    table = paths.table("gold", "model_comparison")
    (spark.createDataFrame(records, schema=schema).write.mode("overwrite")
     .partitionBy("sport")
     .option("replaceWhere", f"sport = '{sport}'")
     .option("mergeSchema", "true")
     .saveAsTable(table))
    return table
