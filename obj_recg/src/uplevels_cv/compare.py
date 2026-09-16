"""Aggregate the per-model MLflow runs into one gold comparison table.

Reads the latest run per model spec from the family experiments, flattens the
shared metric columns, and writes them to
``{catalog}.{schema}.{table_prefix}_comparison`` — the single table a dashboard
or notebook reads to answer "which model wins on accuracy vs. speed vs. size?".
"""

from __future__ import annotations

from uplevels_cv.config import MODEL_ZOO, Paths

# Columns pulled from each run's metrics, in display order.
METRIC_COLS = [
    "detect_map", "detect_map50", "pose_map", "pose_oks",
    "latency_ms_p50", "latency_ms_p95", "params_millions", "size_mb",
    "train_seconds",
]


def collect_rows(experiment_root: str) -> list[dict]:
    """One row per model spec, using its most recent finished run."""
    import mlflow

    rows = []
    for key, spec in MODEL_ZOO.items():
        exp = mlflow.get_experiment_by_name(f"{experiment_root}/{spec.family.value}")
        if exp is None:
            continue
        runs = mlflow.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{key}' and attributes.status = 'FINISHED'",
            order_by=["start_time DESC"], max_results=1,
        )
        if runs.empty:
            continue
        r = runs.iloc[0]
        row = {"model_key": key, "family": spec.family.value, "task": spec.task.value,
               "description": spec.description, "run_id": r["run_id"]}
        for col in METRIC_COLS:
            mcol = f"metrics.{col}"
            row[col] = float(r[mcol]) if mcol in r and r[mcol] == r[mcol] else None
        rows.append(row)
    return rows


def write_comparison(spark, paths: Paths, experiment_root: str) -> str:
    """Write/overwrite the comparison Delta table. Returns its full name."""
    rows = collect_rows(experiment_root)
    if not rows:
        raise RuntimeError("No finished runs found — did the training tasks succeed?")

    table = paths.table("comparison")
    spark.createDataFrame(rows).write.mode("overwrite").option(
        "overwriteSchema", "true").saveAsTable(table.replace("`", ""))
    return table
