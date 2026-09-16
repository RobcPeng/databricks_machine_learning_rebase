"""Run experiment cells and (optionally) log them to MLflow.

Datasets are loaded once into a cache. Each cell then trains its pipeline with
k-fold cross-validation on the train split and reports held-out test metrics.
Every cell returns a flat dict (one leaderboard row) and, when an mlflow module
is passed, is also logged as a nested MLflow run.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split

from .config import ExperimentConfig
from .datasets import DATASETS, load_raw, prepare, split_silver
from .grid import ExperimentSpec
from .pipeline import build_pipeline

# name -> (DatasetSpec, X, y)
DatasetCache = dict[str, tuple[Any, Any, Any]]


def load_datasets(cfg: ExperimentConfig, spark: Optional[Any] = None) -> DatasetCache:
    """Populate the cache. On Databricks, read the cleaned `_silver` UC table;
    off-platform (tests/local), read the public CSV and clean it in-memory."""
    cache: DatasetCache = {}
    for name in cfg.datasets:
        spec = DATASETS[name]
        if spark is not None:
            silver = spark.sql(f"SELECT * FROM {cfg.table(f'{name}_silver')}").toPandas()
            X, y = split_silver(silver)
        else:
            X, y = prepare(spec, load_raw(spec))
        cache[name] = (spec, X, y)
    return cache


def run_one(
    spec: ExperimentSpec,
    cfg: ExperimentConfig,
    cache: DatasetCache,
    mlflow: Optional[Any] = None,
) -> dict:
    dspec, X, y = cache[spec.dataset]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    pipe = build_pipeline(
        dspec, spec.reducer, spec.n_components, spec.clustering, spec.model, cfg.random_state
    )

    t0 = time.time()
    cv = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cfg.cv_folds,
        scoring=["accuracy", "f1", "roc_auc"],
        error_score="raise",
    )
    cv_seconds = time.time() - t0

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        test_auc = float("nan")

    row = {
        "dataset": spec.dataset,
        "reducer": spec.reducer,
        "n_components": spec.n_components if spec.n_components is not None else -1,
        "clustering": spec.clustering,
        "model": spec.model,
        "cv_accuracy": float(np.mean(cv["test_accuracy"])),
        "cv_f1": float(np.mean(cv["test_f1"])),
        "cv_roc_auc": float(np.mean(cv["test_roc_auc"])),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": test_auc,
        "cv_seconds": float(cv_seconds),
    }
    if spec.reducer == "pca":
        red = pipe.named_steps.get("reduce")
        if red is not None and hasattr(red, "explained_variance_ratio_"):
            row["pca_explained_var"] = float(np.sum(red.explained_variance_ratio_))

    if mlflow is not None:
        run_name = f"{spec.dataset}|{spec.reducer}|k{row['n_components']}|{spec.clustering}|{spec.model}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(
                {
                    "dataset": spec.dataset,
                    "reducer": spec.reducer,
                    "n_components": row["n_components"],
                    "clustering": spec.clustering,
                    "model": spec.model,
                    "cv_folds": cfg.cv_folds,
                    "test_size": cfg.test_size,
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in row.items() if isinstance(v, float) and not np.isnan(v)}
            )
            mlflow.set_tags(
                {"dataset": spec.dataset, "reducer": spec.reducer, "model": spec.model}
            )
    return row
