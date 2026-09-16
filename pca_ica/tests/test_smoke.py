"""Local smoke tests — no network, no Databricks.

Run with `pytest` from the repo root (see requirements.txt). These exercise the
registries, the grid expansion, and an end-to-end pipeline fit on synthetic data
shaped like each registered dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dr_bench import (
    DATASETS,
    MODELS,
    REDUCERS,
    ExperimentConfig,
    build_pipeline,
    expand_grid,
    make_model,
    make_reducer,
)
from dr_bench.datasets import prepare, split_silver, to_silver


def _synth(spec, n=150, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(60, 12, n) for c in spec.numeric_cols}
    for c in spec.categorical_cols:
        data[c] = rng.choice(["A", "B", "C"], n)
    if spec.positive_label is not None:
        data[spec.target_col] = rng.choice([spec.positive_label, "Other"], n)
    else:
        data[spec.target_col] = rng.normal(300000, 60000, n)
    return pd.DataFrame(data)


@pytest.mark.parametrize("name", list(DATASETS))
def test_prepare_and_silver_roundtrip(name):
    spec = DATASETS[name]
    df = _synth(spec)
    X, y = prepare(spec, df)
    assert list(X.columns) == spec.numeric_cols + spec.categorical_cols
    assert set(y.unique()) <= {0, 1}

    silver = to_silver(spec, df)
    assert "label" in silver.columns
    X2, y2 = split_silver(silver)
    assert "label" not in X2.columns
    assert len(y2) == len(df)


def test_expand_grid_collapses_none_reducer():
    cfg = ExperimentConfig(
        datasets=["campus_placement"],
        reducers=["none", "pca"],
        n_components=[5, 8],
        clustering=["none", "kmeans"],
        models=["logistic_regression", "knn"],
    )
    specs = expand_grid(cfg)
    # per (clustering, model): none(1) + pca×2 = 3 cells; × 2 clustering × 2 models = 12
    assert len(specs) == 12
    assert sum(1 for s in specs if s.reducer == "none") == 4  # 2 clustering × 2 models


def test_all_models_and_reducers_build():
    for m in MODELS:
        assert make_model(m, 42) is not None
    for r in REDUCERS:
        make_reducer(r, 5, 42)  # should not raise


@pytest.mark.parametrize("reducer", REDUCERS)
@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "mlp"])
def test_pipeline_fits_and_predicts(reducer, model):
    spec = DATASETS["campus_placement"]
    df = _synth(spec, n=160)
    X, y = prepare(spec, df)
    pipe = build_pipeline(spec, reducer, 4, "kmeans", model, 42)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)
