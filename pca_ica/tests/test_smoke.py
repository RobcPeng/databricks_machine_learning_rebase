"""Local smoke tests — no network, no Databricks.

Run with `pytest` from the repo root (see requirements.txt). These exercise the
registries, grid expansion, and an end-to-end pipeline fit on synthetic data
shaped like each registered dataset (including filters and target columns).
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
    prepare,
    split_silver,
    to_silver,
)


def _synth(spec, n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(50, 15, n) for c in spec.numeric_cols}
    for c in spec.categorical_cols:
        data[c] = rng.choice(["A", "B", "C", "D"], n)
    for col, vals in spec.filters.items():
        # mostly allowed values so rows survive the filter
        data[col] = rng.choice([str(vals[0])] * 5 + ["ZZ"], n)
    tcol = spec.target_col[0]
    if spec.target_kind == "binarize_median":
        data[tcol] = rng.normal(80000, 25000, n)
    else:
        data[tcol] = rng.choice([spec.positive_label, "Other"], n)
    return pd.DataFrame(data)


@pytest.mark.parametrize("name", list(DATASETS))
def test_prepare_and_silver_roundtrip(name):
    spec = DATASETS[name]
    df = _synth(spec)
    X, y = prepare(spec, df)
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) > 0

    silver = to_silver(spec, df)
    assert "label" in silver.columns
    X2, y2 = split_silver(silver)
    assert "label" not in X2.columns
    assert len(X2) == len(y2)


def test_expand_grid_collapses_none_reducer():
    cfg = ExperimentConfig(
        datasets=["nscg_2023"],
        reducers=["none", "pca"],
        n_components=[5, 8],
        clustering=["none", "kmeans"],
        models=["logistic_regression", "knn"],
    )
    specs = expand_grid(cfg)
    # per (clustering, model): none(1) + pca x2 = 3; x 2 clustering x 2 models = 12
    assert len(specs) == 12
    assert sum(1 for s in specs if s.reducer == "none") == 4


def test_all_models_and_reducers_build():
    for m in MODELS:
        assert make_model(m, 42) is not None
    for r in REDUCERS:
        make_reducer(r, 5, 42)


@pytest.mark.parametrize("reducer", REDUCERS)
@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "mlp"])
def test_pipeline_fits_and_predicts(reducer, model):
    spec = DATASETS["nscg_2023"]
    X, y = prepare(spec, _synth(spec, n=400))
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pipe = build_pipeline(numeric, categorical, reducer, 4, "kmeans", model, 42)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)
