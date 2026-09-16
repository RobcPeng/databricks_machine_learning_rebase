"""Assemble one end-to-end sklearn Pipeline for a grid cell.

    preprocess  ->  [reduce]  ->  [cluster features]  ->  model

Preprocessing imputes and scales numeric columns and one-hot encodes categorical
columns (dense output for the downstream reducer). The reduce and cluster steps
are omitted when set to "none". Column lists are passed in explicitly so any
dataset schema works without a hard-coded feature list.
"""

from __future__ import annotations

from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clustering import make_cluster_featurizer
from .models import make_model
from .reducers import make_reducer


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)]
    )


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    reducer: str,
    n_components: Optional[int],
    clustering: str,
    model: str,
    random_state: int,
) -> Pipeline:
    steps = [("preprocess", build_preprocessor(numeric_cols, categorical_cols))]

    reduce_step = make_reducer(reducer, n_components, random_state)
    if reduce_step != "passthrough":
        steps.append(("reduce", reduce_step))

    cluster_step = make_cluster_featurizer(clustering, random_state)
    if cluster_step is not None:
        steps.append(("cluster", cluster_step))

    steps.append(("model", make_model(model, random_state)))
    return Pipeline(steps)
