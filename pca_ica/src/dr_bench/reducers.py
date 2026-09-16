"""Dimensionality-reduction step of the pipeline.

`DimReducer` wraps PCA / FastICA / GaussianRandomProjection behind one estimator
so the pipeline builder doesn't branch, and so `n_components` is clamped to the
feature count *at fit time* (one-hot encoding makes the width unknown until
then). For PCA it also exposes `explained_variance_ratio_` for logging.
"""

from __future__ import annotations

from typing import Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

REDUCERS = ["none", "pca", "ica", "random_projection"]


class DimReducer(BaseEstimator, TransformerMixin):
    def __init__(self, kind: str = "pca", n_components: Optional[int] = 10, random_state: int = 42):
        self.kind = kind
        self.n_components = n_components
        self.random_state = random_state

    def _build(self, n_features: int):
        k = self.n_components if self.n_components is not None else n_features
        k = max(1, min(int(k), n_features))  # clamp: PCA/ICA require k <= n_features
        if self.kind == "pca":
            return PCA(n_components=k, random_state=self.random_state)
        if self.kind == "ica":
            return FastICA(
                n_components=k,
                random_state=self.random_state,
                whiten="unit-variance",
                max_iter=1000,
                tol=1e-3,
            )
        if self.kind == "random_projection":
            # RP is happy to project up as well as down, but keep it <= n_features here.
            return GaussianRandomProjection(n_components=k, random_state=self.random_state)
        raise ValueError(f"Unknown reducer kind: {self.kind}")

    def fit(self, X, y=None):
        self.impl_ = self._build(X.shape[1]).fit(X)
        if hasattr(self.impl_, "explained_variance_ratio_"):
            self.explained_variance_ratio_ = self.impl_.explained_variance_ratio_
        return self

    def transform(self, X):
        return self.impl_.transform(X)


def make_reducer(
    name: Optional[str], n_components: Optional[int], random_state: int
) -> Union[str, DimReducer]:
    """Return a transformer, or the string "passthrough" for the `none` case."""
    if name in (None, "none"):
        return "passthrough"
    if name not in REDUCERS:
        raise ValueError(f"Unknown reducer: {name}. Options: {REDUCERS}")
    return DimReducer(kind=name, n_components=n_components, random_state=random_state)
