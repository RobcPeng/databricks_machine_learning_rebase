"""Optional clustering-derived features.

The classic "cluster, then learn on the augmented representation" move:
fit KMeans or a Gaussian Mixture on the (already reduced) features and append
one-hot cluster membership as extra columns. `none` skips the step entirely.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

CLUSTERINGS = ["none", "kmeans", "gmm"]


class ClusterFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, kind: str = "kmeans", n_clusters: int = 4, random_state: int = 42):
        self.kind = kind
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.kind == "kmeans":
            self.model_ = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            ).fit(X)
        elif self.kind == "gmm":
            self.model_ = GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            ).fit(X)
        else:
            raise ValueError(f"Unknown clustering kind: {self.kind}")
        return self

    def transform(self, X):
        X = np.asarray(X)
        labels = self.model_.predict(X)
        onehot = np.zeros((X.shape[0], self.n_clusters), dtype=float)
        onehot[np.arange(X.shape[0]), labels] = 1.0
        return np.hstack([X, onehot])


def make_cluster_featurizer(
    name: Optional[str], random_state: int, n_clusters: int = 4
) -> Optional[ClusterFeaturizer]:
    """Return a featurizer, or None for the `none` case (step is dropped)."""
    if name in (None, "none"):
        return None
    if name not in CLUSTERINGS:
        raise ValueError(f"Unknown clustering: {name}. Options: {CLUSTERINGS}")
    return ClusterFeaturizer(kind=name, n_clusters=n_clusters, random_state=random_state)
