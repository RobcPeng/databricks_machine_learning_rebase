"""Experiment configuration.

The grid (datasets/reducers/models/…) lives in `conf/experiments.yml`. The
Databricks-specific bits (catalog/schema/table_prefix) come in as job
parameters and override the YAML, so the same grid can target any environment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ExperimentConfig:
    # --- where results land (overridden by job params / widgets) ---
    catalog: str = "rpeng_upleveling"
    schema: str = "dimensionality_reduction"
    table_prefix: str = "00_experiment_type"

    # --- the grid (from conf/experiments.yml) ---
    task: str = "classification"
    datasets: list[str] = field(default_factory=lambda: ["campus_placement", "engineering_salary"])
    reducers: list[str] = field(default_factory=lambda: ["none", "pca", "ica", "random_projection"])
    n_components: list[Optional[int]] = field(default_factory=lambda: [10])
    clustering: list[str] = field(default_factory=lambda: ["none", "kmeans"])
    models: list[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "svm_rbf",
            "knn",
            "random_forest",
            "xgboost",
            "mlp",
        ]
    )

    # --- training knobs ---
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    register_best: bool = False

    def table(self, name: str) -> str:
        """Backtick-quoted fully-qualified table name.

        The prefix starts with a digit (`00_…`), so every identifier must be
        quoted or Spark SQL will reject it.
        """
        return f"`{self.catalog}`.`{self.schema}`.`{self.table_prefix}_{name}`"

    def experiment_leaf(self) -> str:
        """Leaf name for the MLflow experiment (the folder is the user's home)."""
        return self.table_prefix


def load_config(path: str, **overrides: Any) -> ExperimentConfig:
    """Load the grid YAML, then apply non-null overrides (e.g. from widgets)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None and v != ""})
    known = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return ExperimentConfig(**{k: v for k, v in raw.items() if k in known})
