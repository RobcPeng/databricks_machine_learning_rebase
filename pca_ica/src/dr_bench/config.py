"""Experiment configuration.

The grid (datasets/reducers/models/…) is defined in `conf/experiments.yml`. The
Databricks placement (catalog/schema_prefix) is supplied as job parameters and
overrides the YAML.

Tables are laid out as a medallion across three schemas under one catalog:

    <schema_prefix>_00_landing    raw source, landed verbatim   (bronze)
    <schema_prefix>_01_cleansed   filtered/cleaned + label      (silver)
    <schema_prefix>_02_curated    leaderboard + comparison      (gold)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# layer -> schema-name suffix (the number orders the schemas in the catalog)
LAYER_SUFFIXES = {"landing": "00_landing", "cleansed": "01_cleansed", "curated": "02_curated"}


@dataclass
class ExperimentConfig:
    catalog: str = "rpeng_upleveling"
    schema_prefix: str = "dimensionality_reduction"

    task: str = "classification"
    datasets: list[str] = field(
        default_factory=lambda: ["nscg_2023", "scorecard_2026", "acs_co_2024", "pisa_2022"]
    )
    reducers: list[str] = field(default_factory=lambda: ["none", "pca", "ica", "random_projection"])
    n_components: list[Optional[int]] = field(default_factory=lambda: [10])
    clustering: list[str] = field(default_factory=lambda: ["none", "kmeans"])
    models: list[str] = field(
        default_factory=lambda: [
            "logistic_regression", "svm_rbf", "knn", "random_forest", "xgboost", "mlp",
        ]
    )

    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    sample_rows: int = 6000  # stratified cap per dataset; 0 disables
    register_best: bool = False

    def schema_for(self, layer: str) -> str:
        return f"{self.schema_prefix}_{LAYER_SUFFIXES[layer]}"

    def all_schemas(self) -> list[str]:
        return [self.schema_for(layer) for layer in LAYER_SUFFIXES]

    def table(self, layer: str, name: str) -> str:
        """Backtick-quoted fully-qualified table name for a medallion layer."""
        return f"`{self.catalog}`.`{self.schema_for(layer)}`.`{name}`"

    def experiment_leaf(self) -> str:
        return "dr_benchmark"


def load_config(path: str, **overrides: Any) -> ExperimentConfig:
    """Load the grid YAML and apply non-empty overrides (e.g. from widgets)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None and v != ""})
    known = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return ExperimentConfig(**{k: v for k, v in raw.items() if k in known})
