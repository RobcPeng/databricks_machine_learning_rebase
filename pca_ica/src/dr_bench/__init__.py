"""dr_bench — config-driven harness for comparing classic ML models across
datasets and dimensionality-reduction techniques.

The grid it sweeps is the cross product of:

    datasets  ×  reducers (none/PCA/ICA/RP)  ×  clustering features  ×  models

Each cell is trained with cross-validation and a held-out test split, and logged
as one MLflow run. The package is plain scikit-learn and runs both locally (for
tests) and on Databricks serverless (for the sweep).
"""

from .config import ExperimentConfig, load_config
from .datasets import DATASETS, DatasetSpec, load_raw, prepare, split_silver, to_silver
from .reducers import REDUCERS, make_reducer
from .clustering import CLUSTERINGS, make_cluster_featurizer
from .models import MODELS, make_model
from .pipeline import build_pipeline, build_preprocessor
from .grid import ExperimentSpec, expand_grid
from .experiment import load_datasets, run_one

__all__ = [
    "ExperimentConfig",
    "load_config",
    "DATASETS",
    "DatasetSpec",
    "load_raw",
    "prepare",
    "to_silver",
    "split_silver",
    "REDUCERS",
    "make_reducer",
    "CLUSTERINGS",
    "make_cluster_featurizer",
    "MODELS",
    "make_model",
    "build_pipeline",
    "build_preprocessor",
    "ExperimentSpec",
    "expand_grid",
    "load_datasets",
    "run_one",
]
