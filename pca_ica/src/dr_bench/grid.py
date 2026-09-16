"""Expand a config into the list of experiment cells to run.

Cross product of datasets × reducers × n_components × clustering × models. The
`none` reducer ignores n_components and collapses to a single cell rather than
one per n_components value.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

from .config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    reducer: str
    n_components: Optional[int]
    clustering: str
    model: str


def expand_grid(cfg: ExperimentConfig) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for dataset, clustering, model in itertools.product(cfg.datasets, cfg.clustering, cfg.models):
        for reducer in cfg.reducers:
            if reducer in (None, "none"):
                specs.append(ExperimentSpec(dataset, "none", None, clustering, model))
            else:
                for k in cfg.n_components:
                    specs.append(ExperimentSpec(dataset, reducer, k, clustering, model))
    # De-dupe while preserving order (frozen dataclass is hashable).
    return list(dict.fromkeys(specs))
