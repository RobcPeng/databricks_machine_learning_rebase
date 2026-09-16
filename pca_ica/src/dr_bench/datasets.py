"""Dataset registry.

Two public, no-auth higher-ed datasets, both framed as the same question —
*does a graduate land a good placement outcome?* — so a single classification
leaderboard is comparable across them.

    campus_placement    Kaggle "Campus Recruitment" (215 MBA grads).
                        Target: status (Placed / Not Placed).
    engineering_salary  Aspiring Minds AMEO 2015 (2,998 engineering grads).
                        Target: Salary, binarized at the median into an
                        "above-median starting salary" outcome.

To add a dataset: append a DatasetSpec below. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class DatasetSpec:
    name: str
    source_url: str
    target_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    drop_cols: list[str] = field(default_factory=list)
    # Binary-label derivation (pick one):
    positive_label: Optional[str] = None  # label == this -> 1  (e.g. "Placed")
    binarize_at_median: bool = False       # value > median -> 1  (e.g. salary)
    description: str = ""

    def make_target(self, s: pd.Series) -> pd.Series:
        if self.positive_label is not None:
            return (s.astype(str).str.strip() == self.positive_label).astype(int)
        if self.binarize_at_median:
            v = pd.to_numeric(s, errors="coerce")
            return (v > v.median()).astype(int)
        raise ValueError(f"{self.name}: no target rule (set positive_label or binarize_at_median)")


DATASETS: dict[str, DatasetSpec] = {
    "campus_placement": DatasetSpec(
        name="campus_placement",
        source_url=(
            "https://raw.githubusercontent.com/MainakRepositor/Datasets/"
            "master/Placement_Data_Full_Class.csv"
        ),
        target_col="status",
        positive_label="Placed",
        numeric_cols=["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"],
        categorical_cols=[
            "gender",
            "ssc_b",
            "hsc_b",
            "hsc_s",
            "degree_t",
            "workex",
            "specialisation",
        ],
        # sl_no is an id; salary leaks the target (only present when Placed).
        drop_cols=["sl_no", "salary"],
        description="Campus recruitment: will a graduating MBA student be placed?",
    ),
    "engineering_salary": DatasetSpec(
        name="engineering_salary",
        source_url=(
            "https://raw.githubusercontent.com/anuragsingh2207/dsba-jupyter-notebook/"
            "master/lpm/Revision/Dataset/Engineering_graduate_salary.csv"
        ),
        target_col="Salary",
        binarize_at_median=True,
        numeric_cols=[
            "10percentage",
            "12graduation",
            "12percentage",
            "CollegeTier",
            "collegeGPA",
            "CollegeCityTier",
            "GraduationYear",
            "Logical",
            "Quant",
            "Domain",
            "agreeableness",
            "openess_to_experience",
        ],
        categorical_cols=["Gender"],
        # ID / CollegeID / CollegeCityID are identifiers, not signal.
        drop_cols=["ID", "CollegeID", "CollegeCityID"],
        description="AMEO: does an engineering grad land an above-median starting salary?",
    ),
}


def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    """Read the raw CSV straight from its public URL (used locally / in tests)."""
    return pd.read_csv(spec.source_url)


def prepare(spec: DatasetSpec, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw (bronze) frame into (features, binary target) per the spec."""
    df = df.copy()
    y = spec.make_target(df[spec.target_col]).astype(int)
    X = df[spec.numeric_cols + spec.categorical_cols].copy()
    for c in spec.numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in spec.categorical_cols:
        X[c] = X[c].astype(str)
    return X, y


def to_silver(spec: DatasetSpec, bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned, analysis-ready frame: feature columns + a binary `label`.

    This is exactly what the experiment harness trains on, so silver is the
    contract between ingestion and modeling.
    """
    X, y = prepare(spec, bronze_df)
    silver = X.copy()
    silver["label"] = y.to_numpy()
    return silver


def split_silver(silver_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a silver frame back into (features, label)."""
    y = silver_df["label"].astype(int)
    X = silver_df.drop(columns=["label"])
    return X, y
