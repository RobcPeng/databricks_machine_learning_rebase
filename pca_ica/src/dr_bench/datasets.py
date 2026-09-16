"""Dataset registry.

Four public US-anchored datasets, each a different "track" for predicting an
earnings/attainment outcome, all reduced to one binary classification task so a
single leaderboard is comparable across them. See docs/DATASETS.md for the full
data dictionary, sources, and target definitions.

    nscg_2023      NSF National Survey of College Graduates 2023. Per graduate.
    scorecard_2026 US Dept. of Education College Scorecard. Per institution.
    acs_co_2024    Census ACS PUMS 2024 (Colorado). Per person.
    pisa_2022      OECD PISA 2022 student questionnaire. Per student (opt-in).

To add a dataset: append a DatasetSpec and, if its source needs a new fetch
pattern, a loader in loaders.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .loaders import load_csv_from_zip, load_csv_url, load_sav_from_zip


@dataclass
class DatasetSpec:
    name: str
    loader: str  # "nscg_zip" | "scorecard_csv" | "acs_zip" | "pisa_sav"
    source_url: str
    target_col: list[str]  # first column present is used
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_kind: str = "binarize_median"  # or "label_equals"
    positive_label: Optional[str] = None
    filters: dict[str, list[str]] = field(default_factory=dict)  # keep rows where col value in list
    sentinel_min: float = 9_999_990.0  # numeric values >= this are treated as missing
    archive_member: Optional[str] = None
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "nscg_2023": DatasetSpec(
        name="nscg_2023",
        loader="nscg_zip",
        source_url="https://ncses.nsf.gov/822/assets/0/files/college_grads_2023.zip",
        archive_member="pcg23Public/epcg23.csv",
        target_col=["SALARY"],
        target_kind="binarize_median",
        filters={"LFSTAT": ["1"]},  # employed
        numeric_cols=["AGE", "HRSWK", "WKSWK"],
        categorical_cols=[
            "SEX_2023", "RACECAT", "HISPANIC", "MARSTA", "CTZUSIN", "BTHUS",
            "DGRDG", "MRDG", "BADG", "N2BAMED", "HDCRN21C", "BACRN21C",
            "EMSECSM", "EMSIZE", "EMTP", "N2OCPRMG",
            "JOBSATIS", "SATADV", "SATSAL", "SATSEC",
            "ACTMGT", "ACTRD", "ACTTCH", "ACTCAP",
        ],
        description="NSCG 2023 graduates: above-median annual salary among the employed.",
    ),
    "scorecard_2026": DatasetSpec(
        name="scorecard_2026",
        loader="scorecard_csv",
        source_url=(
            "https://ed-public-download.scorecard.network/downloads/"
            "Most-Recent-Cohorts-All-Data-Elements.csv"
        ),
        target_col=["MD_EARN_WNE_P10", "MD_EARN_WNE_P6", "MN_EARN_WNE_P10"],
        target_kind="binarize_median",
        numeric_cols=[
            "ADM_RATE", "SAT_AVG", "ACTCMMID", "UGDS", "COSTT4_A",
            "TUITIONFEE_IN", "TUITIONFEE_OUT", "PCTPELL", "C150_4",
            "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
        ],
        categorical_cols=["CONTROL", "REGION", "LOCALE", "PREDDEG", "HIGHDEG", "CCSIZSET", "STABBR"],
        description="College Scorecard institutions: above-median graduate earnings 10 years out.",
    ),
    "acs_co_2024": DatasetSpec(
        name="acs_co_2024",
        loader="acs_zip",
        source_url=(
            "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pco.zip"
        ),
        target_col=["WAGP"],
        target_kind="binarize_median",
        filters={"SCHL": ["21", "22", "23", "24"], "ESR": ["1", "2"]},  # bachelor's+, employed
        numeric_cols=["AGEP", "WKHP"],
        categorical_cols=["SCHL", "FOD1P", "OCCP", "SEX", "RAC1P", "MAR", "CIT", "COW", "HISP"],
        description="ACS PUMS 2024 (CO) degree holders: above-median wage income among the employed.",
    ),
    "pisa_2022": DatasetSpec(
        name="pisa_2022",
        loader="pisa_sav",
        source_url="https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip",
        target_col=["BSMJ", "PV1MATH"],
        target_kind="binarize_median",
        numeric_cols=[
            "ESCS", "HOMEPOS", "HISCED", "BMMJ1", "BFMJ2",
            "PV1MATH", "PV1READ", "PV1SCIE", "BELONG", "GROSAGR", "WORKMAST",
        ],
        categorical_cols=["ST004D01T", "IMMIG", "GRADE", "REPEAT", "CNT"],
        description="PISA 2022 students: above-median expected occupational status (BSMJ).",
    ),
}


def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    """Fetch the raw (bronze) frame for a dataset from its public source."""
    if spec.loader == "nscg_zip":
        return load_csv_from_zip(spec.source_url, spec.archive_member, dtype=str)
    if spec.loader == "scorecard_csv":
        return load_csv_url(spec.source_url, dtype=str)
    if spec.loader == "acs_zip":
        return load_csv_from_zip(spec.source_url, spec.archive_member, dtype=str)
    if spec.loader == "pisa_sav":
        # PV1MATH is both a fallback target and a numeric feature, so dedupe
        # (order-preserving) before asking the .sav reader for these columns.
        cols = spec.target_col + spec.numeric_cols + spec.categorical_cols + list(spec.filters)
        cols = list(dict.fromkeys(cols))
        return load_sav_from_zip(spec.source_url, cols, spec.archive_member)
    raise ValueError(f"Unknown loader: {spec.loader}")


def _first_present(cols: list[str], df: pd.DataFrame) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise KeyError(f"None of {cols} present in columns")


def prepare(spec: DatasetSpec, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw (bronze) frame into (features, binary target).

    Applies row filters, derives the binary label, keeps the spec's feature
    columns that are present, coerces types, and treats large numeric sentinels
    as missing.
    """
    df = df.copy()
    for col, vals in spec.filters.items():
        if col in df.columns:
            allowed = {str(v) for v in vals}
            df = df[df[col].astype(str).str.strip().isin(allowed)]

    tcol = _first_present(spec.target_col, df)
    if spec.target_kind == "binarize_median":
        v = pd.to_numeric(df[tcol], errors="coerce")
        v = v.mask(v >= spec.sentinel_min)
        y = (v > v.median()).astype("Int64")
        y[v.isna()] = pd.NA
    elif spec.target_kind == "label_equals":
        y = (df[tcol].astype(str).str.strip() == spec.positive_label).astype("Int64")
    else:
        raise ValueError(f"Unknown target_kind: {spec.target_kind}")

    # Exclude the resolved target from the features. When a target also appears
    # in a feature list (pisa's PV1MATH is a fallback target and a numeric
    # feature), keeping it would leak the label straight into X.
    num = [c for c in spec.numeric_cols if c in df.columns and c != tcol]
    cat = [c for c in spec.categorical_cols if c in df.columns and c != tcol]
    X = df[num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").mask(lambda s: s >= spec.sentinel_min)
    for c in cat:
        X[c] = X[c].astype(str)

    keep = y.notna().to_numpy()
    return X.loc[keep], y.loc[keep].astype(int)


def to_silver(spec: DatasetSpec, bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned, analysis-ready frame: feature columns plus a binary `label`."""
    X, y = prepare(spec, bronze_df)
    silver = X.copy()
    silver["label"] = y.to_numpy()
    return silver


def split_silver(silver_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a silver frame into (features, label)."""
    y = silver_df["label"].astype(int)
    X = silver_df.drop(columns=["label"])
    return X, y
