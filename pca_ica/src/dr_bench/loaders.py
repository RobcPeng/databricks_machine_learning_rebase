"""Fetch public microdata into pandas DataFrames.

Each loader downloads a public archive and returns a DataFrame. Downloads use
the standard library only; PISA additionally requires `pyreadstat` to read SPSS
`.sav` files.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from typing import Optional

import pandas as pd

_USER_AGENT = "dr-bench/1.0 (+https://databricks.com)"


def _download(url: str, suffix: str) -> str:
    """Download a URL to a temp file and return its local path."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        out.write(resp.read())
    return path


def load_csv_url(url: str, dtype: Optional[object] = None) -> pd.DataFrame:
    """Download a CSV from a URL and read it into a DataFrame."""
    path = _download(url, ".csv")
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=False)
    finally:
        os.remove(path)


def load_csv_from_zip(
    url: str, member: Optional[str] = None, dtype: Optional[object] = None
) -> pd.DataFrame:
    """Download a zip archive and read one CSV member into a DataFrame.

    When `member` is None, the first `.csv` entry in the archive is used.
    """
    zip_path = _download(url, ".zip")
    try:
        with zipfile.ZipFile(zip_path) as z:
            name = member
            if name is None:
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise FileNotFoundError(f"No .csv file found in {url}")
                name = csvs[0]
            with z.open(name) as f:
                return pd.read_csv(f, dtype=dtype, low_memory=False)
    finally:
        os.remove(zip_path)


def load_sav_from_zip(
    url: str, columns: list[str], member: Optional[str] = None
) -> pd.DataFrame:
    """Download a zip archive containing an SPSS `.sav` file and read a column subset.

    Reads metadata first and keeps only requested columns that exist in the file,
    so a partial or renamed variable list still loads instead of raising.
    """
    import pyreadstat

    # The extracted .sav can be hundreds of MB, so always clean it up (the
    # original code leaked one temp dir per call). The outer finally covers the
    # dir even if the download fails; the inner one covers the downloaded zip.
    extract_dir = tempfile.mkdtemp()
    try:
        zip_path = _download(url, ".zip")
        try:
            with zipfile.ZipFile(zip_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".sav")]
                if not names:
                    raise FileNotFoundError(f"No .sav file found in {url}")
                name = member or names[0]
                z.extract(name, extract_dir)
                sav_path = os.path.join(extract_dir, name)

            _, meta = pyreadstat.read_sav(sav_path, metadataonly=True)
            available = set(meta.column_names)
            use = [c for c in columns if c in available]
            missing = [c for c in columns if c not in available]
            if missing:
                print(f"load_sav_from_zip: {len(missing)} requested columns absent, skipped: {missing}")
            df, _ = pyreadstat.read_sav(sav_path, usecols=use)
            return df
        finally:
            os.remove(zip_path)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
