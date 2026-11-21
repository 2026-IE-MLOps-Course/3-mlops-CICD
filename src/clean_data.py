# src/clean_data.py
"""
Educational Goal:
- Why this module exists in an MLOps system: To isolate dataset-specific formatting and quality filtering from
  model training, ensuring upstream data anomalies don't cause downstream pipeline crashes.
- Responsibility (separation of concerns): Pure data transformations (standardizing column names,
  deduplication, dropping missing values) and data observability (logging dropped rows). Strictly NO file I/O
  and NO model fitting.
- Pipeline contract (inputs and outputs): Inputs are the raw Pandas DataFrame and the target column name.
  Output is a cleaned Pandas DataFrame that is guaranteed to have the target column, ready for validation and
  splitting.

"""

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def _standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace and replace inner spaces with "_" (in place on a copy owned by the caller)."""
    df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    return df


def _resolve_target(df: pd.DataFrame, target_column: str) -> str:
    """
    Inputs
    - df: data with standardized headers
    - target_column: target name from config.yaml

    Outputs
    - The target column name as it appears in df (forgiving to case drift, e.g. "od" vs "OD")
    """
    target_std = (target_column or "").strip().replace(" ", "_")
    if not target_std:
        raise ValueError("target_column is empty after standardization")
    if target_std in df.columns:
        return target_std

    cols_lower = {c.lower(): c for c in df.columns}
    if target_std.lower() in cols_lower:
        return cols_lower[target_std.lower()]
    raise ValueError(
        f"Fatal: target column '{target_column}' missing after cleaning. "
        "Check problem.target_column in config.yaml and your raw CSV headers"
    )


def clean_dataframe(df_raw: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    One cleaner for both training and inference

    Inputs
    - df_raw: raw DataFrame (never modified)
    - target_column: target name in training mode, None in inference mode

    Outputs
    - df_clean: a new DataFrame with standardized headers and a fresh 0..n-1 index

    Modes
    - Training (target_column set): standardize headers, drop exact duplicates, drop rows with missing target
    - Inference (target_column None): standardize headers only; never drop rows, every row gets a prediction

    Why: where each preparation step belongs
    - Stateless rules for every row (headers, formats): here, shared by main.py and api.py
    - Row filters (duplicates, missing target): here, training mode only
    - Anything that learns from data (imputation, binning, scaling, encoding): src/features.py,
      inside the fitted pipeline saved in model.joblib
    """
    logger.info("Cleaning dataframe")

    if df_raw is None:
        raise ValueError("df_raw is None. Check src/load_data.py and paths.raw_data in config.yaml")
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"df_raw must be a pandas DataFrame, got type={type(df_raw)}")

    df_clean = _standardize_headers(df_raw.copy())
    initial_rows = len(df_clean)

    if target_column is not None:
        # Row filters are training only: dropping rows at inference would silently lose predictions
        # Exact duplicates across all columns (including ID): a unique ID means no row is dropped
        df_clean = df_clean.drop_duplicates()
        # Supervised learning requires a target label for every training row
        df_clean = df_clean.dropna(subset=[_resolve_target(df_clean, target_column)])

    df_clean = df_clean.reset_index(drop=True)

    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.info("Dropped %s rows", dropped_rows)

    logger.info("Rows after cleaning: %s", len(df_clean))
    return df_clean
