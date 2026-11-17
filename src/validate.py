# src/validate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Fail fast on obvious data issues before expensive steps run
- Responsibility (separation of concerns): Only validation checks, no cleaning, features, training, or metrics logic
- Pipeline contract (inputs and outputs): Input is a DataFrame and constraint lists, output is True or a
  raised ValueError (TypeError for a wrong object type)

How it is organised
- validate_dataframe() is the only public function; it calls one small named check per rule
- Validation never fixes data: it only accepts or rejects it
"""

import logging
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _check_frame(df: pd.DataFrame) -> None:
    """Reject None, non-DataFrame and empty inputs (the most common upstream mistakes)."""
    if df is None:
        raise ValueError("Validation failed: df is None. Check src/load_data.py and paths.raw_data in config.yaml")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Validation failed: df must be a pandas DataFrame, got type={type(df)}")
    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty after cleaning")


def _standard_columns(required_columns: List[str]) -> List[str]:
    """
    Inputs
    - required_columns: column names from config.yaml

    Outputs
    - The same names, stripped, spaces replaced by "_" (matches clean_data.py) and de-duplicated in order
    """
    if not required_columns:
        raise ValueError("Validation failed: required_columns is empty")
    names = [str(c).strip().replace(" ", "_") for c in required_columns]
    return list(dict.fromkeys(names))


def _check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Every required column must exist, so downstream steps never fail with a KeyError."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: Missing required columns: {missing}")


def _check_no_missing(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Reject missing feature values (only used when the feature recipe has no imputers)."""
    cols_with_missing = [col for col in required_columns if df[col].isna().any()]
    if cols_with_missing:
        raise ValueError(
            f"Validation failed: Missing values found in required columns: {cols_with_missing}. "
            "Fix in clean_data.py or disable check_missing_values if you use imputers"
        )


def _check_target(df: pd.DataFrame, target_column: str, allowed_values: Optional[List[Any]]) -> None:
    """
    Target policy: always strict once a target is declared

    Inputs
    - df: data with the target column
    - target_column: target name
    - allowed_values: allowed labels (classification) or None (any value)
    """
    if target_column not in df.columns:
        raise ValueError(f"Validation failed: target_column '{target_column}' not found in DataFrame")
    if df[target_column].isna().any():
        raise ValueError(f"Validation failed: target column '{target_column}' contains missing values")
    if allowed_values is None:
        return
    actual_values = set(df[target_column].dropna().unique().tolist())
    if not actual_values.issubset(set(allowed_values)):
        raise ValueError(
            f"Validation failed: Target '{target_column}' has invalid values {sorted(actual_values)}. "
            f"Expected subset of {sorted(set(allowed_values))}"
        )


def _check_non_negative(df: pd.DataFrame, columns: List[str]) -> None:
    """Business rule: quantities such as days of supply can never be negative."""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Validation failed: numeric_non_negative_cols includes missing column '{col}'")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Validation failed: Column '{col}' is in numeric_non_negative_cols but is not numeric")
        if (df[col] < 0).any():
            raise ValueError(f"Validation failed: Column '{col}' contains negative values")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    target_column: Optional[str] = None,
    check_missing_values: bool = True,
    target_allowed_values: Optional[List[Any]] = None,
    numeric_non_negative_cols: Optional[List[str]] = None,
) -> bool:
    """
    Inputs
    - df: DataFrame to validate
    - required_columns: Columns that must exist for downstream steps
    - target_column: Optional target column name for strict target checks (None at inference)
    - check_missing_values: If True, fail when missing values exist in required columns (feature columns)
    - target_allowed_values: Optional list of allowed target values (classification)
    - numeric_non_negative_cols: Optional list of numeric columns that must be >= 0

    Outputs
    - is_valid: True if valid, otherwise raises ValueError (TypeError for wrong object or column types)

    Why this contract matters for reliable ML delivery
    - Validation catches schema drift and bad data early, before training wastes time or produces silent failures
    - Target integrity is non negotiable for supervised learning, missing target must always fail fast
    """
    logger.info("Validating dataframe")

    _check_frame(df)
    columns = _standard_columns(required_columns)
    _check_required_columns(df, columns)

    # False allows missing feature values because imputers in features.py handle them
    if check_missing_values:
        _check_no_missing(df, columns)

    if target_column is not None:
        _check_target(df, target_column, target_allowed_values)

    _check_non_negative(df, numeric_non_negative_cols or [])
    return True
