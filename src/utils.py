"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize repetitive I/O (Input/Output) logic.
- Responsibility (separation of concerns): Only basic CSV read/write and model save/load.
- Pipeline contract: Agnostic plumbing. It does not know business logic or pipeline state.

"""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Read a comma-separated file into a DataFrame

    Inputs:
    - filepath: CSV path (pathlib.Path)
    Outputs:
    - df: Loaded DataFrame
    Why this contract matters for reliable ML delivery:
    - Standardized parsing reduces fragile one-off fixes and improves reproducibility
    - A missing file raises FileNotFoundError (not a generic parsing error), so callers can tell
      "data not pulled yet" (run dvc pull) apart from "data is corrupt"
    """
    logger.info("Loading CSV from %s", filepath)

    if not isinstance(filepath, Path):
        raise TypeError(f"filepath must be a pathlib.Path, got type={type(filepath)}")

    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found at {filepath}. If it is DVC-tracked, run: dvc pull")

    if not filepath.is_file():
        raise ValueError(f"CSV Parsing Error: {filepath} exists but is not a file")

    try:
        return pd.read_csv(filepath, sep=",")
    except Exception as e:
        raise ValueError(
            f"CSV Parsing Error: Failed to read {filepath}. "
            "Check delimiter, encoding, or file corruption. "
            f"Original pandas error: {e}"
        ) from e


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: Output path (parent folders are created)
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Deterministic saving (index=False) prevents alignment bugs downstream
    """
    logger.info("Saving CSV to %s", filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model: Any, filepath: Path) -> None:
    """
    Inputs:
    - model: Trained estimator or pipeline
    - filepath: Output path (parent folders are created)
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Persisted artifacts enable reproducible inference and auditability
    """
    logger.info("Saving model to %s", filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path) -> Any:
    """
    Inputs:
    - filepath: Model artifact path
    Outputs:
    - model: Deserialized estimator
    Why this contract matters for reliable ML delivery:
    - Fail fast on missing artifacts prevents cryptic inference crashes
    """
    logger.info("Loading model from %s", filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {filepath}. "
            "Run the training pipeline first to generate models/model.joblib"
        )

    return joblib.load(filepath)
