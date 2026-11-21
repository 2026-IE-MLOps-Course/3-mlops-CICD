# src/infer.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Separate inference so it can be reused for batch or real-time prediction.
- Responsibility (separation of concerns): Only runs model.predict and formats outputs. No training or metrics.
- Pipeline contract: Inputs are a fitted model and a feature DataFrame. Output is a strictly formatted
  predictions DataFrame.


"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_inputs(model: Any, X_infer: pd.DataFrame) -> None:
    """
    Fail fast before predicting

    Inputs
    - model: the loaded artifact
    - X_infer: features for the new rows

    Outputs
    - None. Raises ValueError (empty data) or TypeError (wrong object types)
    """
    if X_infer is None or len(X_infer) == 0:
        raise ValueError("Fatal: X_infer is empty. Cannot run inference.")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(
            f"Fatal: X_infer must be a pandas DataFrame to retain feature names. Got type={type(X_infer)}")

    if not hasattr(model, "predict"):
        raise TypeError(f"Fatal: model must implement predict(), got type={type(model)}")


def _positive_probability(model: Any, X_infer: pd.DataFrame) -> np.ndarray:
    """
    Inputs
    - model: fitted classifier
    - X_infer: features for the new rows

    Outputs
    - Probability of the positive class (OD=1), one float per row

    Why it matters
    - Defensive shape checks stop a wrong artifact from producing silently shifted scores
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Fatal: include_proba=True but the loaded model does not support predict_proba().")

    proba = model.predict_proba(X_infer)

    if not isinstance(proba, np.ndarray) or proba.ndim != 2 or proba.shape[0] != len(X_infer):
        raise ValueError(f"Fatal: predict_proba returned an invalid shape. Got {getattr(proba, 'shape', None)}")

    if proba.shape[1] < 2:
        raise ValueError(
            f"Fatal: predict_proba must return at least 2 columns for binary classification. Got {proba.shape}")

    return proba[:, 1].astype(float)


def run_inference(model: Any, X_infer: pd.DataFrame, include_proba: bool = False) -> pd.DataFrame:
    """
    Executes model predictions on new, unseen data.

    Inputs:
    - model: Fitted scikit-learn compatible artifact with a .predict() method
    - X_infer: Pandas DataFrame of inference features (never modified)
    - include_proba: If True, adds the probability of the positive class

    Outputs:
    - df_pred: DataFrame with 'prediction' and optionally 'probability', on the same index as X_infer

    Why this contract matters for reliable ML delivery:
    - A consistent schema reduces downstream integration risk for APIs or batch jobs.
    - Preserving the index allows predictions to be joined back to the original database records.
    - Defensive shape checking on probabilities prevents silent downstream crashes.
    """
    logger.info("Running inference")
    _check_inputs(model, X_infer)

    # Preserve the index for downstream joins
    df_pred = pd.DataFrame(index=X_infer.index)
    df_pred["prediction"] = model.predict(X_infer)

    if include_proba:
        df_pred["probability"] = _positive_probability(model, X_infer)

    return df_pred
