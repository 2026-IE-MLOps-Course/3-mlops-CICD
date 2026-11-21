"""
Educational goal
- Encapsulate model training so orchestration (main) stays clean
- Make model choice and hyperparameters configurable via config.yaml
- Return a single fitted Pipeline artifact that can be saved and reused

Design choices for simplicity
- Prefer functions over classes
- Use a small model switch based on model_type
- Keep scikit-learn Pipeline as the core artifact
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# problem_type -> (default model_type, {model_type: estimator class})
SUPPORTED_MODELS: Dict[str, tuple] = {
    "classification": ("logistic_regression", {"logistic_regression": LogisticRegression}),
    "regression": ("linear_regression", {"linear_regression": LinearRegression}),
}

# Keys in training.<problem_type> that steer orchestration, not the estimator constructor
CALIBRATION_KEYS = ("calibration_enabled", "calibration_method", "calibration_cv")


def _check_training_data(X_train: pd.DataFrame, y_train: pd.Series, action: str) -> None:
    """
    Fail fast on empty or misaligned training data

    Inputs
    - X_train, y_train: features and target
    - action: verb used in the error message ("train", "calibrate")

    Outputs
    - None. Raises ValueError with an actionable message
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError(f"X_train is empty. Cannot {action} a model")
    if y_train is None or len(y_train) == 0:
        raise ValueError(f"y_train is empty. Cannot {action} a model")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train rows ({len(X_train)}) do not match y_train rows ({len(y_train)})")


def _estimator_params(model_params: Optional[Dict[str, Any]], random_state: Optional[int]) -> Dict[str, Any]:
    """
    Turn the config section into constructor arguments

    Inputs
    - model_params: training.<problem_type> from config.yaml (never mutated)
    - random_state: seed used when the config does not set one

    Outputs
    - params: a new dictionary without routing keys (model_type, calibration_*)
    """
    params: Dict[str, Any] = dict(model_params) if isinstance(model_params, dict) else {}
    params.pop("model_type", None)
    for key in CALIBRATION_KEYS:
        params.pop(key, None)
    if random_state is not None:
        params.setdefault("random_state", random_state)
    return params


def build_estimator(problem_type: str, model_params: Optional[Dict[str, Any]] = None,
                    random_state: Optional[int] = None) -> BaseEstimator:
    """
    Create the unfitted estimator named in config.yaml

    Inputs
    - problem_type: "classification" or "regression"
    - model_params: training.<problem_type> from config.yaml, including model_type
    - random_state: seed for estimators that accept one

    Outputs
    - An unfitted scikit-learn estimator

    Why it matters
    - Unknown problem types, model types or hyperparameters fail with a message that names the config key
    """
    pt = (problem_type or "").strip().lower()
    if pt not in SUPPORTED_MODELS:
        raise ValueError("Unsupported problem_type. Use 'classification' or 'regression'")

    default_type, registry = SUPPORTED_MODELS[pt]
    model_type = str((model_params or {}).get("model_type") or default_type).strip().lower()
    if model_type not in registry:
        raise ValueError(f"Unsupported {pt} model_type '{model_type}'. Supported: {sorted(registry)}")

    estimator_cls = registry[model_type]
    params = _estimator_params(model_params, random_state)
    if "random_state" not in estimator_cls().get_params():
        params.pop("random_state", None)

    try:
        return estimator_cls(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid {pt} hyperparameters in config.yaml under training.{pt}. "
            f"Constructor error: {e}. Fix the keys or remove the unsupported parameter."
        ) from e


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> Pipeline:
    """
    Train a scikit-learn Pipeline (feature recipe + model) on the training split only

    Inputs
    - X_train: Features for training (target removed)
    - y_train: Target values
    - preprocessor: Unfitted ColumnTransformer describing the feature recipe
    - problem_type: "classification" or "regression"
    - model_params: dictionary from config.yaml training.<problem_type>
    - random_state: seed used when model_params does not set one (estimators that accept it)

    Output
    - A fitted scikit-learn Pipeline with steps "preprocess" and "model"
    """
    logger.info("Training model pipeline for problem_type=%s", problem_type)

    _check_training_data(X_train, y_train, "train")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(f"preprocessor must be a ColumnTransformer. Got type={type(preprocessor)}")

    model = build_estimator(problem_type, model_params, random_state)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    logger.info("Base model training completed")
    return pipeline


def calibrate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """
    Calibrate a fitted classification pipeline so predicted probabilities are more reliable.

    Inputs
    - pipeline: A fitted scikit-learn Pipeline with predict_proba support
    - X_train: Training features used for calibration fitting
    - y_train: Training target used for calibration fitting
    - method: Calibration method supported by scikit-learn ("sigmoid" or "isotonic")
    - cv: Number of folds used by CalibratedClassifierCV

    Output
    - A fitted CalibratedClassifierCV object that behaves like a prediction model

    Why this matters in production
    - A classifier can rank cases reasonably well but still output poorly calibrated probabilities
    - Calibration is useful when probabilities drive business thresholds, triage, or intervention rules
    """
    logger.info("Calibrating model probabilities | method=%s | cv=%s", method, cv)

    if pipeline is None:
        raise ValueError("pipeline cannot be None for calibration")
    _check_training_data(X_train, y_train, "calibrate")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("pipeline must implement predict_proba() for calibration")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'")
    if int(cv) < 2:
        raise ValueError("Calibration cv must be at least 2")

    calibrated_model = CalibratedClassifierCV(estimator=pipeline, method=method, cv=int(cv))
    calibrated_model.fit(X_train, y_train)

    logger.info("Model calibration completed")
    return calibrated_model
