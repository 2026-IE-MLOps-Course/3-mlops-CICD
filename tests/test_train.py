# tests/test_train.py
"""
Educational Goal:
- train_model returns ONE fitted Pipeline (recipe + model) built from config.yaml settings
- Every configuration mistake fails fast with a message that names the config key
"""

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor
from src.train import _estimator_params, build_estimator, calibrate_pipeline, train_model


@pytest.fixture
def data():
    X = pd.DataFrame({"rx_ds": [float(i) for i in range(40)], "A": [i % 2 for i in range(40)]})
    y = pd.Series([int(i > 19) for i in range(40)])
    return X, y


def recipe() -> ColumnTransformer:
    return get_feature_preprocessor(quantile_bin_cols=["rx_ds"], numeric_passthrough_cols=["A"], n_bins=3)


def test_unit_estimator_params_drop_routing_keys_and_keep_config_seed():
    cfg = {"model_type": "logistic_regression", "calibration_enabled": True, "calibration_method": "sigmoid",
           "calibration_cv": 3, "C": 0.5, "random_state": 42}
    assert _estimator_params(cfg, 7) == {"C": 0.5, "random_state": 42}
    assert _estimator_params(None, 7) == {"random_state": 7}
    assert _estimator_params(None, None) == {}
    assert "model_type" in cfg, "the config dictionary must never be mutated"


def test_unit_build_estimator_defaults_and_seed_handling():
    model = build_estimator("classification", {"solver": "liblinear"}, 3)
    assert isinstance(model, LogisticRegression) and model.random_state == 3
    assert isinstance(build_estimator("Regression", None, 3), LinearRegression)  # no random_state argument


@pytest.mark.parametrize("problem_type, params, match", [
    ("clustering", None, "Unsupported problem_type"),
    ("classification", {"model_type": "random_forest"}, "Unsupported classification model_type"),
    ("classification", {"not_a_param": 1}, "Invalid classification hyperparameters"),
])
def test_unit_build_estimator_rejects_bad_config(problem_type, params, match):
    with pytest.raises(ValueError, match=match):
        build_estimator(problem_type, params)


def test_unit_train_model_fails_fast_on_bad_inputs(data):
    X, y = data
    with pytest.raises(ValueError, match="X_train is empty"):
        train_model(X.iloc[:0], y, recipe(), "classification")
    with pytest.raises(ValueError, match="y_train is empty"):
        train_model(X, y.iloc[:0], recipe(), "classification")
    with pytest.raises(ValueError, match="do not match"):
        train_model(X, y.iloc[:5], recipe(), "classification")
    with pytest.raises(TypeError, match="ColumnTransformer"):
        train_model(X, y, "not a recipe", "classification")


def test_unit_calibrate_pipeline_rejects_bad_inputs(data):
    X, y = data
    with pytest.raises(ValueError, match="cannot be None"):
        calibrate_pipeline(None, X, y)
    with pytest.raises(ValueError, match="X_train is empty"):
        calibrate_pipeline(object(), X.iloc[:0], y)
    with pytest.raises(TypeError, match="predict_proba"):
        calibrate_pipeline(object(), X, y)
    fitted = train_model(X, y, recipe(), "classification", None, 0)
    with pytest.raises(ValueError, match="sigmoid"):
        calibrate_pipeline(fitted, X, y, method="beta")
    with pytest.raises(ValueError, match="at least 2"):
        calibrate_pipeline(fitted, X, y, cv=1)


def test_integration_train_then_calibrate_gives_probabilities(data):
    X, y = data
    params = {"model_type": "logistic_regression", "solver": "liblinear", "calibration_enabled": True}
    pipeline = train_model(X, y, recipe(), "classification", params, 42)
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]

    calibrated = calibrate_pipeline(pipeline, X, y, method="sigmoid", cv=2)
    assert isinstance(calibrated, CalibratedClassifierCV)
    proba = calibrated.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_integration_regression_pipeline_predicts(data):
    X, _ = data
    y = X["rx_ds"] * 2.0
    pipeline = train_model(X, y, recipe(), "regression", {"model_type": "linear_regression"}, 0)
    assert len(pipeline.predict(X)) == len(X)
