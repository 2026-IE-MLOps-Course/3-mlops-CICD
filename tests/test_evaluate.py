# tests/test_evaluate.py
"""
Educational Goal:
- evaluate_model returns JSON-safe float metrics that the quality gate can read
- Each failure path (bad data, wrong model, one class only) raises a clear error
- check_quality_gate is the rule that stops a weak model before it is saved
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import evaluate as evaluate_module
from src.evaluate import check_quality_gate, evaluate_calibration, evaluate_model
from src.features import get_feature_preprocessor
from src.train import train_model


class FakeModel:
    """A stand-in model whose outputs we control, to reach the defensive branches."""

    def __init__(self, proba=None):
        self.proba = proba

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return self.proba


class PredictOnly:
    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def xy():
    X = pd.DataFrame({"x": [float(i) for i in range(20)]})
    y = pd.Series([int(i >= 10) for i in range(20)])
    return X, y


def test_unit_calibration_table_and_ece():
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
    table, ece = evaluate_calibration(y, prob, n_bins=4)
    assert list(table.columns) == ["bin", "mean_prob", "observed_rate", "count"]
    assert table["count"].sum() == len(y)
    assert 0.0 <= ece <= 1.0


def test_unit_calibration_rejects_bad_inputs():
    with pytest.raises(ValueError, match="empty"):
        evaluate_calibration(pd.Series([], dtype=float), np.array([]))
    with pytest.raises(ValueError, match="does not match"):
        evaluate_calibration(pd.Series([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_calibration(pd.Series([0, 1]), np.array([0.5, 1.5]))


def test_unit_calibration_undefined_cases_return_nan(monkeypatch):
    table, ece = evaluate_calibration(pd.Series([0, 1, 0, 1]), np.array([0.5] * 4), n_bins=4)
    assert table.empty and math.isnan(ece)

    def _boom(*args, **kwargs):
        raise ValueError("cannot bin")

    monkeypatch.setattr(evaluate_module.pd, "qcut", _boom)
    table, ece = evaluate_calibration(pd.Series([0, 1]), np.array([0.2, 0.8]))
    assert table.empty and math.isnan(ece)


def test_unit_evaluate_rejects_bad_data_and_models(xy):
    X, y = xy
    with pytest.raises(ValueError, match="X_eval is empty"):
        evaluate_model(PredictOnly(), X.iloc[:0], y, "classification")
    with pytest.raises(ValueError, match="y_eval is empty"):
        evaluate_model(PredictOnly(), X, y.iloc[:0], "classification")
    with pytest.raises(ValueError, match="do not match"):
        evaluate_model(PredictOnly(), X.iloc[:10], y.iloc[:5], "classification")
    with pytest.raises(TypeError, match="predict"):
        evaluate_model(object(), X, y, "classification")
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        evaluate_model(PredictOnly(), X, y, "clustering")


def test_unit_evaluate_classification_defensive_branches(xy):
    X, y = xy
    with pytest.raises(ValueError, match="only one class"):
        evaluate_model(PredictOnly(), X, pd.Series([1] * 20), "classification")
    with pytest.raises(TypeError, match="predict_proba"):
        evaluate_model(PredictOnly(), X, y, "classification")
    with pytest.raises(ValueError, match="invalid shape"):
        evaluate_model(FakeModel(np.zeros(20)), X, y, "classification")
    with pytest.raises(ValueError, match="only one probability column"):
        evaluate_model(FakeModel(np.zeros((20, 1))), X, y, "classification")


def test_unit_regression_metrics_return_rmse(xy):
    X, _ = xy
    metrics = evaluate_model(PredictOnly(), X, pd.Series([2.0] * 20), "regression")
    assert metrics == {"rmse": pytest.approx(2.0)}


def test_unit_quality_gate_rules():
    check_quality_gate({"pr_auc": 0.6}, metric="pr_auc", min_value=0.4)
    check_quality_gate({"rmse": 1.0}, metric="rmse", max_value=2.0)
    with pytest.raises(ValueError, match="below the minimum"):
        check_quality_gate({"pr_auc": 0.3}, metric="pr_auc", min_value=0.4)
    with pytest.raises(ValueError, match="above the maximum"):
        check_quality_gate({"rmse": 3.0}, metric="rmse", max_value=2.0)
    with pytest.raises(ValueError, match="not found"):
        check_quality_gate({"roc_auc": 0.9}, metric="pr_auc", min_value=0.4)


def test_integration_trained_pipeline_metrics_are_finite_floats():
    X = pd.DataFrame({"rx_ds": [float(i % 30) for i in range(90)], "A": [i % 2 for i in range(90)]})
    y = pd.Series([int((i % 30) > 15) for i in range(90)])
    pre = get_feature_preprocessor(quantile_bin_cols=["rx_ds"], numeric_passthrough_cols=["A"], n_bins=3)
    model = train_model(X, y, pre, "classification", {"solver": "liblinear"}, 0)
    metrics = evaluate_model(model, X, y, "classification")
    assert set(metrics) == {"pr_auc", "roc_auc", "ece"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    check_quality_gate(metrics, metric="pr_auc", min_value=0.4)
