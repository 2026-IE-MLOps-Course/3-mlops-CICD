# tests/test_infer.py
"""
Educational Goal:
- run_inference keeps the input index, never modifies its input and returns a fixed schema:
  prediction, plus probability when include_proba=True
"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.train import train_model


class FakeModel:
    """A stand-in model whose probability output we control, to reach the defensive branches."""

    def __init__(self, proba=None):
        self.proba = proba

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return self.proba


class NoProba:
    def predict(self, X):
        return np.ones(len(X))


@pytest.fixture
def X():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])


def test_unit_prediction_only_keeps_index(X):
    out = run_inference(FakeModel(), X)
    assert list(out.columns) == ["prediction"]
    assert list(out.index) == [10, 20, 30]


def test_unit_probability_column_when_requested(X):
    out = run_inference(FakeModel(np.array([[0.2, 0.8]] * 3)), X, True)
    assert out["probability"].tolist() == [0.8, 0.8, 0.8]


def test_unit_rejects_bad_inputs(X):
    with pytest.raises(ValueError, match="empty"):
        run_inference(FakeModel(), X.iloc[:0])
    with pytest.raises(TypeError, match="DataFrame"):
        run_inference(FakeModel(), X.to_numpy())
    with pytest.raises(TypeError, match="predict"):
        run_inference(object(), X)


def test_unit_rejects_bad_probability_outputs(X):
    with pytest.raises(TypeError, match="predict_proba"):
        run_inference(NoProba(), X, True)
    with pytest.raises(ValueError, match="invalid shape"):
        run_inference(FakeModel([[0.2, 0.8]] * 3), X, True)  # a list, not an ndarray
    with pytest.raises(ValueError, match="at least 2 columns"):
        run_inference(FakeModel(np.ones((3, 1))), X, True)


def test_integration_pipeline_scores_cleaned_rows_without_touching_input(make_patients, flags):
    train = make_patients(80)
    pre = get_feature_preprocessor(quantile_bin_cols=["rx_ds"], numeric_passthrough_cols=flags,
                                   binary_sum_cols=flags, n_bins=4)
    model = train_model(train.drop(columns=["ID", "OD"]), train["OD"], pre, "classification", None, 0)

    new = clean_dataframe(make_patients(6, with_target=False, start_id=500), target_column=None)
    X_new = new.drop(columns=["ID"])
    X_new.index = [5, 8, 11, 14, 17, 20]
    before = X_new.copy()

    out = run_inference(model, X_new, include_proba=True)
    assert list(out.index) == list(X_new.index)
    assert X_new.equals(before)
    assert set(out["prediction"].unique()) <= {0, 1}
