# tests/test_api.py
"""
Educational Goal:
- Integration tests for the FastAPI service using TestClient
- GitHub Actions runs these on every Pull Request, so a broken API can never be merged or deployed

Notes
- We MUST use `with TestClient(app) as client:` so the API lifespan (startup) runs
- To test predictions without depending on models/model.joblib, we train a tiny model in a fixture
  and place it in app.state, exactly where the lifespan would put it
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.features import get_feature_preprocessor
from src.train import train_model

FLAGS = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "R", "S", "T", "Low_inc", "SURG"]

VALID_RECORD = {
    "ID": "P001", "rx_ds": 12.0,
    "A": 1, "B": 0, "C": 1, "D": 0, "E": 1, "F": 0, "H": 0, "I": 1, "J": 0,
    "K": 0, "L": 1, "M": 0, "N": 0, "R": 1, "S": 0, "T": 1, "Low_inc": 1, "SURG": 0,
}


@pytest.fixture(scope="module")
def tiny_model():
    """A small but real pipeline trained on synthetic rows that match the API contract."""
    n = 60
    X = pd.DataFrame({"rx_ds": [float(i) for i in range(n)]})
    for j, col in enumerate(FLAGS):
        X[col] = [(i + j) % 2 for i in range(n)]
    y = pd.Series([i % 2 for i in range(n)])
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["rx_ds"], numeric_passthrough_cols=FLAGS, binary_sum_cols=FLAGS, n_bins=4)
    return train_model(X, y, preprocessor, "classification", {"solver": "liblinear", "random_state": 42})


@pytest.fixture
def client_with_model(tiny_model):
    with TestClient(app) as client:
        app.state.model_pipeline = tiny_model
        app.state.model_version = "test-model"
        yield client


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_ui_page_served():
    """The browser interface is served by the API itself and calls the real endpoints."""
    with TestClient(app) as client:
        response = client.get("/ui")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "/predict" in response.text and "/health" in response.text


def test_health_reports_503_when_model_missing():
    """Readiness, not just liveness: no model means the service must not receive traffic."""
    with TestClient(app) as client:
        app.state.model_pipeline = None
        response = client.get("/health")
        assert response.status_code == 503


def test_health_ok_when_model_loaded(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert "git_commit" in body


def test_predict_happy_path_returns_prediction_and_probability(client_with_model):
    response = client_with_model.post("/predict", json={"records": [VALID_RECORD]})
    assert response.status_code == 200
    item = response.json()["predictions"][0]
    assert item["ID"] == "P001"
    assert item["prediction"] in {0, 1}
    assert 0.0 <= item["probability"] <= 1.0
    assert "X-Correlation-ID" in response.headers


def test_predict_missing_fields_rejected_by_pydantic(client_with_model):
    response = client_with_model.post("/predict", json={"records": [{"ID": "test_patient_001"}]})
    assert response.status_code == 422


def test_predict_unknown_field_rejected(client_with_model):
    """extra='forbid': a typo in a column name must not silently reach the model."""
    response = client_with_model.post("/predict", json={"records": [{**VALID_RECORD, "rx_dss": 3}]})
    assert response.status_code == 422


def test_predict_business_rule_violation_returns_422(client_with_model):
    """validate.py rejects negative days supply, the API turns that into a client error."""
    response = client_with_model.post("/predict", json={"records": [{**VALID_RECORD, "rx_ds": -5}]})
    assert response.status_code == 422


def test_predict_returns_503_without_model():
    with TestClient(app) as client:
        app.state.model_pipeline = None
        response = client.post("/predict", json={"records": [VALID_RECORD]})
        assert response.status_code == 503
