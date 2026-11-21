# tests/test_api.py
"""
Educational Goal:
- Integration tests for the FastAPI service using TestClient
- GitHub Actions runs these on every Pull Request, so a broken API can never be merged or deployed

Notes
- We MUST use `with TestClient(app) as client:` so the API lifespan (startup) runs
- To test predictions without depending on the real model file, we train a tiny model in a fixture
  and place it in app.state, exactly where the lifespan would put it
- The registry and the tracker are replaced by small fakes: no network, no account
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import PatientRecord, app
from src.clean_data import clean_dataframe
from src.config import load_config
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.train import train_model

REPO_CONFIG = Path(api_module.PROJECT_ROOT) / "config.yaml"

VALID_RECORD = {
    "ID": "P001", "rx_ds": 12.0,
    "A": 1, "B": 0, "C": 1, "D": 0, "E": 1, "F": 0, "H": 0, "I": 1, "J": 0,
    "K": 0, "L": 1, "M": 0, "N": 0, "R": 1, "S": 0, "T": 1, "Low_inc": 1, "SURG": 0,
}


@pytest.fixture(scope="module")
def tiny_model():
    """A small but real pipeline trained on synthetic rows that match the API contract."""
    flags = [k for k in VALID_RECORD if k not in ("ID", "rx_ds")]
    n = 60
    X = pd.DataFrame({"rx_ds": [float(i) for i in range(n)]})
    for j, col in enumerate(flags):
        X[col] = [(i + j) % 2 for i in range(n)]
    y = pd.Series([i % 2 for i in range(n)])
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["rx_ds"], numeric_passthrough_cols=flags, binary_sum_cols=flags, n_bins=4)
    return train_model(X, y, preprocessor, "classification", {"solver": "liblinear"}, 42)


@pytest.fixture
def client_with_model(tiny_model):
    with TestClient(app) as client:
        app.state.model_pipeline = tiny_model
        app.state.model_version = "opioid_classification_model:v7"
        yield client


@pytest.fixture
def cfg_with_model(tmp_path, tiny_model):
    """The real config.yaml, with the model path pointing at a tiny model in tmp_path."""
    cfg = load_config(REPO_CONFIG)
    model_path = tmp_path / "model.joblib"
    joblib.dump(tiny_model, model_path)
    cfg["paths"]["model_artifact"] = str(model_path)
    return cfg


# --------------------------------------------------------
# Unit tests: contract, version reporting, loaders, logging buffer
# --------------------------------------------------------
def test_unit_patient_record_exposes_dataset_column_I():
    record = PatientRecord(**VALID_RECORD)
    assert record.I_ == 1
    assert record.model_dump(by_alias=True)["I"] == 1
    assert "I" in PatientRecord.model_json_schema()["properties"]


def test_unit_git_commit_prefers_image_then_render_then_local(monkeypatch):
    assert api_module.git_commit() == "local"
    monkeypatch.setenv("RENDER_GIT_COMMIT", "render-sha")
    assert api_module.git_commit() == "render-sha"
    monkeypatch.setenv("GIT_COMMIT", "image-sha")
    assert api_module.git_commit() == "image-sha"


def test_unit_local_loader_reports_baked_model_version(cfg_with_model, monkeypatch):
    model, version = api_module._load_model(cfg_with_model)
    assert hasattr(model, "predict") and version == "local"
    monkeypatch.setenv("MODEL_VERSION", "opioid_classification_model:v7")
    assert api_module._load_model(cfg_with_model)[1] == "opioid_classification_model:v7"


def test_unit_local_loader_fails_without_model_file(cfg_with_model, tmp_path):
    cfg_with_model["paths"]["model_artifact"] = str(tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="python -m src.main"):
        api_module._load_model(cfg_with_model)


def test_unit_registry_loader_needs_entity(cfg_with_model, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("MODEL_SOURCE", "wandb")
    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        api_module._load_model(cfg_with_model)


def test_unit_registry_loader_returns_model_and_concrete_version(cfg_with_model, monkeypatch):
    calls = []

    def fake_download(cfg, alias, output, api=None):
        calls.append(alias)
        joblib.dump("model-object", output)
        return "opioid_classification_model:v7"

    monkeypatch.setenv("MODEL_SOURCE", "wandb")
    monkeypatch.setattr(api_module.registry, "download_by_alias", fake_download)

    model, version = api_module._load_model(cfg_with_model)
    assert calls == [None]  # None: src/registry.py reads WANDB_MODEL_ALIAS (default prod)
    assert (model, version) == ("model-object", "opioid_classification_model:v7")


def test_unit_buffer_flushes_at_batch_size_inline_or_in_background(monkeypatch):
    sent = []
    monkeypatch.setattr(api_module.tracking, "log_prediction_batch", lambda cfg, batch: sent.append(len(batch)))
    cfg = {"api": {"log_batch_size": 2}, "wandb": {"project": "p"}}
    api_module.LOG_BUFFER[:] = [{"a": 1}]
    api_module._flush_buffer(cfg)
    assert sent == [] and len(api_module.LOG_BUFFER) == 1  # below the batch size

    api_module._flush_buffer(cfg, force=True)
    assert sent == [1] and api_module.LOG_BUFFER == []

    api_module.LOG_BUFFER[:] = [{"a": 1}, {"a": 2}]
    tasks = BackgroundTasks()
    api_module._flush_buffer(cfg, background_tasks=tasks)
    assert len(tasks.tasks) == 1 and api_module.LOG_BUFFER == []


# --------------------------------------------------------
# Integration tests: real HTTP calls through TestClient
# --------------------------------------------------------
def test_integration_root_points_to_docs_and_ui_is_gone():
    with TestClient(app) as client:
        assert "/docs" in client.get("/").json()["message"]
        assert client.get("/ui").status_code == 404
        assert client.get("/docs").status_code == 200


def test_integration_startup_failure_reports_503(monkeypatch):
    def _broken(cfg):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(api_module, "_load_model", _broken)
    with TestClient(app) as client:
        assert app.state.model_version == "startup_error"
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"records": [VALID_RECORD]}).status_code == 503


def test_integration_startup_loads_model_and_health_reports_version_and_commit(monkeypatch, tiny_model):
    monkeypatch.setattr(api_module, "_load_model", lambda cfg: (tiny_model, "opioid_classification_model:v7"))
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True,
                    "model_version": "opioid_classification_model:v7", "git_commit": "abc1234"}


def test_integration_predict_returns_prediction_probability_and_version(client_with_model):
    response = client_with_model.post("/predict", json={"records": [VALID_RECORD]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "opioid_classification_model:v7"
    item = body["predictions"][0]
    assert item["ID"] == "P001" and item["prediction"] in {0, 1}
    assert 0.0 <= item["probability"] <= 1.0
    assert "X-Correlation-ID" in response.headers


@pytest.mark.parametrize("payload", [
    {"records": [{"ID": "P001"}]},                                   # missing fields
    {"records": [{**VALID_RECORD, "rx_dss": 3}]},                    # extra (typo) field
    {"records": [{**VALID_RECORD, "rx_ds": "twelve"}]},              # wrong type
    {"records": [{**VALID_RECORD, "rx_ds": -5}]},                    # business rule (validate.py)
    {},                                                              # empty body
])
def test_integration_predict_rejects_bad_requests_with_422(client_with_model, payload):
    assert client_with_model.post("/predict", json=payload).status_code == 422


def test_integration_unexpected_model_error_returns_500(client_with_model):
    class Broken:
        def predict(self, X):
            raise RuntimeError("corrupt artifact")

    app.state.model_pipeline = Broken()
    assert client_with_model.post("/predict", json={"records": [VALID_RECORD]}).status_code == 500


def test_integration_api_matches_batch_path_and_keeps_duplicates(client_with_model, tiny_model):
    records = [VALID_RECORD, VALID_RECORD]
    response = client_with_model.post("/predict", json={"records": records})
    api_probas = [p["probability"] for p in response.json()["predictions"]]

    df_batch = clean_dataframe(pd.DataFrame(records), target_column=None)
    batch = run_inference(tiny_model, df_batch.drop(columns=["ID"]), include_proba=True)
    assert api_probas == pytest.approx(batch["probability"].tolist())
