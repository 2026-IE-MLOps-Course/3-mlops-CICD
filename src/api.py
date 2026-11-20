# src/api.py
"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI
- Use Pydantic to enforce a strict data contract (types and required fields)
- Reuse existing pipeline modules to prevent training-serving skew
- Add lightweight observability: request logs (Layer 1) and batched prediction logs to the tracker (Layer 2)
- The tracker's library is never imported here: registry reads live in src/registry.py, writes in src/tracking.py

Key principle
- NO NEW ML LOGIC IN THIS FILE
- This file only does HTTP, schema validation, startup loading, observability and routing

Run it locally
- uvicorn src.api:app --reload
- Open http://127.0.0.1:8000/docs
"""

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from src import registry, tracking
from src.clean_data import clean_dataframe
from src.config import get_feature_columns, load_config, require_section, require_str, resolve_repo_path
from src.infer import run_inference
from src.logger import configure_logging
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Secrets and deployment switches come from environment variables (.env locally, Render settings in the cloud)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


# -------------------------------------------------------------------
# 1) Pydantic schemas (the strict API contract)
# -------------------------------------------------------------------
class PatientRecord(BaseModel):
    """
    The API contract for one patient
    - Missing fields or wrong types are rejected with HTTP 422 before any ML code runs
    - extra="forbid" rejects unknown fields, so a typo cannot silently change model inputs
    - The dataset column "I" is the field I_ with alias "I" (a bare "I" is ambiguous in code, flake8 E741);
      clients still send and see "I"
    """
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "ID": "P001",
                "rx_ds": 12.0,
                "A": 1, "B": 0, "C": 1, "D": 0, "E": 1, "F": 0,
                "H": 0, "I": 1, "J": 0, "K": 0, "L": 1, "M": 0,
                "N": 0, "R": 1, "S": 0, "T": 1, "Low_inc": 1, "SURG": 0,
            }
        },
    )

    ID: str
    rx_ds: float
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int
    H: int
    I_: int = Field(alias="I")
    J: int
    K: int
    L: int
    M: int
    N: int
    R: int
    S: int
    T: int
    Low_inc: int
    SURG: int


class PredictRequest(BaseModel):
    """A batch of patient records; one prediction comes back per record, in the same order."""
    model_config = ConfigDict(extra="forbid")
    records: List[PatientRecord]


class PredictionItem(BaseModel):
    """One scored record: the caller's ID, the 0/1 decision and the event probability."""
    ID: str
    prediction: int
    probability: Optional[float] = None


class PredictResponse(BaseModel):
    """Predictions plus the concrete model version that produced them (for audit and rollback)."""
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    """What /health reports: is a model loaded, which model version, built from which Git commit."""
    status: str
    model_loaded: bool
    model_version: str
    git_commit: str


# -------------------------------------------------------------------
# 2) Model loading (local file or the model registry)
# -------------------------------------------------------------------
def _load_from_registry(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Download the registry version that carries the alias (local development only)

    Inputs
    - cfg: config.yaml (registry names and paths.model_artifact)

    Outputs
    - (model, model_version), where model_version is concrete, e.g. "opioid_classification_model:v7"

    Why it matters
    - The alias "prod" moves; logging the resolved version says exactly which model answered
    - src/registry.py resolves the alias exactly as the release workflow does
    """
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.joblib"
        version = registry.download_by_alias(cfg, None, model_path)
        logger.info("Loaded model from the registry: %s", version)
        return joblib.load(model_path), version


def _load_model(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Load the model once at startup

    Inputs
    - cfg: config.yaml as a dictionary
    - Environment: MODEL_SOURCE=local (default, the release image) or registry.SOURCE_NAME, here
      "wandb" (local development)

    Outputs
    - (model, model_version)
      local: models/model.joblib, version from MODEL_VERSION (baked into the image at release time)
      registry: the version with alias WANDB_MODEL_ALIAS (default "prod"), resolved by src/registry.py
    """
    model_source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    if model_source == registry.SOURCE_NAME:
        return _load_from_registry(cfg)

    model_path = resolve_repo_path(PROJECT_ROOT, require_str(require_section(cfg, "paths"), "model_artifact"))
    logger.info("Loading local model artifact: %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing at {model_path}. Run: python -m src.main")
    return joblib.load(model_path), os.getenv("MODEL_VERSION", "local")


def git_commit() -> str:
    """
    Outputs
    - The Git commit the running code was built from

    Why it matters
    - The deploy smoke test waits until /health reports the released commit
    - GIT_COMMIT is baked into the release image; RENDER_GIT_COMMIT only exists on Git-backed Render services
    """
    return os.getenv("GIT_COMMIT") or os.getenv("RENDER_GIT_COMMIT") or "local"


# -------------------------------------------------------------------
# 3) Lifespan: load config and model ONCE at startup, flush logs at shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Inputs
    - app: the FastAPI application

    Outputs
    - Startup: config and model stored on app.state; shutdown: buffered prediction logs flushed

    Why it matters
    - Loading once at startup keeps every request fast; a failed load is reported by /health (503)
    """
    app.state.global_config = {}
    app.state.model_pipeline = None
    app.state.model_version = "unloaded"

    try:
        cfg = load_config(PROJECT_ROOT / "config.yaml")
        app.state.global_config = cfg
        # Console only: container platforms such as Render collect stdout as the service log
        configure_logging(log_level=str((cfg.get("logging") or {}).get("level", "INFO")))
        app.state.model_pipeline, app.state.model_version = _load_model(cfg)
        logger.info("Startup complete | model_version=%s", app.state.model_version)
    except Exception:
        # The API still starts so /health can report the problem (503) instead of crashing silently
        logger.exception("Startup failed, the model is not loaded")
        app.state.model_version = "startup_error"

    yield

    # Do not lose buffered prediction logs when the server stops
    _flush_buffer(app.state.global_config, force=True)
    logger.info("API shutdown complete")


app = FastAPI(title="Opioid Risk Predictor API", version="1.0.0", lifespan=lifespan)


# -------------------------------------------------------------------
# 4) Observability
# -------------------------------------------------------------------
# Layer 1: every HTTP request gets a correlation ID, status code and latency in the logs
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """
    Inputs
    - request: incoming HTTP request; call_next: the endpoint that handles it

    Outputs
    - The endpoint's response, with an X-Correlation-ID header

    Why it matters
    - One log line per request (path, status, latency, model version) is the first layer of monitoring
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    response = await call_next(request)

    logger.info(
        "correlation_id=%s path=%s method=%s status=%s latency_s=%.4f model_version=%s",
        correlation_id, request.url.path, request.method, response.status_code,
        time.time() - start_time, getattr(app.state, "model_version", "unloaded"),
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# Layer 2: prediction rows are buffered and sent to the tracker in batches (src/tracking.py)
LOG_BUFFER: List[Dict[str, Any]] = []
BUFFER_LOCK = Lock()  # protects the buffer when requests arrive at the same time


def _flush_buffer(cfg: Dict[str, Any], force: bool = False,
                  background_tasks: Optional[BackgroundTasks] = None) -> None:
    """Send the buffer once it reaches api.log_batch_size (or always when force=True)."""
    batch_size = int((cfg.get("api") or {}).get("log_batch_size", 20))
    with BUFFER_LOCK:
        if not LOG_BUFFER or (len(LOG_BUFFER) < batch_size and not force):
            return
        batch = LOG_BUFFER.copy()
        LOG_BUFFER.clear()
    if background_tasks is not None:
        background_tasks.add_task(tracking.log_prediction_batch, cfg, batch)
    else:
        tracking.log_prediction_batch(cfg, batch)


# -------------------------------------------------------------------
# 5) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    """
    Outputs
    - A pointer to the interactive documentation

    Why it matters
    - /docs (Swagger UI, generated from the Pydantic contract) is the only UI this service ships
    """
    return {"message": "Open /docs to try the API; /health reports the model version and Git commit"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Readiness check used by Docker HEALTHCHECK, Render and the deploy smoke test

    Outputs
    - 200 with status, model_version and git_commit when the model is loaded; 503 otherwise

    Why it matters
    - git_commit and model_version let the deploy workflow confirm the NEW release is live
    """
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    body = HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=getattr(app.state, "model_version", "unloaded"),
        git_commit=git_commit(),
    )
    if not model_loaded:
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


def _score_records(model: Any, cfg: Dict[str, Any], records: List[Dict[str, Any]]) -> Tuple[List[str], pd.DataFrame]:
    """
    Same clean -> validate -> infer path as the batch pipeline in main.py (zero training-serving skew)

    Inputs
    - model: loaded pipeline; cfg: config.yaml; records: request rows with dataset column names

    Outputs
    - (ids, df_pred) with columns prediction and, for classification, probability
    """
    df_clean = clean_dataframe(pd.DataFrame(records), target_column=None)
    validation_cfg = require_section(cfg, "validation")
    validate_dataframe(
        df=df_clean,
        required_columns=get_feature_columns(cfg),
        check_missing_values=bool(validation_cfg.get("check_missing_values", False)),
        numeric_non_negative_cols=validation_cfg.get("numeric_non_negative_cols", []),
    )
    problem_cfg = require_section(cfg, "problem")
    id_col = require_str(problem_cfg, "identifier_column")
    is_classification = require_str(problem_cfg, "problem_type").lower() == "classification"
    include_proba = is_classification and bool((cfg.get("run") or {}).get("include_proba_if_classification", True))
    df_pred = run_inference(model, df_clean.drop(columns=[id_col]), include_proba)
    return df_clean[id_col].astype(str).tolist(), df_pred


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, background_tasks: BackgroundTasks) -> PredictResponse:
    """
    Inputs
    - req: validated batch of PatientRecord (422 before this runs if the contract is broken)
    - request: carries the correlation ID; background_tasks: sends prediction logs after the response

    Outputs
    - One PredictionItem per record plus the concrete model_version

    Why it matters
    - Business-rule violations (e.g. negative rx_ds) return 422, real failures 500, a missing model 503
    """
    model = getattr(app.state, "model_pipeline", None)
    cfg = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health and the startup logs")

    records = [r.model_dump(by_alias=True) for r in req.records]
    try:
        start = time.time()
        ids, df_pred = _score_records(model, cfg, records)
        latency = time.time() - start
    except ValueError as e:
        logger.warning("Validation error: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    predictions: List[PredictionItem] = []
    correlation_id = getattr(request.state, "correlation_id", "missing")
    with BUFFER_LOCK:
        for i, row_id in enumerate(ids):
            pred = int(df_pred.iloc[i]["prediction"])
            proba = float(df_pred.iloc[i]["probability"]) if "probability" in df_pred.columns else None
            predictions.append(PredictionItem(ID=row_id, prediction=pred, probability=proba))
            LOG_BUFFER.append({
                "correlation_id": correlation_id, "timestamp": time.time(), "model_version": model_version,
                "latency_s": latency, "prediction": pred, "probability": proba, **records[i],
            })

    _flush_buffer(cfg, background_tasks=background_tasks)
    return PredictResponse(model_version=model_version, predictions=predictions)
