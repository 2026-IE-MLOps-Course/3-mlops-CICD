# src/api.py
"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI
- Use Pydantic to enforce a strict data contract (types and required fields)
- Reuse existing pipeline modules to prevent training-serving skew
- Add lightweight observability: request logs (Layer 1) and batched prediction logs to W&B (Layer 2)

Key principle
- NO NEW ML LOGIC IN THIS FILE
- This file only does HTTP, schema validation, startup loading, observability and routing

Run it locally
- uvicorn src.api:app --reload
- Open http://127.0.0.1:8000/docs
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import wandb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

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
    """
    model_config = ConfigDict(
        extra="forbid",
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
    I: int  # noqa: E741  (column name comes from the dataset)
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
    model_config = ConfigDict(extra="forbid")
    records: List[PatientRecord]


class PredictionItem(BaseModel):
    ID: str
    prediction: int
    probability: Optional[float] = None


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    git_commit: str


# -------------------------------------------------------------------
# 2) Model loading (local file or W&B Model Registry)
# -------------------------------------------------------------------
def _load_model(cfg: Dict[str, Any]):
    """
    Return (model, model_version)

    MODEL_SOURCE=local  -> load models/model.joblib (path from config.yaml)
    MODEL_SOURCE=wandb  -> download the artifact with alias WANDB_MODEL_ALIAS (default "prod")
    """
    model_source = os.getenv("MODEL_SOURCE", "local").strip().lower()

    if model_source == "wandb":
        entity = os.getenv("WANDB_ENTITY")
        alias = os.getenv("WANDB_MODEL_ALIAS", "prod")
        wandb_cfg = cfg.get("wandb") or {}
        project = wandb_cfg.get("project")
        artifact_name = wandb_cfg.get("model_artifact_name")
        if not entity or not project or not artifact_name:
            raise ValueError("MODEL_SOURCE=wandb needs WANDB_ENTITY (.env) plus wandb.project and "
                             "wandb.model_artifact_name (config.yaml)")

        artifact_path = f"{entity}/{project}/{artifact_name}:{alias}"
        logger.info("Downloading model from W&B registry: %s", artifact_path)
        # wandb reads WANDB_API_KEY from the environment automatically
        artifact_dir = wandb.Api().artifact(artifact_path).download()
        model_path = Path(artifact_dir) / Path(require_str(cfg["paths"], "model_artifact")).name
        return joblib.load(model_path), artifact_path

    model_path = resolve_repo_path(PROJECT_ROOT, require_str(require_section(cfg, "paths"), "model_artifact"))
    logger.info("Loading local model artifact: %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing at {model_path}. Run: python -m src.main")
    return joblib.load(model_path), model_path.name


# -------------------------------------------------------------------
# 3) Lifespan: load config and model ONCE at startup, flush logs at shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
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
async def log_requests(request: Request, call_next):
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


# Layer 2: prediction rows are buffered and sent to W&B in batches (a Table per batch)
LOG_BUFFER: List[Dict[str, Any]] = []
BUFFER_LOCK = Lock()  # protects the buffer when requests arrive at the same time


def flush_logs_to_wandb(batch_data: List[Dict[str, Any]], project_name: str) -> None:
    """Send one batch of prediction logs to W&B. Never raises: monitoring must not break serving."""
    if not batch_data or os.getenv("WANDB_MODE", "").lower() == "disabled":
        return
    try:
        run = wandb.init(entity=os.getenv("WANDB_ENTITY") or None, project=project_name,
                         job_type="inference-batch", reinit=True)
        run.log({"inference_logs": wandb.Table(dataframe=pd.DataFrame(batch_data))})
        run.finish()
        logger.info("Flushed %s prediction logs to W&B", len(batch_data))
    except Exception as e:
        logger.error("Failed to flush prediction logs to W&B: %s", e)


def _flush_buffer(cfg: Dict[str, Any], force: bool = False,
                  background_tasks: Optional[BackgroundTasks] = None) -> None:
    batch_size = int((cfg.get("api") or {}).get("wandb_log_batch_size", 20))
    project = (cfg.get("wandb") or {}).get("project", "mlops-project")
    with BUFFER_LOCK:
        if not LOG_BUFFER or (len(LOG_BUFFER) < batch_size and not force):
            return
        batch = LOG_BUFFER.copy()
        LOG_BUFFER.clear()
    if background_tasks is not None:
        background_tasks.add_task(flush_logs_to_wandb, batch, project)
    else:
        flush_logs_to_wandb(batch, project)


# -------------------------------------------------------------------
# 5) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Use /health or /docs to test the API"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Readiness check used by Docker HEALTHCHECK, Render and the deploy smoke test
    - 200 only when the model is loaded
    - git_commit lets the deploy workflow confirm the NEW version is live
    """
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    body = HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=getattr(app.state, "model_version", "unloaded"),
        git_commit=os.getenv("RENDER_GIT_COMMIT", "local"),
    )
    if not model_loaded:
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, background_tasks: BackgroundTasks) -> PredictResponse:
    model = getattr(app.state, "model_pipeline", None)
    cfg = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health and the startup logs")

    try:
        start = time.time()
        records = [r.model_dump() for r in req.records]

        # Same clean -> validate -> infer path as the batch pipeline in main.py (zero training-serving skew)
        df_clean = clean_dataframe(pd.DataFrame(records), target_column=None)
        validation_cfg = require_section(cfg, "validation")
        validate_dataframe(
            df=df_clean,
            required_columns=get_feature_columns(cfg),
            check_missing_values=bool(validation_cfg.get("check_missing_values", False)),
            numeric_non_negative_cols=validation_cfg.get("numeric_non_negative_cols", []),
        )

        id_col = require_str(require_section(cfg, "problem"), "identifier_column")
        ids = df_clean[id_col].astype(str).tolist()
        is_classification = require_str(cfg["problem"], "problem_type").lower() == "classification"
        include_proba = is_classification and bool((cfg.get("run") or {}).get("include_proba_if_classification", True))

        df_pred = run_inference(model=model, X_infer=df_clean.drop(columns=[id_col]), include_proba=include_proba)
        latency = time.time() - start
    except ValueError as e:
        # Business-rule violations (e.g. negative rx_ds) are client errors, not server errors
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
            proba = float(df_pred.iloc[i]["proba"]) if "proba" in df_pred.columns else None
            predictions.append(PredictionItem(ID=row_id, prediction=pred, probability=proba))
            LOG_BUFFER.append({
                "correlation_id": correlation_id, "timestamp": time.time(), "model_version": model_version,
                "latency_s": latency, "prediction": pred, "probability": proba, **records[i],
            })

    _flush_buffer(cfg, background_tasks=background_tasks)
    return PredictResponse(model_version=model_version, predictions=predictions)
