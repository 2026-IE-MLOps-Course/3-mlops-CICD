# src/tracking.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Keep experiment tracking (Weights & Biases, W&B)
  in one place so main.py reads like a recipe, not like a W&B tutorial
- Responsibility (separation of concerns): Start and finish W&B runs, log metrics, tables,
  plots and file artifacts. No training or evaluation logic
- Pipeline contract: Every function accepts run=None and then does nothing.
  This is how the pipeline runs with W&B switched off (wandb.enabled: false or WANDB_MODE=disabled)

Teaching note
- Tracking is optional infrastructure. The pipeline must never fail because tracking is off
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb
import yaml

from src.evaluate import evaluate_calibration

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _wandb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the wandb section of config.yaml, or an empty dict when it is missing or malformed."""
    value = cfg.get("wandb")
    return value if isinstance(value, dict) else {}


def _clean_list(value: Any) -> List[str]:
    """Return the non-empty strings of a list (config tags), or an empty list."""
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def git_commit(repo_root: Path = PROJECT_ROOT) -> str:
    """
    Inputs
    - repo_root: folder inside the Git checkout

    Outputs
    - The current commit SHA; GITHUB_SHA in CI when git is unavailable; "unknown" otherwise

    Why it matters
    - A registry version must name the exact code that produced it
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                                text=True, check=True, timeout=10)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.getenv("GITHUB_SHA", "unknown")


def dvc_pointer_md5(data_path: Path) -> str:
    """
    Inputs
    - data_path: a DVC-tracked file, e.g. data/raw/opioid_raw_data.csv (its pointer is <file>.dvc)

    Outputs
    - The md5 recorded in the pointer, or "unknown" when there is no readable pointer

    Why it matters
    - The md5 links a model version to the exact data version on the DVC remote (dvc pull restores it)
    """
    pointer = Path(f"{data_path}.dvc")
    try:
        outs = (yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}).get("outs") or [{}]
        return str(outs[0].get("md5") or "unknown")
    except (OSError, yaml.YAMLError, AttributeError):
        return "unknown"


def run_lineage(cfg: Dict[str, Any], repo_root: Path = PROJECT_ROOT) -> Dict[str, str]:
    """
    Inputs
    - cfg: full configuration (paths.raw_data names the data file)
    - repo_root: repository root

    Outputs
    - {"git_commit": <sha>, "raw_data_md5": <md5>}, stored on the run config
    """
    raw = str((cfg.get("paths") or {}).get("raw_data") or "")
    return {
        "git_commit": git_commit(repo_root),
        "raw_data_md5": dvc_pointer_md5(repo_root / raw) if raw else "unknown",
    }


def _text_or_none(value: Any) -> Optional[str]:
    """Return a stripped string, or None for missing or blank values (W&B then uses its default)."""
    text = str(value or "").strip()
    return text or None


def _init_kwargs(wb: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the wandb section of config.yaml into wandb.init() keyword arguments (empty -> None)."""
    project = _text_or_none(wb.get("project"))
    if project is None:
        raise ValueError("config.yaml: wandb.project must be a non-empty string when wandb.enabled is true")
    return {
        "project": project,
        "name": _text_or_none(wb.get("name")),
        "job_type": _text_or_none(wb.get("job_type")) or "training-pipeline",
        "group": _text_or_none(wb.get("group")),
        "notes": _text_or_none(wb.get("notes")),
        "tags": _clean_list(wb.get("tags")) or None,
    }


def start_run(cfg: Dict[str, Any]) -> Optional[Any]:
    """
    Inputs
    - cfg: Full configuration dictionary (the whole config is stored on the run for lineage)

    Outputs
    - run: An active W&B run, or None when tracking is disabled

    Why it matters
    - The run config also records the Git commit and the raw-data md5 (from the .dvc pointer),
      so every registry version traces back to its code and its data
    """
    wb = _wandb_cfg(cfg)
    if not bool(wb.get("enabled", False)):
        logger.info("W&B disabled in config.yaml, continuing without experiment tracking")
        return None

    kwargs = _init_kwargs(wb)
    lineage = run_lineage(cfg)
    run = wandb.init(config={**cfg, "lineage": lineage}, **kwargs)
    run.summary["entrypoint"] = "python -m src.main"
    logger.info("Initialized W&B run | name=%s | project=%s | commit=%s | raw_data_md5=%s",
                run.name, kwargs["project"], lineage["git_commit"], lineage["raw_data_md5"])
    return run


def log_metrics(run: Optional[Any], metrics: Dict[str, Any]) -> None:
    """
    Inputs: run (W&B run or None) and metrics (flat dictionary of numbers)
    Outputs: None. Does nothing when run is None, so the pipeline runs without W&B
    """
    if run is None:
        return
    run.log({k: float(v) for k, v in metrics.items()})


def log_table(run: Optional[Any], key: str, df: pd.DataFrame) -> None:
    """
    Inputs: run (W&B run or None), key (panel name) and df (small DataFrame)
    Outputs: None. Logs an interactive W&B Table; does nothing when run is None or df is empty
    """
    if run is None or df is None or df.empty:
        return
    run.log({key: wandb.Table(dataframe=df)})


def log_file_artifact(run: Optional[Any], *, name: str, artifact_type: str, file_path: Path, description: str) -> None:
    """
    Version a file (model, dataset, predictions) as a W&B Artifact

    Inputs: run (or None), name, artifact_type ("model", "dataset", ...), file_path, description
    Outputs: None. A new artifact version (v0, v1, ...) in the W&B project

    Why this matters
    - Artifacts give lineage: which data and code produced which model
    - The API later downloads the model artifact that carries the alias "prod"
    """
    if run is None:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type, description=description)
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)
    logger.info("Logged W&B artifact | name=%s | type=%s", name, artifact_type)


def log_classification_plots(run: Optional[Any], cfg: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series,
                             model: Any, stage_name: str) -> None:
    """
    Inputs: run (or None), cfg (wandb.log_* flags), validation split, fitted model, stage_name (panel suffix)
    Outputs: None. Logs the optional ROC, PR, confusion matrix and calibration table
    """
    if run is None or not hasattr(model, "predict_proba"):
        return

    wb = _wandb_cfg(cfg)
    class_names = wb.get("class_names")
    y_probas = model.predict_proba(X_val)

    if wb.get("log_auc_plots", False):
        run.log({
            f"plots/roc_curve_val_{stage_name}": wandb.plot.roc_curve(
                y_true=y_val.tolist(), y_probas=y_probas, labels=class_names,
                title=f"Validation ROC Curve ({stage_name})"),
            f"plots/pr_curve_val_{stage_name}": wandb.plot.pr_curve(
                y_true=y_val.tolist(), y_probas=y_probas, labels=class_names,
                title=f"Validation Precision-Recall Curve ({stage_name})"),
        })

    if wb.get("log_confusion_matrix", False):
        y_pred = model.predict(X_val)
        run.log({
            f"plots/confusion_matrix_val_{stage_name}": wandb.plot.confusion_matrix(
                probs=None, y_true=y_val.tolist(), preds=list(y_pred), class_names=class_names,
                title=f"Validation Confusion Matrix ({stage_name})"),
        })

    if wb.get("log_calibration_table", False):
        n_bins = int((cfg.get("evaluation") or {}).get("calibration_bins", 10))
        table, _ = evaluate_calibration(y_true=y_val, y_prob=y_probas[:, 1], n_bins=n_bins)
        log_table(run, f"tables/calibration_val_{stage_name}", table)


def finish_run(run: Optional[Any], exit_code: int = 0) -> None:
    """
    Inputs: run (or None) and exit_code (1 marks the run as failed)
    Outputs: None. Closes the run so W&B uploads everything
    """
    if run is None:
        return
    run.finish(exit_code=exit_code)


def wandb_flag(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    """
    Inputs: cfg, key and default
    Outputs: the boolean switch wandb.<key> from config.yaml
    """
    return bool(_wandb_cfg(cfg).get(key, default))


def wandb_setting(cfg: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """
    Inputs: cfg, key and default
    Outputs: the value wandb.<key> from config.yaml, or default
    """
    return _wandb_cfg(cfg).get(key, default)
