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
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb

from src.evaluate import evaluate_calibration

logger = logging.getLogger(__name__)


def _wandb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    value = cfg.get("wandb")
    return value if isinstance(value, dict) else {}


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def start_run(cfg: Dict[str, Any]):
    """
    Inputs
    - cfg: Full configuration dictionary (the whole config is stored on the run for lineage)

    Outputs
    - run: An active W&B run, or None when tracking is disabled
    """
    wb = _wandb_cfg(cfg)
    if not bool(wb.get("enabled", False)):
        logger.info("W&B disabled in config.yaml, continuing without experiment tracking")
        return None

    project = str(wb.get("project") or "").strip()
    if not project:
        raise ValueError("config.yaml: wandb.project must be a non-empty string when wandb.enabled is true")

    run = wandb.init(
        project=project,
        name=str(wb.get("name") or "").strip() or None,
        job_type=str(wb.get("job_type") or "training-pipeline").strip(),
        group=str(wb.get("group") or "").strip() or None,
        notes=str(wb.get("notes") or "").strip() or None,
        tags=_clean_list(wb.get("tags")) or None,
        config=cfg,
    )
    run.summary["entrypoint"] = "python -m src.main"
    logger.info("Initialized W&B run | name=%s | project=%s", run.name, project)
    return run


def log_metrics(run, metrics: Dict[str, Any]) -> None:
    """Log a flat dictionary of numbers. Does nothing when run is None."""
    if run is None:
        return
    run.log({k: float(v) for k, v in metrics.items()})


def log_table(run, key: str, df: pd.DataFrame) -> None:
    """Log a small DataFrame as an interactive W&B Table. Does nothing when run is None."""
    if run is None or df is None or df.empty:
        return
    run.log({key: wandb.Table(dataframe=df)})


def log_file_artifact(run, *, name: str, artifact_type: str, file_path: Path, description: str) -> None:
    """
    Version a file (model, dataset, predictions) as a W&B Artifact

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


def log_classification_plots(run, cfg: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series,
                             model, stage_name: str) -> None:
    """Log optional ROC, PR, confusion matrix and calibration table, controlled by config flags."""
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


def finish_run(run, exit_code: int = 0) -> None:
    """Close the run so W&B uploads everything. exit_code=1 marks the run as failed."""
    if run is None:
        return
    run.finish(exit_code=exit_code)


def wandb_flag(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    """Read a boolean switch from the wandb section of config.yaml."""
    return bool(_wandb_cfg(cfg).get(key, default))


def wandb_setting(cfg: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """Read any value from the wandb section of config.yaml."""
    return _wandb_cfg(cfg).get(key, default)
