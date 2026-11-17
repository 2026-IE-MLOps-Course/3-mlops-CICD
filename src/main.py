# src/main.py
"""
Educational Goal:
- Show how an MLOps repo separates orchestration (main) from implementation (src modules)
- Show how a system becomes configurable without editing code: runtime settings live in config.yaml
- Establish a single "golden path" entry point that CI and deployment can run the same way

Run it
- python -m src.main
- python -m src.main --config config.yaml

What this module owns
- Read config.yaml and .env, set up logging and experiment tracking
- Orchestrate: load -> clean -> validate -> split -> feature recipe -> train -> evaluate
  -> quality gate -> test audit -> save -> infer

What this module does not own
- Config parsing rules (src/config.py)
- Data cleaning, validation, features, training, evaluation, inference (one module each)
- W&B details (src/tracking.py)
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src import tracking
from src.clean_data import clean_dataframe
from src.config import (
    get_feature_columns,
    load_config,
    require_float,
    require_int,
    require_list,
    require_section,
    require_str,
    resolve_repo_path,
)
from src.evaluate import check_quality_gate, evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logger import configure_logging
from src.train import calibrate_pipeline, train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SplitResult = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]


# -----------------------------
# Data splitting
# -----------------------------
def _two_step_split(X: pd.DataFrame, y: pd.Series, test_size: float, relative_val_size: float,
                    random_state: int, stratify: bool) -> SplitResult:
    """
    Inputs
    - X, y: full features and target
    - test_size: share of ALL rows for test
    - relative_val_size: share of the remaining rows for validation
    - random_state: seed; stratify: keep the class share in every split

    Outputs
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state,
        stratify=y_temp if stratify else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> SplitResult:
    """
    Split into train, validation and test (sizes are shares of the WHOLE dataset)

    Inputs
    - X, y: features and target
    - test_size, val_size: shares of all rows, for example 0.15 and 0.15
    - random_state: seed, so the same data always gives the same split
    - stratify: True for classification, keeps the event rate equal in every split

    Outputs
    - X_train, X_val, X_test, y_train, y_val, y_test

    Why it matters
    - Train is for learning
    - Validation is for decisions during development (and for the quality gate)
    - Test is the final audit, used once per run and never for tuning
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError("Split sizes must satisfy: 0 < test_size, 0 < val_size, and test_size + val_size < 1")

    relative_val_size = val_size / (1.0 - test_size)
    try:
        return _two_step_split(X, y, test_size, relative_val_size, random_state, stratify)
    except ValueError as e:
        if not stratify:
            raise
        logger.warning("Stratified split failed, falling back to non-stratified split | error=%s", e)
        return _two_step_split(X, y, test_size, relative_val_size, random_state, False)


# -----------------------------
# Settings (read once, fail fast before any heavy work)
# -----------------------------
def _path(paths_cfg: Dict[str, Any], key: str) -> Path:
    """Resolve one paths.<key> entry relative to the repo root."""
    return resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, key))


def read_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs
    - cfg: the parsed config.yaml

    Outputs
    - settings: flat dictionary with every value the pipeline needs (paths resolved, types checked)

    Why it matters
    - A typo in config.yaml stops the run in the first second, not after training
    """
    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    features_cfg = require_section(cfg, "features")
    validation_cfg = require_section(cfg, "validation")
    training_cfg = require_section(cfg, "training")
    for section in ("split", "run", "logging"):
        require_section(cfg, section)
    evaluation_cfg = cfg.get("evaluation") or {}

    problem_type = require_str(problem_cfg, "problem_type").lower()
    if problem_type not in {"classification", "regression"}:
        raise ValueError("config.yaml: problem.problem_type must be 'classification' or 'regression'")

    feature_cols = get_feature_columns(cfg)
    if not feature_cols:
        raise ValueError("config.yaml: features must define at least 1 column across the feature lists")

    model_params = training_cfg.get(problem_type) or {}
    if not isinstance(model_params, dict):
        raise ValueError(f"config.yaml: training.{problem_type} must be a mapping")

    return {
        "problem_type": problem_type,
        "target": require_str(problem_cfg, "target_column"),
        "id_col": require_str(problem_cfg, "identifier_column"),
        "raw": _path(paths_cfg, "raw_data"), "processed": _path(paths_cfg, "processed_data"),
        "model": _path(paths_cfg, "model_artifact"), "inference": _path(paths_cfg, "inference_data"),
        "predictions": _path(paths_cfg, "predictions_artifact"), "log_file": _path(paths_cfg, "log_file"),
        "feature_cols": feature_cols,
        "quantile_bin_cols": require_list(features_cfg, "quantile_bin"),
        "non_negative_cols": require_list(validation_cfg, "numeric_non_negative_cols"),
        "check_missing": bool(validation_cfg.get("check_missing_values", False)),
        "model_params": model_params,
        "gate": evaluation_cfg.get("quality_gate") or {},
        "evaluate_on_test": bool(evaluation_cfg.get("evaluate_on_test", False)),
    }


# -----------------------------
# Pipeline steps (each one calls the module that owns the logic)
# -----------------------------
def load_and_clean(s: Dict[str, Any], run: Any) -> pd.DataFrame:
    """
    Load -> clean -> save processed data

    Inputs
    - s: settings from read_settings, run: W&B run or None

    Outputs
    - df_clean: the training data after the stateless cleaning rules (also saved to paths.processed_data)
    """
    logger.info("1) LOAD raw data")
    df_raw = load_raw_data(s["raw"])
    tracking.log_metrics(run, {"data/raw_rows": df_raw.shape[0], "data/raw_cols": df_raw.shape[1]})

    logger.info("2) CLEAN training data")
    df_clean = clean_dataframe(df_raw, target_column=s["target"])
    tracking.log_metrics(run, {"data/clean_rows": df_clean.shape[0], "data/clean_cols": df_clean.shape[1]})

    logger.info("3) SAVE processed data")
    save_csv(df_clean, s["processed"])
    return df_clean


def build_candidates(cfg: Dict[str, Any], s: Dict[str, Any], X_train: pd.DataFrame,
                     y_train: pd.Series) -> List[Tuple[str, Any]]:
    """
    Feature recipe -> train -> optional calibration

    Inputs
    - cfg: full config, s: settings, X_train and y_train: training split only

    Outputs
    - candidates: [(stage_name, model)]; the LAST one is the model we intend to ship
    """
    for col in s["quantile_bin_cols"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise ValueError(f"Column '{col}' must be numeric for quantile binning. Found {X_train[col].dtype}")

    logger.info("6) BUILD feature recipe")
    features_cfg = require_section(cfg, "features")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=s["quantile_bin_cols"],
        categorical_onehot_cols=require_list(features_cfg, "categorical_onehot"),
        numeric_passthrough_cols=require_list(features_cfg, "numeric_passthrough"),
        binary_sum_cols=require_list(features_cfg, "binary_sum_cols"),
        n_bins=require_int(features_cfg, "n_bins"),
    )

    logger.info("7) TRAIN model pipeline")
    params = s["model_params"]
    model = train_model(X_train, y_train, preprocessor, s["problem_type"], params,
                        require_int(require_section(cfg, "split"), "random_state"))
    candidates = [("before_calibration", model)]

    if s["problem_type"] == "classification" and bool(params.get("calibration_enabled", False)):
        logger.info("7b) CALIBRATE classification probabilities")
        calibrated = calibrate_pipeline(
            pipeline=model, X_train=X_train, y_train=y_train,
            method=str(params.get("calibration_method", "sigmoid")).strip().lower(),
            cv=int(params.get("calibration_cv", 3)),
        )
        candidates.append(("after_calibration", calibrated))
    return candidates


def evaluate_candidates(cfg: Dict[str, Any], s: Dict[str, Any], candidates: List[Tuple[str, Any]],
                        X_val: pd.DataFrame, y_val: pd.Series, run: Any) -> Dict[str, float]:
    """
    Evaluate every candidate on the validation split and log the comparison

    Inputs
    - cfg, s: config and settings; candidates: from build_candidates; X_val, y_val: validation split; run

    Outputs
    - The validation metrics of the LAST candidate (the one the quality gate judges)
    """
    logger.info("8) EVALUATE on validation split")
    comparison = []
    for stage_name, model_obj in candidates:
        val_metrics = evaluate_model(model_obj, X_val, y_val, s["problem_type"])
        logger.info("Validation metrics [%s]: %s", stage_name, val_metrics)
        tracking.log_metrics(run, {f"metrics/val_{stage_name}/{k}": v for k, v in val_metrics.items()})
        if s["problem_type"] == "classification":
            tracking.log_classification_plots(run, cfg, X_val, y_val, model_obj, stage_name)
        comparison.append({"stage": stage_name, **val_metrics})

    if len(comparison) > 1:
        tracking.log_table(run, "tables/metrics_comparison_val", pd.DataFrame(comparison))
    return {k: v for k, v in comparison[-1].items() if k != "stage"}


def audit_on_test(s: Dict[str, Any], model: Any, X_test: pd.DataFrame, y_test: pd.Series, run: Any) -> None:
    """
    Inputs
    - s: settings; model: the gated model; X_test, y_test: the untouched test split; run: W&B run or None

    Outputs
    - None. Logs the test metrics once when evaluation.evaluate_on_test is true (report only, never tune on it)
    """
    if not s["evaluate_on_test"]:
        return
    logger.info("10) AUDIT on test split (report only, never tune on it)")
    test_metrics = evaluate_model(model, X_test, y_test, s["problem_type"])
    logger.info("Test metrics (final audit): %s", test_metrics)
    tracking.log_metrics(run, {f"metrics/test/{k}": v for k, v in test_metrics.items()})


def save_and_register(cfg: Dict[str, Any], s: Dict[str, Any], model: Any, run: Any) -> str:
    """
    Save the gated model and log it (plus the processed data) as W&B artifacts

    Inputs
    - cfg, s: config and settings; model: the gated model; run: W&B run or None

    Outputs
    - artifact_name: the registry name, reused for the predictions artifact
    """
    logger.info("11) SAVE model artifact")
    save_model(model, s["model"])

    artifact_name = str(tracking.wandb_setting(cfg, "model_artifact_name", "model"))
    tracking.log_file_artifact(
        run, name=artifact_name, artifact_type="model", file_path=s["model"],
        description="Scikit-learn pipeline (preprocessing + model), passed the quality gate",
    )
    if tracking.wandb_flag(cfg, "log_processed_data"):
        tracking.log_file_artifact(
            run, name=f"{artifact_name}-processed-data", artifact_type="dataset",
            file_path=s["processed"], description="Cleaned training dataset",
        )
    return artifact_name


def batch_inference(cfg: Dict[str, Any], s: Dict[str, Any], model: Any) -> pd.DataFrame:
    """
    Score the inference file with the same clean -> validate -> infer path as the API

    Inputs
    - cfg, s: config and settings; model: the gated model

    Outputs
    - df_predictions: identifier, prediction and (classification) probability per input row
    """
    logger.info("12) INFER on new data file")
    df_infer = clean_dataframe(load_raw_data(s["inference"]), target_column=None)
    validate_dataframe(df=df_infer, required_columns=s["feature_cols"], check_missing_values=s["check_missing"],
                       numeric_non_negative_cols=s["non_negative_cols"])

    include_proba = s["problem_type"] == "classification" and bool(
        (cfg.get("run") or {}).get("include_proba_if_classification", True))
    df_predictions = run_inference(model, df_infer.drop(columns=[s["id_col"]], errors="ignore"), include_proba)
    if s["id_col"] in df_infer.columns:
        df_predictions.insert(0, s["id_col"], df_infer[s["id_col"]].values)
    return df_predictions


def log_predictions(cfg: Dict[str, Any], s: Dict[str, Any], df_predictions: pd.DataFrame,
                    artifact_name: str, run: Any) -> None:
    """
    Inputs
    - cfg, s: config and settings; df_predictions: batch output; artifact_name: model registry name; run

    Outputs
    - None. Writes the predictions CSV and, when enabled, a W&B table and artifact
    """
    save_csv(df_predictions, s["predictions"])
    if tracking.wandb_flag(cfg, "log_predictions_table"):
        n_rows = int(tracking.wandb_setting(cfg, "predictions_table_rows", 200))
        tracking.log_table(run, "tables/predictions_preview", df_predictions.head(n_rows))
    if tracking.wandb_flag(cfg, "log_predictions"):
        tracking.log_file_artifact(
            run, name=f"{artifact_name}-predictions", artifact_type="predictions",
            file_path=s["predictions"], description="Batch inference outputs",
        )


def parse_args() -> argparse.Namespace:
    """
    Command line interface

    Why: lets CI or a teammate point the pipeline at another config file

    Outputs
    - Parsed arguments with .config (Path)
    """
    parser = argparse.ArgumentParser(description="Run the end-to-end training and batch inference pipeline")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml",
                        help="Path to config.yaml (default: repo root config.yaml)")
    return parser.parse_args()


def main(config_path: Optional[Path] = None) -> None:
    """
    Entry point: python -m src.main [--config path]. Reads like the recipe: every step in order
    Inputs: config_path (default: repo root config.yaml)
    Outputs: None. Writes processed data, model and predictions; raises on any failure (W&B run marked failed)
    Why: one entry point for CI (dvc repro), teammates and the grader; the quality gate sits before
    save_and_register, so a weak model is never saved, registered or deployed
    """
    # Secrets (for example WANDB_API_KEY) come from .env, never from config.yaml
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    cfg = load_config(config_path or PROJECT_ROOT / "config.yaml")
    s = read_settings(cfg)
    configure_logging(log_level=require_str(require_section(cfg, "logging"), "level"), log_file=s["log_file"])
    is_classification = s["problem_type"] == "classification"
    split_cfg, gate = require_section(cfg, "split"), s["gate"]

    run = tracking.start_run(cfg)
    try:
        df_clean = load_and_clean(s, run)
        logger.info("4) VALIDATE training data (before the split: bad data never reaches training)")
        validate_dataframe(df_clean, [s["target"]] + s["feature_cols"], s["target"], s["check_missing"],
                           [0, 1] if is_classification else None, s["non_negative_cols"])

        logger.info("5) SPLIT train/val/test (the identifier is kept for traceability, never used as a feature)")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_clean.drop(columns=[s["target"], s["id_col"]], errors="ignore"), df_clean[s["target"]],
            require_float(split_cfg, "test_size"), require_float(split_cfg, "val_size"),
            require_int(split_cfg, "random_state"), is_classification)

        candidates = build_candidates(cfg, s, X_train, y_train)
        final_model = candidates[-1][1]
        val_metrics = evaluate_candidates(cfg, s, candidates, X_val, y_val, run)
        if bool(gate.get("enabled", False)):
            logger.info("9) QUALITY GATE on validation metrics")
            check_quality_gate(val_metrics, metric=require_str(gate, "metric"),
                               min_value=gate.get("min_value"), max_value=gate.get("max_value"))
        else:
            logger.warning("9) QUALITY GATE disabled in config.yaml, any model will be saved")

        audit_on_test(s, final_model, X_test, y_test, run)
        artifact_name = save_and_register(cfg, s, final_model, run)
        log_predictions(cfg, s, batch_inference(cfg, s, final_model), artifact_name, run)
        logger.info("Done | processed=%s | model=%s | predictions=%s", s["processed"], s["model"], s["predictions"])
    except Exception:
        # logger.exception keeps the full stack trace in the log file
        logger.exception("Pipeline failed")
        tracking.finish_run(run, exit_code=1)
        raise
    tracking.finish_run(run)


if __name__ == "__main__":
    main(parse_args().config)
