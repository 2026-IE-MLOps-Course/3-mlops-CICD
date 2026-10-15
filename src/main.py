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
from typing import Optional, Tuple

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


# -----------------------------
# Data splitting
# -----------------------------
def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split into train, validation, test

    Why it matters
    - Train is for learning
    - Validation is for decisions during development (and for the quality gate)
    - Test is the final audit, used once per run and never for tuning
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            "Split sizes must satisfy: 0 < test_size, 0 < val_size, and test_size + val_size < 1"
        )

    relative_val_size = val_size / (1.0 - test_size)

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state,
            stratify=y_temp if stratify else None,
        )
    except ValueError as e:
        logger.warning("Stratified split failed, falling back to non-stratified split | error=%s", e)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def parse_args() -> argparse.Namespace:
    """Command line interface: lets CI or a teammate point the pipeline at another config file."""
    parser = argparse.ArgumentParser(description="Run the end-to-end training and batch inference pipeline")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml",
                        help="Path to config.yaml (default: repo root config.yaml)")
    return parser.parse_args()


def main(config_path: Optional[Path] = None) -> None:
    # Secrets (for example WANDB_API_KEY) come from .env, never from config.yaml
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    # -----------------------------
    # 0) Read and check configuration (fail fast before any heavy work)
    # -----------------------------
    cfg = load_config(config_path or PROJECT_ROOT / "config.yaml")

    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    split_cfg = require_section(cfg, "split")
    features_cfg = require_section(cfg, "features")
    validation_cfg = require_section(cfg, "validation")
    run_cfg = require_section(cfg, "run")
    training_cfg = require_section(cfg, "training")
    logging_cfg = require_section(cfg, "logging")
    evaluation_cfg = cfg.get("evaluation") or {}

    configure_logging(
        log_level=require_str(logging_cfg, "level"),
        log_file=resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "log_file")),
    )

    problem_type = require_str(problem_cfg, "problem_type").lower()
    if problem_type not in {"classification", "regression"}:
        raise ValueError("config.yaml: problem.problem_type must be 'classification' or 'regression'")

    target_column = require_str(problem_cfg, "target_column")
    identifier_column = require_str(problem_cfg, "identifier_column")

    raw_data_path = resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "raw_data"))
    processed_data_path = resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "processed_data"))
    model_artifact_path = resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "model_artifact"))
    inference_data_path = resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "inference_data"))
    predictions_path = resolve_repo_path(PROJECT_ROOT, require_str(paths_cfg, "predictions_artifact"))

    feature_cols = get_feature_columns(cfg)
    if not feature_cols:
        raise ValueError("config.yaml: features must define at least 1 column across the feature lists")
    quantile_bin_cols = require_list(features_cfg, "quantile_bin")

    numeric_non_negative_cols = require_list(validation_cfg, "numeric_non_negative_cols")
    check_missing_values = bool(validation_cfg.get("check_missing_values", False))

    model_params = training_cfg.get(problem_type) or {}
    if not isinstance(model_params, dict):
        raise ValueError(f"config.yaml: training.{problem_type} must be a mapping")

    gate_cfg = evaluation_cfg.get("quality_gate") or {}

    run = tracking.start_run(cfg)

    try:
        logger.info("Starting pipeline")

        # -----------------------------
        # 1) Load raw training data
        # -----------------------------
        logger.info("1) LOAD raw data")
        df_raw = load_raw_data(raw_data_path)
        tracking.log_metrics(run, {"data/raw_rows": df_raw.shape[0], "data/raw_cols": df_raw.shape[1]})

        # -----------------------------
        # 2) Clean (stateless, safe before the split)
        # -----------------------------
        logger.info("2) CLEAN training data")
        df_clean = clean_dataframe(df_raw, target_column=target_column)
        tracking.log_metrics(run, {"data/clean_rows": df_clean.shape[0], "data/clean_cols": df_clean.shape[1]})

        logger.info("3) SAVE processed data")
        save_csv(df_clean, processed_data_path)

        # -----------------------------
        # 4) Validate (data quality gate)
        # -----------------------------
        logger.info("4) VALIDATE training data")
        validate_dataframe(
            df=df_clean,
            required_columns=[target_column] + feature_cols,
            check_missing_values=check_missing_values,
            target_column=target_column,
            target_allowed_values=[0, 1] if problem_type == "classification" else None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        # -----------------------------
        # 5) Split before any fitting (leakage gate)
        # -----------------------------
        logger.info("5) SPLIT train/val/test")
        X_full = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        # The identifier is kept for traceability but never used as a feature
        X_features = X_full.drop(columns=[identifier_column], errors="ignore")

        X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
            X_features, y,
            test_size=require_float(split_cfg, "test_size"),
            val_size=require_float(split_cfg, "val_size"),
            random_state=require_int(split_cfg, "random_state"),
            stratify=(problem_type == "classification"),
        )
        logger.info("Split sizes | train=%s | val=%s | test=%s", X_train.shape, X_val.shape, X_test.shape)

        for col in quantile_bin_cols:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                raise ValueError(f"Column '{col}' must be numeric for quantile binning. Found {X_train[col].dtype}")

        # -----------------------------
        # 6) Feature recipe (unfitted) and 7) training
        # -----------------------------
        logger.info("6) BUILD feature recipe")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=quantile_bin_cols,
            categorical_onehot_cols=require_list(features_cfg, "categorical_onehot"),
            numeric_passthrough_cols=require_list(features_cfg, "numeric_passthrough"),
            binary_sum_cols=require_list(features_cfg, "binary_sum_cols"),
            n_bins=require_int(features_cfg, "n_bins"),
        )

        logger.info("7) TRAIN model pipeline")
        final_model = train_model(
            X_train=X_train, y_train=y_train, preprocessor=preprocessor,
            problem_type=problem_type, model_params=model_params,
        )
        candidates = [("before_calibration", final_model)]

        if problem_type == "classification" and bool(model_params.get("calibration_enabled", False)):
            logger.info("7b) CALIBRATE classification probabilities")
            final_model = calibrate_pipeline(
                pipeline=final_model, X_train=X_train, y_train=y_train,
                method=str(model_params.get("calibration_method", "sigmoid")).strip().lower(),
                cv=int(model_params.get("calibration_cv", 3)),
            )
            candidates.append(("after_calibration", final_model))

        # -----------------------------
        # 8) Evaluate on validation
        # -----------------------------
        logger.info("8) EVALUATE on validation split")
        comparison = []
        for stage_name, model_obj in candidates:
            val_metrics = evaluate_model(model=model_obj, X_eval=X_val, y_eval=y_val, problem_type=problem_type)
            logger.info("Validation metrics [%s]: %s", stage_name, val_metrics)
            tracking.log_metrics(run, {f"metrics/val_{stage_name}/{k}": v for k, v in val_metrics.items()})
            if problem_type == "classification":
                tracking.log_classification_plots(run, cfg, X_val, y_val, model_obj, stage_name)
            comparison.append({"stage": stage_name, **val_metrics})

        if len(comparison) > 1:
            tracking.log_table(run, "tables/metrics_comparison_val", pd.DataFrame(comparison))

        # The last candidate is the model we intend to ship
        final_val_metrics = {k: v for k, v in comparison[-1].items() if k != "stage"}

        # -----------------------------
        # 9) Model quality gate (fail fast: a weak model is never saved or registered)
        # -----------------------------
        if bool(gate_cfg.get("enabled", False)):
            logger.info("9) QUALITY GATE on validation metrics")
            check_quality_gate(
                final_val_metrics,
                metric=require_str(gate_cfg, "metric"),
                min_value=gate_cfg.get("min_value"),
                max_value=gate_cfg.get("max_value"),
            )
        else:
            logger.warning("9) QUALITY GATE disabled in config.yaml, any model will be saved")

        # -----------------------------
        # 10) Final audit on the untouched test split (report only, never tune on it)
        # -----------------------------
        if bool(evaluation_cfg.get("evaluate_on_test", False)):
            logger.info("10) AUDIT on test split")
            test_metrics = evaluate_model(model=final_model, X_eval=X_test, y_eval=y_test, problem_type=problem_type)
            logger.info("Test metrics (final audit): %s", test_metrics)
            tracking.log_metrics(run, {f"metrics/test/{k}": v for k, v in test_metrics.items()})

        # -----------------------------
        # 11) Save and register the model artifact
        # -----------------------------
        logger.info("11) SAVE model artifact")
        save_model(final_model, model_artifact_path)

        artifact_name = str(tracking.wandb_setting(cfg, "model_artifact_name", "model"))
        tracking.log_file_artifact(
            run, name=artifact_name, artifact_type="model", file_path=model_artifact_path,
            description="Scikit-learn pipeline (preprocessing + model), passed the quality gate",
        )
        if tracking.wandb_flag(cfg, "log_processed_data"):
            tracking.log_file_artifact(
                run, name=f"{artifact_name}-processed-data", artifact_type="dataset",
                file_path=processed_data_path, description="Cleaned training dataset",
            )

        # -----------------------------
        # 12) Batch inference on new data (same clean + validate path as the API)
        # -----------------------------
        logger.info("12) INFER on new data file")
        df_infer = clean_dataframe(load_raw_data(inference_data_path), target_column=None)
        validate_dataframe(
            df=df_infer,
            required_columns=feature_cols,
            check_missing_values=check_missing_values,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        include_proba = problem_type == "classification" and bool(
            run_cfg.get("include_proba_if_classification", True))
        df_predictions = run_inference(
            model=final_model,
            X_infer=df_infer.drop(columns=[identifier_column], errors="ignore"),
            include_proba=include_proba,
        )
        if identifier_column in df_infer.columns:
            df_predictions.insert(0, identifier_column, df_infer[identifier_column].values)

        save_csv(df_predictions, predictions_path)

        if tracking.wandb_flag(cfg, "log_predictions_table"):
            n_rows = int(tracking.wandb_setting(cfg, "predictions_table_rows", 200))
            tracking.log_table(run, "tables/predictions_preview", df_predictions.head(n_rows))
        if tracking.wandb_flag(cfg, "log_predictions"):
            tracking.log_file_artifact(
                run, name=f"{artifact_name}-predictions", artifact_type="predictions",
                file_path=predictions_path, description="Batch inference outputs",
            )

        logger.info("Done | processed=%s | model=%s | predictions=%s",
                    processed_data_path, model_artifact_path, predictions_path)

    except Exception:
        # logger.exception keeps the full stack trace in the log file
        logger.exception("Pipeline failed")
        tracking.finish_run(run, exit_code=1)
        raise

    tracking.finish_run(run)


if __name__ == "__main__":
    main(parse_args().config)
