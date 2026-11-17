# tests/test_main.py
"""
Educational Goal:
- Unit test the split contract: split_data(X, y, test_size, val_size, random_state, stratify)
- Run the whole pipeline end to end on tiny synthetic CSVs in tmp_path (never the real data/ folder)
- Prove the quality gate stops a weak model BEFORE it is saved
- Spy on delegation: main.py orchestrates, the modules do the work
"""

import runpy
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml
from sklearn.calibration import CalibratedClassifierCV

import src.main as main_module
from src.utils import load_model


# --------------------------------------------------------
# Helpers: synthetic inputs and an in-memory config with the real schema
# --------------------------------------------------------
def make_config(tmp_path: Path, flags, problem_type: str = "classification") -> Dict[str, Any]:
    """Every path points into tmp_path, so the real data/, models/ and reports/ folders are never touched."""
    return {
        "paths": {
            "raw_data": str(tmp_path / "raw.csv"),
            "processed_data": str(tmp_path / "out" / "clean.csv"),
            "model_artifact": str(tmp_path / "out" / "model.joblib"),
            "inference_data": str(tmp_path / "infer.csv"),
            "predictions_artifact": str(tmp_path / "out" / "predictions.csv"),
            "log_file": str(tmp_path / "out" / "pipeline.log"),
        },
        "logging": {"level": "INFO"},
        "problem": {"target_column": "OD", "problem_type": problem_type, "identifier_column": "ID"},
        "split": {"test_size": 0.15, "val_size": 0.15, "random_state": 42},
        "training": {
            "classification": {"model_type": "logistic_regression", "max_iter": 200, "solver": "liblinear",
                               "class_weight": "balanced", "calibration_enabled": True,
                               "calibration_method": "sigmoid", "calibration_cv": 3},
            "regression": {"model_type": "linear_regression"},
        },
        "features": {"quantile_bin": ["rx_ds"], "categorical_onehot": [], "numeric_passthrough": flags,
                     "binary_sum_cols": flags, "n_bins": 4},
        "validation": {"numeric_non_negative_cols": ["rx_ds"], "check_missing_values": False},
        "evaluation": {"evaluate_on_test": False, "quality_gate": {"enabled": False}},
        "run": {"include_proba_if_classification": True},
    }


@pytest.fixture
def pipeline_cfg(tmp_path, monkeypatch, make_patients, flags):
    """Write tiny raw and inference CSVs and make main() read the in-memory config."""
    cfg = make_config(tmp_path, flags)
    make_patients(200).to_csv(cfg["paths"]["raw_data"], index=False)
    make_patients(10, with_target=False, start_id=900).to_csv(cfg["paths"]["inference_data"], index=False)
    monkeypatch.setattr(main_module, "load_config", lambda _path: cfg)
    return cfg


def spy(monkeypatch, name: str) -> Dict[str, int]:
    """Count calls to main_module.<name> while still running the real function."""
    calls = {"n": 0}
    original = getattr(main_module, name)

    def _wrapper(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, name, _wrapper)
    return calls


# --------------------------------------------------------
# 1) Unit tests: split contract
# --------------------------------------------------------
@pytest.fixture
def split_xy():
    X = pd.DataFrame({"f1": range(100), "f2": range(100)})
    y = pd.Series([0] * 80 + [1] * 20, name="target")
    return X, y


def test_unit_split_data_sizes_are_shares_of_the_whole_dataset(split_xy):
    X_train, X_val, X_test, y_train, y_val, y_test = main_module.split_data(*split_xy, 0.10, 0.20, 42, False)
    assert (len(X_train), len(X_val), len(X_test)) == (70, 20, 10)
    assert (len(y_train), len(y_val), len(y_test)) == (70, 20, 10)
    assert not (set(X_train.index) & set(X_val.index) or set(X_train.index) & set(X_test.index))


def test_unit_split_data_stratifies_and_is_deterministic(split_xy):
    first = main_module.split_data(*split_xy, 0.20, 0.20, 7, True)
    second = main_module.split_data(*split_xy, 0.20, 0.20, 7, True)
    assert all(list(a.index) == list(b.index) for a, b in zip(first[:3], second[:3]))
    assert [float(part.mean()) for part in first[3:]] == pytest.approx([0.2, 0.2, 0.2])


@pytest.mark.parametrize("test_size, val_size", [(0.0, 0.2), (0.2, 0.0), (0.5, 0.5)])
def test_unit_split_data_rejects_invalid_sizes(split_xy, test_size, val_size):
    with pytest.raises(ValueError, match="Split sizes"):
        main_module.split_data(*split_xy, test_size, val_size, 42, False)


def test_unit_split_data_falls_back_when_stratification_is_impossible(split_xy):
    X, _ = split_xy
    y = pd.Series([0] * 99 + [1])  # one positive row cannot be stratified
    parts = main_module.split_data(X, y, 0.10, 0.20, 42, True)
    assert sum(len(p) for p in parts[:3]) == 100


def test_unit_split_data_reraises_when_not_stratified():
    X, y = pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1])
    with pytest.raises(ValueError):
        main_module.split_data(X, y, 0.4, 0.4, 42, False)


# --------------------------------------------------------
# 2) Unit tests: settings and gate
# --------------------------------------------------------
def test_unit_read_settings_rejects_bad_config(tmp_path, flags):
    cfg = make_config(tmp_path, flags, problem_type="clustering")
    with pytest.raises(ValueError, match="problem_type"):
        main_module.read_settings(cfg)

    cfg = make_config(tmp_path, [])
    cfg["features"]["quantile_bin"] = []
    with pytest.raises(ValueError, match="at least 1 column"):
        main_module.read_settings(cfg)

    cfg = make_config(tmp_path, flags)
    cfg["training"]["classification"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="must be a mapping"):
        main_module.read_settings(cfg)

    cfg = make_config(tmp_path, flags)
    del cfg["run"]
    with pytest.raises(ValueError, match="'run'"):
        main_module.read_settings(cfg)


def test_unit_quality_gate_disabled_or_passing_lets_the_model_through():
    assert main_module.apply_quality_gate({"enabled": False}, {"pr_auc": 0.1}) is None
    assert main_module.apply_quality_gate({"enabled": True, "metric": "pr_auc", "min_value": 0.4},
                                          {"pr_auc": 0.6}) is None


def test_unit_build_candidates_rejects_non_numeric_binning_column(tmp_path, flags):
    cfg = make_config(tmp_path, flags)
    settings = main_module.read_settings(cfg)
    X = pd.DataFrame({"rx_ds": ["a", "b"]})
    with pytest.raises(ValueError, match="must be numeric"):
        main_module.build_candidates(cfg, settings, X, pd.Series([0, 1]))


def test_unit_parse_args_reads_config_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["src.main", "--config", str(tmp_path / "c.yaml")])
    assert main_module.parse_args().config == tmp_path / "c.yaml"


# --------------------------------------------------------
# 3) Integration tests: the whole pipeline
# --------------------------------------------------------
def test_integration_main_writes_clean_data_model_and_predictions(pipeline_cfg):
    main_module.main()

    paths = pipeline_cfg["paths"]
    assert Path(paths["processed_data"]).exists()
    assert isinstance(load_model(Path(paths["model_artifact"])), CalibratedClassifierCV)
    preds = pd.read_csv(paths["predictions_artifact"])
    assert list(preds.columns) == ["ID", "prediction", "probability"]
    assert len(preds) == 10


def test_integration_main_evaluates_twice_and_infers_once(pipeline_cfg, monkeypatch):
    evaluate_calls = spy(monkeypatch, "evaluate_model")
    infer_calls = spy(monkeypatch, "run_inference")
    main_module.main()
    assert evaluate_calls["n"] == 2  # before and after calibration
    assert infer_calls["n"] == 1


def test_integration_test_split_is_audited_once_when_enabled(pipeline_cfg, monkeypatch):
    pipeline_cfg["evaluation"]["evaluate_on_test"] = True
    evaluate_calls = spy(monkeypatch, "evaluate_model")
    main_module.main()
    assert evaluate_calls["n"] == 3


def test_integration_quality_gate_failure_stops_before_saving(pipeline_cfg, monkeypatch):
    pipeline_cfg["evaluation"]["quality_gate"] = {"enabled": True, "metric": "pr_auc", "min_value": 1.01}
    artifacts = spy(monkeypatch, "save_and_register")
    with pytest.raises(ValueError, match="Quality gate FAILED"):
        main_module.main()
    assert not Path(pipeline_cfg["paths"]["model_artifact"]).exists()
    assert artifacts["n"] == 0, "no file saved and nothing logged to the W&B registry"


def test_integration_missing_raw_data_fails_fast(pipeline_cfg):
    Path(pipeline_cfg["paths"]["raw_data"]).unlink()
    with pytest.raises(FileNotFoundError):
        main_module.main()


def test_integration_regression_pipeline_without_id_column(pipeline_cfg, tmp_path, flags, make_patients):
    pipeline_cfg.update(make_config(tmp_path, flags, problem_type="regression"))
    pipeline_cfg["wandb"] = {"enabled": False, "log_processed_data": True, "log_predictions": True,
                             "log_predictions_table": True}
    make_patients(10, with_target=False).drop(columns=["ID"]).to_csv(
        pipeline_cfg["paths"]["inference_data"], index=False)

    main_module.main()

    preds = pd.read_csv(pipeline_cfg["paths"]["predictions_artifact"])
    assert list(preds.columns) == ["prediction"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_integration_cli_runs_pipeline_from_a_config_file(tmp_path, monkeypatch, make_patients, flags):
    cfg = make_config(tmp_path, flags)
    make_patients(200).to_csv(cfg["paths"]["raw_data"], index=False)
    make_patients(10, with_target=False).to_csv(cfg["paths"]["inference_data"], index=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["src.main", "--config", str(config_file)])
    runpy.run_module("src.main", run_name="__main__")

    assert Path(cfg["paths"]["predictions_artifact"]).exists()
