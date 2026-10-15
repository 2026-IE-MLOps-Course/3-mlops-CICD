# tests/test_tracking.py
"""
Educational Goal:
- Prove that experiment tracking is optional: with W&B disabled every helper is a safe no-op
"""

from pathlib import Path

import pandas as pd

from src import tracking


def test_start_run_returns_none_when_disabled():
    assert tracking.start_run({"wandb": {"enabled": False}}) is None
    assert tracking.start_run({}) is None


def test_helpers_are_no_ops_without_a_run(tmp_path: Path):
    f = tmp_path / "model.joblib"
    f.write_text("x", encoding="utf-8")
    tracking.log_metrics(None, {"a": 1})
    tracking.log_table(None, "t", pd.DataFrame({"a": [1]}))
    tracking.log_file_artifact(None, name="m", artifact_type="model", file_path=f, description="d")
    tracking.log_classification_plots(None, {}, pd.DataFrame(), pd.Series(dtype=int), object(), "s")
    tracking.finish_run(None)


def test_wandb_settings_have_safe_defaults():
    assert tracking.wandb_flag({}, "log_predictions") is False
    assert tracking.wandb_setting({"wandb": {"project": "p"}}, "project") == "p"
    assert tracking.wandb_setting({}, "missing", "default") == "default"
