# tests/test_tracking.py
"""
Educational Goal:
- Prove that experiment tracking is optional: with W&B disabled every helper is a safe no-op
- Prove what IS sent when tracking is on, using a fake run object (no network, no account)
- Prove lineage: the run config records the Git commit and the raw-data md5 from the .dvc pointer
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import tracking


class FakeRun:
    """Records every call the pipeline makes, like a W&B run would."""

    def __init__(self):
        self.logged, self.artifacts, self.summary = [], [], {}
        self.name, self.exit_code = "fake-run", None

    def log(self, data):
        self.logged.append(data)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeArtifact:
    def __init__(self, name, type, description):
        self.name, self.type, self.files = name, type, []

    def add_file(self, path):
        self.files.append(path)


@pytest.fixture
def fake_wandb(monkeypatch):
    """Replace the wandb functions tracking.py calls; returns the dict of wandb.init kwargs."""
    init_kwargs = {}

    def _init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    monkeypatch.setattr(tracking.wandb, "init", _init)
    monkeypatch.setattr(tracking.wandb, "Artifact", FakeArtifact)
    monkeypatch.setattr(tracking.wandb, "Table", lambda dataframe: ("table", len(dataframe)))
    monkeypatch.setattr(tracking.wandb, "plot", SimpleNamespace(
        roc_curve=lambda **kw: "roc", pr_curve=lambda **kw: "pr", confusion_matrix=lambda **kw: "cm"))
    return init_kwargs


def test_unit_helpers_are_no_ops_without_a_run(tmp_path: Path):
    assert tracking.start_run({"wandb": {"enabled": False}}) is None
    assert tracking.start_run({}) is None
    f = tmp_path / "model.joblib"
    f.write_text("x", encoding="utf-8")
    tracking.log_metrics(None, {"a": 1})
    tracking.log_table(None, "t", pd.DataFrame({"a": [1]}))
    tracking.log_file_artifact(None, name="m", artifact_type="model", file_path=f, description="d")
    tracking.log_classification_plots(None, {}, pd.DataFrame(), pd.Series(dtype=int), object(), "s")
    tracking.finish_run(None)


def test_unit_settings_have_safe_defaults():
    assert tracking.wandb_flag({}, "log_predictions") is False
    assert tracking.wandb_flag({"wandb": "not a mapping"}, "log_predictions") is False
    assert tracking.wandb_setting({"wandb": {"project": "p"}}, "project") == "p"
    assert tracking.wandb_setting({}, "missing", "default") == "default"


def test_unit_dvc_pointer_md5_reads_pointer_or_reports_unknown(tmp_path: Path):
    data = tmp_path / "raw.csv"
    assert tracking.dvc_pointer_md5(data) == "unknown"  # no pointer yet
    Path(f"{data}.dvc").write_text("outs:\n- md5: abc123\n  path: raw.csv\n", encoding="utf-8")
    assert tracking.dvc_pointer_md5(data) == "abc123"
    Path(f"{data}.dvc").write_text("- just a list\n", encoding="utf-8")
    assert tracking.dvc_pointer_md5(data) == "unknown"


def test_unit_git_commit_falls_back_to_ci_variable(monkeypatch):
    def _no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(tracking.subprocess, "run", _no_git)
    monkeypatch.setenv("GITHUB_SHA", "ci-sha")
    assert tracking.git_commit() == "ci-sha"


def test_unit_run_lineage_links_commit_and_data(tmp_path: Path):
    (tmp_path / "raw.csv.dvc").write_text("outs:\n- md5: f00d\n", encoding="utf-8")
    lineage = tracking.run_lineage({"paths": {"raw_data": "raw.csv"}}, repo_root=tmp_path)
    assert lineage["raw_data_md5"] == "f00d"
    assert tracking.run_lineage({}, repo_root=tmp_path)["raw_data_md5"] == "unknown"


def test_unit_git_commit_reads_head_with_git(monkeypatch):
    def _fake_git(cmd, **kwargs):
        assert cmd == ["git", "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(tracking.subprocess, "run", _fake_git)
    assert tracking.git_commit() == "abc123"


def test_unit_start_run_requires_a_project(fake_wandb):
    with pytest.raises(ValueError, match="wandb.project"):
        tracking.start_run({"wandb": {"enabled": True, "project": "  "}})


def test_integration_start_run_records_config_lineage_and_options(fake_wandb):
    cfg = {"wandb": {"enabled": True, "project": "p", "name": " ", "tags": ["a", "", None]},
           "paths": {"raw_data": "raw.csv"}}
    run = tracking.start_run(cfg)
    assert isinstance(run, FakeRun)
    assert fake_wandb["project"] == "p" and fake_wandb["name"] is None
    assert fake_wandb["job_type"] == "training-pipeline" and fake_wandb["tags"] == ["a"]
    tracking.start_run({"wandb": {"enabled": True, "project": "p", "tags": "not-a-list"}})
    assert fake_wandb["tags"] is None
    assert set(fake_wandb["config"]["lineage"]) == {"git_commit", "raw_data_md5"}
    assert "lineage" not in cfg, "the caller's config is never mutated"
    assert run.summary["entrypoint"] == "python -m src.main"


def test_integration_logging_helpers_send_metrics_tables_artifacts_and_plots(fake_wandb, tmp_path: Path):
    run = FakeRun()
    tracking.log_metrics(run, {"pr_auc": np.float64(0.6)})
    tracking.log_table(run, "t", pd.DataFrame({"a": [1, 2]}))
    tracking.log_table(run, "empty", pd.DataFrame())
    f = tmp_path / "model.joblib"
    f.write_text("x", encoding="utf-8")
    tracking.log_file_artifact(run, name="m", artifact_type="model", file_path=f, description="d")

    class Model:
        def predict_proba(self, X):
            return np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])

        def predict(self, X):
            return np.array([0, 1, 0, 1])

    cfg = {"wandb": {"log_auc_plots": True, "log_confusion_matrix": True, "log_calibration_table": True,
                     "class_names": ["No event", "Event"]}, "evaluation": {"calibration_bins": 2}}
    tracking.log_classification_plots(run, cfg, pd.DataFrame({"x": range(4)}), pd.Series([0, 1, 0, 1]), Model(), "s")
    tracking.log_classification_plots(run, cfg, pd.DataFrame(), pd.Series(dtype=int), object(), "no_proba")
    n_logged = len(run.logged)
    tracking.log_classification_plots(run, {"wandb": {}}, pd.DataFrame({"x": range(4)}), pd.Series([0, 1, 0, 1]),
                                      Model(), "flags_off")
    assert len(run.logged) == n_logged, "every plot is opt-in through config.yaml"
    tracking.finish_run(run, exit_code=1)

    assert run.logged[0] == {"pr_auc": 0.6} and isinstance(run.logged[0]["pr_auc"], float)
    assert run.logged[1] == {"t": ("table", 2)}
    assert run.artifacts[0].type == "model" and run.artifacts[0].files == [str(f)]
    keys = {k for entry in run.logged for k in entry}
    assert {"plots/roc_curve_val_s", "plots/confusion_matrix_val_s", "tables/calibration_val_s"} <= keys
    assert run.exit_code == 1
