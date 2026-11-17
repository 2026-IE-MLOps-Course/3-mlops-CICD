# tests/test_utils.py
"""
Educational Goal:
- Prove the I/O helpers keep their contract: round trips are lossless and every bad path fails fast
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils import load_csv, load_model, save_csv, save_model


def test_unit_save_csv_creates_folders_and_load_csv_reads_back(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "nested" / "folder" / "out.csv"
    save_csv(df, path)
    pd.testing.assert_frame_equal(load_csv(path), df)


def test_unit_load_csv_rejects_non_path_type():
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_csv("not_a_path.csv")


def test_unit_load_csv_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_csv(tmp_path / "missing.csv")


def test_unit_load_csv_rejects_directory_path(tmp_path: Path):
    with pytest.raises(ValueError, match="not a file"):
        load_csv(tmp_path)


def test_unit_load_csv_wraps_parser_errors(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV Parsing Error"):
        load_csv(empty)


def test_unit_load_model_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Run the training pipeline"):
        load_model(tmp_path / "missing.joblib")


def test_integration_model_roundtrip_predicts_the_same(tmp_path: Path):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(X, y)
    path = tmp_path / "m" / "model.joblib"
    save_model(model, path)
    assert (load_model(path).predict(X) == model.predict(X)).all()
