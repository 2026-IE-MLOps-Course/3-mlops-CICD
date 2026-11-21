# tests/test_load_data.py
"""
Educational Goal:
- Prove ingestion fails fast with an actionable message (missing file, folder, empty file)
- Tests build tiny CSVs in tmp_path: they never read the real data/ folder
"""

from pathlib import Path

import pytest

from src.load_data import load_raw_data


def test_unit_load_raw_data_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw_data(tmp_path / "missing.csv")


def test_unit_load_raw_data_raises_if_path_is_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="is a directory"):
        load_raw_data(tmp_path)


def test_unit_load_raw_data_raises_on_header_only_csv(tmp_path: Path):
    path = tmp_path / "header_only.csv"
    path.write_text("ID,OD,rx_ds\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero rows"):
        load_raw_data(path)


def test_unit_load_raw_data_raises_on_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_data(path)


def test_integration_load_raw_data_reads_what_save_wrote(tmp_path: Path, make_patients):
    df = make_patients(25)
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    loaded = load_raw_data(path)
    assert loaded.shape == df.shape
    assert list(loaded.columns) == list(df.columns)
