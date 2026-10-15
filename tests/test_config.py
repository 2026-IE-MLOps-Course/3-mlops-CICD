# tests/test_config.py
"""
Educational Goal:
- Prove that bad configuration fails fast with a clear message, before any data is loaded
"""

from pathlib import Path

import pytest

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

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_config_yaml_loads_and_has_required_sections():
    """The committed config.yaml must always be valid: CI catches a broken edit before merge."""
    cfg = load_config(REPO_ROOT / "config.yaml")
    for section in ["paths", "problem", "split", "features", "validation", "training", "logging", "run"]:
        assert isinstance(require_section(cfg, section), dict)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_non_mapping_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dictionary"):
        load_config(bad)


def test_require_helpers_return_clean_values():
    section = {"name": "  OD ", "size": "0.2", "seed": "42", "cols": ["a", " b ", "", None]}
    assert require_str(section, "name") == "OD"
    assert require_float(section, "size") == pytest.approx(0.2)
    assert require_int(section, "seed") == 42
    assert require_list(section, "cols") == ["a", "b"]
    assert require_list(section, "missing") == []


@pytest.mark.parametrize(
    "func, section, key",
    [
        (require_str, {"x": ""}, "x"),
        (require_float, {"x": "abc"}, "x"),
        (require_int, {"x": None}, "x"),
        (require_list, {"x": "not-a-list"}, "x"),
    ],
)
def test_require_helpers_raise_on_bad_values(func, section, key):
    with pytest.raises(ValueError):
        func(section, key)


def test_require_section_missing_raises():
    with pytest.raises(ValueError, match="paths"):
        require_section({}, "paths")


def test_resolve_repo_path_is_relative_to_root(tmp_path):
    assert resolve_repo_path(tmp_path, "data/raw.csv") == tmp_path / "data" / "raw.csv"
    with pytest.raises(ValueError):
        resolve_repo_path(tmp_path, "  ")


def test_get_feature_columns_dedupes_in_order():
    cfg = {"features": {"quantile_bin": ["a"], "numeric_passthrough": ["b", "c"], "binary_sum_cols": ["b", "d"]}}
    assert get_feature_columns(cfg) == ["a", "b", "c", "d"]
