# tests/test_clean_data.py
"""
Educational Goal:
- One cleaner for training and inference: headers are standardised in both modes,
  rows are only filtered in training mode (every inference row must get a prediction)
"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe
from src.validate import validate_dataframe


def test_unit_training_mode_drops_duplicates_and_missing_target():
    df = pd.DataFrame({" OD ": [1, 1, np.nan, 0], "rx ds": [5, 5, 3, 2]})
    out = clean_dataframe(df, target_column="OD")
    assert list(out.columns) == ["OD", "rx_ds"]
    assert len(out) == 2
    assert list(out.index) == [0, 1]
    assert list(df.columns) == [" OD ", "rx ds"], "the input must never be modified"


def test_unit_inference_mode_keeps_every_row_including_duplicates():
    df = pd.DataFrame({"ID": ["a", "a"], "rx_ds": [1.0, 1.0]})
    out = clean_dataframe(df, target_column=None)
    assert len(out) == 2


def test_unit_target_lookup_forgives_case_drift():
    df = pd.DataFrame({"od": [1, 0], "rx_ds": [1, 2]})
    assert len(clean_dataframe(df, target_column="OD")) == 2


def test_unit_missing_target_raises():
    with pytest.raises(ValueError, match="missing after cleaning"):
        clean_dataframe(pd.DataFrame({"x": [1]}), target_column="OD")


def test_unit_blank_target_name_raises():
    with pytest.raises(ValueError, match="empty after standardization"):
        clean_dataframe(pd.DataFrame({"x": [1]}), target_column="   ")


def test_unit_rejects_none_and_non_dataframe():
    with pytest.raises(ValueError, match="df_raw is None"):
        clean_dataframe(None, target_column="OD")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        clean_dataframe([1, 2, 3], target_column="OD")


def test_integration_cleaned_training_data_passes_validation(make_patients, flags):
    df = make_patients(30)
    df.columns = [f" {c} " for c in df.columns]  # messy headers from an export
    out = clean_dataframe(df, target_column="OD")
    assert validate_dataframe(out, ["OD", "rx_ds"] + flags, "OD", target_allowed_values=[0, 1]) is True
