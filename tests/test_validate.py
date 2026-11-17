# tests/test_validate.py
"""
Educational Goal:
- Every validation rule has a test that triggers it, so a silent regression is impossible
- Validation never fixes data: it returns True or raises
"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe
from src.validate import validate_dataframe


@pytest.fixture
def good_df() -> pd.DataFrame:
    return pd.DataFrame({"OD": [0, 1, 0], "rx_ds": [1.0, 2.0, 0.0], "A": [1, 0, 1]})


def test_unit_valid_dataframe_returns_true(good_df):
    assert validate_dataframe(good_df, ["OD", "rx_ds", "A"], "OD", target_allowed_values=[0, 1],
                              numeric_non_negative_cols=["rx_ds"]) is True


def test_unit_target_without_allowed_values_accepts_any_label(good_df):
    good_df["OD"] = [0.5, 2.0, 3.0]
    assert validate_dataframe(good_df, ["OD"], "OD") is True


@pytest.mark.parametrize("bad, error", [(None, ValueError), ([1, 2], TypeError), (pd.DataFrame(), ValueError)])
def test_unit_rejects_none_wrong_type_and_empty(bad, error):
    with pytest.raises(error):
        validate_dataframe(bad, ["OD"])


def test_unit_rejects_empty_required_columns(good_df):
    with pytest.raises(ValueError, match="required_columns is empty"):
        validate_dataframe(good_df, [])


def test_unit_rejects_missing_columns_after_name_standardisation(good_df):
    assert validate_dataframe(good_df, [" rx ds "]) is True  # same rule as clean_data.py headers
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(good_df, ["rx_ds", "B"])


def test_unit_missing_values_policy(good_df):
    good_df.loc[0, "A"] = np.nan
    assert validate_dataframe(good_df, ["A"], check_missing_values=False) is True
    with pytest.raises(ValueError, match="Missing values found"):
        validate_dataframe(good_df, ["A"], check_missing_values=True)


def test_unit_target_rules(good_df):
    with pytest.raises(ValueError, match="not found"):
        validate_dataframe(good_df, ["A"], "target")
    with pytest.raises(ValueError, match="invalid values"):
        validate_dataframe(good_df.assign(OD=[0, 1, 2]), ["OD"], "OD", target_allowed_values=[0, 1])
    with pytest.raises(ValueError, match="contains missing values"):
        validate_dataframe(good_df.assign(OD=[0, np.nan, 1]), ["A"], "OD")


def test_unit_non_negative_rules(good_df):
    with pytest.raises(ValueError, match="missing column"):
        validate_dataframe(good_df, ["A"], numeric_non_negative_cols=["Z"])
    with pytest.raises(TypeError, match="not numeric"):
        validate_dataframe(good_df.assign(rx_ds=["a", "b", "c"]), ["A"], numeric_non_negative_cols=["rx_ds"])
    with pytest.raises(ValueError, match="negative values"):
        validate_dataframe(good_df.assign(rx_ds=[1.0, -1.0, 0.0]), ["A"], numeric_non_negative_cols=["rx_ds"])


def test_integration_inference_rows_from_cleaner_pass_without_target(make_patients, flags):
    df = clean_dataframe(make_patients(5, with_target=False), target_column=None)
    assert validate_dataframe(df, flags + ["rx_ds"], numeric_non_negative_cols=["rx_ds"]) is True
