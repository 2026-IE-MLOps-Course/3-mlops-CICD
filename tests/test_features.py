# tests/test_features.py
"""
Educational Goal:
- features.py returns an UNFITTED recipe built from config lists; fitting happens in train.py only
- The derived binary_sum feature is computed inside the pipeline (no training-serving skew)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from src.features import _binary_sum_names_out, _row_sum_numpy, get_feature_preprocessor
from src.train import train_model


def test_unit_raises_on_empty_config():
    with pytest.raises(ValueError, match="No feature columns"):
        get_feature_preprocessor()


def test_unit_raises_on_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        get_feature_preprocessor(quantile_bin_cols=["rx_ds"], n_bins=1)


def test_unit_registers_every_configured_block_unfitted():
    pre = get_feature_preprocessor(
        quantile_bin_cols=["rx_ds"], categorical_onehot_cols=["sex"], numeric_passthrough_cols=["A"],
        binary_sum_cols=["A", "B"], n_bins=3,
    )
    assert isinstance(pre, ColumnTransformer)
    assert [name for name, _, _ in pre.transformers] == ["quantile_bins", "cat_ohe", "binary_sum", "num_scaled"]
    assert pre.remainder == "drop"
    with pytest.raises(Exception):
        check_is_fitted(pre)


def test_unit_only_configured_blocks_are_built():
    pre = get_feature_preprocessor(numeric_passthrough_cols=["A"])
    assert [name for name, _, _ in pre.transformers] == ["num_scaled"]


def test_unit_row_sum_handles_2d_and_1d_inputs():
    assert _row_sum_numpy(np.array([[1, 0, 1], [0, 0, 0]])).tolist() == [[2], [0]]
    assert _row_sum_numpy(np.array([1, 0, 1])).tolist() == [[1], [0], [1]]
    assert _binary_sum_names_out(None, ["A", "B"]).tolist() == ["binary_sum"]


def test_integration_recipe_fits_inside_pipeline_and_names_features():
    X = pd.DataFrame({"rx_ds": [1.0, 5.0, 9.0, 12.0, 30.0, 60.0], "sex": ["f", "m", "f", "m", "f", "m"],
                      "A": [1, 0, 1, 0, 1, 0], "B": [0, 0, 1, 1, 0, 1]})
    y = pd.Series([0, 0, 1, 0, 1, 1])
    pre = get_feature_preprocessor(quantile_bin_cols=["rx_ds"], categorical_onehot_cols=["sex"],
                                   numeric_passthrough_cols=["A"], binary_sum_cols=["A", "B"], n_bins=2)
    pipeline = train_model(X, y, pre, "classification", None, 0)
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert "binary_sum__binary_sum" in names
    assert len(pipeline.predict(X)) == len(X)
