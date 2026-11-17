# src/features.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Define feature engineering rules as a reusable recipe that can be
  fitted safely on training data only
- Responsibility (separation of concerns): Build and return a ColumnTransformer recipe. No file I/O, no .fit()
  calls here
- Pipeline contract (inputs and outputs): Inputs are configuration lists. Output is an unfitted ColumnTransformer

"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _row_sum_numpy(X) -> np.ndarray:
    """
    Inputs
    - X: 2D array like (DataFrame or ndarray) of indicator columns

    Outputs
    - sums: 2D numpy array of shape (n_rows, 1)

    Why this contract matters for reliable ML delivery
    - Derived features must be computed the same way in training and inference
    """
    X_np = np.asarray(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)

    sums = np.sum(X_np, axis=1).reshape(-1, 1)
    return sums


def _binary_sum_names_out(estimator, input_features) -> np.ndarray:
    """
    Provides a stable column name for the derived binary sum feature
    Must be a top level function so the pipeline can be safely pickled and saved to disk
    """
    return np.array(["binary_sum"])


def _quantile_block(cols: List[str], n_bins: int) -> Pipeline:
    """
    Impute -> quantile bin -> scale

    Inputs
    - cols: numeric columns to bucket (used by ColumnTransformer, listed here for readability)
    - n_bins: number of quantile buckets

    Outputs
    - Unfitted Pipeline; the median and the bin edges are learned on training data only
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("qbin", KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
            )),
            ("scale", StandardScaler()),
        ]
    )


def _onehot_block() -> Pipeline:
    """
    Impute -> one hot encode

    Outputs
    - Unfitted Pipeline; handle_unknown="ignore" avoids crashes when new categories appear at inference time
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _binary_sum_block() -> Pipeline:
    """
    Impute -> row sum of indicator flags -> scale

    Outputs
    - Unfitted Pipeline; the derived feature is computed inside the model, so training and serving agree
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("sum", FunctionTransformer(func=_row_sum_numpy, validate=False,
                                        feature_names_out=_binary_sum_names_out)),
            ("scale", StandardScaler()),
        ]
    )


def _numeric_block() -> Pipeline:
    """
    Impute -> scale

    Outputs
    - Unfitted Pipeline for plain numeric features
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    binary_sum_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Inputs
    - quantile_bin_cols: Numeric columns to bucket into quantiles
    - categorical_onehot_cols: Categorical columns to one hot encode
    - numeric_passthrough_cols: Numeric columns to keep as numeric features
    - binary_sum_cols: Indicator columns to sum into one derived feature
    - n_bins: Number of quantile buckets

    Outputs
    - preprocessor: Unfitted ColumnTransformer recipe

    Why this contract matters for reliable ML delivery
    - This function defines the recipe only, fitting happens later on X_train only
    """
    logger.info("Building feature recipe from configuration")

    if n_bins < 2:
        raise ValueError("Fatal: n_bins must be >= 2 for quantile binning")

    # Order is part of the saved artifact: keep it stable
    transformers = []
    if quantile_bin_cols:
        transformers.append(("quantile_bins", _quantile_block(quantile_bin_cols, n_bins), quantile_bin_cols))
    if categorical_onehot_cols:
        transformers.append(("cat_ohe", _onehot_block(), categorical_onehot_cols))
    if binary_sum_cols:
        transformers.append(("binary_sum", _binary_sum_block(), binary_sum_cols))
    if numeric_passthrough_cols:
        transformers.append(("num_scaled", _numeric_block(), numeric_passthrough_cols))

    if not transformers:
        raise ValueError("Fatal: No feature columns configured for the preprocessor")

    # remainder="drop" is a strict gatekeeper
    # Only configured columns enter the model, unexpected columns are ignored rather than silently changing
    # the feature space
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
