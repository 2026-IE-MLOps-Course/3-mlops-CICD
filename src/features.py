# src/features.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate stateful feature engineering rules (like binning or scaling) from the execution of those rules.
- Responsibility (separation of concerns): Define the preprocessing recipe as a ColumnTransformer. No file I/O, no .fit() calls here.
- Pipeline contract: Inputs are configuration lists. Output is a scikit-learn ColumnTransformer.

Why this prevents leakage:
- The recipe is fitted ONLY inside `pipeline.fit(X_train, y_train)` inside train.py.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Build a preprocessing recipe safely.
    """
    print("[features.get_feature_preprocessor] Building feature recipe from configuration")

    if n_bins < 2:
        raise ValueError("Fatal: n_bins must be >= 2 for quantile binning.")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    if not (quantile_bin_cols or numeric_passthrough_cols or categorical_onehot_cols):
        raise ValueError(
            "Fatal: No feature columns configured for the preprocessor.")

    transformers = []

    if quantile_bin_cols:
        quantile_binner = KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="quantile",
        )
        transformers.append(
            ("quantile_bins", quantile_binner, quantile_bin_cols))

    if categorical_onehot_cols:
        try:
            onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat_ohe", onehot, categorical_onehot_cols))

    if numeric_passthrough_cols:
        transformers.append(
            ("num_pass", "passthrough", numeric_passthrough_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
