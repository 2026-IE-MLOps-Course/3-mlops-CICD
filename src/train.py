# src/train.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulate training so models are reproducible and swappable without rewiring the pipeline.
- Responsibility (separation of concerns): Combines the feature recipe and algorithm into a Pipeline.
- Pipeline contract: Inputs are train split, problem type, and preprocessor. Output is a fully fitted Pipeline artifact.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str
) -> Pipeline:
    """
    Inputs:
    - X_train: Training features
    - y_train: Training target
    - preprocessor: The configured ColumnTransformer recipe
    - problem_type: "regression" or "classification"
    Outputs:
    - pipeline: Trained scikit-learn Pipeline object
    """
    print(
        f"[train.train_model] Training model pipeline for problem_type={problem_type}")

    pt = (problem_type or "").strip().lower()

    if pt == "classification":
        model = LogisticRegression(max_iter=500)
    elif pt == "regression":
        model = LinearRegression()
    else:
        # Actionable error message
        raise ValueError(
            f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'.")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline
