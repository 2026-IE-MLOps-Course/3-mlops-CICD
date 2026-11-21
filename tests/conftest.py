# tests/conftest.py
"""
Shared test setup

Why this file exists
- Tests must be hermetic: no internet, no W&B account, no secrets, no real data/ or models/ files
- These environment variables are set BEFORE src.api is imported, so a developer's local .env
  (for example MODEL_SOURCE=wandb) cannot change test behaviour
- Small synthetic datasets are built here once, so every test file uses the same shape as the real data

Naming convention (graded)
- test_unit_*: one function, one behaviour, tiny inputs
- test_integration_*: several modules working together (for example clean -> validate -> infer)
"""

import os
from typing import Callable, List

import pandas as pd
import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["MODEL_SOURCE"] = "local"
for _name in ("MODEL_VERSION", "GIT_COMMIT", "RENDER_GIT_COMMIT", "WANDB_ENTITY"):
    os.environ.pop(_name, None)

FLAGS: List[str] = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "R", "S", "T",
                    "Low_inc", "SURG"]


@pytest.fixture
def flags() -> List[str]:
    """The 18 indicator columns of the opioid dataset."""
    return list(FLAGS)


@pytest.fixture
def make_patients() -> Callable[..., pd.DataFrame]:
    """
    Factory for synthetic patient rows with the real column names

    Inputs (of the returned function)
    - n: number of rows
    - with_target: add the OD column (training) or not (inference)
    - start_id: first ID value

    Outputs
    - DataFrame: ID, rx_ds, the 18 flags and (optionally) OD; OD depends on rx_ds so models can learn
    """

    def _make(n: int = 120, with_target: bool = True, start_id: int = 0) -> pd.DataFrame:
        df = pd.DataFrame({"ID": [f"P{start_id + i:04d}" for i in range(n)],
                           "rx_ds": [float((i * 7) % 90) for i in range(n)]})
        for j, col in enumerate(FLAGS):
            df[col] = [(i + j) % 2 for i in range(n)]
        if with_target:
            df["OD"] = [int(((i * 7) % 90) > 45 or i % 9 == 0) for i in range(n)]
        return df

    return _make
