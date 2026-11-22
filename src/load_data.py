# src/load_data.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, predictable entry point for raw data ingestion
- Responsibility (separation of concerns): Only loads raw data, no cleaning, validation, training, or evaluation
- Pipeline contract (inputs and outputs): Input is a raw CSV path, output is a raw DataFrame

"""

import logging
from pathlib import Path
import pandas as pd

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load the raw training or inference CSV and fail fast on a missing, folder or empty path

    Inputs:
    - raw_data_path: Path to the raw CSV file
    Outputs:
    - df_raw: DataFrame loaded from raw_data_path
    Why this contract matters for reliable ML delivery:
    - A stable ingestion interface prevents downstream rewiring when data sources evolve
    """
    logger.info("Loading raw data from %s", raw_data_path)

    # 1) Pipeline Guardrail: Missing Data Dependency
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Ingestion Error: The raw data file was not found at {raw_data_path}. "
            "Run dvc pull to download the versioned data, or check paths.raw_data in config.yaml."
        )

    # 2) Pipeline Guardrail: Not a File
    if not raw_data_path.is_file():
        raise ValueError(
            f"Ingestion Error: {raw_data_path} is a directory, not a file. "
            "Check paths.raw_data in config.yaml"
        )

    # 3) Execute the load via Utility
    df_raw = load_csv(raw_data_path)

    # 4) Pipeline Guardrail: Empty Data
    if df_raw.empty:
        raise ValueError(
            f"Ingestion Error: The file at {raw_data_path} loaded but contains zero rows. "
            "Check your data source export."
        )

    # 5) Observability
    logger.info("Loaded dataframe shape: %s", df_raw.shape)

    return df_raw
