# src/config.py
"""
Educational Goal:
- Why this module exists in an MLOps system: One place that reads config.yaml, so the
  training pipeline (main.py), the API (api.py) and the notebook all read settings the same way
- Responsibility (separation of concerns): Load YAML, check required keys and types, resolve paths.
  No data loading, no training, no W&B
- Pipeline contract: Input is a path to config.yaml, output is a plain Python dictionary

Why the API imports from here and not from main.py
- main.py is the training "factory". The API must not depend on training code
- Keeping config helpers here lets the serving container stay small and decoupled
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Inputs
    - config_path: Path to config.yaml

    Outputs
    - cfg: Configuration dictionary

    Why this contract matters for reliable ML delivery
    - Fail fast with a clear message when config.yaml is missing or malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse into a dictionary at the top level")

    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Return a required top-level section, or raise an error that names the missing section."""
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml must contain a top-level '{section}' mapping")
    return value


def require_str(section: Dict[str, Any], key: str) -> str:
    """Return a non-empty string value from a config section."""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: '{key}' must be a non-empty string")
    return value.strip()


def require_float(section: Dict[str, Any], key: str) -> float:
    """Return a numeric value from a config section as float."""
    value = section.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"config.yaml: '{key}' must be a number. Got '{value}'") from e


def require_int(section: Dict[str, Any], key: str) -> int:
    """Return an integer value from a config section."""
    value = section.get(key)
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    """Return a list of non-empty strings. A missing key returns an empty list."""
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """
    Resolve a config path relative to the repo root

    Why this matters
    - The pipeline behaves the same no matter which folder you run it from
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("config.yaml: path values must be non-empty strings")
    return Path(project_root) / relative_path.strip()


def get_feature_columns(cfg: Dict[str, Any]) -> List[str]:
    """
    Return every configured feature column once, in a stable order

    Why this matters
    - Training, batch inference and the API must agree on the same input columns
    - One helper avoids three slightly different copies of the same logic (DRY)
    """
    features_cfg = require_section(cfg, "features")
    cols = (
        require_list(features_cfg, "quantile_bin")
        + require_list(features_cfg, "categorical_onehot")
        + require_list(features_cfg, "numeric_passthrough")
        + require_list(features_cfg, "binary_sum_cols")
    )
    return list(dict.fromkeys(cols))
