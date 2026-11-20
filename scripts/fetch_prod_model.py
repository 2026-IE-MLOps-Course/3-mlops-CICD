# scripts/fetch_prod_model.py
"""
Resolve the W&B alias "prod" to a concrete model version and download it

Why this script exists
- "Promote with a label, ship with a release": a person moves the alias prod in W&B,
  then the release workflow (deploy.yml) runs this script and bakes THAT version into the image
- The alias moves; the version it points at (for example opioid_classification_model:v7) does not.
  The concrete version is printed, written to $GITHUB_OUTPUT and passed to docker build as MODEL_VERSION

Usage
- python scripts/fetch_prod_model.py                 (reads config.yaml and WANDB_ENTITY)
- python scripts/fetch_prod_model.py --alias staging --output /tmp/model.joblib

Needs
- WANDB_API_KEY (read by wandb) and WANDB_ENTITY in the environment (.env locally, secrets/variables in CI)
- Read-only: it never logs, tags or moves anything in W&B

Dependencies: Python standard library, PyYAML and the registry client (wandb). The registry logic itself
lives in src/registry.py, shared with the API (MODEL_SOURCE=wandb), so both resolve "prod" the same way
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_github_output(version: str, github_output: Optional[str]) -> None:
    """
    Inputs: version and the $GITHUB_OUTPUT file path (None outside GitHub Actions)
    Outputs: None. Appends MODEL_VERSION=<version> so later workflow steps can read it
    """
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"MODEL_VERSION={version}\n")


def main(argv: Optional[List[str]] = None, api: Any = None) -> str:
    """
    Inputs
    - argv: command line arguments (None = sys.argv)
    - api: injected registry client (tests); None lets src/registry.py create it

    Outputs
    - The concrete model version (also printed as MODEL_VERSION=...)
    """
    parser = argparse.ArgumentParser(description="Download the model version that carries an alias")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--alias", default="prod")
    parser.add_argument("--output", type=Path, default=None, help="default: paths.model_artifact")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    output = args.output or REPO_ROOT / str((cfg.get("paths") or {}).get("model_artifact") or "")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))  # "python scripts/fetch_prod_model.py" runs with scripts/ on the path
    from src import registry  # shared with the API, so both resolve the alias the same way

    version = registry.download_by_alias(cfg, args.alias, output, api=api)
    print(f"Resolved alias {args.alias} -> {version} | saved to {output}")
    print(f"MODEL_VERSION={version}")
    write_github_output(version, os.getenv("GITHUB_OUTPUT"))
    return version


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        sys.exit(f"FAILED: {error}")
