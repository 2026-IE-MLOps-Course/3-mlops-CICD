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

Dependencies: Python standard library plus wandb (which brings PyYAML)
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_settings(config_path: Path) -> Dict[str, str]:
    """
    Inputs
    - config_path: config.yaml

    Outputs
    - {"project", "artifact_name", "model_file"} where model_file is paths.model_artifact

    Why: the registry names live in config.yaml, the same place main.py reads them from
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    wandb_cfg = cfg.get("wandb") or {}
    settings = {
        "project": str(wandb_cfg.get("project") or ""),
        "artifact_name": str(wandb_cfg.get("model_artifact_name") or ""),
        "model_file": str((cfg.get("paths") or {}).get("model_artifact") or ""),
    }
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise ValueError(f"{config_path}: missing wandb.project, wandb.model_artifact_name or "
                         f"paths.model_artifact ({missing})")
    return settings


def artifact_ref(entity: str, project: str, artifact_name: str, alias: str) -> str:
    """
    Inputs: W&B entity, project, artifact name and alias
    Outputs: the full reference, e.g. "team/opioid-risk-classification/opioid_classification_model:prod"
    """
    if not entity:
        raise ValueError("WANDB_ENTITY is not set (.env locally, a repository variable in GitHub Actions)")
    return f"{entity}/{project}/{artifact_name}:{alias}"


def fetch_model(api: Any, ref: str, model_file_name: str, output: Path) -> str:
    """
    Inputs
    - api: a wandb.Api() object (a fake one in tests)
    - ref: entity/project/name:alias
    - model_file_name: file inside the artifact (model.joblib)
    - output: where to put the model

    Outputs
    - The concrete version, e.g. "opioid_classification_model:v7"

    Raises
    - SystemExit with a clear message when no version carries the alias or the file is missing
    """
    try:
        artifact = api.artifact(ref)
    except Exception as exc:  # wandb raises CommError for an unknown alias; the message is what matters
        raise SystemExit(f"No model version carries '{ref.rsplit(':', 1)[1]}' ({ref}). "
                         f"Promote a gated version in W&B first. Details: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(artifact.download(root=tmp)) / model_file_name
        if not source.is_file():
            raise SystemExit(f"{ref} has no file named {model_file_name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)

    return f"{ref.split('/')[-1].split(':')[0]}:{artifact.version}"


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
    - api: injected W&B API object (tests); None creates wandb.Api()

    Outputs
    - The concrete model version (also printed as MODEL_VERSION=...)
    """
    parser = argparse.ArgumentParser(description="Download the W&B model version that carries an alias")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--alias", default="prod")
    parser.add_argument("--output", type=Path, default=None, help="default: paths.model_artifact")
    args = parser.parse_args(argv)

    settings = read_settings(args.config)
    ref = artifact_ref(os.getenv("WANDB_ENTITY", ""), settings["project"], settings["artifact_name"], args.alias)
    output = args.output or REPO_ROOT / settings["model_file"]

    if api is None:
        import wandb  # imported here so --help and the tests work without W&B credentials

        api = wandb.Api()

    version = fetch_model(api, ref, Path(settings["model_file"]).name, output)
    print(f"Resolved {ref} -> {version} | saved to {output}")
    print(f"MODEL_VERSION={version}")
    write_github_output(version, os.getenv("GITHUB_OUTPUT"))
    return version


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        sys.exit(f"FAILED: {error}")
