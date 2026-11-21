# src/registry.py
"""
Educational Goal:
- Why this module exists in an MLOps system: ONE place reads the model registry, so the local API
  (MODEL_SOURCE=wandb) and the release workflow (scripts/fetch_prod_model.py) resolve "prod" the same way
- Responsibility (separation of concerns): resolve an alias to a concrete model version and download
  that version's model file. Read-only: it never logs, tags or moves anything
- Writing to the tracker (runs, metrics, model versions) lives in src/tracking.py

Backend contract (swap Weights & Biases, W&B, for another registry by replacing this file only)
- SOURCE_NAME: the MODEL_SOURCE value that makes the API load from the registry
- download_by_alias(cfg, alias, output, api) -> "<model name>:<concrete version>"
- Standard library only at import time: deploy.yml runs the release script with just the registry client installed

Teaching note
- The alias (prod) moves; the version it resolves to (for example opioid_classification_model:v7) does not.
  Always report the concrete version, never the alias
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

SOURCE_NAME = "wandb"


def registry_settings(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Inputs
    - cfg: config.yaml as a dictionary

    Outputs
    - {"project", "model_name", "model_file"}: W&B project, model artifact name, file name inside the version

    Why: the registry names live in config.yaml, the same place main.py reads them from
    """
    wandb_cfg = cfg.get("wandb") or {}
    settings = {
        "project": str(wandb_cfg.get("project") or ""),
        "model_name": str(wandb_cfg.get("model_artifact_name") or ""),
        "model_file": Path(str((cfg.get("paths") or {}).get("model_artifact") or "")).name,
    }
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise ValueError(f"config.yaml: missing wandb.project, wandb.model_artifact_name or "
                         f"paths.model_artifact ({missing})")
    return settings


def artifact_ref(entity: str, project: str, model_name: str, alias: str) -> str:
    """
    Inputs: W&B entity, project, model artifact name and alias
    Outputs: the full reference, e.g. "team/opioid-risk-classification/opioid_classification_model:prod"
    """
    if not entity:
        raise ValueError("WANDB_ENTITY is not set (.env locally, a repository variable in GitHub Actions)")
    return f"{entity}/{project}/{model_name}:{alias}"


def download_by_alias(cfg: Dict[str, Any], alias: Optional[str], output: Path, api: Any = None) -> str:
    """
    Resolve an alias to a concrete model version and copy its model file to output

    Inputs
    - cfg: config.yaml as a dictionary
    - alias: for example "prod"; None reads WANDB_MODEL_ALIAS (default "prod")
    - output: where to put the model file
    - api: injected registry client (tests); None creates wandb.Api(), which reads WANDB_API_KEY

    Outputs
    - The concrete version, e.g. "opioid_classification_model:v7"

    Raises
    - ValueError for missing settings; SystemExit with a clear message when no version carries
      the alias or the version has no model file
    """
    settings = registry_settings(cfg)
    alias = alias or os.getenv("WANDB_MODEL_ALIAS", "prod")
    ref = artifact_ref(os.getenv("WANDB_ENTITY", ""), settings["project"], settings["model_name"], alias)

    if api is None:
        import wandb  # imported here so this module loads without W&B (release script, tests)

        api = wandb.Api()

    try:
        artifact = api.artifact(ref)
    except Exception as exc:  # wandb raises CommError for an unknown alias; the message is what matters
        raise SystemExit(f"No model version carries '{alias}' ({ref}). "
                         f"Promote a gated version in W&B first. Details: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(artifact.download(root=tmp)) / settings["model_file"]
        if not source.is_file():
            raise SystemExit(f"{ref} has no file named {settings['model_file']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)

    return f"{settings['model_name']}:{artifact.version}"
