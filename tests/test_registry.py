# tests/test_registry.py
"""
Educational Goal:
- The registry module resolves an alias (prod) to a CONCRETE version and downloads that model file;
  the API (MODEL_SOURCE=wandb) and the release script both rely on it
- It fails loudly when settings are missing or nobody promoted a version
- A fake registry client replaces the network: no account, no secrets
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src import registry

CFG = {
    "paths": {"model_artifact": "models/model.joblib"},
    "wandb": {"project": "opioid-risk-classification", "model_artifact_name": "opioid_classification_model"},
}


class FakeApi:
    """Serves one model version for one alias, like the W&B registry."""

    def __init__(self, version="v7", alias="prod", files=("model.joblib",)):
        self.version, self.alias, self.files, self.requested = version, alias, files, []

    def artifact(self, ref):
        self.requested.append(ref)
        if not ref.endswith(f":{self.alias}"):
            raise RuntimeError("artifact not found")

        def download(root):
            for name in self.files:
                (Path(root) / name).write_text("model-bytes", encoding="utf-8")
            return root

        return SimpleNamespace(version=self.version, download=download)


def test_unit_settings_reject_incomplete_config():
    assert registry.registry_settings(CFG) == {"project": "opioid-risk-classification",
                                               "model_name": "opioid_classification_model",
                                               "model_file": "model.joblib"}
    with pytest.raises(ValueError, match="model_artifact_name"):
        registry.registry_settings({"wandb": {"project": "p"}})


def test_unit_artifact_ref_needs_entity():
    assert registry.artifact_ref("team", "proj", "model", "prod") == "team/proj/model:prod"
    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        registry.artifact_ref("", "proj", "model", "prod")


def test_unit_download_fails_clearly_without_promoted_version_or_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "team")
    with pytest.raises(SystemExit, match="Promote a gated version"):
        registry.download_by_alias(CFG, "prod", tmp_path / "m.joblib", api=FakeApi(alias="staging"))
    with pytest.raises(SystemExit, match="no file named"):
        registry.download_by_alias(CFG, "prod", tmp_path / "m.joblib", api=FakeApi(files=()))


def test_integration_download_resolves_alias_from_env_to_concrete_version(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_MODEL_ALIAS", "prod")
    api = FakeApi(version="v7")

    import wandb
    monkeypatch.setattr(wandb, "Api", lambda: api)  # the default client (api=None) is wandb.Api()

    output = tmp_path / "baked" / "model.joblib"
    version = registry.download_by_alias(CFG, None, output)

    assert version == "opioid_classification_model:v7"
    assert api.requested == ["team/opioid-risk-classification/opioid_classification_model:prod"]
    assert output.read_text(encoding="utf-8") == "model-bytes"
