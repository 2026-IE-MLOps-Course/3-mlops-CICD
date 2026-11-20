# tests/test_fetch_prod_model.py
"""
Educational Goal:
- The release script resolves the alias prod to a CONCRETE version (through src/registry.py),
  downloads that model and hands the version to the workflow
- The registry logic itself is tested in tests/test_registry.py
- A fake W&B API object replaces the network: no account, no secrets
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_prod_model.py"
spec = importlib.util.spec_from_file_location("fetch_prod_model", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class FakeApi:
    """Serves one artifact version for one alias, like the W&B registry."""

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


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"model_artifact": "out/model.joblib"},
        "wandb": {"project": "opioid-risk-classification", "model_artifact_name": "opioid_classification_model"},
    }), encoding="utf-8")
    return path


def test_integration_main_downloads_model_and_writes_github_output(config_file, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_ENTITY", "team")
    github_output = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    output = tmp_path / "baked" / "model.joblib"
    api = FakeApi(version="v7")

    version = fetch.main(["--config", str(config_file), "--output", str(output)], api=api)

    assert version == "opioid_classification_model:v7"
    assert api.requested == ["team/opioid-risk-classification/opioid_classification_model:prod"]
    assert output.read_text(encoding="utf-8") == "model-bytes"
    assert github_output.read_text(encoding="utf-8") == "MODEL_VERSION=opioid_classification_model:v7\n"
    assert "MODEL_VERSION=opioid_classification_model:v7" in capsys.readouterr().out
