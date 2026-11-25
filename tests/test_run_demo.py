"""Exercise real DVC transfers against local storage; never contact an external service."""

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from dvc.exceptions import DvcException
from dvc.repo import Repo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_demo.py"
spec = importlib.util.spec_from_file_location("run_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Create a standalone DVC project with two known CSVs and a local remote."""
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.delenv("DAGSHUB_USER", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    remote = tmp_path / "remote"
    with Repo.init(str(root), no_scm=True) as repo:
        for pointer in demo.DATA_POINTERS:
            csv = root / pointer.removesuffix(".dvc")
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_bytes(b"ID,value\nP001,1\n")
            repo.add(str(csv))
        with repo.config.edit() as config:
            config["core"]["remote"] = "dagshub"
            config["remote"] = {"dagshub": {"url": str(remote)}}
    return root, remote


def test_integration_publish_rebuilds_cache_uploads_and_verifies_a_fresh_download(project, capsys):
    root, remote = project
    pointers = {name: (root / name).read_bytes() for name in demo.DATA_POINTERS}
    shutil.rmtree(root / ".dvc" / "cache")

    demo.publish_data(root, prompt=False)

    assert list(remote.rglob("*"))
    assert {name: (root / name).read_bytes() for name in demo.DATA_POINTERS} == pointers
    assert "both CSVs downloaded into an empty cache" in capsys.readouterr().out
    with Repo(str(root)) as repo:
        assert repo.status(cloud=True) == {}


def test_integration_fresh_download_detects_missing_remote_despite_populated_local_cache(project):
    root, remote = project
    with pytest.raises(DvcException):
        demo.verify_download(root, "dagshub", {"url": str(remote)})


def test_integration_changed_reference_data_is_rejected_before_upload(project):
    root, remote = project
    (root / "data/raw/opioid_raw_data.csv").write_bytes(b"changed data")

    with pytest.raises(ValueError, match="differs from its recorded version"):
        demo.publish_data(root, prompt=False)

    assert not remote.exists()


def test_integration_private_prompt_saves_credentials_only_in_local_config(project, monkeypatch, capsys):
    root, _ = project
    monkeypatch.setattr(demo.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "teacher")
    monkeypatch.setattr(demo.getpass, "getpass", lambda _: "test-only-secret")
    with Repo(str(root)) as repo:
        with repo.config.edit() as config:
            config["remote"]["dagshub"]["url"] = "https://dagshub.com/owner/example.dvc"
        assert demo.configure_auth(repo, root, prompt=True) == "dagshub"
    with Repo(str(root)) as repo:
        assert repo.config["remote"]["dagshub"]["password"] == "test-only-secret"
        assert repo.config["remote"]["dagshub"]["user"] == "teacher"
    assert "test-only-secret" not in (root / ".dvc/config").read_text()
    assert "test-only-secret" not in capsys.readouterr().out
    assert (root / ".dvc/config.local").stat().st_mode & 0o777 == 0o600


def test_integration_missing_credentials_stop_before_upload_or_training(project, monkeypatch):
    root, remote = project
    monkeypatch.setattr(demo, "REPO_ROOT", root)
    with Repo(str(root)) as repo:
        with repo.config.edit() as config:
            config["remote"]["dagshub"]["url"] = "https://dagshub.com/owner/example.dvc"
    training = []
    monkeypatch.setattr(demo.subprocess, "run", lambda *args, **kwargs: training.append(args))

    with pytest.raises(ValueError, match="write token is missing"):
        demo.main(["--no-prompt"])

    assert training == []
    assert not (root / ".dvc/config.local").exists()
    assert not remote.exists()


@pytest.mark.parametrize("data_only", [False, True])
def test_integration_demo_trains_once_after_upload_or_stops_after_data_verification(
        project, monkeypatch, data_only):
    root, _ = project
    monkeypatch.setattr(demo, "REPO_ROOT", root)
    calls = []

    def training(command, *, cwd, check):
        with Repo(str(root)) as repo:
            assert repo.status(cloud=True) == {}
        calls.append((command, cwd, check))

    monkeypatch.setattr(demo.subprocess, "run", training)
    demo.main(["--data-only", "--no-prompt"] if data_only else ["--no-prompt"])

    assert len(calls) == (0 if data_only else 1)
    if calls:
        assert calls[0] == ([demo.sys.executable, "-m", "dvc", "repro", "--force"], root, True)
