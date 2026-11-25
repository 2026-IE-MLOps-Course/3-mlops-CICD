"""Upload the reference CSVs, verify a fresh download, then run the training demo.

Run with the existing mlops environment: python scripts/run_demo.py
Use --data-only to publish and verify the data without training.
DagsHub credentials are saved only in ignored .dvc/config.local. CI continues
to use dvc pull/repro directly and never calls this publishing command.
"""

import argparse
import getpass
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dvc.exceptions import DvcException
from dvc.repo import Repo

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_POINTERS = (
    "data/raw/opioid_raw_data.csv.dvc",
    "data/inference/opioid_infer_01.csv.dvc",
)


def verify_data(root):
    """Require the exact reference bytes before caching or publishing either CSV."""
    for target in DATA_POINTERS:
        pointer = root / target
        output = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
        data = pointer.parent / output["path"]
        if not data.is_file():
            raise FileNotFoundError(f"Missing reference data: {data}. Run dvc pull or restore the supplied CSV.")
        if hashlib.md5(data.read_bytes()).hexdigest() != output["md5"]:
            raise ValueError(f"{data.name} differs from its recorded version. "
                             "Restore the original CSV before publishing.")


def configure_auth(repo, root, prompt):
    """Read existing DVC credentials, or collect and save them without echoing a token."""
    remote = repo.config["core"].get("remote")
    if not remote:
        raise ValueError("No DVC remote is configured in .dvc/config.")
    settings = repo.config["remote"][remote]
    url = urlparse(settings["url"])
    print(f"Data destination: {settings['url']}", flush=True)
    if url.hostname != "dagshub.com":
        return remote

    password = os.getenv("DAGSHUB_TOKEN") or settings.get("password")
    user = os.getenv("DAGSHUB_USER") or settings.get("user") or url.path.strip("/").split("/")[0]
    if not password:
        if not prompt or not sys.stdin.isatty():
            raise ValueError("DagsHub write token is missing. Run this command in your terminal to enter it privately; "
                             "it will be saved in ignored .dvc/config.local.")
        user = input(f"DagsHub username [{user}]: ").strip() or user
        password = getpass.getpass("DagsHub token (hidden): ").strip()
        if not password:
            raise ValueError("No DagsHub token entered; nothing was uploaded.")

    local = root / ".dvc" / "config.local"
    local.touch(mode=0o600, exist_ok=True)
    local.chmod(0o600)
    with repo.config.edit("local") as config:
        credentials = config.setdefault("remote", {}).setdefault(remote, {})
        credentials.update(auth="basic", user=user, password=password)
    return remote


def verify_download(root, remote, settings):
    """Pull into an empty temporary repository, without relying on the user's cache."""
    with tempfile.TemporaryDirectory(prefix="mlops-data-check-") as directory:
        scratch = Path(directory)
        Repo.init(str(scratch), no_scm=True).close()
        for target in DATA_POINTERS:
            destination = scratch / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / target, destination)
        config = {"core": {"remote": remote}, "remote": {remote: dict(settings)}}
        with Repo(str(scratch), config=config) as check:
            check.pull(targets=[str(scratch / target) for target in DATA_POINTERS])
            verify_data(scratch)
    print("Verified: both CSVs downloaded into an empty cache and match their recorded fingerprints.", flush=True)


def publish_data(root, prompt=True):
    """Publish only the two reference datasets and prove their remote copies can be retrieved."""
    verify_data(root)
    with Repo(str(root)) as repo:
        remote = configure_auth(repo, root, prompt)
        for target in DATA_POINTERS:
            repo.commit(target=str(root / target))
        print("Uploading the two reference CSVs...", flush=True)
        repo.push(targets=[str(root / target) for target in DATA_POINTERS], remote=remote)
        verify_download(root, remote, repo.config["remote"][remote])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="upload and verify data without training")
    parser.add_argument("--no-prompt", action="store_true", help="require credentials to be configured already")
    args = parser.parse_args(argv)
    publish_data(REPO_ROOT, prompt=not args.no_prompt)
    if not args.data_only:
        print("Data upload verified. Running training, the quality gate and W&B tracking...", flush=True)
        subprocess.run([sys.executable, "-m", "dvc", "repro", "--force"], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (DvcException, ValueError, OSError, EOFError, subprocess.CalledProcessError) as error:
        sys.exit(f"FAILED: {error}")
