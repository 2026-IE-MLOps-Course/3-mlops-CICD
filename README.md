[![CI](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml)
[![CD](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/deploy.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/deploy.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Opioid Risk Prediction: MLOps Reference Project (Repo 3 of 3: CI/CD)

> Ranking patients by calibrated risk lets care managers reach about three times more high-risk patients per call.

**Author:** Ivan Diaz
**Course:** MLOps, Master in Business Analytics and Data Science (IE University)
**Stage:** Repo 3 of 3. Sessions 10 to 14 (configuration, logging, W&B, FastAPI, Docker, CI/CD, Render)

| Repo | What it teaches | Sessions |
|---|---|---|
| `1-mlops-kickoff-repo` | Project scaffold, Git workflow, business case | 1 to 3 |
| `2-mlops-modularized-repo` | Notebook to tested Python modules | 4 to 9 |
| `3-mlops-CICD` (this repo) | Config, logging, tracking, serving, containers, CI/CD | 10 to 14 |

This repository is a teaching reference: study the engineering pattern and apply the same logic to your own dataset and model.

### How to use this reference

- **Read and compare, do not copy.** Your group project starts from the course skeleton `4-mlops-project-skeleton`, with your own data, target, metric and business case. Copied code from this repo is easy to spot and teaches you nothing
- **Use it to answer "what does good look like?"** For each graded practice, open the matching file here (section 10 maps them), read why it is built that way (module docstrings), then build your own version
- **Run it before you build yours.** The quick start below takes about 10 minutes: watching the gate, the tests and a release work end to end makes your own design decisions easier
- **Follow the order of the sessions:** configuration and logging, then tracking, then serving and Docker, then CI/CD (section 13 gives a reading order)

---

## 1. Business case

### Context

Clinical care managers call patients who receive opioid prescriptions to offer preventive support. They can make a limited number of calls each week, so the order of the call list decides who gets help first. Today the list is not risk-ranked, so most calls reach patients who were never at high risk of the outcome `OD` (opioid use disorder event).

### Problem and baseline

Baseline: an untargeted call list reaches about **18 high-risk patients per 100 calls**, because 18.1 % of the 1000 patients in the dataset have `OD = 1` (measured on the data, 2026-11-17). Every call spent on a low-risk patient is a call not made to a high-risk one. The decision we support is "who do we call this week, and in what order", made by the care manager, not by the model.

### Goal and success metrics

| Type | Metric | Baseline | Target |
|---|---|---|---|
| Business KPI | Outreach yield: true high-risk patients per 100 calls, calling the top 20 % by risk score | 18 per 100 (untargeted list) | at least 40 per 100 |
| ML metric (quality gate) | `pr_auc` on the validation split (`config.yaml > evaluation.quality_gate`) | 0.18 (positive rate, a random ranking) | >= 0.40 (minimum) |
| ML metric (reported) | `roc_auc`, `ece` (calibration error) | not gated | reported in W&B |

Why `pr_auc`: only about 18 % of patients are positive, so accuracy would look good for a model that predicts "no risk" for everyone. PR AUC rewards ranking the rare positives first, which is exactly what the call list needs. Current model, validation: `pr_auc` 0.61, `roc_auc` 0.84, `ece` 0.08. On the untouched test split, calling the top 20 % gives 53 true high-risk patients per 100 calls (59 % of all of them).

### Solution

A calibrated logistic regression scores every patient with a probability of `OD`. Two serving modes share one pipeline: a weekly batch job (`python -m src.main` writes `reports/predictions.csv` for the call list) and a real-time API (`POST /predict`) for a single patient during a consultation. The care manager keeps the final decision. Fallback: if the model is unavailable (`/health` returns 503) or a release fails its smoke test, the team keeps the previous release running (rollback by tag) or returns to the manual, untargeted list. Not for: diagnosis, treatment decisions, autonomous clinical action or causal claims; this is a decision-support teaching example, not a clinical tool.

### Data

Source: a synthetic, de-identified teaching dataset of 1000 patients (owner: the course; no real patient appears in it). Grain: one row per patient. Target `OD` (1 = opioid use disorder event). Features known at prediction time: `rx_ds` (opioid days supplied) and 18 diagnosis and context flags (for example `Low_inc`, `SURG`). The identifier `ID` is kept for traceability and never used as a feature. Split: 70 % train, 15 % validation, 15 % test, stratified on `OD`, seed 42. Personal data: none here; with real patient data the pipeline would need pseudonymised IDs, a data processing agreement and GDPR (General Data Protection Regulation) access controls before any training run. Data versions live on DagsHub through DVC, never in Git.

### Value, cost and risks

Value: at the same weekly call capacity, about three times more high-risk patients are reached (53 versus 18 per 100 calls on the test split). Cost: free tiers for Render, W&B, DagsHub and GitHub Actions; estimated one day of engineering per model update and one care-manager review of the call cut-off per quarter.

| Risk | Mitigation |
|---|---|
| Data quality: missing or negative `rx_ds`, wrong labels, schema drift | `src/validate.py` rejects the run or the request (HTTP 422); imputers handle missing values inside the pipeline |
| Model error: false negatives (a high-risk patient not called) | Gate on `pr_auc` >= 0.40 before a model is saved; care managers can always add a patient by hand |
| Model error: false positives (calls to low-risk patients) | Calibrated probabilities (`ece` reported) let the team set the call cut-off by capacity |
| Bias against an income group (`Low_inc`) | Compare yield by group before each release; open gap below |
| Operational: outage or a bad release | Smoke test before and after deploy; rollback by tag redeploys the previous image (code and model together) |
| Operational: privacy and compliance with real data | Synthetic data only; secrets in `.env` and GitHub Secrets, never in Git or the image |

### Open gaps

- The yield of the current manual list is assumed to be the positive rate; it has not been measured in practice
- The 40 per 100 target is a teaching estimate, to be agreed with the care-manager lead
- No fairness check by `Low_inc` group yet, and no drift monitoring beyond prediction logging in W&B
- Synthetic data: results say nothing about real patients

---

## 2. Live links

| What | Link |
|---|---|
| Live API (Render) | `https://<new-render-service>.onrender.com` (paste the new image-backed service URL; try it at `/docs`) |
| Experiment tracking and model registry (W&B) | https://wandb.ai/idiazl/opioid-risk-classification |
| Data versions (DagsHub) | https://dagshub.com/phidesigner/3-mlops-CICD |
| Business value video | TODO: video link |

---

## 3. Quick start (about 10 minutes)

```bash
# 1. Create and activate the environment (exact versions, same as CI and Docker)
conda-lock install --name mlops conda-lock.yml   # or: conda env create -f environment.yml
conda activate mlops

# 2. Pull the exact data versions (the DagsHub repo is public: no account or token needed to read)
dvc pull

# 3. Add your secrets (never commit .env)
cp .env.example .env            # then edit .env

# 4. Check the code
flake8
pytest --cov=src --cov-branch

# 5. Run the full pipeline: train, quality gate, test audit, save, batch inference
dvc repro                       # or: python -m src.main

# 6. Serve the model locally and test it
uvicorn src.api:app --reload
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

> **Windows:** always run inside the activated environment (`conda activate mlops`, then `python -m src.main`), or use `conda run -n mlops python -m src.main`. Calling the environment's `python.exe` directly, for example from a VS Code interpreter that did not activate the environment, can crash with `0xC06D007F` during model calibration because Windows cannot find the environment's maths libraries.

Outputs of step 5:

- `data/processed/clean.csv` cleaned training data
- `models/model.joblib` the deployable pipeline (preprocessing + model in one file)
- `reports/predictions.csv` batch predictions with patient `ID`
- `logs/pipeline.log` audit trail of every run

If the quality gate fails, the pipeline stops and **no model is saved**. That is the intended behaviour.

**Run this reference in your own accounts** (use Ivan's data for this rehearsal, then build your group project separately):

1. Create your own GitHub repository from this template, if available, or copy the repository into a new one under your account and clone it locally; create both `main` and `dev`, and require the `lint` and `test-and-build` checks on both using the branch protection settings in section 9
2. Follow the quick start above to install the environment and download Ivan's data with `dvc pull`; create your own public DagsHub repository, run `dvc remote modify dagshub url https://dagshub.com/<your-user>/<your-repo>.dvc`, configure your local push credentials as described in section 9, then run `dvc push` and commit `.dvc/config`
3. Copy `.env.example` to `.env` as shown above and enter your own `WANDB_API_KEY` and `WANDB_ENTITY`, with `WANDB_MODE=online`; `.env` configures local runs only, not GitHub Actions or Render
4. Create a project in your own W&B account and set `wandb.project` in `config.yaml` to its name, with `wandb.enabled: true`; run `python -m src.main` at least twice, changing a model setting between runs to produce different model files, compare the gated versions in W&B, and assign `prod` to one that passed the quality gate; use your actual version numbers, which need not be `v2` and `v3`
5. In your GitHub repository, open Settings > Secrets and variables > Actions and fill in the existing section 9 table with your own account values; follow its first-release steps to add the Render settings when the new service exists
6. Follow section 9's first-release steps for the GitHub Release, public GHCR image, Render image-backed service and deploy hook; skip old-service cleanup if you have no old service, and keep all pull requests within your own repository

---

## 4. How the pipeline works

```text
config.yaml + .env
      |
      v
load -> clean -> validate -> split -> feature recipe -> train -> calibrate
                                                                    |
                    batch inference <- save + register <- test audit <- quality gate <- evaluate (validation)
```

| Module | One job |
|---|---|
| `src/config.py` | Read and check `config.yaml` (shared by pipeline, API and notebook) |
| `src/load_data.py` | Load raw CSV, fail fast if missing or empty |
| `src/clean_data.py` | Stateless cleaning (column names). Training only: drop duplicates and missing target |
| `src/validate.py` | Data quality gate (schema, target values, business rules) |
| `src/features.py` | Unfitted feature recipe (`ColumnTransformer`) |
| `src/train.py` | Fit the pipeline on the train split only, optional calibration |
| `src/evaluate.py` | Metrics and the model quality gate |
| `src/infer.py` | Predictions with a fixed output schema |
| `src/tracking.py` | Writes to Weights & Biases (W&B): runs, metrics, plots, artifacts, API prediction batches. Safe no-op when disabled |
| `src/registry.py` | Reads the W&B registry: alias `prod` -> concrete version -> model file. Used by the API and the release |
| `src/logger.py` | One logging setup for the whole process |
| `src/main.py` | Orchestrator: the single entry point `python -m src.main` |
| `src/api.py` | FastAPI service. Reuses clean, validate, infer. No new ML logic |

### Key rules this repo enforces

- **Split before fit:** every step that learns numbers (imputers, bins, scaler, model) is fitted on the train split only
- **Validation for decisions, test for the audit:** the quality gate uses validation, the test split is reported once
- **One artifact:** preprocessing and model are saved together, so training and serving cannot drift apart

### Where each preparation step runs

One question decides it: does the step learn anything from the data?

| Step type | Examples | Where it runs | Training | Inference |
|---|---|---|---|---|
| Stateless rule for every row | Rename headers, align types, units and text case | `clean_dataframe()` in `src/clean_data.py`, called by `main.py` and `api.py` | Yes | Yes |
| Row filter | Drop duplicates, drop rows with a missing target, fix wrong labels | `clean_dataframe(target_column=...)` | Yes | Never: every input row gets a prediction |
| Check, not a transform | Required columns, non-negative `rx_ds` | `src/validate.py` | Yes | Yes |
| Learns from data (stateful) | Impute, quantile bins, scale, one-hot encode, derived `binary_sum`, model | `src/features.py` + `src/train.py`, saved together in `models/model.joblib` | Fit on the train split | Replayed from the file, never refitted |

The pipeline artifact keeps the fitted values the same. Shared functions keep the stateless rules the same. Docker and `conda-lock.yml` keep the library versions the same, so the file loads and behaves as it did in training. You need all three.
- **Config, not code:** paths, features, thresholds and hyperparameters live in `config.yaml`
- **Secrets never in Git:** `.env` locally, Render environment variables and GitHub Secrets in the cloud

---

## 5. Configuration and secrets

`config.yaml` holds every non-secret runtime setting. Change behaviour here, not in code. For example, to test a new feature list, edit `features:` and rerun `python -m src.main`.

`.env` holds secrets and deployment switches. Start from `.env.example`:

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_MODE=online          # online | offline | disabled
MODEL_SOURCE=local         # local | wandb
WANDB_MODEL_ALIAS=prod
```

### Environment files

| File | Who reads it | Purpose |
|---|---|---|
| `environment.yml` | Humans | Short list of top-level packages |
| `conda-lock.yml` | CI and Docker | Exact Linux versions with hashes (no "works on my machine") |

After any change to `environment.yml`, regenerate and commit the lock file:

```bash
conda-lock -p linux-64 -f environment.yml
```

---

## 6. Experiment tracking and model registry (W&B)

`python -m src.main` logs data shapes, validation and test metrics, ROC and PR curves, a calibration table, and three artifacts: the model, the processed data and the predictions. Each run's config also records `lineage.git_commit` and `lineage.raw_data_md5` (from the `.dvc` pointer), so every model version links to its code and its data.

The model is logged **only after the quality gate passes**, so every version in the registry is a gated model.

**Promote with a label, ship with a release:**

1. Open W&B > project `opioid-risk-classification` > Artifacts > `opioid_classification_model`
2. Pick the version you want (check its metrics and `lineage`) and move the alias `prod` to it
3. Publish a GitHub Release from `main` (section 9). `deploy.yml` resolves `prod` to that concrete version (for example `opioid_classification_model:v7`), bakes it into the release image and deploys the image. Moving the alias alone changes nothing in production

**Rollback:** redeploy the previous release (Actions > "CD - Deploy to Render" > Run workflow > tag). Code and model roll back together because they are one image. Then move `prod` back to the version that release carries (its `/health` shows it), so the next release does not ship the rolled-back model again.

Local development can follow the alias instead of a file: set `MODEL_SOURCE=wandb` in `.env` and the API downloads the version that carries `WANDB_MODEL_ALIAS` (default `prod`) at startup. `scripts/fetch_prod_model.py` does the same download on its own (read-only) and prints the concrete version. Both use `src/registry.py`, so they always resolve the alias the same way.

**Where W&B lives:** only in `src/tracking.py` (writes) and `src/registry.py` (reads). `main.py`, `api.py` and `scripts/fetch_prod_model.py` never import `wandb`. To use another tracker with a model registry (for example MLflow), replace those two files, keeping the function names listed at the top of each, then update `environment.yml` and `conda-lock.yml`, the `wandb` section of `config.yaml`, `.env.example`, and the secrets in `deploy.yml`.

---

## 7. Serving with FastAPI

```bash
uvicorn src.api:app --reload
```

- `GET /health` returns 200 only when the model is loaded (readiness). It also returns `model_version` (the concrete registry version, for example `opioid_classification_model:v7`) and `git_commit` (the commit the image was built from)
- `POST /predict` validates the JSON with Pydantic, then runs the same `clean_dataframe`, `validate_dataframe` and `run_inference` as the pipeline. The response carries `model_version` too
- `GET /docs` interactive Swagger page to try requests: the only UI this service ships
- `GET /` points to `/docs`
- Every response carries an `X-Correlation-ID` header, also written to the logs

Example request: [`scripts/sample_request.json`](scripts/sample_request.json)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
```

Error codes you should expect: `422` for a bad payload (missing field, extra field, wrong type, empty body) or a business-rule violation (for example negative `rx_ds`), `503` if the model is not loaded, `500` for an unexpected model error.

---

## 8. Docker

```bash
# Bake the promoted model into a local image, exactly like a release
python scripts/fetch_prod_model.py                      # prints MODEL_VERSION=opioid_classification_model:vN
docker build --build-arg MODEL_VERSION=opioid_classification_model:vN \
             --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t mlops-api:local .
docker run -p 8000:8000 --name mlops-api mlops-api:local
python scripts/smoke_test.py --url http://127.0.0.1:8000 --expect-commit $(git rev-parse HEAD)
```

The Dockerfile has two stages. The builder installs `conda-lock.yml` exactly, then keeps only the packages the API needs (the dependency closure of its imports, at the locked versions). The runtime stage, a slim Debian base, receives that environment plus three explicit copies: `config.yaml`, the serving modules of `src/` and `models/model.joblib`. The allow-list in `.dockerignore` decides which modules: the API and what it imports (`api`, `clean_data`, `validate`, `infer`, `config`, `logger`, `tracking`, `registry`) plus `features.py`, because the saved model's preprocessing calls its functions by module path (it is replayed, never refitted). Training code (`main`, `train`, `load_data`, `utils`, `evaluate`) stays in the repository for local training and CI but is not in the image. The result is about 0.9 GB instead of 4.3 GB, with no conda, training code, tests, notebooks, data or secrets, and it runs as a non-root user. It runs with `MODEL_SOURCE=local` and the baked model; `MODEL_VERSION` and `GIT_COMMIT` are set at build time and reported by `/health`, and the image carries the labels `org.opencontainers.image.revision` and `io.github.mlops.model-version`. Pass `--env-file .env` only if you want prediction logs sent to W&B.

---

## 9. CI/CD

```text
feature/* --PR--> dev --PR--> main --(person moves W&B alias prod, then publishes a GitHub Release)--> deploy.yml
              CI          CI         guard -> CI on the tag -> build image (code + prod model) -> smoke test -> push ghcr.io -> Render -> live smoke test
```

### Data and pipeline versioning: DagsHub, `.dvc` pointers, `dvc.yaml` and `dvc.lock`

The raw and inference CSV files live on DagsHub (`https://dagshub.com/phidesigner/3-mlops-CICD.dvc`). Git stores only their `.dvc` pointer files (`data/raw/opioid_raw_data.csv.dvc` and `data/inference/opioid_infer_01.csv.dvc`); the CSVs themselves are ignored by Git. `dvc.yaml` declares the whole pipeline as one stage: its inputs (`src/`, `config.yaml`, the raw and inference CSV files) and its outputs (`clean.csv`, `model.joblib`, `predictions.csv`). `dvc.lock` records the md5 of each one. Commit `.dvc` pointers and `dvc.lock` with every change to data, settings or code, and any commit tells you exactly which data its model came from.

```bash
dvc pull       # download the exact data versions recorded in this commit
dvc repro      # rerun only if an input changed, then update dvc.lock
dvc status     # which input changed since dvc.lock was written
dvc push       # upload any new data version to DagsHub
dvc dag        # draw the stages
git add data/raw/opioid_raw_data.csv.dvc data/inference/opioid_infer_01.csv.dvc dvc.lock && git commit -m "Record pipeline run"
```

First time: run `dvc pull`, then `dvc repro` (the remote is public for reading). To `dvc push` new data you need write access: configure your DagsHub token locally with `dvc remote modify dagshub --local auth basic`, `... --local user <user>` and `... --local password <token>` (written to `.dvc/config.local`, ignored by Git, never in `.env`). DVC is part of `environment.yml` and `conda-lock.yml`.

> **Do not re-save the CSVs.** `dvc pull` returns the exact bytes recorded in the `.dvc` pointer. Opening a CSV in Excel or a text editor and saving it, or converting its line endings (Windows CRLF vs Linux LF), changes the md5 even if no value changed: `dvc status` then reports modified data and `dvc repro` rewrites the pointer to a version DagsHub does not have. If that happens, restore the original with `dvc checkout`.

### Continuous Integration: `.github/workflows/ci.yml`

Runs on every Pull Request to `dev` or `main` and on every push to `main`:

1. **Lint:** `flake8` (fails in seconds)
2. **Environment:** rebuild from `conda-lock.yml`, with the downloaded packages cached (new cache key when `conda-lock.yml` changes)
3. **Pipeline:** `dvc pull`, then `dvc repro` runs `python -m src.main` with W&B disabled. The model quality gate runs here and its result is written to the run summary. This is a reproducibility check: the model CI trains is never shipped
4. **Tests:** `pytest` with coverage, minimum 90 % branch coverage (`--cov-branch --cov-fail-under=90`)
5. **Evidence:** `logs/pipeline.log`, `reports/predictions.csv`, `coverage.xml` and the `dvc.lock` written by the run are kept for 30 days as the artifact `pipeline-evidence-<commit>`, pass or fail
6. **Docker:** the image must build (it is never pushed from CI)

The workflow token can only read the repository (`permissions: contents: read`), and a new push to a Pull Request cancels its previous run.

### Continuous Deployment: `.github/workflows/deploy.yml`

A person decides **what** ships (moves the W&B alias `prod`) and **when** (publishes a GitHub Release from `main`). On `release: published`:

1. **guard** (`release-guard.yml`): fails unless the tag commit is on `origin/main`
2. **ci**: reruns the full `ci.yml` on the tag
3. **publish-image** (the only job with `packages: write`): `scripts/fetch_prod_model.py` resolves `prod` to a concrete version and downloads it; the image is built with `MODEL_VERSION` and `GIT_COMMIT`, run locally and smoke-tested (`/health` must report the tag commit and that model version, `/predict` must answer), then pushed to `ghcr.io/<owner>/<repo>` as `X.Y.Z`, `X.Y`, `latest` and `sha-<short>`
4. **deploy**: calls the Render deploy hook with `imgURL=ghcr.io/<owner>/<repo>:X.Y.Z`, then runs `scripts/smoke_test.py` against the live URL until `/health` reports the tag commit

The release build disables BuildKit provenance so the first push publishes the image with its repository source label, rather than creating a package from an unlinked attestation first. If a push reports `denied`, check the package's repository link and Actions access: an administrator can connect the package to this repository and grant it write access under **Package settings > Manage Actions access** (or enable inherited repository permissions). The registry error alone does not confirm the cause. After correcting any access issue, re-run the failed jobs. A re-run uses the original release's workflow; changes to `deploy.yml` apply only to releases containing those changes.

### One-time setup

**Render** (New > Web Service > **Existing image**, Free plan):

- Image URL: a fixed release tag such as `ghcr.io/<owner>/<repo>:3.0.0`, never `latest` (after that, the deploy hook passes the exact `X.Y.Z` of each release). The package is public, so Render needs no registry credential (for a private package, add one: a GitHub token with only `read:packages`)
- Settings > Auto-Deploy: **Off** (only releases deploy)
- Health Check Path: `/health`
- Environment variables: `WANDB_API_KEY` and `WANDB_ENTITY`, used only to send prediction logs to W&B. The model is baked into the image, so no `MODEL_SOURCE` or alias settings are needed
- Settings > Deploy Hook: copy the URL

**GitHub** (Settings > Secrets and variables > Actions):

| Name | Kind | Used by | Value |
|---|---|---|---|
| `DAGSHUB_TOKEN` | Secret | `ci.yml` (`dvc pull`) | DagsHub token (read access is enough) |
| `RENDER_DEPLOY_HOOK_URL` | Secret | `deploy.yml` | Render deploy hook URL (treat it like a password) |
| `WANDB_API_KEY` | Secret | `deploy.yml` (`fetch_prod_model.py`) | W&B API key |
| `DAGSHUB_USER` | Variable | `ci.yml` | DagsHub username, for example `phidesigner` |
| `RENDER_SERVICE_URL` | Variable | `deploy.yml` | `https://<your-service>.onrender.com` |
| `WANDB_ENTITY` | Variable | `deploy.yml` | W&B user or team that owns the project |

Pull requests from forks receive no secrets, so work on internal branches (`dev`, `feature/*`).

**First release on an image-backed service** (the service needs an image before it can start, and the first image comes from the first release):

1. Delete the old deploy hook secret: `gh secret delete RENDER_DEPLOY_HOOK_URL` (it belongs to the old service)
2. Publish the release (see "How to release"). guard, ci and publish-image run and push `ghcr.io/<owner>/<repo>:X.Y.Z`; the deploy job stops at "Check required deploy settings", so nothing is deployed
3. Render > New > Web Service > Existing image: `ghcr.io/<owner>/<repo>:X.Y.Z`, with the settings above. Copy its deploy hook
4. Set the new secret and variable: `gh secret set RENDER_DEPLOY_HOOK_URL` and `gh variable set RENDER_SERVICE_URL --body https://<new-service>.onrender.com`
5. Actions > the release run > **Re-run failed jobs**: the deploy job now calls the new hook and smoke-tests the live URL, so the release run ends green
6. Suspend the old service (Render > old service > Settings > Suspend), and update the live link in section 2

**GitHub branch protection** (Settings > Branches, for `main` and `dev`):

- Require a pull request and 1 approval
- Require status checks to pass: `lint` and `test-and-build`

This is a solo reference repo, so it requires 0 approvals on `main` and `dev`; student teams require 1 approval on both.

### How to release

1. Merge `dev` into `main` through a Pull Request (CI must be green)
2. In W&B, move the alias `prod` to the gated version you want to ship
3. Update the changelog below
4. GitHub > Releases > Draft a new release > tag `v3.0.0` on `main` > Publish
5. Watch Actions > "CD - Deploy to Render"; when it is green, `/health` shows the new `git_commit` and `model_version`

**Rollback:** Actions > "CD - Deploy to Render" > Run workflow > tag, for example `v3.0.0`. The workflow skips guard, CI and the image build, redeploys `ghcr.io/<owner>/<repo>:3.0.0` and smoke-tests it. Then move `prod` back in W&B to the version that release carries.

Rollback log (rehearsals and real rollbacks, newest first):

| Date | From | Back to | Why |
|---|---|---|---|
| (none yet) | | | |

Free Render services sleep after inactivity. The first request can take about a minute.

---

## 10. Repository structure

```text
.
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml                 # quality gate for every PR (reused by deploy.yml)
│       ├── deploy.yml             # release: build, test and push one image, deploy it to Render
│       └── release-guard.yml      # a release tag must be on main
├── config.yaml                    # all non-secret settings
├── .env.example                   # list of required secrets (no real values)
├── environment.yml                # human-readable dependencies
├── conda-lock.yml                 # exact dependencies for CI and Docker
├── Dockerfile / .dockerignore     # lean serving image: inference + monitoring modules only
├── .flake8 / pytest.ini           # quality tool settings
├── data/
│   ├── raw/opioid_raw_data.csv.dvc          # pointer in Git; the CSV comes from DagsHub (dvc pull)
│   └── inference/opioid_infer_01.csv.dvc
├── docs/DEMO.md                   # 10-minute class demo script (screenshots in docs/demo/)
├── notebooks/                     # exploration only, imports from src/
├── scripts/
│   ├── fetch_prod_model.py        # alias prod -> concrete version -> models/model.joblib (via src/registry.py)
│   ├── smoke_test.py              # checks a running API (local, Docker, Render)
│   └── sample_request.json
├── src/                           # production code (see section 4)
├── dvc.yaml / dvc.lock            # the pipeline as one DVC stage, with md5 of every input and output
└── tests/                         # test_unit_* and test_integration_* per module, 100 % branch coverage
```

Generated and ignored by Git: `models/`, `reports/`, `logs/`, `wandb/`, `artifacts/`.

---

## 11. Model card

| Item | Value |
|---|---|
| Model | Calibrated logistic regression inside one scikit-learn Pipeline |
| Target | `OD`, opioid use disorder indicator |
| Inputs | `rx_ds` (opioid days supplied, binned), 18 indicator flags, number of flags per patient |
| Output | `prediction` (0 or 1) and `probability` of the positive class |
| Intended use | Rank patients for preventive outreach by care managers; a person makes every decision |
| Training data | 1000 synthetic, de-identified records. 70% train, 15% validation, 15% test, stratified |
| Validation metrics (seed 42) | `pr_auc` 0.61 (gate >= 0.40), `roc_auc` 0.84, `ece` 0.08 after calibration |
| Test audit (seed 42) | `pr_auc` 0.55, `roc_auc` 0.80; top 20 % by score: 53 true high-risk patients per 100 calls |
| Version served | The registry version that carried `prod` when the release was built; `/health` reports it |
| Limitations and risks | Small synthetic dataset, possible bias by income group (`Low_inc`, no fairness check yet), no drift monitoring yet |
| Not for | Diagnosis, treatment decisions, causal interpretation |

---

## 12. Changelog

Format: [Keep a Changelog](https://keepachangelog.com), versions follow [Semantic Versioning](https://semver.org).

### [3.0.0] - 2026-11-17

Breaking changes (update any code or notebook that calls `src` directly):

- `main.three_way_split(X, y, *, ...)` is now `main.split_data(X, y, test_size, val_size, random_state, stratify)` (positional)
- `train_model(X_train, y_train, preprocessor, problem_type, model_params=None, random_state=None)`: new `random_state`, used when the config sets none
- `validate_dataframe(df, required_columns, target_column=None, check_missing_values=True, ...)`: `target_column` moved to third place
- `run_inference(model, X_infer, include_proba=False)`: `include_proba` is positional; the probability column is now `probability` (was `proba`), also in `reports/predictions.csv`
- `load_csv` raises `FileNotFoundError` for a missing file (was `ValueError`)
- Deployment: Render now runs a release image from `ghcr.io` instead of building from the Git commit. A release bakes the W&B `prod` model into one image with the code; rollback redeploys an older image (code and model together)

Added and changed:

- `scripts/fetch_prod_model.py`: resolves the alias `prod` to a concrete version and downloads it; `Dockerfile` takes `MODEL_VERSION` and `GIT_COMMIT` build arguments
- `/health` and `/predict` report the concrete `model_version`; `git_commit` reads `GIT_COMMIT`, then `RENDER_GIT_COMMIT`
- `deploy.yml`: guard (tag on `main`) -> CI on the tag -> build, smoke-test and push the image -> Render deploys that image -> live smoke test; manual run with a tag rolls back
- W&B runs record `lineage.git_commit` and `lineage.raw_data_md5`
- W&B code only in `src/tracking.py` (writes, now also API prediction batches) and the new `src/registry.py` (reads, shared by the API and `fetch_prod_model.py`); helpers `tracking.flag` and `tracking.setting` (were `wandb_flag`, `wandb_setting`); config key `api.log_batch_size` (was `api.wandb_log_batch_size`)
- CI: 90 % branch coverage floor; DVC installed from `conda-lock.yml`; the CI model is no longer uploaded (`gated-model` artifact removed)
- Serving image holds only inference and monitoring code: `.dockerignore` lists the serving modules of `src/` (plus `features.py`, needed to load the saved preprocessing) and the Dockerfile copies `config.yaml`, `src/` and `models/model.joblib` explicitly; `tracking.py` imports `evaluate_calibration` only inside the training-only calibration-table branch
- Removed `GET /ui` and `src/static/index.html`: `/docs` is the only UI
- Tests renamed `test_unit_*` and `test_integration_*`; 100 % branch coverage of `src/`
- Every function at cyclomatic complexity 10 or less; lazy `%s` logging; no linter suppressions
- `data/raw/opioid_raw_data.csv` and `data/inference/opioid_infer_01.csv` are no longer tracked by Git (2.2.0 added the pointers but left the CSVs in Git); notebook outputs cleared

### [2.2.0] - 2026-11-13

- CI: cached conda packages keyed on `conda-lock.yml`, read-only token, cancel superseded Pull Request runs, quality gate result in the run summary, and a `pipeline-evidence-<commit>` artifact (pipeline log, predictions, coverage) kept 30 days
- DVC: moved `data/raw/opioid_raw_data.csv` and `data/inference/opioid_infer_01.csv` out of Git into DagsHub (`https://dagshub.com/phidesigner/3-mlops-CICD.dvc`) via `.dvc` pointer files; `dvc.yaml` runs the whole pipeline as one stage and `dvc.lock` records the md5 of the data, `config.yaml`, `src/` and every output. CI runs `dvc pull` then `dvc repro` and uploads `models/model.joblib` as the `gated-model` artifact for CD container packaging
- CD: every published release also pushes a versioned serving image to GitHub Container Registry
- Actions updated to their Node 24 majors: `actions/checkout@v6`, `actions/setup-python@v6`

### [2.1.1] - 2026-11-10

- Fixed: inference no longer drops duplicate rows, so every input row gets a prediction
- Added a training-serving symmetry test (API and batch path return the same probability) and the README table on where each preparation step runs

### [2.1.0] - 2026-10-22

- Added `GET /ui`: minimal web interface (`src/static/index.html`) to test the deployed model with a CSV upload

### [2.0.0] - 2026-10-09

- Added `deploy.yml`: Render deploys only after a published GitHub Release, followed by an automatic smoke test
- Added model quality gate and a one-time test-split audit to the pipeline
- Added `src/config.py` and `src/tracking.py`. `main.py` is shorter and `api.py` no longer imports training code
- CI now lints, runs the real pipeline, enforces 75% coverage and builds the image
- Feature config now uses each indicator flag (validation ROC AUC 0.60 to 0.84). Test split raised from 5% to 15%
- Synthetic data committed so the project runs right after `git clone`
- Pinned Docker base image and Python 3.12. Added `.env.example`, `.flake8`, Pull Request template

### [1.4.0] - 2026-03-19

- Render deployment access instructions

### [1.3.0] - 2026-03-18

- FastAPI serving, W&B model loading, Docker

### [1.0.0] - 2026-03-10

- Initial modular MLOps teaching reference

---

## 13. Suggested reading order

1. `README.md` and `config.yaml`
2. `src/main.py` (the story), then each module it calls
3. `src/api.py`
4. `tests/`
5. `.github/workflows/ci.yml` and `deploy.yml`

## License

MIT. See `LICENSE`.
