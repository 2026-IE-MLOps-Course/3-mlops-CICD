[![CI](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml)
[![CD](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/deploy.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/deploy.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Opioid Risk Prediction: MLOps Reference Project (Repo 3 of 3: CI/CD)

**Author:** Ivan Diaz
**Course:** MLOps, Master in Business Analytics and Data Science (IE University)
**Stage:** Repo 3 of 3. Sessions 10 to 14 (configuration, logging, W&B, FastAPI, Docker, CI/CD, Render)

| Repo | What it teaches | Sessions |
|---|---|---|
| `1-mlops-kickoff-repo` | Project scaffold, Git workflow, business case | 1 to 3 |
| `2-mlops-modularized-repo` | Notebook to tested Python modules | 4 to 9 |
| `3-mlops-CICD` (this repo) | Config, logging, tracking, serving, containers, CI/CD | 10 to 14 |

This repository is a teaching reference. Study the engineering pattern and apply the same logic to your own dataset and model.

---

## 1. Business objective

Identify patients with higher opioid use disorder risk early enough to support preventive intervention.

- **User:** Clinical care managers who use the probability score to prioritise outreach
- **In scope:** A repeatable, auditable pipeline that produces risk scores, served through an API
- **Out of scope:** Diagnosis, treatment decisions, autonomous clinical action, causal claims

This is a decision-support teaching example. It is not a clinical tool.

## 2. Success metrics

| Type | Metric | Target | Where it is enforced |
|---|---|---|---|
| Business KPI | Outreach yield (true high-risk patients per 100 contacts) | Better than current rule-based list | Business review |
| Technical (primary) | PR AUC (Precision-Recall Area Under the Curve) on validation | ≥ 0.40 (baseline = positive rate ≈ 0.18) | `config.yaml > evaluation.quality_gate` |
| Technical (secondary) | ROC AUC, Expected Calibration Error (ECE) | Reported | W&B run and `logs/pipeline.log` |

Why PR AUC: only about 18% of patients are positive. Accuracy would look good for a model that predicts "no risk" for everyone.

---

## 3. Quick start (about 10 minutes)

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate mlops

# 2. Add your secrets (never commit .env)
cp .env.example .env            # then edit .env

# 3. Check the code
flake8
pytest

# 4. Run the full pipeline: train, quality gate, test audit, save, batch inference
python -m src.main

# 5. Serve the model locally and test it
uvicorn src.api:app --reload
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

> **Windows:** always run inside the activated environment (`conda activate mlops`, then `python -m src.main`), or use `conda run -n mlops python -m src.main`. Calling the environment's `python.exe` directly, for example from a VS Code interpreter that did not activate the environment, can crash with `0xC06D007F` during model calibration because Windows cannot find the environment's maths libraries.

Outputs of step 4:

- `data/processed/clean.csv` cleaned training data
- `models/model.joblib` the deployable pipeline (preprocessing + model in one file)
- `reports/predictions.csv` batch predictions with patient `ID`
- `logs/pipeline.log` audit trail of every run

If the quality gate fails, the pipeline stops and **no model is saved**. That is the intended behaviour.

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
| `src/clean_data.py` | Stateless cleaning (column names, duplicates, missing target) |
| `src/validate.py` | Data quality gate (schema, target values, business rules) |
| `src/features.py` | Unfitted feature recipe (`ColumnTransformer`) |
| `src/train.py` | Fit the pipeline on the train split only, optional calibration |
| `src/evaluate.py` | Metrics and the model quality gate |
| `src/infer.py` | Predictions with a fixed output schema |
| `src/tracking.py` | Weights & Biases (W&B) runs, metrics, artifacts. Safe no-op when disabled |
| `src/logger.py` | One logging setup for the whole process |
| `src/main.py` | Orchestrator: the single entry point `python -m src.main` |
| `src/api.py` | FastAPI service. Reuses clean, validate, infer. No new ML logic |

### Key rules this repo enforces

- **Split before fit:** every step that learns numbers (imputers, bins, scaler, model) is fitted on the train split only
- **Validation for decisions, test for the audit:** the quality gate uses validation, the test split is reported once
- **One artifact:** preprocessing and model are saved together, so training and serving cannot drift apart
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

`python -m src.main` logs data shapes, validation and test metrics, ROC and PR curves, a calibration table, and three artifacts: the model, the processed data and the predictions.

To promote a model to production:

1. Open W&B > your project > Artifacts > `opioid_classification_model`
2. Select the version you want and add the alias `prod`
3. Set `MODEL_SOURCE=wandb` (locally in `.env`, on Render in Environment). The API downloads the `prod` model at startup

Rollback of the model only: move the `prod` alias back to the previous version and restart the Render service.

---

## 7. Serving with FastAPI

```bash
uvicorn src.api:app --reload
```

- `GET /health` returns 200 only when the model is loaded (readiness). It also returns `git_commit`
- `POST /predict` validates the JSON with Pydantic, then runs the same `clean_dataframe`, `validate_dataframe` and `run_inference` as the pipeline
- `GET /docs` interactive Swagger page to try requests
- Every response carries an `X-Correlation-ID` header, also written to the logs

Example request: [`scripts/sample_request.json`](scripts/sample_request.json)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
```

Error codes you should expect: `422` for a bad payload or a business-rule violation (for example negative `rx_ds`), `503` if the model is not loaded.

---

## 8. Docker

```bash
docker build -t mlops-api:latest .
docker run -p 8000:8000 --env-file .env --name mlops-api mlops-api:latest
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

The image contains only `config.yaml`, `src/`, `models/` and the locked environment (see `.dockerignore`). No tests, notebooks, data or secrets.

---

## 9. CI/CD

```text
feature/* --PR--> dev --PR--> main --(human publishes a GitHub Release)--> deploy.yml --> Render
              CI          CI                                              CI again + smoke test
```

### Continuous Integration: `.github/workflows/ci.yml`

Runs on every Pull Request to `dev` or `main` and on every push to `main`:

1. **Lint:** `flake8` (fails in seconds)
2. **Environment:** rebuild from `conda-lock.yml`
3. **Pipeline:** `python -m src.main` with W&B disabled. The model quality gate runs here
4. **Tests:** `pytest` with coverage, minimum 75%
5. **Docker:** the image must build

### Continuous Deployment: `.github/workflows/deploy.yml`

Runs only when a person publishes a GitHub Release (or runs it manually):

1. Reruns the full CI on the released commit
2. Calls the Render deploy hook for that exact commit
3. Runs `scripts/smoke_test.py` until `/health` reports the new commit, then calls `/predict`

### One-time setup

**Render** (New > Web Service > connect this repo, Docker runtime, Free plan):

- Environment variables: `WANDB_API_KEY`, `WANDB_ENTITY`, `MODEL_SOURCE=wandb`, `WANDB_MODEL_ALIAS=prod`
- Health Check Path: `/health`
- Settings > Auto-Deploy: **Off** (only releases deploy)
- Settings > Deploy Hook: copy the URL

**GitHub** (Settings > Secrets and variables > Actions):

- Secret `RENDER_DEPLOY_HOOK_URL` = the deploy hook URL (treat it like a password)
- Variable `RENDER_SERVICE_URL` = `https://<your-service>.onrender.com`

**GitHub branch protection** (Settings > Branches, for `main` and `dev`):

- Require a pull request and 1 approval
- Require status checks to pass: `lint` and `test-and-build`

### How to release

1. Merge `dev` into `main` through a Pull Request (CI must be green)
2. Update the changelog below
3. GitHub > Releases > Draft a new release > tag `v2.1.0` from `main` > Publish
4. Watch Actions > "CD - Deploy to Render"

**Rollback:** Actions > "CD - Deploy to Render" > Run workflow > pick the previous tag.

Free Render services sleep after inactivity. The first request can take about a minute.

---

## 10. Repository structure

```text
.
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml                 # quality gate for every PR
│       └── deploy.yml             # release-gated deploy to Render
├── config.yaml                    # all non-secret settings
├── .env.example                   # list of required secrets (no real values)
├── environment.yml                # human-readable dependencies
├── conda-lock.yml                 # exact dependencies for CI and Docker
├── Dockerfile / .dockerignore     # lean serving image
├── .flake8 / pytest.ini           # quality tool settings
├── data/
│   ├── raw/opioid_raw_data.csv    # synthetic teaching data (committed on purpose)
│   └── inference/opioid_infer_01.csv
├── notebooks/                     # exploration only, imports from src/
├── scripts/
│   ├── smoke_test.py              # checks a running API (local, Docker, Render)
│   └── sample_request.json
├── src/                           # production code (see section 4)
└── tests/                         # one test file per module + API tests
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
| Data | 1000 synthetic, de-identified records. 70% train, 15% validation, 15% test, stratified |
| Validation (seed 42) | PR AUC ≈ 0.61, ROC AUC ≈ 0.84, ECE ≈ 0.08 after calibration |
| Test audit (seed 42) | PR AUC ≈ 0.55, ROC AUC ≈ 0.80 |
| Limitations | Small synthetic dataset, possible bias by income group (`Low_inc`), no drift monitoring yet |
| Not for | Diagnosis, treatment decisions, causal interpretation |

---

## 12. Changelog

Format: [Keep a Changelog](https://keepachangelog.com), versions follow [Semantic Versioning](https://semver.org).

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
