[![CI](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-success)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/2026-IE-MLOps-Course/3-mlops-CICD)

# Opioid Risk Prediction MLOps Project

Author: Ivan Diaz  
Course context: MLOps: From Notebook to Production  
Repository: `2026-IE-MLOps-Course/3-mlops-CICD`

## Overview

This repository shows how to turn a notebook-first machine learning project into a modular, testable, reproducible, and deployable MLOps system.

The project predicts opioid use disorder risk from synthetic healthcare-style records. It includes:

- a modular `src/` pipeline
- centralized configuration in `config.yaml`
- unit tests with `pytest`
- experiment tracking and artifact logging with Weights & Biases
- a FastAPI inference service
- Docker packaging for serving
- Continuous Integration with GitHub Actions
- deployment to Render

This repository is designed as a teaching reference. Students should study the engineering pattern and adapt the same logic to their own project, dataset, and model.

## Business objective

The goal is to identify patients with higher opioid use disorder risk early enough to support preventive intervention.

This project is a decision-support example for teaching MLOps. It is not a clinical decision system, not a diagnostic tool, and not a substitute for medical judgment.

## What this project demonstrates

- moving from notebooks to a `src/` layout
- separating data loading, cleaning, validation, feature engineering, training, evaluation, and inference
- using one config file as the main source of non-secret runtime settings
- saving a single serialized pipeline artifact for consistent training and serving
- exposing model inference through a documented web API
- packaging the service in Docker
- validating quality with automated tests in Continuous Integration
- serving the API on Render
- logging both training artifacts and inference telemetry to Weights & Biases

## Project architecture

The end-to-end workflow is:

1. Load raw data  
2. Clean and standardize columns  
3. Validate required columns and numeric constraints  
4. Build features  
5. Train a classification pipeline  
6. Evaluate model quality  
7. Save the model artifact  
8. Run batch inference  
9. Serve the model through FastAPI locally or on Render

### Main modules

- `src/load_data.py` loads raw data
- `src/clean_data.py` standardizes and cleans the dataframe
- `src/validate.py` applies fail-fast checks before expensive steps run
- `src/features.py` defines feature transformations
- `src/train.py` trains the model pipeline
- `src/evaluate.py` computes evaluation metrics
- `src/infer.py` generates predictions and probabilities
- `src/logger.py` centralizes logging setup
- `src/main.py` orchestrates the full pipeline
- `src/api.py` exposes the model as a FastAPI service

## Relevant repository structure

Only the files and folders below are important for understanding and running the project:

```text
.
├── Dockerfile
├── LICENSE
├── README.md
├── conda-lock.yml
├── config.yaml
├── environment.yml
├── pytest.ini
├── artifacts/
│   └── opioid_classification_model:v1/
│       └── model.joblib
├── data/
│   ├── raw/
│   │   └── opioid_raw_data.csv
│   ├── processed/
│   │   └── clean.csv
│   └── inference/
│       └── opioid_infer_01.csv
├── logs/
│   └── pipeline.log
├── models/
│   └── model.joblib
├── notebooks/
│   ├── 00_opioid_analysis_vLegacy.ipynb
│   └── 01_opioid_analysis_vExp.ipynb
├── reports/
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py
```

## Tech stack

- Python 3.9+
- Conda for environment management
- scikit-learn for model training
- FastAPI and Pydantic for serving and request validation
- Weights & Biases for experiment tracking and model artifacts
- Docker for containerized serving
- GitHub Actions for Continuous Integration
- Render for cloud deployment

## Configuration

`config.yaml` is the single source of truth for non-secret runtime settings.

It currently defines:

- file paths for raw data, processed data, model artifact, inference input, predictions output, and logs
- classification problem settings such as target and identifier columns
- data split parameters
- training settings for logistic regression and probability calibration
- feature groups such as `quantile_bin` and `binary_sum_cols`
- validation rules such as non-negative numeric columns
- evaluation settings
- runtime inference behavior
- Weights & Biases project and artifact settings

Secrets must stay in `.env`, not in `config.yaml`.

Example `.env` file:

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
MODEL_SOURCE=local
WANDB_MODEL_ALIAS=prod
```

## Setup

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate mlops-modul
```

If you want exact locked dependencies, use `conda-lock.yml` as your reproducibility reference.

### 2) Run the tests

```bash
pytest -q
```

### 3) Run the full pipeline

```bash
python -m src.main
```

Expected main outputs:

- `data/processed/clean.csv`
- `models/model.joblib`
- `reports/predictions.csv`
- `logs/pipeline.log`

### 4) Explore the notebook version

```bash
jupyter notebook notebooks/01_opioid_analysis_vExp.ipynb
```

Use the notebooks for exploration, not as the production pipeline entry point.

## FastAPI service

The inference service is defined in `src/api.py`.

### Run locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Local URLs

- Home: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Accessing the Render deployment

Your deployed service base URL is:

```text
https://three-mlops-cicd.onrender.com
```

Use these endpoints:

### 1) Open the API docs in the browser

```text
https://three-mlops-cicd.onrender.com/docs
```

This is the easiest way for students to inspect the schema and test requests.

### 2) Check service health

```text
https://three-mlops-cicd.onrender.com/health
```

A healthy response should indicate that the service is up and the model is loaded.

### 3) Use the home route

```text
https://three-mlops-cicd.onrender.com/
```

The root route is a minimal guidance endpoint that tells users to use `/health` or `/docs`.

## Prediction request format

The deployed API expects a JSON payload with a `records` list. Each record must match the Pydantic contract in `src/api.py`.

### Example request body

```json
{
  "records": [
    {
      "ID": "P001",
      "rx_ds": 12.0,
      "A": 1,
      "B": 0,
      "C": 1,
      "D": 0,
      "E": 1,
      "F": 0,
      "H": 0,
      "I": 1,
      "J": 0,
      "K": 0,
      "L": 1,
      "M": 0,
      "N": 0,
      "R": 1,
      "S": 0,
      "T": 1,
      "Low_inc": 1,
      "SURG": 0
    }
  ]
}
```

### Test with curl against Render

```bash
curl -X POST "https://three-mlops-cicd.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "ID": "P001",
        "rx_ds": 12.0,
        "A": 1,
        "B": 0,
        "C": 1,
        "D": 0,
        "E": 1,
        "F": 0,
        "H": 0,
        "I": 1,
        "J": 0,
        "K": 0,
        "L": 1,
        "M": 0,
        "N": 0,
        "R": 1,
        "S": 0,
        "T": 1,
        "Low_inc": 1,
        "SURG": 0
      }
    ]
  }'
```

### Test with Python

```python
import requests

url = "https://three-mlops-cicd.onrender.com/predict"
payload = {
    "records": [
        {
            "ID": "P001",
            "rx_ds": 12.0,
            "A": 1,
            "B": 0,
            "C": 1,
            "D": 0,
            "E": 1,
            "F": 0,
            "H": 0,
            "I": 1,
            "J": 0,
            "K": 0,
            "L": 1,
            "M": 0,
            "N": 0,
            "R": 1,
            "S": 0,
            "T": 1,
            "Low_inc": 1,
            "SURG": 0,
        }
    ]
}

response = requests.post(url, json=payload, timeout=30)
print(response.status_code)
print(response.json())
```

## Expected API behavior

- `GET /health` returns service health and model version information
- `POST /predict` returns predictions and, when enabled, probabilities
- the API attaches an `X-Correlation-ID` response header for request tracing
- when configured, inference logs are batched and sent to Weights & Biases

## Docker

This repository includes a Dockerfile for serving the API in a container.

### Build the image

```bash
docker build -t mlops-api:latest .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env -e MODEL_SOURCE=wandb --name mlops-api-container mlops-api:latest
```

### Test the containerized API

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Continuous Integration

The repository includes a GitHub Actions workflow under `.github/workflows/ci.yml`.

Its role is to automatically validate code quality and project integrity after code changes.

In practical terms, students should understand that Continuous Integration checks whether the project can still be installed, imported, and tested after each push or pull request.

## Testing strategy

The `tests/` folder covers the major pipeline modules:

- data loading
- cleaning
- validation
- feature logic
- training
- evaluation
- inference
- API behavior
- orchestration behavior in `main.py`

This matters because MLOps is not only about training a model. It is about proving that the full system remains reliable as the codebase evolves.

## Weights & Biases usage

This project uses Weights & Biases for:

- experiment tracking
- metrics logging
- plot logging
- prediction previews
- model artifact versioning
- optional inference telemetry logging from the API

The configuration in `config.yaml` shows that the current project name is `opioid-risk-classification` and the registered model artifact name is `opioid_classification_model`.

## Reproducibility rules in this repo

- use `config.yaml` instead of hardcoding paths and settings across files
- keep secrets in `.env`
- run the orchestrator with `python -m src.main`
- run tests before pushing code
- serialize one deployable model pipeline artifact
- keep notebooks for exploration, not production orchestration

## Brief model card

### Model name

Opioid Risk Prediction Pipeline

### Model type

Binary classification pipeline built with scikit-learn

### Intended use

Support educational demonstration of how a healthcare-style risk model can be trained, evaluated, tracked, packaged, and served using MLOps practices.

### Primary users

Students, instructors, and technical reviewers studying modular MLOps workflows.

### Prediction target

`OD`, the opioid use disorder target column defined in `config.yaml`

### Inputs

Structured tabular features such as `rx_ds`, binary indicator variables, and socioeconomic or clinical proxy features included in the API schema and feature configuration.

### Output

A binary prediction and, when enabled, a probability score.

### Training data

Synthetic or de-identified healthcare-style tabular records stored in `data/raw/opioid_raw_data.csv`.

### Main limitations

- this is a teaching project, not a validated medical product
- performance depends on the representativeness of the source data
- historical patterns may encode bias
- deployment monitoring is lightweight and should not be treated as full production governance

### Out of scope

- diagnosis
- treatment decisions
- autonomous clinical action
- causal interpretation of risk factors

## Changelog

### [Unreleased]

- refine deployment documentation for student handoff
- keep README aligned with the latest repo structure and Render access pattern

### [1.4.0] - 2026-03-19

- added Render deployment access instructions
- documented the live `/docs`, `/health`, and `/predict` usage pattern
- aligned README structure with the final teaching repo tree

### [1.3.0] - 2026-03-18

- added FastAPI serving guidance for local and cloud inference
- documented Weights & Biases based model loading and telemetry concepts
- clarified containerized API execution with Docker

### [1.2.0] - 2026-03-16

- strengthened observability guidance around request tracing and inference logging
- clarified testing coverage across pipeline modules

### [1.1.0] - 2026-03-14

- documented calibration-aware classification workflow
- clarified configuration-driven training and evaluation behavior

### [1.0.0] - 2026-03-10

- initial modular MLOps teaching reference released
- included pipeline orchestration, testing, experiment tracking, and artifact saving

## Recommended student reading order

For students new to the repo, this is the simplest reading order:

1. `README.md`  
2. `config.yaml`  
3. `src/main.py`  
4. `src/load_data.py` and `src/clean_data.py`  
5. `src/validate.py` and `src/features.py`  
6. `src/train.py`, `src/evaluate.py`, and `src/infer.py`  
7. `src/api.py`  
8. `tests/`

## License

This project includes an MIT License. Review `LICENSE` for details.
