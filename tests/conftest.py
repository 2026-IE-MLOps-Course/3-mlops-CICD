# tests/conftest.py
"""
Shared test setup

Why this file exists
- Tests must be hermetic: no internet, no W&B account, no secrets
- These environment variables are set BEFORE src.api is imported, so a developer's local .env
  (for example MODEL_SOURCE=wandb) cannot change test behaviour
"""

import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["MODEL_SOURCE"] = "local"
