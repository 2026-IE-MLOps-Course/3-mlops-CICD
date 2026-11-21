# --------------------------------------------------------
# Serving container only
# Packages the FastAPI inference service, NOT training code.
#
# Release image = code + ONE promoted model ("promote with a label, ship with a release")
#   deploy.yml runs scripts/fetch_prod_model.py (W&B alias prod -> e.g. opioid_classification_model:v7),
#   then builds with:
#     docker build --build-arg MODEL_VERSION=opioid_classification_model:v7 \
#                  --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t mlops-api:local .
#   Rollback = redeploy the previous release image: code and model roll back together
#
# Two stages keep the image small
#   1. builder: conda + conda-lock install the exact environment from conda-lock.yml
#   2. runtime: a slim Debian base receives ONLY the finished environment, the code and the model
#      (no conda, no package cache, no compilers)
# --------------------------------------------------------

# ===== Stage 1: builder =====
# Pin exact base image tags: "latest" can change under you and break a build that worked yesterday
FROM condaforge/miniforge3:25.9.1-0 AS builder

# The lockfile alone, so this slow layer is cached until conda-lock.yml changes
# (generated locally with: conda-lock -p linux-64 -f environment.yml)
COPY conda-lock.yml /tmp/conda-lock.yml

# Install conda-lock, then the exact frozen environment at a fixed prefix
# (conda environments hard-code their path, so the runtime stage uses the same one)
RUN conda install -n base -c conda-forge conda-lock -y && \
    conda-lock install -p /opt/conda/envs/mlops /tmp/conda-lock.yml

# Keep only what the API imports: these runtime roots plus every package they depend on (read from
# conda-meta, so versions stay exactly as locked). Notebook, test and DVC tooling is removed:
# it is needed on laptops and in CI, never in the serving image
ARG RUNTIME_ROOTS="python pandas numpy scikit-learn joblib pyyaml python-dotenv wandb fastapi uvicorn pydantic"
RUN /opt/conda/envs/mlops/bin/python - <<'EOF' > /tmp/unneeded.txt
import glob, json, os
meta = {}
for path in glob.glob("/opt/conda/envs/mlops/conda-meta/*.json"):
    record = json.load(open(path))
    meta[record["name"]] = [dep.split()[0] for dep in record.get("depends", [])]
keep, todo = set(), os.environ["RUNTIME_ROOTS"].split()
while todo:
    name = todo.pop()
    if name in meta and name not in keep:
        keep.add(name)
        todo.extend(meta[name])
print("\n".join(sorted(set(meta) - keep)))
EOF
RUN conda remove -p /opt/conda/envs/mlops --force -y $(cat /tmp/unneeded.txt) && \
    conda clean -afy && \
    cd /opt/conda/envs/mlops && \
    find . -name "__pycache__" -type d -prune -exec rm -rf {} + && \
    find . -name "*.a" -type f -delete && \
    find lib/python3.12/site-packages -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + && \
    rm -rf include share/doc share/man share/jupyter/lab/staging conda-meta/*.json.bak

# ===== Stage 2: runtime =====
FROM debian:12.12-slim

WORKDIR /app

# PYTHONUNBUFFERED: logs reach Render immediately; PYTHONDONTWRITEBYTECODE: no .pyc files at runtime
# PYTHONPATH: "import src.api" works from /app; PATH: the locked environment comes first
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/opt/conda/envs/mlops/bin:$PATH

COPY --from=builder /opt/conda/envs/mlops /opt/conda/envs/mlops

# Copy only what .dockerignore allows: config.yaml, src/ and models/ (the baked model)
COPY . .

# Traceability, baked in at build time: /health reports both values
# MODEL_VERSION: the registry version the model came from; GIT_COMMIT: the commit the code came from
ARG MODEL_VERSION=unknown
ARG GIT_COMMIT=local
ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_COMMIT=${GIT_COMMIT} \
    MODEL_SOURCE=local
LABEL org.opencontainers.image.revision=${GIT_COMMIT} \
      org.opencontainers.image.title="opioid-risk-api" \
      io.github.mlops.model-version=${MODEL_VERSION}

# Fail the build (not the deploy) if the trimmed environment misses anything the API imports
RUN python -c "import src.api"

# Do not run the API as root
RUN useradd --create-home --uid 1000 app
USER app

# Expose a default port (mostly for local documentation; Render sets $PORT)
EXPOSE 8000

# Readiness: /health returns 200 only when the model is loaded (Python, so the slim base needs no curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health', timeout=5)" || exit 1

# Start the FastAPI server using uvicorn
# 'sh -c' expands Render's dynamic $PORT
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
