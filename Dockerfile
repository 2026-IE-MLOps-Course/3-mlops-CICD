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
# --------------------------------------------------------

# Pin an exact base image tag: "latest" can change under you and break a build that worked yesterday
FROM continuumio/miniconda3:26.5.3-1

# Set the working directory
WORKDIR /app

# This ensures that logs are flushed immediately, which is crucial
# for real-time monitoring and debugging in production environments like Render
ENV PYTHONUNBUFFERED=1

# This is important in a containerized environment where we want to minimize
# unnecessary file writes and ensure that the application runs with the latest
# code without worrying about stale bytecode files. It also helps to reduce the
# image size by not generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python can resolve absolute imports like `src.api`
# Crucial for the FastAPI application to run correctly, especially when using a
# modular project structure
ENV PYTHONPATH=/app

# Copy the pre-compiled lockfile first to leverage Docker layer caching.
# This file was generated locally using: conda-lock -p linux-64 -f environment.yml
COPY conda-lock.yml .

# 1. Install conda-lock itself into the container
# 2. Use conda-lock to install the exact frozen dependencies
# 3. Install curl explicitly for the Docker HEALTHCHECK
# 4. Clean caches to reduce image size
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

# We set the PATH to include the mlops environment so that when we run commands like `uvicorn`,
# it uses the correct Python environment with all dependencies installed
ENV PATH=/opt/conda/envs/mlops/bin:$PATH

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

# Expose a default port (mostly for local documentation)
EXPOSE 8000

# Tell Docker/Render how to test if the API is actually healthy
# Crucial for production deployments to ensure that the service is running
# and responsive before accepting traffic. It helps with auto-scaling and load balancing.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Start the FastAPI server using uvicorn
# We use 'sh -c' to allow environment variable expansion for Render's dynamic $PORT
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]