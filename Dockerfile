FROM node:20-bullseye-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# System basics
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Install minimal requirements for the web app (from repo root)
COPY requirements-web.txt ./requirements-web.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements-web.txt

# Copy the project
COPY . .

# Bring in the compiled React UI
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Writable cache and outputs
VOLUME ["/cache/huggingface", "/app/results", "/app/logs"]

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
