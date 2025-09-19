# SwarmAgentic — Docker + Local Quick Start

This guide shows how to clone the repo, set up a local environment, and build/run the Docker images for the backend (FastAPI) and the frontend (React + Nginx).

The commands assume you are in the repository root (the folder that contains `Dockerfile` and `docker-compose.yml`).

## 1) Clone the Repo

```bash
git clone https://github.com/YOUR_GITHUB_ACCOUNT/SwarmAgentic.git
cd SwarmAgentic
```

## 2) Local (no Docker) — Run the Backend Only

Prereqs: Python 3.10+ recommended, pip.

```bash
python -m pip install -r requirements-web.txt
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

Optional environment variables:

- `BERT_MODEL_ID` — HF model repo id or local path (default is a small model).
- `TRAINING_TEXT_PATH` — path to your domain corpus file (e.g., `data/agent_ops_handbook.txt`).
- `OPENAI_API_KEY` — enables LLM‑assisted PSO mutations.
- `TEXT_PSO_*` — pacing controls, e.g., `TEXT_PSO_PAUSE`, `TEXT_PSO_POP`, `TEXT_PSO_ITERS`.

On Windows PowerShell:

```powershell
$env:BERT_MODEL_ID = "prajjwal1/bert-tiny"
$env:TRAINING_TEXT_PATH = "C:\\path\\to\\your_corpus.txt"
$env:OPENAI_API_KEY = "sk-..."  # optional
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## 3) Docker — Backend Only (FastAPI)

Build the image:

```bash
docker build -t swarmagentic-web:latest .
```

Run the container (Linux/macOS):

```bash
docker run --rm -p 8000:8000 \
  -e HF_HOME=/cache/huggingface \
  -e BERT_MODEL_ID=prajjwal1/bert-tiny \
  -v $(pwd)/.cache/huggingface:/cache/huggingface \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  swarmagentic-web:latest
```

Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 `
  -e HF_HOME=/cache/huggingface `
  -e BERT_MODEL_ID=prajjwal1/bert-tiny `
  -v ${PWD}\.cache\huggingface:/cache/huggingface `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\logs:/app/logs `
  swarmagentic-web:latest
```

Open: http://localhost:8000

## 4) Docker Compose — Backend + Frontend

This repo includes `docker-compose.yml` to run the backend (`web`) and the built static frontend (`frontend`).

Build & start:

```bash
docker compose up -d --build web frontend
```

Open:

- Frontend (Nginx): http://localhost:3000
- Backend API: http://localhost:8000

Notes:

- If you see a build error related to a `desktop` service, start just the `web` and `frontend` services as shown above, or comment out the `desktop` section in `docker-compose.yml`.
- The frontend’s Nginx proxies `/api/*` and `/ws/stream` to the backend service by name (`web:8000`).

## 5) Environment Variables (Docker)

You can pass env vars with `-e` (for `docker run`) or set them under `environment:` in `docker-compose.yml`.

Common:

- `BERT_MODEL_ID` — HF model id (e.g., `prajjwal1/bert-tiny`, `textattack/bert-base-uncased-MRPC`)
- `TRAINING_TEXT_PATH` — path inside the container (e.g., `/app/data/agent_ops_handbook.txt`)
- `OPENAI_API_KEY` — optional, for LLM PSO
- `TEXT_PSO_PAUSE` — default pause between iterations (seconds)
- `TEXT_PSO_POP` — population size
- `TEXT_PSO_ITERS` — iterations

## 6) Data & Caches

To avoid re-downloading models and to persist results/logs between runs, the Docker setups mount these directories:

- `./.cache/huggingface` → `/cache/huggingface`
- `./results` → `/app/results`
- `./logs` → `/app/logs`

You can safely delete them if you need a clean slate.

## 7) Troubleshooting

- Docker Desktop must be running (Windows/macOS).
- Run commands from the repo root: it contains `Dockerfile` and `docker-compose.yml`.
- If `docker compose up` complains about a missing path for `desktop`, bring up only `web` and `frontend`.
- If the backend doesn’t start, check logs:
  - `docker compose logs -f web`
- If the frontend can’t reach the API, verify Nginx proxy in `frontend/nginx.conf` and that the backend is healthy.

## 8) Security & Secrets

Never commit secrets. Put your OpenAI key in an environment variable (e.g., set `OPENAI_API_KEY` in your shell or compose) or store it in a local file you **don’t** commit. This repo’s `.gitignore` excludes common caches and secret files (e.g., `OpenAI-APIkey.txt`, `.env`).

