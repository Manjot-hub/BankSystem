# Banking Customer Support Chatbot

A production-style banking support chatbot built with Python, FastAPI, LangGraph, Chroma, and guardrails. The project demonstrates a full RAG pipeline with local fallback embeddings, evaluation gating, API security, and a multi-agent architecture pattern suitable for agentic AI interviews and portfolio work.

## Overview

This project includes:

- Retrieval-augmented generation (RAG) over banking policy and FAQ content
- Local embedding fallback without requiring an OpenAI API key for development
- ChromaDB vector storage with metadata filtering
- LangGraph orchestration for retrieval and generation flow
- Guardrails for PII detection, prompt injection checks, and financial advice blocking
- FastAPI API with auth, rate limiting, metrics, and readiness checks
- Evaluation harness with a golden set and metric-based pass criteria
- Tool-calling pattern for banking actions such as balance checks and transfers
- Multi-agent routing patterns for supervisor and specialized agents

## Tech Stack

- Python 3.11+
- FastAPI
- LangGraph
- LangChain
- ChromaDB
- PostgreSQL-ready config
- Redis-ready config
- OpenTelemetry + Prometheus
- pytest
- sentence-transformers

## Project Structure

```text
.
├── src/
│   └── bank_chatbot/
│       ├── agents/
│       ├── api/
│       ├── config/
│       ├── data/
│       ├── eval/
│       ├── guardrails/
│       ├── rag/
│       └── tools/
├── data/
│   ├── chroma_db/
│   ├── eval/
│   └── synthetic/
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── deploy.ps1
├── requirements.txt
├── pyproject.toml
├── README.md
└── RAILWAY_DEPLOYMENT_GUIDE.md
```

## Local Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Configure environment variables

Copy the example file:

```bash
copy .env.example .env
```

Then update values in `.env` as needed for local development.

## Run the app locally

```bash
python -m uvicorn src.bank_chatbot.api.server:app --host 0.0.0.0 --port 8000
```

## Run tests

```bash
python -m pytest -q
```

## API Endpoints

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /guardrails/status`
- `GET /config`
- `POST /chat`

## Example chat request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_token" \
  -d '{"message":"What is the funds availability policy?"}'
```

## Security Notes

This project includes:

- API key authentication for protected routes
- rate limiting
- prompt injection detection
- PII detection patterns
- output validation guardrails
- local-only secret handling via `.env`

Production deployments should move secrets to a secure secret manager such as Railway variables, AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager.

## Production Deployment Notes

This project is compatible with Railway deployment without Docker. See the guide in [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md) for full Railway setup steps.

## License

This project is for educational and portfolio use.

## Status

The project is currently a strong agentic AI portfolio project with a clean architecture, local fallback support, guardrails, evaluation, and deployment-ready structure.
