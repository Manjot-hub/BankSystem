$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if (-not (Test-Path "data/synthetic/processed_chunks.json")) {
    python src/bank_chatbot/data/ingestion.py
}

$port = if ($env:PORT) { $env:PORT } else { "8000" }
python -m uvicorn src.bank_chatbot.api.server:app --host 0.0.0.0 --port $port
