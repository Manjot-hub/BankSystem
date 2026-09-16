from fastapi.testclient import TestClient

from bank_chatbot.api import server as server_module
from bank_chatbot.api.rate_limiter import TokenBucketRateLimiter
from bank_chatbot.api.server import app


client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": server_module.settings.AUTH_TOKEN}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint_returns_response():
    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "What is the funds availability policy?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"]
    assert payload["retrieved_count"] > 0


def test_chat_endpoint_blocks_pii():
    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "My email is john.doe@example.com and my SSN is 123-45-6789"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["guardrail_flags"]
    assert payload["retrieved_count"] == 0


def test_chat_endpoint_requires_auth():
    response = client.post(
        "/chat",
        json={"message": "What is the funds availability policy?"},
    )

    assert response.status_code == 401


def test_chat_endpoint_rate_limited(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "rate_limiter",
        TokenBucketRateLimiter(rate_per_minute=60, burst=1),
    )

    first = client.post("/chat", headers=AUTH_HEADERS, json={"message": "What is the funds availability policy?"})
    second = client.post("/chat", headers=AUTH_HEADERS, json={"message": "What is the funds availability policy?"})

    assert first.status_code == 200
    assert second.status_code == 429
