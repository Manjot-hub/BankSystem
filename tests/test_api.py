"""
API Endpoint Unit Tests for Banking Chatbot
"""
import pytest
from fastapi.testclient import TestClient
from src.bank_chatbot.api.server import app, rate_limiter, settings

# Secure headers matching APIKeyAuthMiddleware settings
AUTH_HEADERS = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance for testing."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Ensure health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint_requires_auth(client):
    """Ensure protected chat endpoint rejects unauthenticated requests with 401."""
    response = client.post(
        "/chat",
        json={"message": "What is the funds availability policy?"},
    )
    assert response.status_code == 401


def test_chat_endpoint_returns_response(client):
    """Ensure authenticated chat endpoint returns valid RAG response."""
    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "What is the funds availability policy?", "user_id": "usr_000001"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "response" in payload
    assert payload["retrieved_count"] >= 0


def test_chat_endpoint_blocks_pii(client):
    """Ensure input guardrails block prompt containing sensitive PII."""
    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "My SSN is 123-45-6789 and my email is john@example.com", "user_id": "usr_000001"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["guardrail_flags"]) > 0
    assert payload["retrieved_count"] == 0


def test_chat_endpoint_rate_limited(client):
    """Ensure rate limiter returns 429 when token bucket is exhausted."""
    # Temporarily set burst capacity to 1 token and refill rate to 0
    rate_limiter.burst = 1
    rate_limiter.rate_per_minute = 0
    rate_limiter.refill_rate = 0.0
    rate_limiter._buckets.clear()

    payload = {"message": "Hello", "user_id": "usr_000001"}

    # First request consumes the single available token (200 OK)
    res1 = client.post("/chat", json=payload, headers=AUTH_HEADERS)
    
    # Second request immediately exceeds the rate limit (429 Too Many Requests)
    res2 = client.post("/chat", json=payload, headers=AUTH_HEADERS)

    assert res2.status_code == 429