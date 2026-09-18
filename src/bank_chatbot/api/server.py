"""
FastAPI Server for Banking Chatbot

Exposes single-agent tool execution, RAG pipeline, guardrails, and observability.
"""
import os
import uvicorn
import time
import uuid
import structlog
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, REGISTRY, Histogram, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.pipeline import RAGPipeline
from src.bank_chatbot.agents.tool_agent import ToolCallingAgent
from src.bank_chatbot.guardrails.guardrails import BankingGuardrails
from src.bank_chatbot.api.rate_limiter import TokenBucketRateLimiter
from src.bank_chatbot.api.auth import APIKeyAuthMiddleware

logger = structlog.get_logger()
settings = get_settings()

# Initialize Singletons
pipeline = None
agent = None
guardrails = None

def get_or_create_counter(name: str, documentation: str, labelnames: list[str]):
    """Prevent DuplicateTimeseries errors during Uvicorn auto-reloads."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)

def get_or_create_histogram(name: str, documentation: str, labelnames: list[str] = None):
    """Prevent DuplicateTimeseries errors during Uvicorn auto-reloads."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames or [])

# Prometheus metrics (re-load safe)
REQUEST_COUNT = get_or_create_counter(
    "bank_chatbot_requests_total",
    "Total number of requests",
    ["endpoint", "status_code"]
)
REQUEST_LATENCY = get_or_create_histogram(
    "bank_chatbot_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
GUARDRAIL_BLOCKS = get_or_create_counter(
    "bank_chatbot_guardrail_blocks_total",
    "Total number of guardrail blocks",
    ["reason"]
)
RATE_LIMIT_BLOCKS = get_or_create_counter(
    "bank_chatbot_rate_limit_blocks_total",
    "Total number of rate limit blocks",
    ["client"]
)

rate_limiter = TokenBucketRateLimiter(
    rate_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst=settings.RATE_LIMIT_BURST,
)


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="usr_001", max_length=100)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=100)


class ChatResponse(BaseModel):
    """Response model for basic RAG endpoint."""
    response: str
    citations: list[dict]
    confidence: float
    guardrail_flags: list[str]
    retrieved_count: int
    request_id: str
    latency_ms: float


class AgentChatResponse(BaseModel):
    """Response model for Single Agent endpoint."""
    response: str
    tool_calls: list[dict]
    tool_results: list[dict]
    error: Optional[str] = None
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, agent, guardrails
    logger.info("Starting single-agent banking chatbot server")
    
    # Initialize heavy components inside lifespan
    guardrails = BankingGuardrails()
    pipeline = RAGPipeline()
    agent = ToolCallingAgent()
    
    yield
    logger.info("Stopping banking chatbot server")


app = FastAPI(
    title="Banking Customer Support Chatbot API",
    description="Single-Agent Banking Assistant with RAG, Tool Execution, and State Memory",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    APIKeyAuthMiddleware,
    api_key=settings.AUTH_TOKEN,
    protected_paths=["/chat", "/agent/chat"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to chat requests."""
    if request.url.path not in ["/chat", "/agent/chat"]:
        return await call_next(request)

    client_host = request.client.host if request.client else "unknown"
    decision = rate_limiter.check(client_host)

    if not decision.allowed:
        RATE_LIMIT_BLOCKS.labels(client=client_host).inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
            headers={
                "Retry-After": str(max(1, int(decision.reset_after))),
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + decision.reset_after)),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + decision.reset_after))
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        status_code=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest):
    """Main Single-Agent Endpoint: Handles tools, balances, transfers, and persistent memory."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # 1. Input Guardrail Verification
    guardrail_result = guardrails.process_message(request.message)
    if not guardrail_result["allowed"]:
        GUARDRAIL_BLOCKS.labels(reason="input_blocked").inc()
        return AgentChatResponse(
            response="I'm sorry, but I cannot process that request due to security policies.",
            tool_calls=[],
            tool_results=[],
            error="Guardrail Blocked",
            request_id=request_id,
            latency_ms=(time.time() - start_time) * 1000,
        )

    # 2. Invoke Single-Agent Graph
    try:
        result = agent.invoke(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        latency_ms = (time.time() - start_time) * 1000

        return AgentChatResponse(
            response=result["answer"],
            tool_calls=result["tool_calls"],
            tool_results=result["tool_results"],
            error=result["error"],
            request_id=request_id,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error("Single-Agent invocation error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Legacy/Direct RAG Endpoint for static policy lookup."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    guardrail_result = guardrails.process_message(request.message)
    if not guardrail_result["allowed"]:
        return ChatResponse(
            response="I'm sorry, but I can't process that request due to security policies.",
            citations=[],
            confidence=0.0,
            guardrail_flags=guardrail_result["flags"],
            retrieved_count=0,
            request_id=request_id,
            latency_ms=(time.time() - start_time) * 1000,
        )

    rag_result = pipeline.invoke(
        query=request.message,
        user_id=request.user_id,
        session_id=request.session_id,
    )

    return ChatResponse(
        response=rag_result["answer"],
        citations=rag_result["citations"],
        confidence=rag_result["confidence"],
        guardrail_flags=rag_result["guardrail_flags"],
        retrieved_count=rag_result["retrieved_count"],
        request_id=request_id,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bank-chatbot", "timestamp": time.time()}


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "agent_active": agent is not None}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.bank_chatbot.api.server:app", host="0.0.0.0", port=port)