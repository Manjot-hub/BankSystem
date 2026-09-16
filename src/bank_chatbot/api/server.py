"""
FastAPI Server for Banking Chatbot

Exposes the RAG pipeline, guardrails, and observability endpoints.
"""
import time
import uuid
import structlog
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.pipeline import RAGPipeline
from src.bank_chatbot.guardrails.guardrails import BankingGuardrails
from src.bank_chatbot.api.rate_limiter import TokenBucketRateLimiter
from src.bank_chatbot.api.auth import APIKeyAuthMiddleware


logger = structlog.get_logger()

settings = get_settings()
pipeline = RAGPipeline()
guardrails = BankingGuardrails()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "bank_chatbot_requests_total",
    "Total number of requests",
    ["endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "bank_chatbot_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
RAG_LATENCY = Histogram(
    "bank_chatbot_rag_latency_seconds",
    "RAG pipeline latency in seconds",
)
GUARDRAIL_BLOCKS = Counter(
    "bank_chatbot_guardrail_blocks_total",
    "Total number of guardrail blocks",
    ["reason"]
)
RATE_LIMIT_BLOCKS = Counter(
    "bank_chatbot_rate_limit_blocks_total",
    "Total number of rate limit blocks",
    ["client"]
)


rate_limiter = TokenBucketRateLimiter(
    rate_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst=settings.RATE_LIMIT_BURST,
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous", max_length=100)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=100)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    citations: list[dict]
    confidence: float
    guardrail_flags: list[str]
    retrieved_count: int
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    logger.info("Starting banking chatbot server")
    yield
    logger.info("Stopping banking chatbot server")


app = FastAPI(
    title="Banking Customer Support Chatbot API",
    description="Production-grade banking chatbot with RAG, guardrails, and observability",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    APIKeyAuthMiddleware,
    api_key=settings.AUTH_TOKEN,
    protected_paths=["/chat"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to chat requests."""
    if request.url.path != "/chat":
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bank-chatbot",
        "version": "0.1.0",
        "timestamp": time.time(),
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check vector store connectivity
        stats = pipeline.vector_store.get_collection_stats()
        return {
            "status": "ready",
            "vector_store": stats,
            "guardrails_enabled": guardrails.rails is not None,
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "Processing chat request",
        request_id=request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        message_length=len(request.message),
    )

    # 1. Guardrails check
    guardrail_result = guardrails.process_message(request.message)

    if not guardrail_result["allowed"]:
        GUARDRAIL_BLOCKS.labels(reason="input_blocked").inc()
        logger.warning(
            "Request blocked by guardrails",
            request_id=request_id,
            flags=guardrail_result["flags"],
        )

        return ChatResponse(
            response="I'm sorry, but I can't process that request due to security policies. Please contact our support team for assistance.",
            citations=[],
            confidence=0.0,
            guardrail_flags=guardrail_result["flags"],
            retrieved_count=0,
            request_id=request_id,
            latency_ms=(time.time() - start_time) * 1000,
        )

    # 2. RAG pipeline
    rag_start = time.time()
    try:
        rag_result = pipeline.invoke(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(
            "RAG pipeline failed",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal error")

    RAG_LATENCY.observe(time.time() - rag_start)

    # 3. Response validation
    response_validation = guardrails.validate_response(rag_result["answer"])

    if not response_validation["allowed"]:
        GUARDRAIL_BLOCKS.labels(reason="output_blocked").inc()
        logger.warning(
            "Response blocked by guardrails",
            request_id=request_id,
            flags=response_validation["flags"],
        )
        final_response = "I'm sorry, but I can't provide that information. Please contact our support team for assistance."
        guardrail_flags = rag_result["guardrail_flags"] + response_validation["flags"]
    else:
        final_response = rag_result["answer"]
        guardrail_flags = rag_result["guardrail_flags"]

    latency_ms = (time.time() - start_time) * 1000

    logger.info(
        "Chat request completed",
        request_id=request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        confidence=rag_result["confidence"],
        retrieved_count=rag_result["retrieved_count"],
        latency_ms=latency_ms,
    )

    return ChatResponse(
        response=final_response,
        citations=rag_result["citations"],
        confidence=rag_result["confidence"],
        guardrail_flags=guardrail_flags,
        retrieved_count=rag_result["retrieved_count"],
        request_id=request_id,
        latency_ms=latency_ms,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/guardrails/status")
async def guardrails_status():
    """Return guardrail policy summary."""
    return {
        "guardrails": guardrails.get_policy_summary(),
        "blocked_patterns": {
            "pii": True,
            "prompt_injection": True,
            "financial_advice": True,
            "forbidden_content": True,
        },
    }


@app.get("/config")
async def config_status():
    """Return active configuration (non-sensitive only)."""
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL_PRIMARY,
        "vector_store": settings.CHROMA_COLLECTION_NAME,
        "feature_flags": {
            "rag": settings.FEATURE_RAG_ENABLED,
            "tools": settings.FEATURE_TOOLS_ENABLED,
            "multi_agent": settings.FEATURE_MULTI_AGENT_ENABLED,
            "human_handoff": settings.FEATURE_HUMAN_HANDOFF_ENABLED,
        },
        "guardrails_config_dir": settings.GUARDRAILS_CONFIG_DIR,
    }


# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.bank_chatbot.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )