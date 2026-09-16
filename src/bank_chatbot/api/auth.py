"""
API key authentication middleware for the banking chatbot.
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require an API key for protected routes."""

    def __init__(self, app, api_key: str, protected_paths: list[str] = None):
        super().__init__(app)
        self.api_key = api_key
        self.protected_paths = set(protected_paths or ["/chat"])

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in self.protected_paths:
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        authorization = request.headers.get("authorization", "")

        if authorization.startswith("Bearer "):
            api_key = authorization.split(" ", 1)[1]

        if not api_key or api_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
