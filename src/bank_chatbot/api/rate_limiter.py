"""
Rate limiting utilities for the banking chatbot API.
"""
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class RateLimitDecision:
    allowed: bool
    remaining: int
    reset_after: float


class TokenBucketRateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate_per_minute: int, burst: int):
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self.refill_rate = rate_per_minute / 60.0
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()

        with self._lock:
            tokens, updated_at = self._buckets.get(key, (float(self.burst), now))
            elapsed = now - updated_at
            tokens = min(float(self.burst), tokens + elapsed * self.refill_rate)

            if tokens >= 1.0:
                tokens -= 1.0
                allowed = True
                reset_after = 0.0
            else:
                allowed = False
                reset_after = (1.0 - tokens) / self.refill_rate

            self._buckets[key] = (tokens, now)
            remaining = max(0, int(tokens))

        return RateLimitDecision(
            allowed=allowed,
            remaining=remaining,
            reset_after=reset_after,
        )
