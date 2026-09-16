from bank_chatbot.api.rate_limiter import TokenBucketRateLimiter


def test_rate_limiter_allows_initial_burst():
    limiter = TokenBucketRateLimiter(rate_per_minute=60, burst=3)

    assert limiter.check("client-1").allowed
    assert limiter.check("client-1").allowed
    assert limiter.check("client-1").allowed
    assert not limiter.check("client-1").allowed


def test_rate_limiter_refills_over_time(monkeypatch):
    limiter = TokenBucketRateLimiter(rate_per_minute=60, burst=1)
    now = [0.0]

    monkeypatch.setattr("bank_chatbot.api.rate_limiter.time.monotonic", lambda: now[0])

    assert limiter.check("client-1").allowed
    assert not limiter.check("client-1").allowed

    now[0] = 1.0
    assert limiter.check("client-1").allowed
