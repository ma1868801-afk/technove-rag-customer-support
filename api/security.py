"""
Security helpers for the FastAPI/WhatsApp layer (production review point
#7, API side): verify that inbound webhook calls really came from Meta,
and rate-limit by WhatsApp sender so one number can't hammer the pipeline
(and your Gemini/embedding bill) with rapid-fire messages.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time

from src import config


def verify_whatsapp_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Validate Meta's `X-Hub-Signature-256: sha256=<hex>` header against the
    raw request body using the app secret. If WHATSAPP_APP_SECRET is unset
    (e.g. local dev without a configured Meta app yet), verification is
    skipped -- set it before going to production.
    """
    if not config.WHATSAPP_APP_SECRET:
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        config.WHATSAPP_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)


class SlidingWindowRateLimiter:
    """A small in-memory per-key sliding-window limiter.

    Fine for a single-process deployment; for multiple workers/instances,
    swap the in-memory dict for Redis (e.g. `INCR` + `EXPIRE`) behind the
    same `allow(key)` interface.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(hits) >= self.max_requests:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


whatsapp_rate_limiter = SlidingWindowRateLimiter(
    max_requests=config.WHATSAPP_RATE_LIMIT_MAX_REQUESTS,
    window_seconds=config.WHATSAPP_RATE_LIMIT_WINDOW_SECONDS,
)
