"""
Observability (production review point #4: "no logging of what got
retrieved, what got reranked out, latency per stage").

Two things live here:
  - `setup_logging()`: configures either structured JSON logs (production)
    or plain text logs (local dev), controlled by LOG_JSON / LOG_LEVEL.
  - `stage_timer`: a context manager that logs a structured event with the
    stage name, duration, and any extra fields you pass -- used around
    every node in the LangGraph pipeline (translate, retrieve, rerank,
    generate, translate-back) so a single conversation turn produces a
    readable trace of what happened and how long each step took.

For deeper tracing (full prompt/response capture, a UI to browse traces),
set the standard LangSmith env vars (LANGCHAIN_TRACING_V2=true,
LANGCHAIN_API_KEY=...) -- LangChain/LangGraph pick those up automatically,
no code changes needed here.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

from src import config


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    root = logging.getLogger("technova_rag")
    root.setLevel(config.LOG_LEVEL)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if config.LOG_JSON:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root.addHandler(handler)
    root.propagate = False


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"extra_fields": fields})


@contextmanager
def stage_timer(logger: logging.Logger, stage: str, **fields: Any) -> Iterator[dict]:
    """
    Usage:
        with stage_timer(logger, "retrieve", user_id=user_id) as ctx:
            ...
            ctx["candidate_count"] = len(candidates)   # add fields mid-stage

    Logs one structured "stage_complete" event on exit with duration_ms and
    every field collected, or "stage_failed" (still with duration_ms) if the
    block raises -- the exception is re-raised unchanged either way.
    """
    ctx: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield ctx
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            logger,
            "stage_failed",
            stage=stage,
            duration_ms=duration_ms,
            error=str(exc),
            **fields,
            **ctx,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            logger,
            "stage_complete",
            stage=stage,
            duration_ms=duration_ms,
            **fields,
            **ctx,
        )
