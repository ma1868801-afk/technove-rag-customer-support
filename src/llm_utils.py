"""
Reliability wrapper around every LLM call in the system (point #2 from the
production review: "no error handling around the Groq/Gemini calls").

`safe_invoke` gives every LLM call in the graph:
  - a hard timeout (a hung request can't stall a WhatsApp reply forever)
  - exponential-backoff retries on transient failures (rate limits, network
    blips, 5xx from the provider)
  - a guard against a "successful" call that returns an empty/blank response
    (which otherwise silently propagates as a blank WhatsApp message)

Every call site should go through this instead of calling `llm.invoke(...)`
directly.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

logger = logging.getLogger("technova_rag.llm")

# A small shared thread pool just for enforcing per-call timeouts around the
# (synchronous) langchain `.invoke()` calls.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-call")


class LLMCallError(Exception):
    """Raised when an LLM call fails, or returns an empty response, after all retries."""


def _invoke_with_timeout(
    llm: BaseChatModel, messages: List[BaseMessage], timeout: float, **kwargs: Any
):
    future = _executor.submit(llm.invoke, messages, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM call exceeded {timeout}s timeout") from exc


@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1, min=config.LLM_RETRY_MIN_WAIT, max=config.LLM_RETRY_MAX_WAIT
    ),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _safe_invoke_inner(llm: BaseChatModel, messages: List[BaseMessage], **kwargs: Any):
    response = _invoke_with_timeout(llm, messages, config.LLM_TIMEOUT_SECONDS, **kwargs)
    content = getattr(response, "content", None)
    if content is None or not str(content).strip():
        raise LLMCallError("LLM returned an empty response")
    return response


def safe_invoke(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    *,
    fallback: str | None = None,
    call_name: str = "llm_call",
    **kwargs: Any,
) -> str:
    """
    Invoke `llm` with retries + timeout, returning the response text.

    If every attempt fails and `fallback` is provided, logs the failure and
    returns `fallback` instead of raising -- callers that have a sane
    default (e.g. "just pass the original text through untranslated")
    should use this. Callers with no safe fallback should catch
    LLMCallError themselves.
    """
    try:
        response = _safe_invoke_inner(llm, messages, **kwargs)
        return str(response.content).strip()
    except Exception as exc:  # noqa: BLE001 -- intentionally broad, this is the top-level guard
        logger.error("%s failed after %d attempts: %s", call_name, config.LLM_MAX_RETRIES, exc)
        if fallback is not None:
            return fallback
        raise LLMCallError(f"{call_name} failed after retries: {exc}") from exc
