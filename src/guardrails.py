"""
Input guardrails (production review point #7: "nothing stops prompt
injection via a crafted customer message, nothing validates input length").

Two layers, matching how this class of problem is actually mitigated:

1. Validation/sanitization here -- reject empty/oversized input, strip
   control characters, and flag (for logging/monitoring) messages that
   look like an injection attempt.

2. Prompt structure in `translation.py` / `rag_pipeline.py` -- every system
   prompt explicitly tells the model the customer's message is DATA to
   answer from/about, never an instruction to follow, and user content is
   always wrapped in explicit delimiters so it can't be confused with the
   system prompt. That's the real defense; the heuristic check below is a
   monitoring signal on top of it, not a hard filter -- blocking on it
   outright would false-positive on legitimate customers ("please ignore
   my previous message, I meant...").
"""
from __future__ import annotations

import logging
import re
import unicodedata

from src import config

logger = logging.getLogger("technova_rag.guardrails")

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|the)? ?previous instructions",
        r"disregard (all|any|the)? ?(system|previous) instructions",
        r"you are now",
        r"act as (a|an) ",
        r"reveal your (system prompt|instructions)",
        r"</?system>",
        r"</?(assistant|user)>",
    ]
]


class GuardrailViolation(Exception):
    """Raised when input fails a hard guardrail (empty / too long)."""


def _strip_control_chars(text: str) -> str:
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )


def validate_and_clean(text: str) -> str:
    """
    Hard checks (raise on failure) + cleanup. Call this on every inbound
    message before it touches the graph.
    """
    if text is None:
        raise GuardrailViolation("Empty message.")

    cleaned = _strip_control_chars(text).strip()

    if not cleaned:
        raise GuardrailViolation("Empty message.")

    if len(cleaned) > config.MAX_INPUT_CHARS:
        raise GuardrailViolation(
            f"Message too long ({len(cleaned)} chars, max {config.MAX_INPUT_CHARS})."
        )

    if looks_like_injection(cleaned):
        logger.warning("Possible prompt-injection pattern in inbound message", extra={
            "extra_fields": {"preview": cleaned[:120]}
        })
        # Not blocked -- see module docstring. Logged for monitoring/alerting.

    return cleaned


def looks_like_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def wrap_user_content(text: str) -> str:
    """
    Wrap raw customer text in explicit delimiters for inclusion in a prompt,
    so the model can't mistake it for system/developer instructions.
    """
    return f"<customer_message>\n{text}\n</customer_message>"
