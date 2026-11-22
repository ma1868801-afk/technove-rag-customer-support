"""
Query Rewriting + Query Expansion.

Query Rewriting
---------------
Kept here for standalone/CLI use (e.g. the eval script, or a future
English-only integration). In the WhatsApp graph this step is absorbed
into `translation.translate_to_english`, which does translation + rewriting
in a single Gemini call to save a round-trip -- see that module.

Query Expansion
---------------
Takes a query and generates several alternative phrasings / related
sub-queries. Retrieving with all of them and merging results covers
vocabulary mismatch (e.g. "phone won't charge" vs "charging issue" vs
"battery not powering on") that a single embedding search can miss.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.llm_utils import safe_invoke

# --------------------------------------------------------------------------- #
# Query Rewriting
# --------------------------------------------------------------------------- #

_REWRITE_SYSTEM_PROMPT = """You rewrite customer questions for a Saudi electronics \
retailer's (TechNova) search system so they retrieve better results from a \
knowledge base of products, FAQs/complaints, and Jeddah branch locations.

Rules:
- Output ONLY the rewritten query text. No preamble, no quotes, no explanation.
- Resolve pronouns and vague references using the conversation history if given.
- Fix obvious typos and expand abbreviations.
- Keep it a single, self-contained question in the same language as the input.
- Do not invent details the user did not imply.
- If the query is already clear, you may return it nearly unchanged."""


def rewrite_query(
    llm: BaseChatModel,
    raw_query: str,
    chat_history: Optional[Sequence[str]] = None,
) -> str:
    history_block = ""
    if chat_history:
        history_block = "Conversation so far:\n" + "\n".join(chat_history) + "\n\n"

    messages = [
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{history_block}User question: {raw_query}\n\nRewritten query:"
        ),
    ]
    return safe_invoke(
        llm,
        messages,
        temperature=config.REWRITE_TEMPERATURE,
        fallback=raw_query,
        call_name="rewrite_query",
    )


# --------------------------------------------------------------------------- #
# Query Expansion
# --------------------------------------------------------------------------- #

_EXPANSION_SYSTEM_PROMPT = """You generate alternative search-query phrasings for a \
retrieval system over a Saudi electronics retailer's (TechNova) knowledge base \
(products, FAQs/complaints, Jeddah branches).

Given one query, produce {n} alternative phrasings that:
- Use different but relevant vocabulary/synonyms (e.g. "won't turn on" <-> "not powering on")
- Cover related angles a user might mean (e.g. a defect question might also relate to warranty or returns)
- Stay closely relevant to the original query's intent -- do not drift topic.
- If the user query asks for a total sum, grand total, or aggregate calculation, decompose it into sub-queries that request the individual price of each specific product.
- Are each a short, standalone search query in the same language as the input.

Respond with ONLY a JSON array of {n} strings. No other text."""


def expand_query(
    llm: BaseChatModel,
    rewritten_query: str,
    n: int = config.NUM_QUERY_EXPANSIONS,
) -> List[str]:
    """Return [rewritten_query] + up to n LLM-generated alternative phrasings."""
    messages = [
        SystemMessage(content=_EXPANSION_SYSTEM_PROMPT.format(n=n)),
        HumanMessage(content=f"Query: {rewritten_query}"),
    ]
    raw = safe_invoke(
        llm,
        messages,
        temperature=config.EXPANSION_TEMPERATURE,
        fallback="[]",  # empty expansion -> we still search with the original query below
        call_name="expand_query",
    )

    variants = _parse_json_string_list(raw)
    if not variants:
        variants = [
            re.sub(r"^[\-\d\.\)\s]+", "", line).strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    all_queries = [rewritten_query] + [v for v in variants if v]
    seen = set()
    deduped = []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q.strip())
    return deduped[: n + 1]


def _parse_json_string_list(raw: str) -> List[str]:
    """Best-effort extraction of a JSON string array from an LLM response."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    candidate = match.group(0) if match else raw
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return []
