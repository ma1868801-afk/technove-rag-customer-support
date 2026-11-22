"""
LangGraph pipeline: translate the customer's message to English, run it
through the RAG tool, translate the answer back into Saudi Arabic.

    START
      |
      v
  translate_in   (Gemini: any language -> clean English question;
                   guardrail validation happens here first)
      |
      +--- error? --> fallback ---------------------+
      v                                              |
   rag_tool       (query expansion -> multi-retrieve  |
                    -> re-rank -> generate; English    |
                    in, English answer out; cache-      |
                    aware)                              |
      |                                                 |
      +--- error? --> fallback  (hardcoded Arabic        |
      |                          apology, no further LLM |
      |                          calls -- avoids           |
      |                          cascading failures)        |
      v                                                      v
  translate_out  (Gemini: English answer -> Saudi Arabic) <--+
      |
      v
     END

State is checkpointed per `user_id` (LangGraph `thread_id`) via a SQLite
checkpointer passed in from the caller, so a WhatsApp conversation's
history survives process restarts. Every node is wrapped in
`stage_timer` for structured, per-turn observability.
"""
from __future__ import annotations

import logging
from typing import List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src import guardrails
from src.guardrails import GuardrailViolation
from src.logging_utils import stage_timer
from src.rag_pipeline import TechNovaRAG
from src.translation import translate_to_english, translate_to_saudi_arabic, detect_language

logger = logging.getLogger("technova_rag.graph")

# Hardcoded (no LLM call) so a failure deep in the pipeline can never cascade
# into the fallback path itself failing.
FALLBACK_ARABIC_MESSAGE = (
    "عذرًا، واجهنا مشكلة تقنية بسيطة الحين. ممكن تعيد سؤالك؟ "
    "أو تتواصل مباشرة مع أقرب فرع لك وفريقنا يساعدك."
)
FALLBACK_ENGLISH_MESSAGE = (
    "Sorry, we hit a small technical issue just now. Could you try your "
    "question again? Or reach out directly to your nearest branch and our "
    "team can help."
)


class ConversationState(TypedDict, total=False):
    raw_input: str
    user_id: str
    chat_history: List[str]
    english_query: str
    rag_answer: str
    detected_language: str
    rag_sources: List[dict]
    final_answer: str
    error: Optional[str]


def build_graph(llm: BaseChatModel, rag: TechNovaRAG, checkpointer=None):
    """Wire up the graph with `llm` and `rag` injected via closures, and
    compile it with the given checkpointer (None -> no persistence, state
    only lives for a single `.invoke()` call -- fine for the CLI/eval, not
    for the WhatsApp API where conversations span many requests)."""

    def translate_in(state: ConversationState) -> dict:
      with stage_timer(logger, "translate_in", user_id=state.get("user_id")) as ctx:
        lang = detect_language(state.get("raw_input", ""))
        ctx["detected_language"] = lang

        try:
            cleaned = guardrails.validate_and_clean(state["raw_input"])
        except GuardrailViolation as exc:
            ctx["guardrail_violation"] = str(exc)
            return {"error": f"guardrail: {exc}", "detected_language": lang}

        english = translate_to_english(llm, cleaned, state.get("chat_history"))
        ctx["english_query_preview"] = english[:80]
        return {"english_query": english, "detected_language": lang, "error": None}

    def rag_tool(state: ConversationState) -> dict:
        with stage_timer(logger, "rag_tool", user_id=state.get("user_id")) as ctx:
            try:
                result = rag.answer_english_query(state["english_query"])
            except Exception as exc:  # noqa: BLE001 -- last line of defense before the fallback node
                logger.error("rag_tool failed: %s", exc)
                ctx["error"] = str(exc)
                return {"error": f"rag_tool: {exc}"}

            
            ctx["candidate_count"] = result.candidate_count
            ctx["source_count"] = len(result.sources_summary())
            return {
                "rag_answer": result.answer,
                "rag_sources": result.sources_summary(),
                "error": None,
            }

    def _append_history(state: ConversationState, assistant_line: str) -> list:
        # Chat history is bookkept inside the graph (not by callers) so it
        # rides along in the checkpointed state automatically turn-over-turn
        # -- the CLI/API only ever need to pass `raw_input` + `user_id`.
        history = list(state.get("chat_history", []))
        history.append(f"User: {state.get('raw_input', '')}")
        history.append(f"Assistant: {assistant_line}")
        return history[-8:]

    def translate_out(state: ConversationState) -> dict:
     with stage_timer(logger, "translate_out", user_id=state.get("user_id")) as ctx:
        if state.get("detected_language") == "ar":
            final = translate_to_saudi_arabic(llm, state["rag_answer"])
        else:
            final = state["rag_answer"]  # already English -- no LLM call needed
        ctx["translated_to_arabic"] = state.get("detected_language") == "ar"
        return {
            "final_answer": final,
            "chat_history": _append_history(state, state["rag_answer"]),
        }

    def fallback(state: ConversationState) -> dict:
     logger.warning(
        "Routing to fallback node",
        extra={
            "extra_fields": {
                "user_id": state.get("user_id"),
                "error": state.get("error"),
            }
        },
     )
     message = (
        FALLBACK_ARABIC_MESSAGE
        if state.get("detected_language") == "ar"
        else FALLBACK_ENGLISH_MESSAGE
     )
     return {
        "final_answer": message,
        "chat_history": _append_history(state, "[error -- fallback response sent]"),
    }

    graph = StateGraph(ConversationState)
    graph.add_node("translate_in", translate_in)
    graph.add_node("rag_tool", rag_tool)
    graph.add_node("translate_out", translate_out)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("translate_in")
    graph.add_conditional_edges(
        "translate_in", lambda s: "fallback" if s.get("error") else "rag_tool"
    )
    graph.add_conditional_edges(
        "rag_tool", lambda s: "fallback" if s.get("error") else "translate_out"
    )
    graph.add_edge("translate_out", END)
    graph.add_edge("fallback", END)

    return graph.compile(checkpointer=checkpointer)
