"""
FastAPI app: WhatsApp Cloud API webhook + a plain REST /chat endpoint for
testing/other integrations, both backed by the same LangGraph pipeline
(src/graph.py).

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8080

For local WhatsApp testing, expose this with a tunnel (e.g. `ngrok http
8080`) and point Meta's webhook config at https://<tunnel>/webhook.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from api import security, whatsapp_client
from src import config, guardrails
from src.bootstrap import build_app as build_pipeline
from src.graph import FALLBACK_ARABIC_MESSAGE
from src.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger("technova_rag.api")

limiter = Limiter(key_func=get_remote_address)

# Simple in-process fallback rate-limit message for slow-down replies.
_SLOW_DOWN_MESSAGE = "وصلتنا رسائل كثيرة منك بوقت قصير، تكرم تصبر شوي وتعيد المحاولة 🙏"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with build_pipeline() as (graph_app, rag, vectorstore):
        app.state.graph_app = graph_app
        app.state.rag = rag
        app.state.vectorstore = vectorstore
        logger.info("FastAPI app ready.")
        yield
    logger.info("FastAPI app shutting down.")


app = FastAPI(title="TechNova Support Assistant", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --------------------------------------------------------------------------- #
# Health / readiness
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness/readiness probe: reports whether the pipeline finished loading."""
    ready = hasattr(request.app.state, "graph_app")
    return {"status": "ok" if ready else "starting"}



# --------------------------------------------------------------------------- #
# REST endpoint (direct testing / non-WhatsApp integrations)
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=config.MAX_INPUT_CHARS)
    user_id: str = Field(
        ..., min_length=1, description="Stable conversation id, e.g. a phone number or session id."
    )


class ChatResponse(BaseModel):
    answer: str
    english_query: str
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(config.RATE_LIMIT_PER_MINUTE)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    try:
        cleaned = guardrails.validate_and_clean(body.message)
    except guardrails.GuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    graph_app = request.app.state.graph_app
    thread_config = {"configurable": {"thread_id": body.user_id}}

    result_state = await run_in_threadpool(
        graph_app.invoke,
        {"raw_input": cleaned, "user_id": body.user_id},
        config=thread_config,
    )

    return ChatResponse(
        answer=result_state.get("final_answer", ""),
        english_query=result_state.get("english_query", ""),
        sources=result_state.get("rag_sources", []),
    )


# --------------------------------------------------------------------------- #
# WhatsApp Cloud API webhook
# --------------------------------------------------------------------------- #
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta's one-time webhook verification handshake."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == config.WHATSAPP_VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed.")


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not security.verify_whatsapp_signature(raw_body, signature):
        logger.warning("Rejected webhook call with invalid signature.")
        raise HTTPException(status_code=403, detail="Invalid signature.")

    payload = await request.json()
    message = whatsapp_client.extract_inbound_message(payload)
    if message is None:
        # Not a new text message (status callback, media, reaction, etc.) -- ack and ignore.
        return Response(status_code=200)

    sender = message["from"]

    if not security.whatsapp_rate_limiter.allow(sender):
        logger.warning("Rate-limited WhatsApp sender %s", sender)
        background_tasks.add_task(whatsapp_client.send_text_message, sender, _SLOW_DOWN_MESSAGE)
        return Response(status_code=200)

    # Meta expects a fast ack and will retry (and eventually flag the
    # webhook unhealthy) if it doesn't get one quickly -- so we return 200
    # immediately and do the actual LLM/RAG work in the background, then
    # push the reply via the Graph API ourselves.
    background_tasks.add_task(_handle_message_and_reply, request.app, sender, message["text"])
    return Response(status_code=200)


async def _handle_message_and_reply(app: FastAPI, sender: str, text: str) -> None:
    graph_app = app.state.graph_app
    thread_config = {"configurable": {"thread_id": sender}}

    try:
        cleaned = guardrails.validate_and_clean(text)
        result_state = await run_in_threadpool(
            graph_app.invoke,
            {"raw_input": cleaned, "user_id": sender},
            config=thread_config,
        )
        reply = result_state.get("final_answer") or FALLBACK_ARABIC_MESSAGE
    except guardrails.GuardrailViolation as exc:
        logger.info("Guardrail rejected message from %s: %s", sender, exc)
        reply = FALLBACK_ARABIC_MESSAGE
    except Exception:
        logger.exception("Unhandled error processing message from %s", sender)
        reply = FALLBACK_ARABIC_MESSAGE

    await whatsapp_client.send_text_message(sender, reply)
