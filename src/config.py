"""
Central configuration for the TechNova RAG + WhatsApp system.

All tunable constants live here so ingestion, the LangGraph pipeline, the
FastAPI app, and the eval script stay in sync without magic numbers
scattered across files. Everything is overridable via environment
variables / .env -- no code changes needed to tune the system.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _bool(env_var: str, default: str = "false") -> bool:
    return os.getenv(env_var, default).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# LLM -- Gemini (primary: translation + RAG generation used by the graph)
# --------------------------------------------------------------------------- #
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# --------------------------------------------------------------------------- #
# LLM -- Groq (optional: kept for the original standalone CLI / eval script,
# not used by the WhatsApp graph unless you wire it in yourself)
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

# --------------------------------------------------------------------------- #
# Embedding model (the query is translated to English before it hits the
# store, so an English-only embedding model is fine -- no need for a
# multilingual one as long as the translate node runs first)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------------------------------- #
# Cross-encoder re-ranker
# --------------------------------------------------------------------------- #
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------------------------------- #
# ChromaDB
# --------------------------------------------------------------------------- #
# "local"  -> embedded/persistent Chroma on disk (fine for one process)
# "server" -> connect to a standalone `chroma run` server via CHROMA_HOST/PORT
#             (needed once you scale the API past a single worker/process)
CHROMA_MODE = os.getenv("CHROMA_MODE", "local")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "technova_kb")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# --------------------------------------------------------------------------- #
# Source data + re-ingestion
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_FILE = DATA_DIR / "01_products.md"
COMPLAINTS_FILE = DATA_DIR / "02_complaints_faq.md"
BRANCHES_FILE = DATA_DIR / "03_branches_jeddah.md"
# Records a hash of the data/ files at last ingest time, so the API can
# auto-detect "the knowledge base changed, rebuild before serving".
INGEST_HASH_FILE = Path(CHROMA_PERSIST_DIR) / ".ingest_hash.json"
AUTO_REBUILD_ON_STARTUP = _bool("AUTO_REBUILD_ON_STARTUP", "true")

# --------------------------------------------------------------------------- #
# Retrieval / advanced-RAG pipeline knobs
# --------------------------------------------------------------------------- #
NUM_QUERY_EXPANSIONS = int(os.getenv("NUM_QUERY_EXPANSIONS", "3"))
K_PER_QUERY = int(os.getenv("K_PER_QUERY", "6"))
RERANK_CANDIDATE_POOL = int(os.getenv("RERANK_CANDIDATE_POOL", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

REWRITE_TEMPERATURE = float(os.getenv("REWRITE_TEMPERATURE", "0.0"))
EXPANSION_TEMPERATURE = float(os.getenv("EXPANSION_TEMPERATURE", "0.4"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.0"))

# --------------------------------------------------------------------------- #
# Reliability: retries / timeouts for every LLM call
# --------------------------------------------------------------------------- #
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
LLM_RETRY_MIN_WAIT = float(os.getenv("LLM_RETRY_MIN_WAIT", "1"))
LLM_RETRY_MAX_WAIT = float(os.getenv("LLM_RETRY_MAX_WAIT", "8"))


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1500"))

# --------------------------------------------------------------------------- #
# Conversation checkpointing (LangGraph state persistence)
# --------------------------------------------------------------------------- #
CHECKPOINT_DB_PATH = os.getenv(
    "CHECKPOINT_DB_PATH", str(PROJECT_ROOT / "conversation_state.sqlite")
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = _bool("LOG_JSON", "true")

# --------------------------------------------------------------------------- #
# WhatsApp Cloud API (Meta) -- used by api/main.py
# --------------------------------------------------------------------------- #
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")

# --------------------------------------------------------------------------- #
# API rate limiting
# --------------------------------------------------------------------------- #
# Public REST endpoint (/chat), keyed by client IP via slowapi.
RATE_LIMIT_PER_MINUTE = os.getenv("RATE_LIMIT_PER_MINUTE", "20/minute")
# WhatsApp webhook, keyed by the sender's WhatsApp ID (a phone number can't
# be blocked by IP -- every message arrives from Meta's own servers).
WHATSAPP_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("WHATSAPP_RATE_LIMIT_MAX_REQUESTS", "15"))
WHATSAPP_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("WHATSAPP_RATE_LIMIT_WINDOW_SECONDS", "60"))
