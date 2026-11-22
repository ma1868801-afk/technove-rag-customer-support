"""
Centralized startup wiring, shared by the CLI (src/cli.py) and the FastAPI
app (api/main.py): build the embedding model, the vector store, the
Gemini LLM, the reranker, the semantic cache, the RAG pipeline, and the
compiled LangGraph app -- once, in one place, so the two entry points
can't drift out of sync.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src import config
from src.graph import build_graph
from src.ingest import build_vectorstore, get_embeddings, needs_rebuild
from src.logging_utils import setup_logging
from src.rag_pipeline import TechNovaRAG
from src.reranker import Reranker

logger = logging.getLogger("technova_rag.bootstrap")


def build_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )
    return ChatGroq(
        model=config.GROQ_MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
    
    )

@contextmanager
def build_app(rebuild_vectorstore: bool = False) -> Iterator[Tuple]:
    """
    Build every component and yield (graph_app, rag, vectorstore).

    Use as a context manager so the SQLite checkpointer connection behind
    the graph is closed cleanly on shutdown:

        with build_app() as (graph_app, rag, vectorstore):
            ...

    `rebuild_vectorstore` forces a rebuild regardless of the data/ hash;
    otherwise the vector store auto-rebuilds only when data/ changed since
    the last ingest (see AUTO_REBUILD_ON_STARTUP / src/ingest.py).
    """
    setup_logging()

    should_rebuild = rebuild_vectorstore or (
        config.AUTO_REBUILD_ON_STARTUP and needs_rebuild()
    )
    if should_rebuild:
        logger.info("Knowledge base changed or missing -- rebuilding vector store.")

    logger.info("Loading embedding model (%s)...", config.EMBEDDING_MODEL_NAME)
    embeddings = get_embeddings()

    logger.info("Loading/building vector store...")
    vectorstore = build_vectorstore(rebuild=should_rebuild, embeddings=embeddings)

    logger.info("Loading re-ranker (%s)...", config.RERANKER_MODEL_NAME)
    reranker = Reranker()


    llm = build_llm()
    rag = TechNovaRAG(vectorstore=vectorstore, llm=llm, reranker=reranker)

    # Imported lazily so a missing sqlite checkpoint package doesn't block
    # anything that only needs the RAG pipeline itself (e.g. the eval script).
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH) as checkpointer:
        graph_app = build_graph(llm=llm, rag=rag, checkpointer=checkpointer)
        logger.info("System ready.")
        yield graph_app, rag, vectorstore
