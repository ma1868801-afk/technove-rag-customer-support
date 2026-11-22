"""
Ingestion pipeline for the TechNova knowledge base.

Rather than blindly running a generic character splitter over the three
markdown files (which would slice a product's price away from its
description, or a FAQ answer away from its question), this module parses
each file according to its own structure and produces ONE chunk per
semantic unit:

    01_products.md        -> one chunk per product ("### Product Name")
    02_complaints_faq.md  -> one chunk per Q/A pair ("**Q: ...**")
    03_branches_jeddah.md -> one chunk per branch / reference table
                             ("## Section")

Each chunk keeps rich metadata (source file, doc_type, category, title)
so the retriever and the re-ranker have something to reason about beyond
raw text, and so the final answer can cite exactly where a fact came from.

This module also tracks a hash of data/ so the API/CLI can auto-detect
"the knowledge base changed since last ingest, rebuild before serving"
(production review point #8) instead of silently serving a stale index.

Run directly to (re)build the persisted Chroma collection:

    python -m src.ingest --rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

# --------------------------------------------------------------------------- #
# Markdown parsing helpers
# --------------------------------------------------------------------------- #

_H2_SPLIT = re.compile(r"(?m)^##\s+(.*?)\s*$")
_H3_SPLIT = re.compile(r"(?m)^###\s+(.*?)\s*$")
_Q_SPLIT = re.compile(r"(?m)^\*\*Q:\s*(.*?)\*\*\s*$")


def _split_h2_sections(text: str) -> List[tuple[str, str]]:
    """Split a markdown document into (header, body) pairs on '## ' headers."""
    matches = list(_H2_SPLIT.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((header, body))
    return sections


def parse_products(path: Path) -> List[Document]:
    """One Document per '### Product Name' entry, tagged with its '## Category'."""
    text = path.read_text(encoding="utf-8")
    docs: List[Document] = []

    for category, body in _split_h2_sections(text):
        matches = list(_H3_SPLIT.finditer(body))
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            item_body = body[start:end].strip()
            item_body = re.sub(r"\n?-{3,}\s*$", "", item_body).strip()

            content = f"Product: {title}\nCategory: {category}\n{item_body}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "doc_type": "product",
                        "category": category,
                        "title": title,
                    },
                )
            )
    return docs


def parse_complaints(path: Path) -> List[Document]:
    """One Document per Q/A pair, tagged with its '## Section' and 'Category:' line."""
    text = path.read_text(encoding="utf-8")
    docs: List[Document] = []

    for section, body in _split_h2_sections(text):
        matches = list(_Q_SPLIT.finditer(body))
        for i, m in enumerate(matches):
            question = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            qa_body = body[start:end].strip()
            qa_body = re.sub(r"\n?-{3,}\s*$", "", qa_body).strip()

            cat_match = re.search(r"(?m)^Category:\s*(.*?)\s*$", qa_body)
            category = cat_match.group(1).strip() if cat_match else section

            content = f"Question: {question}\nSection: {section}\n{qa_body}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "doc_type": "faq",
                        "category": category,
                        "title": question,
                    },
                )
            )
    return docs


def parse_branches(path: Path) -> List[Document]:
    """One Document per '## Section' (each branch, plus the reference table)."""
    text = path.read_text(encoding="utf-8")
    docs: List[Document] = []

    for title, body in _split_h2_sections(text):
        body = re.sub(r"\n?-{3,}\s*$", "", body).strip()
        content = f"{title}\n{body}"
        doc_type = "branch_reference_table" if "Reference Table" in title else "branch"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "doc_type": doc_type,
                    "category": "Branches",
                    "title": title,
                },
            )
        )
    return docs


def build_documents() -> List[Document]:
    docs: List[Document] = []
    docs += parse_products(config.PRODUCTS_FILE)
    docs += parse_complaints(config.COMPLAINTS_FILE)
    docs += parse_branches(config.BRANCHES_FILE)
    return docs


# --------------------------------------------------------------------------- #
# Change detection (auto-rebuild when data/ changes)
# --------------------------------------------------------------------------- #

def compute_data_hash() -> str:
    """SHA-256 over the concatenated bytes of every source file, order-stable."""
    hasher = hashlib.sha256()
    for path in (config.PRODUCTS_FILE, config.COMPLAINTS_FILE, config.BRANCHES_FILE):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _read_stored_hash() -> str | None:
    hash_file = config.INGEST_HASH_FILE
    if not hash_file.exists():
        return None
    try:
        return json.loads(hash_file.read_text(encoding="utf-8")).get("hash")
    except (json.JSONDecodeError, OSError):
        return None


def _write_stored_hash(hash_value: str) -> None:
    hash_file = config.INGEST_HASH_FILE
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(json.dumps({"hash": hash_value}), encoding="utf-8")


def needs_rebuild() -> bool:
    """True if data/ has changed since the last successful ingest, or no index exists yet."""
    persist_dir = Path(config.CHROMA_PERSIST_DIR)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        return True
    return _read_stored_hash() != compute_data_hash()


# --------------------------------------------------------------------------- #
# Embeddings + vector store
# --------------------------------------------------------------------------- #

def get_embeddings() -> HuggingFaceEmbeddings:
    """all-MiniLM-L6-v2 sentence embeddings, run locally via sentence-transformers."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def _chroma_client_kwargs() -> dict:
    """
    "local"  -> embedded/on-disk Chroma (fine for a single process).
    "server" -> connect to a standalone `chroma run` server, needed once you
                scale the API past one worker/process (Chroma's local mode
                does not support concurrent writers).
    """
    if config.CHROMA_MODE == "server":
        import chromadb

        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        return {"client": client}
    return {"persist_directory": str(config.CHROMA_PERSIST_DIR)}


def build_vectorstore(rebuild: bool = False, embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """
    Build (or load) the persisted Chroma collection.

    If `rebuild` is True (or the data/ hash no longer matches the last
    ingest), any existing local collection directory is deleted first so
    stale chunks can never linger after the source markdown changes.

    Pass an existing `embeddings` instance (e.g. one also shared with the
    semantic response cache) to avoid loading the sentence-transformers
    model twice at startup.
    """
    persist_dir = Path(config.CHROMA_PERSIST_DIR)
    current_hash = compute_data_hash()
    stale = _read_stored_hash() != current_hash

    if config.CHROMA_MODE == "local" and (rebuild or stale) and persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = embeddings or get_embeddings()
    already_built = (
        config.CHROMA_MODE == "server"
        or (persist_dir.exists() and any(persist_dir.iterdir()))
    )

    if already_built and not rebuild and not stale:
        return Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            **_chroma_client_kwargs(),
        )

    documents = build_documents()
    if not documents:
        raise RuntimeError("No documents parsed from data/ -- check the source files.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        **_chroma_client_kwargs(),
    )
    _write_stored_hash(current_hash)
    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TechNova Chroma vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the persisted collection from scratch.",
    )
    args = parser.parse_args()

    docs = build_documents()
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.metadata["doc_type"]] = by_type.get(d.metadata["doc_type"], 0) + 1

    print(f"Parsed {len(docs)} chunks from data/:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  - {doc_type}: {count}")

    print(
        f"\nEmbedding with '{config.EMBEDDING_MODEL_NAME}' and persisting to "
        f"'{config.CHROMA_PERSIST_DIR}' (collection='{config.CHROMA_COLLECTION_NAME}')..."
    )
    build_vectorstore(rebuild=args.rebuild)
    print("Done.")


if __name__ == "__main__":
    main()
