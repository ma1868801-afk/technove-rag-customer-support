"""
Retrieval evaluation harness (production review point #5: "no way to know
if a prompt/model change made retrieval better or worse").

Runs every case in eval_set.json through the real pipeline (translation
for Arabic cases, then query expansion -> multi-retrieve -> re-rank) and
checks whether each case's `expected_titles` show up:

  - anywhere in the merged candidate pool (pre-rerank)  -> "retrieval hit"
  - in the final top-K after re-ranking                 -> "rerank hit"

Comparing the two tells you WHERE a miss happens: a retrieval miss means
the embedding step never found the right chunk at all (fix: better
chunking, more expansion variants, a stronger embedding model); a
retrieval-hit-but-rerank-miss means the right chunk was found but the
cross-encoder ranked it too low (fix: tune RERANK_CANDIDATE_POOL/FINAL_TOP_K,
or reconsider the reranker model).

Requires a real GOOGLE_API_KEY (for the Arabic cases) and network access
to huggingface.co for the embedding/reranker models -- run this on your
machine, not in a network-restricted sandbox.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 3
    python -m eval.run_eval --output eval/results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src import config
from src.bootstrap import build_llm
from src.ingest import build_vectorstore, get_embeddings
from src.rag_pipeline import TechNovaRAG
from src.reranker import Reranker
from src.translation import translate_to_english

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def _hit(titles_in_pool: list[str], expected_titles: list[str]) -> bool:
    pool = set(titles_in_pool)
    return any(t in pool for t in expected_titles)


def _reciprocal_rank(ranked_titles: list[str], expected_titles: list[str]) -> float:
    expected = set(expected_titles)
    for i, title in enumerate(ranked_titles, start=1):
        if title in expected:
            return 1.0 / i
    return 0.0


def run(k: int, output_path: Path | None) -> None:
    cases: list[dict[str, Any]] = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    print(f"Loaded {len(cases)} eval cases from {EVAL_SET_PATH.name}")
    print("Building embeddings, vector store, reranker, and LLM...")

    embeddings = get_embeddings()
    vectorstore = build_vectorstore(embeddings=embeddings)
    reranker = Reranker()
    llm = build_llm()
    # No cache here -- we want every case to actually exercise retrieval,
    # not short-circuit on a near-duplicate from an earlier case.
    rag = TechNovaRAG(vectorstore=vectorstore, llm=llm, reranker=reranker, final_top_k=k)

    results = []
    for case in cases:
        start = time.perf_counter()

        if case["lang"] == "ar":
            english_query = translate_to_english(llm, case["query"])
        else:
            english_query = case["query"]

        expanded, candidates, reranked = rag.retrieve_for_query(english_query)

        candidate_titles = [d.metadata.get("title") for d in candidates]
        reranked_titles = [d.metadata.get("title") for d, _ in reranked]

        retrieval_hit = _hit(candidate_titles, case["expected_titles"])
        rerank_hit = _hit(reranked_titles, case["expected_titles"])
        rr = _reciprocal_rank(reranked_titles, case["expected_titles"])

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        results.append(
            {
                "id": case["id"],
                "lang": case["lang"],
                "query": case["query"],
                "english_query": english_query,
                "expected_titles": case["expected_titles"],
                "top_k_titles": reranked_titles,
                "retrieval_hit": retrieval_hit,
                "rerank_hit": rerank_hit,
                "reciprocal_rank": rr,
                "latency_ms": latency_ms,
            }
        )
        status = "OK  " if rerank_hit else "MISS"
        print(f"  [{status}] {case['id']:5s} ({latency_ms:>7.1f} ms)  {case['query'][:70]}")
        if case["lang"] != "en" and not rerank_hit:
            # Non-English miss: show the translated query right away, since
            # the first thing worth checking is whether Gemini translated
            # it well or drifted off-topic before retrieval even ran.
            print(f"         -> translated to: \"{english_query}\"")

    _print_summary(results, k)

    if output_path:
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull results written to {output_path}")


def _print_summary(results: list[dict], k: int) -> None:
    n = len(results)
    retrieval_hit_rate = sum(r["retrieval_hit"] for r in results) / n
    rerank_hit_rate = sum(r["rerank_hit"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"SUMMARY  (n={n}, top-k={k})")
    print("=" * 60)
    print(f"  Retrieval hit-rate (pre-rerank pool) : {retrieval_hit_rate:.1%}")
    print(f"  Rerank hit-rate    (final top-{k})     : {rerank_hit_rate:.1%}")
    print(f"  Mean Reciprocal Rank                 : {mrr:.3f}")
    print(f"  Avg latency / case                   : {avg_latency:.0f} ms")

    misses = [r for r in results if not r["rerank_hit"]]
    if misses:
          print(f"\n  Missed cases ({len(misses)}):")
          for m in misses:
            stage = "found pre-rerank but dropped by reranker" if m["retrieval_hit"] else "never retrieved at all"
            print(f"    - {m['id']} [{m['lang']}]  ({stage})")
            print(f"        query      : \"{m['query'][:70]}\"")
            if m["lang"] != "en":
                print(f"        translated : \"{m['english_query']}\"")
            print(f"        expected   : {m['expected_titles']}")
            print(f"        got top-k  : {m['top_k_titles']}")

    by_lang: dict[str, list[dict]] = {}
    for r in results:
        by_lang.setdefault(r["lang"], []).append(r)
    print("\n  By language:")
    for lang, rows in sorted(by_lang.items()):
        rate = sum(r["rerank_hit"] for r in rows) / len(rows)
        print(f"    - {lang}: {rate:.1%} ({len(rows)} cases)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TechNova RAG retrieval quality.")
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K, help="Top-k for rerank hit-rate.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write full JSON results.")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    run(k=args.k, output_path=output_path)


if __name__ == "__main__":
    main()
