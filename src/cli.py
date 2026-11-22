"""
Interactive terminal chat for the TechNova WhatsApp-style assistant.

Run with:
    python main.py
    python main.py --verbose      # show translated query + rerank sources
    python main.py --rebuild      # force-rebuild the Chroma collection first

Conversation history is bookkept inside the graph itself (see
src/graph.py) and persisted per `thread_id` by the SQLite checkpointer --
this CLI only needs to pass the raw input and a stable user_id each turn.
"""
from __future__ import annotations

import argparse
import uuid

from src.bootstrap import build_app


def _print_verbose(state: dict) -> None:
    print(f"\n  English query : {state.get('english_query')}")
    print("  Sources:")
    for s in state.get("rag_sources", []):
        print(
            f"    - [{s['relevance_score']:>7}] {s['category']} / "
            f"{s['title']} ({s['source_file']})"
        )
    print(f"  English answer: {state.get('rag_answer')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the TechNova support assistant.")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline internals.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store first.")
    args = parser.parse_args()

    with build_app(rebuild_vectorstore=args.rebuild) as (graph_app, _rag, _vectorstore):
        print(
            "\nTechNova support assistant ready -- type in Arabic or English. "
            "Type 'exit' to quit.\n"
        )

        user_id = f"cli-{uuid.uuid4().hex[:8]}"
        thread_config = {"configurable": {"thread_id": user_id}}

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            result_state = graph_app.invoke(
                {"raw_input": user_input, "user_id": user_id},
                config=thread_config,
            )

            if args.verbose:
                _print_verbose(result_state)

            print(f"Assistant: {result_state.get('final_answer')}\n")


if __name__ == "__main__":
    main()
