"""
Evaluation harness: compares retriever configurations on a gold set.

Run:  python -m eval.run_eval

Loads eval/gold.json (a small hand-labeled set of question -> relevant chunk
ids), runs three retriever configs over it, and prints a comparison table. This
is what produces the before/after numbers for your README.

For end-to-end answer quality (faithfulness, answer relevancy) you can layer
Ragas on top — see eval/ragas_eval.py — but the retrieval metrics below need no
extra API calls and are the clearest demonstration of the retrieve-rerank win.
"""

# Import config first so `.env` is loaded before Embedder (which reads
# OPENAI_API_KEY) is constructed.
from src import config
config.require("OPENAI_API_KEY")

import json
from pathlib import Path

from src.embeddings import Embedder
from src.retriever import HybridRetriever
from src.vector_store import VectorStore

from .metrics import aggregate, hit_rate_at_k, mrr_at_k, recall_at_k

K = 5

CONFIGS = {
    "semantic_only":      {"use_keyword": False, "use_rerank": False},
    "hybrid":             {"use_keyword": True,  "use_rerank": False},
    "hybrid_plus_rerank": {"use_keyword": True,  "use_rerank": True},
}


def load_gold() -> list[dict]:
    path = Path("eval/gold.json")
    if not path.exists():
        raise SystemExit("eval/gold.json not found — create it with question/relevant_ids pairs.")
    return json.loads(path.read_text())


def evaluate_config(name: str, flags: dict, gold: list[dict], store, embedder) -> dict:
    retriever = HybridRetriever(store, embedder, **flags)
    per_query = []
    for item in gold:
        relevant = set(item["relevant_doc_ids"])
        retrieved = retriever.retrieve(item["question"], top_k=K)
        # Evaluate at the document level: which doc each retrieved chunk came
        # from. This keeps the gold set stable even if chunking parameters change.
        doc_ids = [r.chunk_id.split("::")[0] for r in retrieved]
        per_query.append({
            f"hit_rate@{K}": hit_rate_at_k(doc_ids, relevant, K),
            f"mrr@{K}": mrr_at_k(doc_ids, relevant, K),
            f"recall@{K}": recall_at_k(doc_ids, relevant, K),
        })
    result = aggregate(per_query)
    result["config"] = name
    return result


def main():
    gold = load_gold()
    store, embedder = VectorStore(), Embedder()
    if store.count() == 0:
        raise SystemExit("Vector store is empty — run `python -m src.ingest` first.")

    rows = [evaluate_config(name, flags, gold, store, embedder)
            for name, flags in CONFIGS.items()]

    # Print a simple comparison table.
    metrics = [f"hit_rate@{K}", f"mrr@{K}", f"recall@{K}"]
    header = f"{'config':<22}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['config']:<22}" + "".join(f"{row[m]:>14.3f}" for m in metrics)
        print(line)


if __name__ == "__main__":
    main()
