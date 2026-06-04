"""
Retrieval evaluation metrics — pure functions, no API calls.

Given a gold set (question -> the chunk/doc ids that should be retrieved), we
measure how good the retriever is. These are the standard first-stage metrics:

- Hit Rate@k: fraction of questions where at least one relevant chunk is in the
  top-k. "Did we surface anything useful at all?"
- MRR@k (Mean Reciprocal Rank): 1/rank of the first relevant chunk, averaged.
  Rewards putting the right chunk near the top — this is what re-ranking moves.
- Recall@k: fraction of all relevant chunks that made it into the top-k.

Reporting these for different retriever configs (semantic-only vs +keyword
vs +rerank) is the single most convincing thing you can show in an interview:
it turns "I added re-ranking" into "re-ranking raised MRR@5 from 0.62 to 0.81".
"""


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def aggregate(per_query: list[dict]) -> dict:
    """Average each metric across all queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
