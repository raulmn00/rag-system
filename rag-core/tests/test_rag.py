"""Tests for the pure, no-API parts of the system."""

from rag_core.chunking import chunk_text
from rag_core.fusion import reciprocal_rank_fusion
from eval.metrics import hit_rate_at_k, mrr_at_k, recall_at_k


# --- chunking --------------------------------------------------------------

def test_chunking_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, doc_id="d", source="d.txt", chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.doc_id == "d" for c in chunks)
    assert all(c.chunk_id.startswith("d::") for c in chunks)


def test_chunking_rejects_bad_overlap():
    try:
        chunk_text("hello world", doc_id="d", source="d", chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- fusion ----------------------------------------------------------------

def test_rrf_rewards_items_ranked_high_in_both_lists():
    a = ["x", "y", "z"]
    b = ["x", "z", "y"]
    fused = reciprocal_rank_fusion([a, b])
    # x is first in both -> should top the fused ranking.
    assert max(fused, key=fused.get) == "x"


# --- metrics ---------------------------------------------------------------

def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["a", "b", "c"], {"b"}, 3) == 0.5   # rank 2 -> 1/2
    assert mrr_at_k(["a", "b", "c"], {"a"}, 3) == 1.0


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
