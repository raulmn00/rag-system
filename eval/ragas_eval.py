"""
End-to-end answer-quality evaluation with Ragas.

While eval/run_eval.py measures *retrieval* quality (did we fetch the right
chunks?), this measures *answer* quality (is the generated answer any good?).
The two are complementary: a fluent answer built on the wrong context is still
wrong, and perfect retrieval wasted by a bad prompt is also wrong.

Metrics (all LLM-judged, so this DOES make API calls — unlike run_eval.py):
  - faithfulness        : is every claim in the answer supported by the
                          retrieved context? (the direct hallucination measure)
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : are the retrieved chunks relevant, and ranked well?
  - context_recall      : does the retrieved context cover the reference answer?
                          (requires ground_truth, which eval/gold.json provides)

Run:
    pip install ragas datasets        # optional extras, see requirements.txt
    export OPENAI_API_KEY=sk-...
    python -m eval.ragas_eval

Note on cost: this calls the judge LLM several times per question. With the
8-item sample set it is a few cents; keep an eye on it if you expand the set.

Ragas's public API has shifted across releases; this targets ragas >= 0.2.
If your installed version differs, the import block below points to what to
adjust.
"""

# Import config first so `.env` is loaded before any LLM call. Both the
# pipeline (deferred-imported below) and Ragas's judge LLM read
# OPENAI_API_KEY from the process environment.
from src import config
config.require("OPENAI_API_KEY")

import json
import sys
from pathlib import Path


def _load_ragas():
    """Import Ragas lazily with a clear message if it isn't installed."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print(
            "Ragas (and `datasets`) are optional and not installed.\n"
            "Install them with:  pip install ragas datasets\n"
            f"Original import error: {exc}"
        )
        sys.exit(1)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return Dataset, evaluate, metrics


def build_samples(gold: list[dict], use_keyword: bool, use_rerank: bool) -> dict:
    """Run the pipeline over the gold questions and assemble the Ragas dataset.

    Ragas 0.2 expects these column names:
      question | answer | contexts (list[str]) | ground_truth
    """
    from src.pipeline import RAGPipeline  # deferred: pulls in the heavy retriever

    pipeline = RAGPipeline(use_keyword=use_keyword, use_rerank=use_rerank)

    questions, answers, contexts, ground_truths = [], [], [], []
    for item in gold:
        result = pipeline.answer(item["question"])
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([s.text for s in result.sources])
        ground_truths.append(item["ground_truth"])
        print(f"  ran: {item['question'][:60]}...")

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    Dataset, evaluate, metrics = _load_ragas()

    gold_path = Path("eval/gold.json")
    if not gold_path.exists():
        raise SystemExit("eval/gold.json not found.")
    gold = json.loads(gold_path.read_text())

    # Evaluate the full pipeline (hybrid + rerank). Flip the flags to compare
    # configurations — e.g. measure how much faithfulness depends on re-ranking.
    print("Running pipeline over the gold set (this calls the LLM)...")
    data = build_samples(gold, use_keyword=True, use_rerank=True)

    dataset = Dataset.from_dict(data)
    print("\nScoring with Ragas (LLM-judged)...")
    scores = evaluate(dataset, metrics=metrics)

    print("\n=== Ragas answer-quality scores ===")
    # `scores` prints as a dict-like of metric -> mean score.
    print(scores)


if __name__ == "__main__":
    main()
