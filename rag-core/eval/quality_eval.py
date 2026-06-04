"""
End-to-end answer-quality evaluation against the gold set.

While eval/run_eval.py measures *retrieval* quality (did we fetch the
right chunks?), this measures *answer* quality (is the generated answer
any good?). The two are complementary: a fluent answer built on the
wrong context is still wrong, and perfect retrieval wasted by a bad
prompt is also wrong.

Metrics (both LLM-judged via rag_core.evaluation.Evaluator):

  - faithfulness     : is every claim in the answer supported by the
                       retrieved context? (the direct hallucination
                       measure)
  - answer_relevancy : does the answer actually address the question?

Implemented in-tree rather than via Ragas — see rag_core/evaluation.py
for the prompts and the math. Same definitions Ragas uses, with two
fewer dependencies and no upstream incompatibilities to dodge.

Run:
    export OPENAI_API_KEY=sk-...
    python -m eval.quality_eval

Cost: 3 LLM judge calls + 1 embedding call per question, on
gpt-4o-mini. With the 8-item sample set that's a fraction of a cent;
keep an eye on it if you expand the set.
"""

# Import config first so `.env` is loaded before any LLM call.
from rag_core import config
config.require("OPENAI_API_KEY")

import json
from pathlib import Path

from rag_core.evaluation import Evaluator


def evaluate_pipeline(
    gold: list[dict],
    use_keyword: bool,
    use_rerank: bool,
) -> dict[str, float]:
    """Run the RAG pipeline over the gold set, score each answer, and
    return mean faithfulness + mean answer_relevancy across the batch.
    """
    # Deferred import — RAGPipeline pulls in the cross-encoder model and
    # the rest of the retriever stack. If the user only wants the eval
    # logic, they shouldn't pay that import cost up front.
    from rag_core.pipeline import RAGPipeline

    pipeline = RAGPipeline(use_keyword=use_keyword, use_rerank=use_rerank)
    evaluator = Evaluator()

    faith_scores: list[float] = []
    relevancy_scores: list[float] = []
    for item in gold:
        question = item["question"]
        result = pipeline.answer(question)
        contexts = [s.text for s in result.sources]
        scores = evaluator.score(question, result.answer, contexts)
        faith_scores.append(scores.faithfulness)
        relevancy_scores.append(scores.answer_relevancy)
        print(
            f"  scored: {question[:60]}... "
            f"faith={scores.faithfulness:.2f} "
            f"relevancy={scores.answer_relevancy:.2f}"
        )

    return {
        "faithfulness": sum(faith_scores) / len(faith_scores),
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores),
    }


def main():
    gold_path = Path("eval/gold.json")
    if not gold_path.exists():
        raise SystemExit("eval/gold.json not found.")
    gold = json.loads(gold_path.read_text())

    print(f"Running pipeline + scoring over {len(gold)} gold questions...")
    averages = evaluate_pipeline(gold, use_keyword=True, use_rerank=True)

    print("\n=== Answer-quality averages (hybrid + rerank) ===")
    for name, value in averages.items():
        print(f"  {name:<20} {value:.3f}")


if __name__ == "__main__":
    main()
