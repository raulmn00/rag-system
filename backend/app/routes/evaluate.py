"""POST /evaluate — reference-free quality scoring of a single answer."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rag_core.evaluation import Evaluator

from ..limiter import limiter


router = APIRouter()
logger = logging.getLogger("rag-backend.evaluate")


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


class EvaluateResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float


@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("10/minute;60/hour;200/day")
def evaluate(request: Request, req: EvaluateRequest) -> EvaluateResponse:
    """Score one (question, answer, contexts) triple with the in-tree
    LLM-judged metrics from rag_core.evaluation:

      - faithfulness: every factual claim in the answer is supported
        by the retrieved contexts. The direct hallucination measure.
      - answer_relevancy: does the answer actually address the
        question? Computed by generating K alternate questions the
        answer would plausibly answer and averaging the cosine
        similarity of their embeddings against the original question.

    context_precision and context_recall (Ragas surfaces them) need a
    ground-truth reference we don't have at request time, so they're
    not part of this endpoint.

    SLOW: ~3 LLM judge calls + 1 batched embeddings call. Expect
    ~3-8s. The caller is expected to surface this with its own
    loading state, separate from /ask.
    """
    try:
        evaluator = Evaluator()
        scores = evaluator.score(
            question=req.question,
            answer=req.answer,
            contexts=req.contexts,
        )
    except Exception:
        # Anything from the evaluator (OpenAI rate limit, bad JSON from
        # the judge, missing env var) gets logged for ops and returned
        # opaquely. The actual exception never reaches the client.
        logger.exception("Evaluator run failed")
        raise HTTPException(
            status_code=502,
            detail="Não foi possível avaliar a resposta. Tente novamente.",
        )

    return EvaluateResponse(
        faithfulness=scores.faithfulness,
        answer_relevancy=scores.answer_relevancy,
    )
