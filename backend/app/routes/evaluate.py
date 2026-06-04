"""POST /evaluate — reference-free Ragas evaluation of a single answer."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


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
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Score one (question, answer, contexts) triple with reference-free Ragas
    metrics:

      - faithfulness: every claim in the answer is supported by the
        retrieved contexts. The direct hallucination measure.
      - answer_relevancy: does the answer actually address the question?

    context_precision and context_recall are deliberately NOT included.
    Both require a reference (ground-truth) answer that we don't have at
    request time — the whole point of /evaluate is to score answers that
    were just generated, in isolation from any gold set.

    SLOW: each metric makes multiple LLM judge calls. Expect ~5-15s per
    request. The caller is expected to surface this with its own loading
    state, separate from /ask.
    """
    # Ragas is an optional extra (see rag-core/pyproject.toml [eval]).
    # If the deploy didn't install it, surface that explicitly rather
    # than crashing the worker.
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Avaliador opcional não está habilitado neste deploy. "
                "Instale com: pip install -e ./rag-core[eval]"
            ),
        )

    dataset = Dataset.from_dict({
        "question": [req.question],
        "answer": [req.answer],
        "contexts": [req.contexts],
    })

    try:
        result = ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        # Ragas 0.2 returns a Result object with a `.to_pandas()` view.
        df = result.to_pandas()
        return EvaluateResponse(
            faithfulness=float(df["faithfulness"].iloc[0]),
            answer_relevancy=float(df["answer_relevancy"].iloc[0]),
        )
    except Exception:
        # Anything from Ragas (LLM failure, schema drift across versions,
        # rate limits) gets logged for ops and returned opaquely. The
        # real exception is never echoed to the client.
        logger.exception("Ragas evaluation failed")
        raise HTTPException(
            status_code=502,
            detail="Não foi possível avaliar a resposta. Tente novamente.",
        )
