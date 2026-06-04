"""POST /ask — retrieve relevant passages and answer the question with citations."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rag_core.pipeline import RAGPipeline

from ..limiter import limiter


router = APIRouter()
logger = logging.getLogger("rag-backend.ask")

# Cap on the question size to keep a bot from forcing the pipeline to feed
# a 100k-character prompt to the embedder and the answer LLM. 2000 chars
# fits any reasonable question by far; well-formed human queries are
# usually <200.
MAX_QUESTION_LENGTH = 2000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    use_keyword: bool = True
    use_rerank: bool = True
    top_k: int = 5


class SourceOut(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@router.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute;60/hour;200/day")
def ask(request: Request, req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    try:
        pipeline = RAGPipeline(use_keyword=req.use_keyword, use_rerank=req.use_rerank)
        result = pipeline.answer(req.question, top_k=req.top_k)
    except Exception:
        # Anything from pipeline construction or .answer (OpenAI errors,
        # Chroma issues, model load failures) gets the same opaque
        # treatment: real cause to the logs, generic message to the client.
        logger.exception("RAGPipeline.answer failed")
        raise HTTPException(
            status_code=502,
            detail="Não foi possível responder. Tente novamente.",
        )

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceOut(chunk_id=s.chunk_id, source=s.source, score=s.score, text=s.text)
            for s in result.sources
        ],
    )
