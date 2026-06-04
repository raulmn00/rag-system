"""POST /ask — retrieve relevant passages and answer the question with citations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_core.pipeline import RAGPipeline


router = APIRouter()


class AskRequest(BaseModel):
    question: str
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
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    pipeline = RAGPipeline(use_keyword=req.use_keyword, use_rerank=req.use_rerank)
    result = pipeline.answer(req.question, top_k=req.top_k)
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceOut(chunk_id=s.chunk_id, source=s.source, score=s.score, text=s.text)
            for s in result.sources
        ],
    )
