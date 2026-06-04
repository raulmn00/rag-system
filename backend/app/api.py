"""
FastAPI wrapper for the RAG pipeline.

  GET  /          — health
  POST /ask       — { "question": "...", "use_rerank": true } -> answer + sources
"""

# Import rag_core.config first so `.env` is loaded before any module
# below tries to read `OPENAI_API_KEY` from the process environment.
# rag_core is installed editably (see backend/pyproject.toml); the
# backend depends on it as a domain library and stays as a thin
# transport layer over its pipeline.
from rag_core import config
config.require("OPENAI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core.pipeline import RAGPipeline

app = FastAPI(title="RAG System", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


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


@app.get("/")
def health():
    return {"status": "ok", "service": "rag-system"}


@app.post("/ask", response_model=AskResponse)
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
