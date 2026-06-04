"""
FastAPI app entry point.

Per-endpoint logic lives in app.routes.* — this file only wires the
application: env loading, CORS, and router includes. New endpoints
register here in one line.
"""

# rag_core.config is imported first so .env is loaded (and the required
# API keys validated) before anything below constructs an OpenAI client.
from rag_core import config
config.require("OPENAI_API_KEY")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import ask, evaluate, health, upload


app = FastAPI(title="RAG System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(upload.router)
app.include_router(evaluate.router)
