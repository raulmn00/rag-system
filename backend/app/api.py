"""
FastAPI app entry point.

Per-endpoint logic lives in app.routes.* — this file only wires the
application: env loading, CORS, rate-limit handler, and router includes.
New endpoints register here in one line.
"""

# rag_core.config is imported first so .env is loaded (and the required
# API keys validated) before anything below constructs an OpenAI client.
from rag_core import config
config.require("OPENAI_API_KEY")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from .limiter import limiter
from .routes import ask, evaluate, health, upload


app = FastAPI(title="RAG System", version="1.0.0")
app.state.limiter = limiter

# Three Vercel aliases plus the per-deploy hash pattern, plus localhost for
# dev. The wide-open allow_origins=["*"] worked while we were just testing,
# but once the API has a public URL it lets any random site call /ask from
# a browser and burn LLM credits on the project's key. Tightening here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://rag-system-delta-five.vercel.app",
        "https://rag-system-raulmn00s-projects.vercel.app",
        "https://rag-system-raulmn00-raulmn00s-projects.vercel.app",
    ],
    # Per-deploy immutable URLs: rag-system-<hash>-raulmn00s-projects.vercel.app
    allow_origin_regex=r"https://rag-system-[a-z0-9]+-raulmn00s-projects\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    _request: Request,
    _exc: RateLimitExceeded,
) -> JSONResponse:
    # slowapi's default response echoes the exact limit ("10 per 1 minute"),
    # which gives an attacker the exact pace they need to slow to. Keep
    # the message hard-coded so the response carries no calibration intel.
    return JSONResponse(
        status_code=429,
        content={"detail": "Muitas requisições. Tente novamente em alguns instantes."},
    )


app.include_router(health.router)
app.include_router(ask.router)
app.include_router(upload.router)
app.include_router(evaluate.router)
