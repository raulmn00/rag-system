"""GET / — liveness/readiness probe for whichever platform runs us."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health():
    return {"status": "ok", "service": "rag-system"}
