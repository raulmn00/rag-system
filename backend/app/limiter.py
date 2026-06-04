"""Per-IP rate limiting wiring.

slowapi exposes a single Limiter that every route can decorate. Keeping
the limiter in its own module (rather than in api.py) avoids the
circular import that would otherwise happen when each route file
imports from api.py to grab the same limiter instance.

Key callable is `get_client_ip`: Cloud Run terminates the TCP connection
at Google Front End, so `request.client.host` is always the GFE — useless
for rate limiting. The real caller's address is in the first entry of
the X-Forwarded-For header. Fall back to the direct connection in local
dev (where there's no proxy).
"""

from fastapi import Request
from slowapi import Limiter


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_client_ip)
