"""Centralized environment configuration.

Importing this module triggers `.env` loading via python-dotenv:
  - Idempotent (safe to import from multiple entry points).
  - Does NOT override variables already set in the process environment —
    production hosts (Railway, Render, Cloud Run) inject env vars
    directly and those must always win over a developer's local `.env`.
  - If no `.env` file exists, `load_dotenv()` is a silent no-op.

Entry points should import this module **before** any module that reads
API keys from `os.environ` (e.g. `src.embeddings`, `src.generation`).
That way the `.env` is populated before any client construction
attempts to read from it.

For entry points that want fail-fast behavior on a missing key, call
`require("OPENAI_API_KEY", ...)` after the import — it prints a clear
message and exits cleanly rather than letting a deep KeyError surface
from inside an SDK constructor.
"""

import os
import sys

from dotenv import load_dotenv

# Load at module import time. By default, dotenv does not override
# existing env vars, which is the correct behavior here: a key already
# set in the environment (e.g. a production secret) takes priority over
# the local `.env` file.
load_dotenv()


def require(*names: str) -> None:
    """Exit with a clear message if any of the named env vars is unset.

    Grouped into a single message so the user can fix every missing key
    in one pass instead of running the command N times. Writes to
    stderr and exits with code 1 — never raises, so it composes cleanly
    with shell-level error handling.
    """
    missing = [name for name in names if not os.environ.get(name)]
    if not missing:
        return

    listed = ", ".join(missing)
    print(
        f"Missing required environment variable(s): {listed}.\n"
        f"Set them in your shell, or copy .env.example to .env and fill in the values.",
        file=sys.stderr,
    )
    sys.exit(1)
