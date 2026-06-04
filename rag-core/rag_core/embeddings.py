"""
Embeddings via the OpenAI API.

We use `text-embedding-3-small` (1536 dims): cheap, fast, and strong enough for
most retrieval tasks. Swapping to `text-embedding-3-large` is a one-line change
when quality matters more than cost. The interface is deliberately tiny so a
local sentence-transformers backend could be dropped in behind the same two
methods.
"""

import os

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"


class Embedder:
    def __init__(self, api_key: str | None = None, model: str = EMBED_MODEL):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (one API call). Order is preserved."""
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
