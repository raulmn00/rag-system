"""
Answer generation, grounded in retrieved chunks and forced to cite sources.

Two anti-hallucination measures:
1. The prompt instructs the model to answer ONLY from the supplied context and
   to say so when the context is insufficient.
2. Each chunk is numbered [1], [2], ... and the model must cite those numbers,
   making every claim traceable back to a source.
"""

import os

from openai import OpenAI

from .retriever import Retrieved

GEN_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You answer questions using ONLY the provided context passages.

Rules:
- Base every statement on the passages. Do not use outside knowledge.
- Cite the passage number(s) in square brackets after each claim, e.g. [2].
- If the passages do not contain the answer, say you don't have enough
  information in the provided sources. Do not guess.
- Be concise and precise."""


class Generator:
    def __init__(self, api_key: str | None = None, model: str = GEN_MODEL):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, question: str, chunks: list[Retrieved]) -> str:
        if not chunks:
            return "No relevant passages were retrieved, so I cannot answer from the sources."

        context = "\n\n".join(
            f"[{i + 1}] (source: {c.source})\n{c.text}"
            for i, c in enumerate(chunks)
        )
        user = f"Context passages:\n\n{context}\n\nQuestion: {question}"

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
