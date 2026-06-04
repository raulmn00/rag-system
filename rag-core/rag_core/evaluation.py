"""Self-contained evaluation metrics for RAG quality.

Two LLM-as-judge metrics, implemented directly against the OpenAI SDK
instead of via a third-party eval framework. The metric definitions
are public knowledge (see Ragas, TruLens, RAGAS paper) — wiring them
ourselves keeps the dependency tree small and the math transparent.

Metrics:

- faithfulness: every factual claim in the answer is supported by the
  retrieved contexts. Direct hallucination measure. Range [0, 1],
  where 1.0 means "every claim was supported" (or, by convention, the
  answer made no checkable claims at all).

- answer_relevancy: the answer actually addresses the question. Range
  [0, 1] — the score is the mean cosine similarity between the
  original question's embedding and the embeddings of K alternate
  questions that the model says this answer plausibly answers.

Cost per /evaluate call (gpt-4o-mini + text-embedding-3-small):
  ~3 chat completions + 1 batched embeddings call ≈ $0.0005.
"""

import json
import os
from dataclasses import dataclass

from openai import OpenAI


JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# Number of paraphrased questions to generate for answer_relevancy.
# Higher = smoother score but more tokens. 4 is the value Ragas defaults to.
RELEVANCY_QUESTIONS_K = 4


@dataclass
class EvaluationScores:
    faithfulness: float
    answer_relevancy: float


# --- Prompts ----------------------------------------------------------------
#
# All judge prompts request structured JSON. We don't use the OpenAI
# response_format=json_schema parameter so we can support older API
# versions; instead we use json_object mode and parse defensively.

_CLAIM_EXTRACTION_PROMPT = """\
You extract atomic factual claims from an answer.

Given the answer below, decompose it into a flat list of self-contained \
factual claims. Each claim should be a single statement that could be \
independently verified. Ignore opinions, hedges, and meta-commentary.

If the answer is a refusal, an apology, or otherwise contains no \
verifiable factual claims, return an empty list.

Answer:
\"\"\"{answer}\"\"\"

Respond as JSON: {{"claims": ["...", "..."]}}"""


_VERDICT_PROMPT = """\
You judge whether claims are supported by a set of context passages.

For each numbered claim below, determine whether the claim is supported \
by the contexts. A claim is "supported" if it follows from the contexts \
(paraphrasing is fine). It is "unsupported" if the contexts don't \
contain enough information to verify it.

Contexts:
{contexts_block}

Claims:
{claims_block}

Respond as JSON: {{"verdicts": [{{"index": <int>, "supported": <bool>}}, ...]}}"""


_ALT_QUESTIONS_PROMPT = """\
You generate alternative questions an answer could be addressing.

Given the answer below, generate exactly {k} diverse questions that this \
answer could plausibly be answering. The questions should differ in \
wording and angle. Don't invent topics that aren't in the answer.

Answer:
\"\"\"{answer}\"\"\"

Respond as JSON: {{"questions": ["...", "..."]}}"""


# --- Evaluator --------------------------------------------------------------


class Evaluator:
    """Computes faithfulness and answer-relevancy via OpenAI.

    Constructed per-request in the backend route; could be cached
    later if /evaluate becomes hot.
    """

    def __init__(
        self,
        api_key: str | None = None,
        judge_model: str = JUDGE_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.judge_model = judge_model
        self.embedding_model = embedding_model

    def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationScores:
        return EvaluationScores(
            faithfulness=self.faithfulness(answer, contexts),
            answer_relevancy=self.answer_relevancy(question, answer),
        )

    # ----- faithfulness ----------------------------------------------------

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        if not answer.strip():
            return 0.0
        claims = self._extract_claims(answer)
        if not claims:
            # No checkable claims in the answer (refusal, apology). By
            # convention this is "vacuously faithful" — the answer
            # didn't hallucinate because it didn't claim anything.
            return 1.0
        if not contexts:
            # Claims exist but no contexts were retrieved — every claim
            # is unsupported by definition.
            return 0.0
        verdicts = self._judge_claims(claims, contexts)
        supported = sum(1 for v in verdicts if v)
        return supported / len(claims)

    def _extract_claims(self, answer: str) -> list[str]:
        prompt = _CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        data = self._chat_json(prompt)
        claims = data.get("claims", [])
        return [c for c in claims if isinstance(c, str) and c.strip()]

    def _judge_claims(self, claims: list[str], contexts: list[str]) -> list[bool]:
        contexts_block = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(contexts)
        )
        claims_block = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(claims)
        )
        prompt = _VERDICT_PROMPT.format(
            contexts_block=contexts_block,
            claims_block=claims_block,
        )
        data = self._chat_json(prompt)
        verdicts_raw = data.get("verdicts", [])

        # Index → bool. Indices the LLM forgot become False (unsupported),
        # which is the conservative default.
        by_index: dict[int, bool] = {}
        for v in verdicts_raw:
            if isinstance(v, dict) and "index" in v and "supported" in v:
                try:
                    by_index[int(v["index"])] = bool(v["supported"])
                except (TypeError, ValueError):
                    continue
        return [by_index.get(i + 1, False) for i in range(len(claims))]

    # ----- answer_relevancy ------------------------------------------------

    def answer_relevancy(self, question: str, answer: str) -> float:
        if not question.strip() or not answer.strip():
            return 0.0
        generated = self._generate_alt_questions(answer, RELEVANCY_QUESTIONS_K)
        if not generated:
            return 0.0
        # One batched embedding call: [original, generated_1, ..., generated_K]
        embeddings = self._embed([question, *generated])
        original_emb = embeddings[0]
        gen_embs = embeddings[1:]
        sims = [_cosine_similarity(original_emb, e) for e in gen_embs]
        score = sum(sims) / len(sims)
        # Clamp to [0, 1]. Cosine sim is technically in [-1, 1] but
        # embedding similarities between same-language text are almost
        # always non-negative; clamp just in case.
        return max(0.0, min(1.0, score))

    def _generate_alt_questions(self, answer: str, k: int) -> list[str]:
        prompt = _ALT_QUESTIONS_PROMPT.format(answer=answer, k=k)
        data = self._chat_json(prompt)
        questions = data.get("questions", [])
        return [q for q in questions if isinstance(q, str) and q.strip()]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # ----- LLM helpers -----------------------------------------------------

    def _chat_json(self, prompt: str) -> dict:
        """Run a judge prompt with JSON mode. Returns the parsed object,
        or an empty dict on failure — every call site guards on .get()
        so a malformed response degrades gracefully (score = 0 etc.)."""
        resp = self.client.chat.completions.create(
            model=self.judge_model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


# --- Math -------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Embeddings are <2k dims; numpy
    would be overkill for the per-request math we do here."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
