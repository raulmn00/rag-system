"""
The RAG pipeline: ties retrieval and generation together.

    answer, sources = RAGPipeline().answer("What is multi-head attention?")

Retrieval flags (use_keyword, use_rerank) are passed through so the eval harness
can A/B the configurations.
"""

from dataclasses import dataclass

from .embeddings import Embedder
from .generation import Generator
from .retriever import HybridRetriever, Retrieved
from .vector_store import VectorStore


@dataclass
class RAGResult:
    answer: str
    sources: list[Retrieved]


class RAGPipeline:
    def __init__(self, use_keyword: bool = True, use_rerank: bool = True):
        self.store = VectorStore()
        self.embedder = Embedder()
        self.retriever = HybridRetriever(
            self.store, self.embedder,
            use_keyword=use_keyword, use_rerank=use_rerank,
        )
        self.generator = Generator()

    def answer(self, question: str, top_k: int = 5) -> RAGResult:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, chunks)
        return RAGResult(answer=answer, sources=chunks)
