"""
Vector store: a thin wrapper over ChromaDB (persistent, local, zero-config).

ChromaDB is chosen for the demo because it runs in-process with on-disk
persistence — no separate server, no SaaS account, nothing to break live.
Qdrant or pgvector would be the production swap; the methods below
(add / query) are the only surface the rest of the system depends on.
"""

import chromadb

from .chunking import Chunk

COLLECTION = "documents"


class VectorStore:
    def __init__(self, path: str = "data/chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "source": c.source} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return the top_k nearest chunks with their cosine distance."""
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            out.append({
                "chunk_id": cid,
                "text": doc,
                "source": meta["source"],
                "doc_id": meta["doc_id"],
                "score": 1.0 - dist,  # cosine distance -> similarity
            })
        return out

    def all_documents(self) -> list[dict]:
        """Every stored chunk — used to build the BM25 keyword index."""
        res = self.collection.get()
        return [
            {"chunk_id": cid, "text": doc, "source": meta["source"], "doc_id": meta["doc_id"]}
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def count(self) -> int:
        return self.collection.count()
