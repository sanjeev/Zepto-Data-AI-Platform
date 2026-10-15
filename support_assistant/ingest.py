"""Ingestion + embedding stage of the RAG pipeline.

load_documents()  -> reads docs/doc_*.txt
chunk_documents() -> one chunk per document (each policy is a single ~500-char
                     paragraph, so splitting further would separate facts that
                     belong together, e.g. a fee from the threshold it applies to)
build_index()     -> embeds every chunk with all-MiniLM-L6-v2 and upserts it into
                     the ChromaDB collection "zepto_policies" (cosine space)

Run once (the API also calls build_index() on startup if the collection is empty):
    python ingest.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOC_TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def load_documents() -> dict[str, str]:
    docs = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(DOCS_DIR.glob("doc_*.txt"))}
    if len(docs) != 8:
        raise RuntimeError(f"Expected 8 corpus documents in {DOCS_DIR}, found {len(docs)}")
    return docs


def chunk_documents(docs: dict[str, str]) -> list[dict]:
    return [
        {
            "id": f"{doc_id}_chunk_0",
            "text": text,
            "metadata": {"doc_id": doc_id, "title": DOC_TITLES.get(doc_id, doc_id), "chunk_index": 0},
        }
        for doc_id, text in docs.items()
    ]


def embed(texts: list[str]) -> list[list[float]]:
    # Normalized vectors so cosine distance in Chroma == 1 - dot product.
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def build_index() -> int:
    chunks = chunk_documents(load_documents())
    get_collection().upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
    )
    return len(chunks)


def ensure_index() -> None:
    if get_collection().count() < 8:
        build_index()


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Top-k chunks by cosine similarity (similarity = 1 - cosine distance)."""
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    return [
        {
            "id": cid,
            "text": doc,
            "metadata": meta,
            "similarity": round(1 - dist, 4),
        }
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    print(f"Collection count: {get_collection().count()}")
    for q in ["How much is the delivery fee?", "Can I return opened shampoo?", "Is phone support available?"]:
        top = retrieve(q)
        print(f"\nQ: {q}")
        for hit in top:
            print(f"  {hit['id']:<15} sim={hit['similarity']:.3f}  [{hit['metadata']['title']}]")
