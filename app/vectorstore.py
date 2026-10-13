"""Pinecone vector DB + sentence-transformers embeddings."""
from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from .config import settings


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index not in pc.list_indexes().names():
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
        )
    return pc.Index(settings.pinecone_index)


def embed(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def upsert(chunks: list[dict], batch: int = 100) -> int:
    vectors = embed([c["text"] for c in chunks])
    items = [
        {"id": c["id"], "values": v,
         "metadata": {"text": c["text"], "source": c["source"]}}
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(items), batch):
        index().upsert(vectors=items[i:i + batch], namespace=settings.namespace)
    return len(items)


def query(text: str, k: int | None = None) -> list[dict]:
    res = index().query(vector=embed([text])[0], top_k=k or settings.top_k,
                        include_metadata=True, namespace=settings.namespace)
    return [
        {"id": m["id"], "score": float(m["score"]),
         "text": m["metadata"]["text"], "source": m["metadata"].get("source", "unknown")}
        for m in res["matches"] if m["score"] >= settings.min_score
    ]
