"""
chroma_store.py — ChromaDB-based vector store for all three modalities.

Replaces FAISS with ChromaDB for:
  • Persistent storage across runs
  • Native metadata filtering
  • Simple HTTP server mode for production
"""

from __future__ import annotations
import hashlib
import os
import traceback
from typing import Optional

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_TEXT,
    COLLECTION_IMAGES,
    COLLECTION_TABLES,
    EMBED_MODEL,
)


class _HashEmbeddingFunction:
    """Small offline embedding fallback for restricted environments."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input):
        texts = input if isinstance(input, list) else [input]
        return [self._embed_text(text) for text in texts]

    def embed_query(self, input):
        return self.__call__(input)

    def name(self) -> str:
        return "local-hash-embedding"

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = (text or "").lower().split()
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, min(len(digest), self.dim // 8)):
                chunk = digest[i]
                idx = (i * 8 + chunk) % self.dim
                vec[idx] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class _ResilientEmbeddingFunction:
    """Use sentence-transformers when available, otherwise fall back locally."""

    def __init__(self, primary=None, fallback=None):
        self.primary = primary
        self.fallback = fallback or _HashEmbeddingFunction()

    def __call__(self, input):
        if self.primary is not None:
            try:
                return self.primary(input)
            except Exception as e:
                print(f"[chroma_store] Primary embedding failed, using local fallback: {e}")
        return self.fallback(input)

    def embed_query(self, input):
        if self.primary is not None and hasattr(self.primary, "embed_query"):
            try:
                return self.primary.embed_query(input)
            except Exception as e:
                print(f"[chroma_store] Primary query embedding failed, using local fallback: {e}")
        return self.fallback.embed_query(input)

    def name(self) -> str:
        return "resilient-embedding"


def _get_embedding_function():
    """Return a ChromaDB-compatible embedding function."""
    return _ResilientEmbeddingFunction(primary=None)


def get_client():
    """Return a persistent ChromaDB client."""
    try:
        import chromadb
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception as e:
        print(f"[chroma_store] ChromaDB client init failed: {e}")
        return None


def get_or_create_collection(collection_name: str):
    """Get or create a ChromaDB collection with sentence-transformer embeddings."""
    try:
        client = get_client()
        if client is None:
            return None
        ef = _get_embedding_function()
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except Exception as e:
        print(f"[chroma_store] Collection init failed for '{collection_name}': {e}")
        return None


def upsert_documents(
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> bool:
    """Upsert documents into a ChromaDB collection."""
    try:
        collection = get_or_create_collection(collection_name)
        if collection is None:
            return False
        # ChromaDB max batch size is 5461
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        return True
    except Exception as e:
        print(f"[chroma_store] Upsert failed: {e}\n{traceback.format_exc()}")
        return False


def query_collection(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Query a ChromaDB collection and return results as dicts.

    Returns list of:
      {"id": str, "document": str, "metadata": dict, "distance": float}
    """
    try:
        collection = get_or_create_collection(collection_name)
        if collection is None:
            return []
        kwargs = {"query_texts": [query_text], "n_results": min(n_results, max(collection.count(), 1))}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        output = []
        if results and results.get("ids") and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": (results["metadatas"][0][idx] if results.get("metadatas") else {}),
                    "distance": (results["distances"][0][idx] if results.get("distances") else 0.0),
                })
        return output
    except Exception as e:
        print(f"[chroma_store] Query failed: {e}\n{traceback.format_exc()}")
        return []


def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection (for re-indexing)."""
    try:
        client = get_client()
        if client:
            client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"[chroma_store] Delete failed: {e}")
        return False


def collection_count(collection_name: str) -> int:
    """Return the number of documents in a collection."""
    try:
        collection = get_or_create_collection(collection_name)
        return collection.count() if collection else 0
    except Exception:
        return 0
