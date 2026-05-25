"""
ChromaDB persistent storage for candidates.
Used for storage, metadata, duplicate prevention, and analytics — NOT RAG retrieval.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from utils.helpers import PROJECT_ROOT, content_hash

CHROMA_PATH = PROJECT_ROOT / "chroma_data"
COLLECTION_NAME = "candidates"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = None
_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def check_duplicate(text: str) -> Optional[dict[str, Any]]:
    """Return existing candidate metadata if content hash matches."""
    coll = get_collection()
    h = content_hash(text)
    results = coll.get(where={"content_hash": h})
    if results and results.get("ids"):
        meta = results["metadatas"][0] if results.get("metadatas") else {}
        return {"id": results["ids"][0], "metadata": meta}
    return None


def add_candidate_record(
    candidate_id: str,
    document_text: str,
    metadata: dict[str, Any],
) -> str:
    """Store candidate document with embedding and metadata."""
    coll = get_collection()
    embedding = embed_texts([document_text[:8000]])[0]
    h = content_hash(document_text)
    metadata = {**metadata, "content_hash": h}
    # Chroma metadata values must be str, int, float, or bool
    flat_meta = _flatten_metadata(metadata)

    coll.upsert(
        ids=[candidate_id],
        documents=[document_text[:10000]],
        embeddings=[embedding],
        metadatas=[flat_meta],
    )
    return candidate_id


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def _unflatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta or {})
    for key in ("skills", "tags", "entities", "scores", "file_types"):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


def get_candidate(candidate_id: str) -> Optional[dict[str, Any]]:
    coll = get_collection()
    results = coll.get(ids=[candidate_id], include=["documents", "metadatas"])
    if not results or not results.get("ids"):
        return None
    return {
        "id": results["ids"][0],
        "document": results["documents"][0] if results.get("documents") else "",
        "metadata": _unflatten_metadata(results["metadatas"][0]),
    }


def list_all_candidates() -> list[dict[str, Any]]:
    coll = get_collection()
    results = coll.get(include=["metadatas", "documents"])
    items = []
    if not results or not results.get("ids"):
        return items
    for i, cid in enumerate(results["ids"]):
        items.append(
            {
                "id": cid,
                "document": (results["documents"][i] or "")[:500],
                "metadata": _unflatten_metadata(results["metadatas"][i]),
            }
        )
    return items


def get_database_stats() -> dict[str, Any]:
    candidates = list_all_candidates()
    type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for c in candidates:
        meta = c.get("metadata", {})
        for ft in meta.get("file_types", []) if isinstance(meta.get("file_types"), list) else []:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        for tag in meta.get("tags", []) if isinstance(meta.get("tags"), list) else []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "total_candidates": len(candidates),
        "type_counts": type_counts,
        "tag_counts": tag_counts,
    }


def find_similar_candidates(text: str, threshold: float = 0.92) -> list[dict[str, Any]]:
    """Duplicate detection via embedding similarity (not RAG Q&A)."""
    coll = get_collection()
    count = coll.count()
    if count == 0:
        return []
    embedding = embed_texts([text[:8000]])[0]
    results = coll.query(
        query_embeddings=[embedding],
        n_results=min(5, count),
        include=["metadatas", "distances"],
    )
    similar = []
    if results and results.get("ids") and results["ids"][0]:
        for i, cid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i] if results.get("distances") else 1.0
            # cosine distance: lower is more similar; convert to similarity
            similarity = 1 - dist
            if similarity >= threshold:
                similar.append(
                    {
                        "id": cid,
                        "similarity": round(similarity, 3),
                        "metadata": _unflatten_metadata(results["metadatas"][0][i]),
                    }
                )
    return similar


def reset_database() -> None:
    global _client
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
    _client = None


def new_candidate_id() -> str:
    return str(uuid.uuid4())
