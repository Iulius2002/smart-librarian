# src/vector_store.py
from __future__ import annotations
import os
import re
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY
from openai import OpenAI

# --- Embeddings (OpenAI) ---
_client = OpenAI(api_key=OPENAI_API_KEY or None)

def embed(texts: List[str]) -> List[List[float]]:
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# --- Chroma setup ---
os.makedirs(CHROMA_DIR, exist_ok=True)
_chroma = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

_COL_NAME = "smart_librarian"
_collection = _chroma.get_or_create_collection(
    name=_COL_NAME,
    metadata={"hnsw:space": "cosine"}
)

# --- API public pentru ingestie ---
def reset_collection():
    try:
        _chroma.delete_collection(_COL_NAME)
    except Exception:
        pass
    global _collection
    _collection = _chroma.get_or_create_collection(name=_COL_NAME, metadata={"hnsw:space": "cosine"})

def add_chunks(chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
    vecs = embed(chunks)
    _collection.add(documents=chunks, embeddings=vecs, metadatas=metadatas, ids=ids)

# --- helper: atașează 'where' doar dacă există filtre reale ---
def _build_query_kwargs(q: str, n_results: int, where: Optional[Dict[str, Any]]):
    kwargs = dict(
        query_embeddings=embed([q]),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where and len(where) > 0:
        # Lăsăm utilizatorul să trimită un 'where' corect (ex: {"$and":[{"author":{"$eq":"Tolkien"}}]})
        kwargs["where"] = where
    # altfel, NU trimitem 'where' deloc (evităm eroarea „Expected where to have exactly one operator, got {}”)
    return kwargs

# --- căutare brută (semantică) ---
def query_raw(q: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kwargs = _build_query_kwargs(q, n_results, where)
    res = _collection.query(**kwargs)
    return res

# --- util: scor lexical simplu (overlap cuvinte) ---
_word_re = re.compile(r"\w+", re.UNICODE)

def _norm_words(s: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(s or "")]

def _lexical_overlap_score(q: str, text: str) -> float:
    q_words = set(_norm_words(q))
    if not q_words:
        return 0.0
    t_words = _norm_words(text)
    if not t_words:
        return 0.0
    hit = sum(1 for w in t_words if w in q_words)
    return hit / max(6, len(t_words))

# --- reranking pe reguli: semantic + boost pe titlu/teme/autor ---
def search_with_rerank(q: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    1) ia top-N semantic (N = k*3, min 12)
    2) calculează scor compozit:
       score = 0.60 * semantic + 0.25 * lexical + boosts(titlu/teme/autor)
    3) ordonează și returnează primele k
    """
    n = max(12, k * 3)
    where = filters if (filters and len(filters) > 0) else None
    raw = query_raw(q, n_results=n, where=where)

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    items: List[Tuple[float, Dict[str, Any], str]] = []
    q_low = q.lower()

    for doc, meta, dist in zip(docs, metas, dists):
        sem = 1.0 - float(dist)
        sem = 0.0 if sem < 0 else (1.0 if sem > 1 else sem)

        lex = _lexical_overlap_score(q, doc)

        boost = 0.0
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        themes_val = meta.get("themes") or ""
        # în ingest, themes este string; îl „expandăm” ușor pentru boost
        themes_list = [t.strip().lower() for t in str(themes_val).split(",") if t.strip()]

        if title:
            t_low = title.lower()
            if q_low.strip() == t_low:
                boost += 0.20
            elif t_low in q_low or q_low in t_low:
                boost += 0.12
            else:
                common = len(set(_norm_words(title)) & set(_norm_words(q)))
                boost += min(0.10, 0.02 * common)

        for th in themes_list:
            if th and th in q_low:
                boost += 0.04

        if author and author.lower() in q_low:
            boost += 0.06

        score = 0.60 * sem + 0.25 * lex + boost
        items.append((score, meta, doc))

    items.sort(key=lambda x: x[0], reverse=True)
    results = [{"document": d, "metadata": m, "score": round(float(s), 4)} for (s, m, d) in items[:k]]
    return results
