from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from .config import CHROMA_DIR, EMBEDDING_MODEL

# Client OpenAI (folosește cheia din .env via config)
client = OpenAI()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def get_collection():
    """
    Creează/recuperează colecția persistentă 'books' în Chroma.
    hnsw:space=cosine ⇒ căutare după similaritate cosinus (1 - cos_sim e 'distance').
    """
    _ensure_dir(CHROMA_DIR)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    return chroma_client.get_or_create_collection(name="books", metadata={"hnsw:space": "cosine"})

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Obține embedding-uri de la OpenAI. Fiecare string → vector (listă de float-uri).
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def add_documents(docs: List[Dict[str, Any]]) -> None:
    """
    docs = listă de dict-uri:
      - id: str (unic)
      - text: str (conținutul pentru căutare)
      - metadata: dict (ex: {"title": "..."} )
    """
    col = get_collection()
    embeddings = embed_texts([d["text"] for d in docs])

    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d.get("metadata", {}) for d in docs],
        embeddings=embeddings
    )

def query_similar(query_text: str, n_results: int = 3):
    """
    Face embedding pe query și interoghează colecția.
    Returnează dict cu ids, documents, metadatas, distances.
    Notă: la 'cosine', distanță mai mică ⇒ mai apropiat (mai relevant).
    """
    col = get_collection()
    query_embedding = embed_texts([query_text])[0]
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return res
