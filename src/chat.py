# src/chat.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from .config import OPENAI_API_KEY, CHAT_MODEL
from .vector_store import search_with_rerank

client = OpenAI(api_key=OPENAI_API_KEY or None)

# ---- Retrieval helper (cu reranking) ----
def build_context_snippets(question: str, k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return search_with_rerank(question, k=k, filters=filters)

def _format_context(snippets: List[Dict[str, Any]]) -> str:
    lines = []
    for s in snippets:
        meta = s.get("metadata", {})
        title = meta.get("title", "N/A")
        author = meta.get("author", "")
        score = s.get("score", 0.0)
        lines.append(f"- {title} ({author}) [score={score}]")
    return "\n".join(lines)

def chat_once(user_question: str, k: int = 3, filters: Optional[Dict[str, Any]] = None, return_usage: bool = False) -> str | Tuple[str, Dict[str, int]]:
    """
    Dacă return_usage=True => întoarce (answer, {"prompt_tokens":..,"completion_tokens":..,"total_tokens":..})
    """
    snippets = build_context_snippets(user_question, k=k, filters=filters)
    context_bullets = _format_context(snippets)

    system = (
        "Ești Smart Librarian, un asistent pentru recomandări de cărți. "
        "Ai acces la fragmente indexate (RAG). "
        "Răspunde în română, clar și prietenos. "
        "Dacă utilizatorul cere o carte anume, oferă un scurt rezumat. "
        "Nu inventa titluri — rămâi la sursele disponibile."
    )
    content = (
        f"Întrebare: {user_question}\n\n"
        f"Fragmente relevante (top-{k}):\n{context_bullets}\n\n"
        f"Instrucțiuni:\n"
        f"- Recomandă 1-2 titluri, justifică pe scurt în funcție de teme/autor.\n"
        f"- Dacă utilizatorul a cerut un titlu concret, include pe scurt esența poveștii.\n"
        f"- Fii concis (max ~8–10 linii)."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content}
        ],
        temperature=0.6,
        max_tokens=450,
    )
    answer = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return (answer, usage_dict) if return_usage else answer
