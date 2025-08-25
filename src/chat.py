# src/chat.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

from openai import OpenAI

from .config import OPENAI_API_KEY, CHAT_MODEL
from .vector_store import search_with_rerank

client = OpenAI(api_key=OPENAI_API_KEY or None)

# ---- Retrieval helper (cu reranking) ----
def build_context_snippets(question: str, k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returnează o listă de fragmente (doc + meta) potrivite întrebării, deja rerankuite.
    Format compatibil cu server.py:
      [{ "document": "...", "metadata": {...}, "score": 0.87 }, ...]
    """
    return search_with_rerank(question, k=k, filters=filters)

# ---- Tool: get_summary_by_title (mock aici, presupunem că e implementat în alt modul) ----
# În proiectul tău, fie ai deja acest tool, fie modelul îl invocă prin instructions.
# Aici presupunem că finalul deja compune un răspuns cu recomandare + (dacă e cazul) rezumat.
def _format_context(snippets: List[Dict[str, Any]]) -> str:
    lines = []
    for s in snippets:
        meta = s.get("metadata", {})
        title = meta.get("title", "N/A")
        author = meta.get("author", "")
        themes = meta.get("themes", [])
        score = s.get("score", 0.0)
        lines.append(f"- {title} ({author}) [score={score}]")
    return "\n".join(lines)

def chat_once(user_question: str, k: int = 3, filters: Optional[Dict[str, Any]] = None) -> str:
    """
    Construiește context scurt din top-K, îi dă modelului cerința de a:
      1) recomanda 1-2 titluri, justificând pe scurt (potrivire pe teme/autor),
      2) dacă e menționat explicit un titlu, include pe scurt esența (rezumat) pentru acel titlu.
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
    return resp.choices[0].message.content.strip()
