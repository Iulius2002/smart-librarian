from __future__ import annotations
import json
import os
import re
import uuid
from typing import List, Dict, Any

from .vector_store import reset_collection, add_chunks
from .config import CHROMA_DIR

SENT_SPLIT = re.compile(r'(?<=[\.\!\?\:])\s+')

def chunk_text(text: str, target_chars: int = 750, overlap: int = 120) -> List[str]:
    """Împarte textul în bucăți ~750c cu overlap ~120c, pe limite de propoziții."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = SENT_SPLIT.split(text)
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= target_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
                # overlap simplu: ultimele ~overlap caractere intră în noul buffer
                if overlap > 0 and len(buf) > overlap:
                    buf = buf[-overlap:]
                else:
                    buf = ""
            buf = (buf + " " + s).strip()
    if buf:
        chunks.append(buf)
    return chunks

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Acceptă: listă, {'books':[...]} sau map titlu->(text/obiect)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "books" in data and isinstance(data["books"], list):
        return data["books"]

    if isinstance(data, dict):
        items: List[Dict[str, Any]] = []
        for title_key, val in data.items():
            if isinstance(val, str):
                items.append({
                    "title": title_key,
                    "author": "",
                    "themes": [],
                    "year": "",
                    "summary_full": val.strip()
                })
            elif isinstance(val, dict):
                title = (val.get("title") or title_key or "").strip()
                author = (val.get("author") or "").strip()
                themes = val.get("themes") or val.get("tags") or []
                if isinstance(themes, str):
                    themes = [themes]
                year = val.get("year") or val.get("published") or ""
                summary = val.get("summary_full") or val.get("summary") or val.get("text") or val.get("short_summary") or ""
                items.append({
                    "title": title,
                    "author": author,
                    "themes": themes,
                    "year": year,
                    "summary_full": (summary or "").strip()
                })
            else:
                continue
        if items:
            return items

    raise AssertionError("book_summaries.json trebuie să fie o listă sau un obiect mapat pe titlu.")

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    reset_collection()

    data_path = os.path.join("data", "book_summaries.json")
    dataset = load_dataset(data_path)

    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for item in dataset:
        title = (item.get("title") or "").strip()
        author = (item.get("author") or "").strip()

        # themes poate fi listă/str/None; pentru Chroma o convertim la string
        raw_themes = item.get("themes") or item.get("tags") or []
        if isinstance(raw_themes, str):
            themes_list = [raw_themes]
        elif isinstance(raw_themes, list):
            themes_list = [str(x) for x in raw_themes if x is not None]
        else:
            themes_list = []
        themes_str = ", ".join([t.strip() for t in themes_list if t.strip()])

        year = item.get("year") or item.get("published") or ""
        lang = item.get("language") or "ro"

        # rezumat complet (fallback pe câmpuri alternative)
        summary = item.get("summary_full") or item.get("summary") or item.get("text") or item.get("short_summary") or ""
        chunks = chunk_text(summary, target_chars=750, overlap=120)

        for i, ch in enumerate(chunks):
            meta = {
                "title": title,           # str
                "author": author,         # str
                "themes": themes_str,     # str (NU listă -> evita eroarea Chroma)
                "year": year,             # int/str (scalar)
                "language": lang,         # str
                "chunk": i,               # int
                "chunks_total": len(chunks)  # int
            }
            all_chunks.append(ch)
            all_metas.append(meta)
            all_ids.append(str(uuid.uuid4()))

    if not all_chunks:
        print("⚠️ Nu există conținut de indexat (book_summaries.json gol?).")
        return

    print(f"Indexez {len(all_chunks)} fragmente…")
    add_chunks(all_chunks, all_metas, all_ids)
    print("✅ Gata — Chroma DB actualizat.")

if __name__ == "__main__":
    main()
