import json, os, re
from .vector_store import add_documents
from .config import CHROMA_DIR

DATA_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "book_summaries.json")

def slugify(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", t).strip("-")

def main():
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"Nu găsesc datasetul: {DATA_JSON}. Creează data/book_summaries.json conform Etapa B3.")

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for title, summary in data.items():
        docs.append({
            "id": slugify(title),
            "text": summary,
            "metadata": {"title": title}
        })

    add_documents(docs)
    print(f"✔ Ingest gata în {CHROMA_DIR}: {len(docs)} documente adăugate.")

if __name__ == "__main__":
    main()
