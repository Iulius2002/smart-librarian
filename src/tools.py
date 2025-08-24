# src/tools.py
import json, os
from typing import Optional, Dict

DATA_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "book_summaries.json")

def _load_data() -> Dict[str, str]:
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"Nu găsesc {DATA_JSON}. Asigură-te că ai creat datasetul în Etapa B.")
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def get_summary_by_title(title: str) -> Optional[str]:
    data = _load_data()
    return data.get(title)

def list_titles():
    return sorted(_load_data().keys())
