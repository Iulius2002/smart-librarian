import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Chat rapid & ieftin pentru prototip (îl vom folosi în Etapa C)
CHAT_MODEL = "gpt-4o-mini"

# --- RAG settings ---
# Directorul unde Chroma persistă colecția (nu comite în git)
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")

# Model de embeddings (bun, ieftin, multi-purpose)
EMBEDDING_MODEL = "text-embedding-3-small"
