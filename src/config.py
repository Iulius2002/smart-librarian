import os
from dotenv import load_dotenv

load_dotenv()

# Chei & directoare
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")

# Modele
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
IMAGE_MODEL = "gpt-image-1"          # pentru /api/image
TTS_MODEL = "gpt-4o-mini-tts"        # pentru /api/tts (opțional)
STT_MODEL = "whisper-1"              # pentru /api/stt (opțional)
