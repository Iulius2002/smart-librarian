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

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",") if o.strip()]
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "2000"))
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "2000"))
MAX_STT_MB = int(os.getenv("MAX_STT_MB", "12"))  # limită upload audio STT (MB)

# CSRF (opțional). Dacă REQUIRE_CSRF=1, toate POST-urile cer headerul x-csrf-token=CSRF_TOKEN
REQUIRE_CSRF = os.getenv("REQUIRE_CSRF", "0") == "1"
CSRF_TOKEN = os.getenv("CSRF_TOKEN", "")