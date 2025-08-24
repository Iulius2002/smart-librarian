import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Alegem modele „entry-level” și ieftine pentru test.
CHAT_MODEL = "gpt-4o-mini"
