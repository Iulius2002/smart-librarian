# Smart Librarian — RAG + Tool + Web UI (FastAPI)

Chatbot pentru recomandări de cărți:
- **RAG** pe **ChromaDB** (vector store local, non-OpenAI)
- **Tool calling**: `get_summary_by_title` (rezumat complet din dataset local)
- **Web UI** tip ChatGPT (FastAPI + HTML/CSS/JS) cu **STT** (dictare, browser) și **TTS** (citire, browser)
- **Opțional server-side**: TTS (`gpt-4o-mini-tts`) și STT (Whisper)

## 1. Cerințe
- Python 3.10+
- Cheie OpenAI în `.env`

## 2. Instalare


pip install -r requirements.txt
Dacă vei face upload audio la /api/stt, mai instalează:

pip install python-multipart

3) Configurare

Creează un fișier .env în rădăcina proiectului:

OPENAI_API_KEY=sk-...cheia-ta...
CHROMA_DIR=data/chroma_db


.env NU se comite în git (e în .gitignore).

4) Dataset minim (10+ cărți)

Fișierele din data/:

book_summaries.json – rezumate complete (folosite la RAG + tool)

book_summaries.md – rezumate scurte (doar pentru afișare/temă)

Poți extinde oricând datasetul; după modificare, rerulează ingestia (pasul următor).

5) Ingestie (construiește vector store)

Creează embeddings și salvează în Chroma:

python -m src.ingest


Creează folderul data/chroma_db/ (persistență locală). Nu îl urca în repo.

6) Teste rapide (CLI)

Retrieval (fără LLM):

python -m src.test_retrieval


Exemple de interogări:

o carte despre libertate și control social

prietenie și magie

război și destine

Chat end-to-end (RAG + LLM + tool):

python -m src.test_chat

7) Pornire server (Web UI)

Pornește backendul FastAPI:

uvicorn src.server:app --reload


Deschide în browser: http://127.0.0.1:8000/

În UI:

scrii sau dictezi (🎙️) întrebarea;

apeși Trimite;

poți activa „🔊 Auto” ca răspunsurile să fie citite de TTS din browser;

alege o voce ro-RO din dropdown (dacă e disponibilă în sistemul tău) pentru pronunție mai naturală.

8) API (pentru integrare)
POST /api/chat

Primește un mesaj, face RAG + LLM + tool și întoarce răspunsul final + sursele (top-3 fragmente).

Request:

{ "message": "Vreau o carte despre prietenie și magie" }


Response:

{
  "answer": "<text final: recomandare + (dacă e posibil) rezumat complet>",
  "sources": [{ "title": "The Hobbit", "preview": "..." }, { "title": "The Lord of the Rings", "preview": "..." }, { "title": "Pride and Prejudice", "preview": "..." }]
}


Are filtru simplu de limbaj nepotrivit; la detecție răspunde politicos fără a apela LLM.

POST /api/tts (opțional – TTS server-side)

Generează audio MP3 din text folosind gpt-4o-mini-tts.

Request (JSON):

{ "text": "Acesta este un test în limba română." }


Response: audio/mpeg

Exemplu:

curl -s -X POST http://127.0.0.1:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Acesta este un test în limba română."}' \
  --output tts.mp3

POST /api/stt (opțional – STT server-side)

Transcrie fișier audio în text folosind Whisper (whisper-1).

Request: multipart/form-data cu câmpul audio
Response:

{ "text": "Transcrierea în limba română..." }


Exemplu:

curl -s -X POST http://127.0.0.1:8000/api/stt \
  -F "audio=@/cale/catre/inregistrare.m4a"


Pentru upload funcțional, asigură-te că ai python-multipart instalat.

9) Arhitectură pe scurt

RAG: text-embedding-3-small → ChromaDB (HNSW, cosine)

Chat:

Interogare → retrieval (top-K fragmente)

LLM alege titlul exact

Se apelează tool get_summary_by_title (rezumat complet din JSON)

LLM compune răspunsul final (recomandare + rezumat)

UI: FastAPI servește pagina; Web Speech API pentru STT/TTS în browser; opțional TTS/STT pe server pentru calitate mai stabilă.

10) Sfaturi & depanare

Nu găsește .env → rulează din rădăcina proiectului; load_dotenv() e apelat în config.py.

ModuleNotFoundError: src... → rulează cu python -m src.nume_script (din rădăcină).

RAG „ratează” → extinde/rafinează rezumatele, adaugă mai multe cărți, rulează din nou ingestia.

TTS sacadat în browser → alege o voce ro-RO; UI-ul redă textul pe propoziții pentru fluiditate.

Costuri: embeddings la ingestie; chat la cerere; opțional TTS/STT server-side.

11) Structura proiectului (orientativ)
smart-librarian/
├─ README.md
├─ requirements.txt
├─ .env               # în gitignore
├─ data/
│  ├─ book_summaries.json
│  ├─ book_summaries.md
│  └─ chroma_db/      # generat de Chroma (în gitignore)
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ vector_store.py
   ├─ ingest.py
   ├─ tools.py
   ├─ chat.py
   ├─ test_retrieval.py
   ├─ test_chat.py
   ├─ moderation.py
   ├─ server.py
   └─ templates/
      └─ index.html