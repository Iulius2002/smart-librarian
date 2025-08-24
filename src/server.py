# src/server.py
import os
import re
import json
import time
import sqlite3
import asyncio
import tempfile
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from openai import OpenAI

from .chat import chat_once, build_context_snippets
from .moderation import contains_profanity
from .config import OPENAI_API_KEY, IMAGE_MODEL, TTS_MODEL, STT_MODEL

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY or None)

# ---------- App & templates ----------
app = FastAPI(title="Smart Librarian")
templates = Jinja2Templates(directory="src/templates")

# ---------- SQLite (persistență sesiuni) ----------
DB_PATH = "data/chat_history.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("""
CREATE TABLE IF NOT EXISTS messages (
  session_id TEXT,
  role       TEXT,
  text       TEXT,
  ts         INTEGER
)
""")
_conn.commit()

def save_msg(session_id: str, role: str, text: str):
    _conn.execute(
        "INSERT INTO messages(session_id, role, text, ts) VALUES(?,?,?,?)",
        (session_id, role, text, int(time.time()))
    )
    _conn.commit()

def fetch_history(session_id: str, limit: int = 12, max_chars: int = 1800) -> List[Dict[str, str]]:
    cur = _conn.execute(
        "SELECT role, text FROM messages WHERE session_id=? ORDER BY ts DESC LIMIT ?",
        (session_id, limit),
    )
    rows = cur.fetchall()
    rows.reverse()  # cele mai vechi primele
    convo = [{"role": r, "text": t} for (r, t) in rows]
    # taie textul total pentru a nu umfla promptul
    total = 0
    pruned = []
    for m in reversed(convo):  # începând de la cele mai noi
        total += len(m["text"])
        if total > max_chars:
            break
        pruned.append(m)
    pruned.reverse()
    return pruned

def history_to_prefix(history: List[Dict[str, str]]) -> str:
    """Împachetăm istoricul într-un prefix simplu, ușor de consumat de chat_once()."""
    if not history:
        return ""
    lines = []
    lines.append("Context conversație anterioară (rezumat brut):")
    for m in history:
        who = "Utilizator" if m["role"] == "user" else "Asistent"
        lines.append(f"- {who}: {m['text']}")
    lines.append("")  # linie goală
    return "\n".join(lines)

# ---------- Models ----------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    text: str
    title: Optional[str] = None

class ResetRequest(BaseModel):
    session_id: str


# ---------- Utils ----------
def _extract_title_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[\"“„‚'«](.+?)[\"”’'»]", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?:Recomand(?:area)?|Cartea|Titlul)\s*:\s*([^\n\.,;:]+)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def _split_by_words(s: str, max_chars: int = 28):
    if not s:
        return
    buf = ""
    for tok in s.split():
        if not buf:
            buf = tok
            continue
        if len(buf) + 1 + len(tok) <= max_chars:
            buf += " " + tok
        else:
            yield buf
            buf = tok
    if buf:
        yield buf

async def _yield_stream_chunks(text: str, chunk_chars: int = 28, delay_sec: float = 0.045):
    for chunk in _split_by_words(text, chunk_chars):
        yield f'data: {json.dumps({"delta": chunk}, ensure_ascii=False)}\n\n'
        await asyncio.sleep(delay_sec)


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "Mesajul este gol."}, status_code=400)
    sid = (req.session_id or "default").strip() or "default"

    # 0) Moderation
    if contains_profanity(msg):
        polite = "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂"
        save_msg(sid, "user", msg)
        save_msg(sid, "assistant", polite)
        return {"answer": polite, "sources": []}

    # 1) Surse (RAG top-3)
    snippets = build_context_snippets(msg, k=3)
    sources: List[Dict[str, str]] = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        preview = (s["document"] or "").replace("\n", " ")[:220]
        sources.append({"title": title, "preview": preview})

    # 2) Istoric + întrebare actuală
    history = fetch_history(sid, limit=12, max_chars=1800)
    prefix = history_to_prefix(history)
    prompt = f"{prefix}Întrebarea curentă: {msg}"

    # 3) Chat + persistare
    save_msg(sid, "user", msg)
    answer = chat_once(prompt, k=3)
    save_msg(sid, "assistant", answer)
    return {"answer": answer, "sources": sources}


@app.get("/api/chat/stream")
def chat_stream(q: str, sid: Optional[str] = None):
    """
    SSE cu memorie pe sesiune:
    - trimite întâi sursele,
    - apoi răspunsul „drip feed”,
    - salvează în SQLite mesajul user + răspunsul asistentului.
    """
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Mesajul este gol.")
    sid = (sid or "default").strip() or "default"

    # Moderation înainte de orice
    if contains_profanity(q):
        async def polite():
            save_msg(sid, "user", q)
            polite_text = "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂"
            save_msg(sid, "assistant", polite_text)
            yield f'data: {json.dumps({"delta": polite_text}, ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(polite(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    async def event_gen():
        # 1) surse
        snippets = build_context_snippets(q, k=3)
        sources: List[Dict[str, str]] = []
        for s in snippets:
            title = s["metadata"].get("title", "N/A")
            preview = (s["document"] or "").replace("\n", " ")[:220]
            sources.append({"title": title, "preview": preview})
        yield 'event: sources\n'
        yield f'data: {json.dumps({"sources": sources}, ensure_ascii=False)}\n\n'

        # 2) istoric + prompt
        history = fetch_history(sid, limit=12, max_chars=1800)
        prefix = history_to_prefix(history)
        prompt = f"{prefix}Întrebarea curentă: {q}"

        # 3) salvează user
        save_msg(sid, "user", q)

        # 4) răspuns final și drip feed
        answer = chat_once(prompt, k=3)
        async for ev in _yield_stream_chunks(answer, chunk_chars=28, delay_sec=0.045):
            yield ev

        # 5) persistă asistent + DONE
        save_msg(sid, "assistant", answer)
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ---------- Session maintenance ----------
@app.post("/api/session/reset")
def reset_session(req: ResetRequest):
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id lipsă")
    _conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    _conn.commit()
    return {"ok": True}


# ---------- TTS ----------
@app.post("/api/tts")
def tts_endpoint(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text gol.")
    try:
        result = client.audio.speech.create(
            model=TTS_MODEL,
            voice="alloy",
            input=text,
        )
        audio_bytes = result.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare TTS: {e}")

    headers = {"Content-Disposition": 'inline; filename="tts.mp3"'}
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)


# ---------- STT ----------
@app.post("/api/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    try:
        raw = await audio.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Fișier audio gol.")

        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
            )
        os.unlink(tmp_path)
        return {"text": getattr(tr, "text", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare STT: {e}")


# ---------- Image Generation ----------
@app.post("/api/image")
def image_endpoint(req: ImageRequest):
    title = (req.title or "").strip() or _extract_title_from_text(req.text or "")
    if not title:
        raise HTTPException(status_code=400, detail="Nu am putut detecta titlul. Trimite 'title' explicit în request.")
    prompt = (
        f"Generează o ilustrație de copertă sugestivă pentru cartea „{title}”. "
        f"Stil modern, clar, cu contrast bun. Fără text mare pe imagine."
    )
    try:
        img = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        b64 = img.data[0].b64_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la generarea imaginii: {e}")

    return {"title": title, "image_b64": b64}
