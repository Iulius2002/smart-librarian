# src/server.py
import os
import re
import json
import time
import sqlite3
import asyncio
import tempfile
import logging
import threading
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, PlainTextResponse
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

# ---------- Logs ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("smart_librarian")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_json(**kwargs):
    try:
        logger.info(json.dumps(kwargs, ensure_ascii=False))
    except Exception:
        pass

# ---------- Metrics (în memorie) ----------
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.req_count = 0
        self.by_path = defaultdict(int)
        self.latency_sum = defaultdict(float)
        self.latency_count = defaultdict(int)
        self.chat_tokens_prompt = 0
        self.chat_tokens_completion = 0
        self.rate_limit_drops = 0

    def track(self, path: str, dur_sec: float):
        with self.lock:
            self.req_count += 1
            self.by_path[path] += 1
            self.latency_sum[path] += dur_sec
            self.latency_count[path] += 1

    def add_tokens(self, prompt: int, completion: int):
        with self.lock:
            self.chat_tokens_prompt += int(prompt or 0)
            self.chat_tokens_completion += int(completion or 0)

    def drop(self):
        with self.lock:
            self.rate_limit_drops += 1

metrics = Metrics()

# ---------- Rate limiting (simplu, în memorie) ----------
WINDOW_SEC = 60
LIMITS = {
    "/api/chat": 30,          # 30 req/min
    "/api/chat/stream": 20,   # 20 stream/min
    "/api/tts": 60,
    "/api/stt": 20,
    "/api/image": 30,
}
_buckets: Dict[Tuple[str, str], deque] = {}

def client_ip(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    if xf:
        return xf.split(",")[0].strip()
    return (req.client.host if req.client else "unknown") or "unknown"

def allow(ip: str, key: str) -> bool:
    limit = LIMITS.get(key)
    if not limit:
        return True
    dq = _buckets.setdefault((ip, key), deque())
    now = time.time()
    while dq and now - dq[0] > WINDOW_SEC:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True

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
    rows.reverse()
    convo = [{"role": r, "text": t} for (r, t) in rows]
    total = 0
    pruned = []
    for m in reversed(convo):
        total += len(m["text"])
        if total > max_chars:
            break
        pruned.append(m)
    pruned.reverse()
    return pruned

def history_to_prefix(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Context conversație anterioară (rezumat brut):"]
    for m in history:
        who = "Utilizator" if m["role"] == "user" else "Asistent"
        lines.append(f"- {who}: {m['text']}")
    lines.append("")
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
            buf = tok; continue
        if len(buf) + 1 + len(tok) <= max_chars:
            buf += " " + tok
        else:
            yield buf; buf = tok
    if buf:
        yield buf

async def _yield_stream_chunks(text: str, chunk_chars: int = 28, delay_sec: float = 0.045):
    for chunk in _split_by_words(text, chunk_chars):
        yield f'data: {json.dumps({"delta": chunk}, ensure_ascii=False)}\n\n'
        await asyncio.sleep(delay_sec)

# ---------- Middleware: timing + logs ----------
@app.middleware("http")
async def log_and_time(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    ip = client_ip(request)
    try:
        response = await call_next(request)
        return response
    finally:
        dur = time.perf_counter() - start
        metrics.track(path, dur)
        log_json(ts=int(time.time()), ip=ip, path=path, method=request.method, latency_ms=int(dur*1000))

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    ip = client_ip(request)
    if not allow(ip, "/api/chat"):
        metrics.drop()
        raise HTTPException(status_code=429, detail="Prea multe cereri. Încearcă peste câteva secunde.")

    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "Mesajul este gol."}, status_code=400)
    sid = (req.session_id or "default").strip() or "default"

    if contains_profanity(msg):
        polite = "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂"
        save_msg(sid, "user", msg); save_msg(sid, "assistant", polite)
        return {"answer": polite, "sources": []}

    snippets = build_context_snippets(msg, k=3)
    sources: List[Dict[str, str]] = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        preview = (s["document"] or "").replace("\n", " ")[:220]
        sources.append({"title": title, "preview": preview})

    history = fetch_history(sid, limit=12, max_chars=1800)
    prefix = history_to_prefix(history)
    prompt = f"{prefix}Întrebarea curentă: {msg}"

    save_msg(sid, "user", msg)
    answer, usage = chat_once(prompt, k=3, return_usage=True)
    save_msg(sid, "assistant", answer)

    metrics.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    log_json(kind="chat", ip=ip, sid=sid, prompt_tokens=usage.get("prompt_tokens", 0),
             completion_tokens=usage.get("completion_tokens", 0))

    return {"answer": answer, "sources": sources}

@app.get("/api/chat/stream")
def chat_stream(q: str, sid: Optional[str] = None, request: Request = None):
    ip = client_ip(request)
    if not allow(ip, "/api/chat/stream"):
        metrics.drop()
        async def too_many():
            yield f'data: {json.dumps({"delta":"⚠️ Prea multe cereri pe minut. Așteaptă puțin și reîncearcă."}, ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(too_many(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Mesajul este gol.")
    sid = (sid or "default").strip() or "default"

    if contains_profanity(q):
        async def polite():
            save_msg(sid, "user", q)
            polite_text = "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂"
            save_msg(sid, "assistant", polite_text)
            yield f'data: {json.dumps({"delta": polite_text}, ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(polite(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

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

        save_msg(sid, "user", q)

        # 3) răspuns final + tokens
        answer, usage = chat_once(prompt, k=3, return_usage=True)
        metrics.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        log_json(kind="chat_stream", ip=ip, sid=sid,
                 prompt_tokens=usage.get("prompt_tokens", 0),
                 completion_tokens=usage.get("completion_tokens", 0))

        async for ev in _yield_stream_chunks(answer, chunk_chars=28, delay_sec=0.045):
            yield ev

        save_msg(sid, "assistant", answer)
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

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
def tts_endpoint(req: TTSRequest, request: Request):
    ip = client_ip(request)
    if not allow(ip, "/api/tts"):
        metrics.drop()
        raise HTTPException(status_code=429, detail="Rate limit depășit pentru /api/tts.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text gol.")
    try:
        result = client.audio.speech.create(model=TTS_MODEL, voice="alloy", input=text)
        audio_bytes = result.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare TTS: {e}")
    headers = {"Content-Disposition": 'inline; filename="tts.mp3"'}
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)

# ---------- STT ----------
@app.post("/api/stt")
async def stt_endpoint(audio: UploadFile = File(...), request: Request = None):
    ip = client_ip(request)
    if not allow(ip, "/api/stt"):
        metrics.drop()
        raise HTTPException(status_code=429, detail="Rate limit depășit pentru /api/stt.")
    try:
        raw = await audio.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Fișier audio gol.")
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw); tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(model=STT_MODEL, file=f)
        os.unlink(tmp_path)
        return {"text": getattr(tr, "text", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare STT: {e}")

# ---------- Image Generation ----------
@app.post("/api/image")
def image_endpoint(req: ImageRequest, request: Request):
    ip = client_ip(request)
    if not allow(ip, "/api/image"):
        metrics.drop()
        raise HTTPException(status_code=429, detail="Rate limit depășit pentru /api/image.")
    title = (req.title or "").strip() or _extract_title_from_text(req.text or "")
    if not title:
        raise HTTPException(status_code=400, detail="Nu am putut detecta titlul. Trimite 'title' explicit în request.")
    prompt = (
        f"Generează o ilustrație de copertă sugestivă pentru cartea „{title}”. "
        f"Stil modern, clar, cu contrast bun. Fără text mare pe imagine."
    )
    try:
        img = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1)
        b64 = img.data[0].b64_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la generarea imaginii: {e}")
    return {"title": title, "image_b64": b64}

# ---------- /metrics (Prometheus-like) ----------
@app.get("/metrics")
def metrics_endpoint():
    with metrics.lock:
        lines = []
        # Requests
        lines.append("# HELP smart_requests_total Număr total de request-uri")
        lines.append("# TYPE smart_requests_total counter")
        for path, cnt in metrics.by_path.items():
            lines.append(f'smart_requests_total{{path="{path}"}} {cnt}')
        # Latency
        lines.append("# HELP smart_request_latency_seconds Timp total pe endpoint (secunde)")
        lines.append("# TYPE smart_request_latency_seconds summary")
        for path, total in metrics.latency_sum.items():
            count = metrics.latency_count.get(path, 1)
            lines.append(f'smart_request_latency_seconds_sum{{path="{path}"}} {total:.6f}')
            lines.append(f'smart_request_latency_seconds_count{{path="{path}"}} {count}')
        # Tokens
        lines.append("# HELP smart_chat_tokens_total Tokeni consumați (prompt/completion)")
        lines.append("# TYPE smart_chat_tokens_total counter")
        lines.append(f'smart_chat_tokens_total{{type="prompt"}} {metrics.chat_tokens_prompt}')
        lines.append(f'smart_chat_tokens_total{{type="completion"}} {metrics.chat_tokens_completion}')
        # Rate limit drops
        lines.append("# HELP smart_rate_limit_drops_total Cereri respinse de rate limit")
        lines.append("# TYPE smart_rate_limit_drops_total counter")
        lines.append(f'smart_rate_limit_drops_total {metrics.rate_limit_drops}')
        return PlainTextResponse("\n".join(lines))
