# src/server.py
import os
import re
import json
import asyncio
import tempfile
from typing import List, Dict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from openai import OpenAI

from .chat import chat_once, build_context_snippets
from .moderation import contains_profanity
from .config import OPENAI_API_KEY, IMAGE_MODEL, TTS_MODEL, STT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY or None)

app = FastAPI(title="Smart Librarian")
templates = Jinja2Templates(directory="src/templates")


# ---------------- Models ----------------
class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    text: str
    title: str | None = None


# ---------------- Utilities ----------------
def _extract_title_from_text(text: str) -> str | None:
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
    """Împarte pe grupuri mici de cuvinte (~25–35 caractere)."""
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
    """Trimite bucăți mici cu un delay scurt între ele (efect typing)."""
    for chunk in _split_by_words(text, chunk_chars):
        yield f'data: {json.dumps({"delta": chunk}, ensure_ascii=False)}\n\n'
        await asyncio.sleep(delay_sec)


# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "Mesajul este gol."}, status_code=400)

    if contains_profanity(msg):
        return {"answer": "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂", "sources": []}

    snippets = build_context_snippets(msg, k=3)
    sources: List[Dict[str, str]] = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        preview = (s["document"] or "").replace("\n", " ")[:220]
        sources.append({"title": title, "preview": preview})

    answer = chat_once(msg, k=3)
    return {"answer": answer, "sources": sources}


@app.get("/api/chat/stream")
def chat_stream(q: str):
    """
    SSE: trimitem întâi sursele, apoi răspunsul în bucăți mici cu delay.
    Notă: răspunsul final vine din chat_once() (după tool-calling), apoi îl „picurăm”.
    """
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Mesajul este gol.")

    if contains_profanity(q):
        async def polite():
            yield f'data: {json.dumps({"delta":"Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂"})}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(polite(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    async def event_gen():
        # 1) Surse (RAG top-3)
        snippets = build_context_snippets(q, k=3)
        sources: List[Dict[str, str]] = []
        for s in snippets:
            title = s["metadata"].get("title", "N/A")
            preview = (s["document"] or "").replace("\n", " ")[:220]
            sources.append({"title": title, "preview": preview})
        yield 'event: sources\n'
        yield f'data: {json.dumps({"sources": sources}, ensure_ascii=False)}\n\n'

        # 2) Răspuns final -> drip feed
        answer = chat_once(q, k=3)
        async for ev in _yield_stream_chunks(answer, chunk_chars=28, delay_sec=0.045):
            yield ev

        # 3) DONE + ping final
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


class TTSRequest(BaseModel):
    text: str


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
