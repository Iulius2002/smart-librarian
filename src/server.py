import os
import re
import tempfile
from typing import List, Dict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
    text: str           # textul răspunsului asistentului (vom extrage titlul)
    title: str | None = None  # opțional, dacă vrei să trimiți direct titlul


# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "Mesajul este gol."}, status_code=400)

    # 0) Filtru limbaj
    if contains_profanity(msg):
        return {"answer": "Prefer să păstrăm conversația politicoasă. Poți reformula te rog? 🙂", "sources": []}

    # 1) Context RAG (top 3) pentru „Sources”
    snippets = build_context_snippets(msg, k=3)
    sources: List[Dict[str, str]] = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        preview = (s["document"] or "").replace("\n", " ")[:220]
        sources.append({"title": title, "preview": preview})

    # 2) Răspuns final (RAG + LLM + tool)
    answer = chat_once(msg, k=3)
    return {"answer": answer, "sources": sources}


@app.post("/api/tts")
def tts_endpoint(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text gol.")
    try:
        # mp3 bytes
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


# ------------ Image Generation ------------
def _extract_title_from_text(text: str) -> str | None:
    """
    Heuristic: caută titlul între ghilimele sau după cuvinte cheie.
    """
    if not text:
        return None
    m = re.search(r"[\"“„‚'«](.+?)[\"”’'»]", text)
    if m:
        return m.group(1).strip()

    # fallback: caută pattern-uri tip „Recomand: 1984” / „Cartea: 1984”
    m2 = re.search(r"(?:Recomand(?:area)?|Cartea|Titlul)\s*:\s*([^\n\.,;:]+)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()

    return None

@app.post("/api/image")
def image_endpoint(req: ImageRequest):
    # Dacă nu primim explicit titlul, încercăm să-l extragem din textul asistentului
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
        b64 = img.data[0].b64_json  # returnăm b64 pentru a fi afișat direct în <img>
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la generarea imaginii: {e}")

    return {"title": title, "image_b64": b64}
