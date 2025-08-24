# src/server.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.templating import Jinja2Templates

from .chat import chat_once, build_context_snippets  # Etapa C
from .config import OPENAI_API_KEY

app = FastAPI(title="Smart Librarian")

# (opțional) montează un director static dacă mai adaugi asset-uri locale
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="src/templates")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        return JSONResponse({"error": "Mesajul este gol."}, status_code=400)

    # 1) Context RAG (top3) — îl afișăm în UI ca "Sources"
    snippets = build_context_snippets(req.message, k=3)
    sources: List[Dict[str, str]] = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        preview = (s["document"] or "").replace("\n", " ")[:220]
        sources.append({"title": title, "preview": preview})

    # 2) Răspunsul final (RAG + LLM + tool local rezumat complet)
    answer = chat_once(req.message, k=3)

    return {"answer": answer, "sources": sources}
