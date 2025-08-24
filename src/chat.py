# src/chat.py
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
from .config import CHAT_MODEL
from .vector_store import query_similar
from .tools import get_summary_by_title

client = OpenAI()

SYSTEM_PROMPT = (
    "Ești Smart Librarian. Recomanzi cărți pe baza intereselor utilizatorului. "
    "Ai acces la fragmente dintr-o colecție (RAG). "
    "1) Folosește contextul ca să alegi un TITLU de carte potrivit. "
    "2) Dacă poți identifica TITLUL EXACT, apelează tool-ul get_summary_by_title(title) "
    "pentru a obține REZUMATUL COMPLET, apoi sintetizează răspunsul final. "
    "Dacă nu ești sigur, cere clarificări."
)

def build_context_snippets(query: str, k: int = 3) -> List[Dict[str, Any]]:
    res = query_similar(query, n_results=k)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for i in range(len(ids)):
        out.append({
            "document": docs[i],
            "metadata": metas[i]
        })
    return out

def _context_block(snippets: List[Dict[str, Any]]) -> str:
    lines = []
    for s in snippets:
        title = s["metadata"].get("title", "N/A")
        short = s["document"][:220].replace("\n", " ")
        lines.append(f"- {title}: {short}...")
    return "\n".join(lines) if lines else "Nicio potrivire în context."

def call_llm_with_tool(user_query: str, snippets: List[Dict[str, Any]]) -> str:
    ctx = _context_block(snippets)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": f"Context RAG (top 3):\n{ctx}"}
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru un titlu exact de carte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Titlul EXACT al cărții recomandate"}
                },
                "required": ["title"]
            }
        }
    }]

    # Primul apel: modelul alege titlul și poate cere tool-ul nostru
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.5,
    )
    msg = resp.choices[0].message

    # Dacă LLM vrea să cheme tool-ul:
    if getattr(msg, "tool_calls", None):
        # reconstruim mesajul assistant cu tool_calls într-un dict compatibil
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls
            ]
        }

        follow_messages = messages + [assistant_msg]

        for tc in msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments or "{}")
                title = args.get("title")
                detailed = get_summary_by_title(title) if title else None

                tool_msg = {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": "get_summary_by_title",
                    "content": detailed or (f"Nu am găsit rezumat pentru titlul '{title}'." if title else "Titlul lipsește.")
                }
                follow_messages.append(tool_msg)

        # Al doilea apel: LLM vede output-ul tool-ului și compune răspunsul final
        resp2 = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=follow_messages,
            temperature=0.5,
        )
        return resp2.choices[0].message.content or "Nu am primit conținut de la model."

    # Dacă nu a cerut tool-ul, întoarcem conținutul direct (poate cere clarificări)
    return msg.content or "Nu am primit conținut de la model."

def chat_once(user_query: str, k: int = 3) -> str:
    snippets = build_context_snippets(user_query, k=k)
    return call_llm_with_tool(user_query, snippets)
