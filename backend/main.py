"""
FastAPI application — Vercel-compatible.

Endpoints:
  GET  /                    → index.html
  GET  /api/status          → LLM backend status
  GET  /api/sample-tickets  → pre-loaded sample tickets
  POST /api/process         → SSE stream of pipeline events
                              (replaces WebSocket for Vercel compatibility)
"""

import json
import sys
from pathlib import Path

# Ensure the backend/ directory is always on sys.path,
# whether the server is started from project root or from backend/.
_BACKEND = Path(__file__).parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from pipeline import run_pipeline

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FRONTEND_DIR = BASE.parent / "frontend"

app = FastAPI(title="Multi-Agent Customer Support", version="1.0.0")

# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sample-tickets")
async def get_sample_tickets():
    with open(DATA_DIR / "sample_tickets.json") as f:
        return JSONResponse(json.load(f))


@app.get("/api/status")
async def get_status():
    import httpx, os
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return {"llm": "ollama", "available": True, "models": models}
    except Exception:
        pass
    # Check Groq
    if os.environ.get("GROQ_API_KEY"):
        return {"llm": "groq", "available": True, "models": ["llama-3.1-8b-instant"]}
    return {"llm": "demo", "available": True, "models": ["demo-mode"]}


# ---------------------------------------------------------------------------
# SSE pipeline — works on Vercel, Railway, Render, and locally
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ticket: str
    plan: str = "Professional"


@app.post("/api/process")
async def process_ticket(body: TicketRequest):
    """
    Streams pipeline events as Server-Sent Events (SSE).
    Each event is a JSON object; the stream ends with data:[DONE].
    """
    ticket_text = body.ticket.strip()
    if not ticket_text:
        raise HTTPException(status_code=400, detail="Empty ticket text")

    async def event_stream():
        try:
            async for update in run_pipeline(ticket_text, body.plan):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables Nginx buffering
            "Connection": "keep-alive",
        },
    )
