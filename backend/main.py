"""
FastAPI application.

Serves the frontend as static files and exposes:
  GET  /api/sample-tickets   — list of pre-loaded sample tickets
  GET  /api/status           — LLM backend status
  WS   /ws/process           — WebSocket endpoint for pipeline streaming
"""

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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
    # Return empty 204 to avoid browser noise
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
    import httpx
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return {"llm": "ollama", "available": True, "models": models}
    except Exception:
        pass
    return {"llm": "demo", "available": True, "models": ["demo-mode"]}


# ---------------------------------------------------------------------------
# WebSocket — pipeline streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        ticket_text = data.get("ticket", "").strip()
        plan = data.get("plan", "Professional")

        if not ticket_text:
            await websocket.send_json({"type": "error", "message": "Empty ticket text"})
            return

        async for update in run_pipeline(ticket_text, plan):
            await websocket.send_json(update)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
