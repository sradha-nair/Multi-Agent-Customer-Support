#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SupportFlow — start the multi-agent customer support pipeline
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r "$BACKEND/requirements.txt" -q
fi

# Optional: start Ollama if installed and not running
if command -v ollama &>/dev/null && ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &>/dev/null &
  sleep 2
fi

echo ""
echo "  SupportFlow — Multi-Agent Customer Support Pipeline"
echo "  ────────────────────────────────────────────────────"
echo "  Open http://localhost:8000 in your browser"
echo "  Press Ctrl+C to stop"
echo ""

cd "$BACKEND"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
