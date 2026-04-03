#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SupportFlow — start the multi-agent customer support pipeline
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r "$SCRIPT_DIR/requirements.txt" -q
fi

# Optional: start Ollama if installed and not running
if command -v ollama &>/dev/null && ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &>/dev/null &
  sleep 2
fi

echo ""
echo "  SupportFlow — Multi-Agent Customer Support Pipeline"
echo "  ─────────────────────────────────────────────────────"
echo "  Open http://localhost:8000 in your browser"
echo ""
echo "  LLM options (auto-detected):"
echo "    Ollama  → install ollama, run: ollama pull llama3.2"
echo "    Groq    → set env var: export GROQ_API_KEY=your_key_here"
echo "    Demo    → works out of the box, no setup required"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
