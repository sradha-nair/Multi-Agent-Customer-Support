import sys
from pathlib import Path

# Ensure project root is importable in Vercel runtime.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app as backend_app

app = backend_app
