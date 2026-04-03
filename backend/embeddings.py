"""
Embedding utilities — three-tier fallback, fully local.

Priority:
  1. sentence-transformers/all-MiniLM-L6-v2  (semantic, best — needs cached model)
  2. scikit-learn TF-IDF with bigrams        (local, no download, good quality)
  3. Jaccard keyword overlap                 (zero-dependency last resort)

sentence-transformers is tried with a 3-second timeout so a blocked proxy
or missing model never stalls startup.  TF-IDF activates immediately otherwise.
"""

import os
import threading
import numpy as np
from typing import Optional

# ─────────────────────────── Backend state ───────────────────────────

_BACKEND: Optional[str] = None   # "sbert" | "tfidf" | "keyword"
_ST_MODEL = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_TFIDF_CORPUS: list[str] = []

_SBERT_TIMEOUT = 3.0   # seconds; keeps startup snappy when model isn't cached


def _try_load_sbert():
    """Attempt to load the sbert model; called in a daemon thread."""
    import io, contextlib
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # Suppress noisy stderr/stdout during model-load attempts
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.encode(["warmup"], normalize_embeddings=True)
    return model


def _detect_backend() -> str:
    global _BACKEND, _ST_MODEL
    if _BACKEND is not None:
        return _BACKEND

    # -- Try sentence-transformers with a hard timeout -------------------
    result: list = [None]
    exc: list = [None]

    def _target():
        try:
            result[0] = _try_load_sbert()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=_SBERT_TIMEOUT)

    if result[0] is not None:
        _ST_MODEL = result[0]
        _BACKEND = "sbert"
        return _BACKEND

    # -- Try scikit-learn TF-IDF ----------------------------------------
    try:
        import sklearn  # noqa: F401
        _BACKEND = "tfidf"
        return _BACKEND
    except ImportError:
        pass

    _BACKEND = "keyword"
    return _BACKEND


# ─────────────────────────── TF-IDF internals ───────────────────────────

def _tfidf_fit(corpus: list[str]):
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _TFIDF_CORPUS
    from sklearn.feature_extraction.text import TfidfVectorizer
    _TFIDF_CORPUS = list(corpus)
    _TFIDF_VECTORIZER = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
        min_df=1,
    )
    _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(corpus)


def _tfidf_encode_dense(texts: list[str]) -> np.ndarray:
    if _TFIDF_VECTORIZER is None:
        raise RuntimeError("TF-IDF not fitted")
    mat = _TFIDF_VECTORIZER.transform(texts)
    dense = np.asarray(mat.todense(), dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


# ─────────────────────────── Public API ───────────────────────────

def fit_corpus(corpus: list[str]):
    """
    Pre-fit TF-IDF on a fixed corpus.  Call once at startup with all known
    text so the vocabulary is rich before incremental queries arrive.
    sentence-transformers doesn't need this (it's a fixed encoder).
    """
    backend = _detect_backend()
    if backend in ("tfidf",):
        _tfidf_fit(corpus)
    # For keyword or sbert: no-op (sbert encodes on-demand; keyword needs nothing)


def encode(texts: list[str]) -> Optional[np.ndarray]:
    """
    Encode texts → L2-normalised float32 array of shape (N, D).
    Returns None when running in keyword-only mode (handled downstream).
    """
    backend = _detect_backend()

    if backend == "sbert":
        try:
            return _ST_MODEL.encode(texts, normalize_embeddings=True)
        except Exception:
            pass  # fall through to tfidf

    # TF-IDF path (primary in restricted envs, fallback from sbert)
    try:
        if _TFIDF_VECTORIZER is None:
            # Fit on whatever we have right now
            _tfidf_fit(texts)
        else:
            # Handle out-of-vocabulary texts gracefully (transform still works)
            pass
        return _tfidf_encode_dense(texts)
    except Exception:
        pass

    return None  # keyword mode


def max_similarity(query_emb: np.ndarray, corpus_embs: np.ndarray) -> float:
    sims = query_emb @ corpus_embs.T
    return float(np.max(sims))


def top_k_indices(
    query_emb: np.ndarray, corpus_embs: np.ndarray, k: int = 3
) -> list[tuple[int, float]]:
    sims = query_emb @ corpus_embs.T
    idxs = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]


# ─────────────────────────── Keyword fallback ───────────────────────────

def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word tokens."""
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fallback_top_k(
    query: str, corpus: list[str], k: int = 3
) -> list[tuple[int, float]]:
    scores = [(i, _keyword_overlap(query, doc)) for i, doc in enumerate(corpus)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def backend_name() -> str:
    return _detect_backend()
