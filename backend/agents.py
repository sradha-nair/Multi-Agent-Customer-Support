"""
All six pipeline agents.

1. NoveltyDetector     — distribution-shift detection (before LLM)
2. ClassifierAgent     — category + priority + confidence
3. ResearcherAgent     — RAG retrieval from knowledge base
4. ResponderAgent      — draft response
5. GroundingChecker    — verify response claims against KB
6. ConfidenceScorer    — final score + routing decision
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional

import embeddings as emb
from llm import LLMClient

DATA_DIR = Path(__file__).parent / "data"

# Load KB articles
with open(DATA_DIR / "kb_articles.json") as f:
    KB_ARTICLES: list[dict] = json.load(f)

# Load sample tickets (used as "known distribution" for novelty detection)
with open(DATA_DIR / "sample_tickets.json") as f:
    KNOWN_TICKETS: list[dict] = json.load(f)

KB_TEXTS = [f"{a['title']} {a['content']}" for a in KB_ARTICLES]
KNOWN_TICKET_TEXTS = [f"{t['subject']} {t['body']}" for t in KNOWN_TICKETS]

# Pre-fit the embedding backend on all static text so TF-IDF vocabulary is rich
emb.fit_corpus(KB_TEXTS + KNOWN_TICKET_TEXTS)

# Pre-compute embeddings once on module load
_KB_EMBS: Optional[np.ndarray] = None
_KNOWN_EMBS: Optional[np.ndarray] = None


def _kb_embs() -> Optional[np.ndarray]:
    global _KB_EMBS
    if _KB_EMBS is None:
        _KB_EMBS = emb.encode(KB_TEXTS)
    return _KB_EMBS


def _known_embs() -> Optional[np.ndarray]:
    global _KNOWN_EMBS
    if _KNOWN_EMBS is None:
        _KNOWN_EMBS = emb.encode(KNOWN_TICKET_TEXTS)
    return _KNOWN_EMBS


# ---------------------------------------------------------------------------
# 1. Novelty Detector
# ---------------------------------------------------------------------------

class NoveltyDetector:
    """
    Compares incoming ticket embedding to the known-ticket distribution.
    A low max-similarity signals a novel ticket that may trigger distributional shift.
    Threshold: tickets with similarity < 0.45 are flagged as novel.
    """
    THRESHOLD = 0.45

    def run(self, ticket: str) -> dict:
        ticket_emb = emb.encode([ticket])
        known = _known_embs()

        if ticket_emb is not None and known is not None:
            max_sim = emb.max_similarity(ticket_emb[0], known)
            method = "semantic"
        else:
            # Fallback: keyword overlap
            scores = [emb._keyword_overlap(ticket, t) for t in KNOWN_TICKET_TEXTS]
            max_sim = max(scores) if scores else 0.0
            method = "keyword"

        is_novel = max_sim < self.THRESHOLD
        nearest_idx = None
        if not is_novel:
            # Find which known ticket it's most similar to
            if ticket_emb is not None and known is not None:
                sims = ticket_emb[0] @ known.T
                nearest_idx = int(np.argmax(sims))
            else:
                scores_list = [emb._keyword_overlap(ticket, t) for t in KNOWN_TICKET_TEXTS]
                nearest_idx = int(np.argmax(scores_list))

        return {
            "agent": "novelty_detector",
            "is_novel": is_novel,
            "similarity_to_known": round(float(max_sim), 3),
            "threshold": self.THRESHOLD,
            "method": method,
            "nearest_ticket": KNOWN_TICKETS[nearest_idx]["subject"] if nearest_idx is not None else None,
            "flag": "NOVEL_TICKET" if is_novel else None,
            "note": (
                "Ticket is significantly different from known patterns — "
                "all downstream confidence scores will be penalised."
                if is_novel else
                f"Ticket matches known distribution (closest: '{KNOWN_TICKETS[nearest_idx]['subject'] if nearest_idx is not None else 'N/A'}')"
            ),
        }


# ---------------------------------------------------------------------------
# 2. Classifier Agent
# ---------------------------------------------------------------------------

class ClassifierAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, ticket: str) -> dict:
        result = await self.llm.classify(ticket)
        return {
            "agent": "classifier",
            **result,
        }


# ---------------------------------------------------------------------------
# 3. Researcher Agent (RAG)
# ---------------------------------------------------------------------------

class ResearcherAgent:
    TOP_K = 3
    # TF-IDF cosine sims are much smaller than sbert sims; use a tiny floor.
    MIN_RELEVANCE = 0.005

    def run(self, ticket: str, category: str) -> dict:
        ticket_emb = emb.encode([ticket])
        kb = _kb_embs()

        if ticket_emb is not None and kb is not None:
            hits = emb.top_k_indices(ticket_emb[0], kb, k=self.TOP_K)
            method = "semantic"
        else:
            hits = emb.fallback_top_k(ticket, KB_TEXTS, k=self.TOP_K)
            method = "keyword"

        retrieved = []
        for idx, score in hits:
            if score >= self.MIN_RELEVANCE:
                retrieved.append({
                    "id": KB_ARTICLES[idx]["id"],
                    "title": KB_ARTICLES[idx]["title"],
                    "category": KB_ARTICLES[idx]["category"],
                    "relevance": round(score, 3),
                    "content": KB_ARTICLES[idx]["content"],
                })

        # Also include category-matched articles not in top-k
        cat_articles = [a for a in KB_ARTICLES if a["category"] == category]
        existing_ids = {r["id"] for r in retrieved}
        for a in cat_articles:
            if a["id"] not in existing_ids and len(retrieved) < 4:
                retrieved.append({
                    "id": a["id"],
                    "title": a["title"],
                    "category": a["category"],
                    "relevance": 0.30,
                    "content": a["content"],
                })

        return {
            "agent": "researcher",
            "method": method,
            "retrieved_count": len(retrieved),
            "articles": retrieved,
            "kb_context": "\n\n".join(
                f"[{r['id']}] {r['title']}:\n{r['content']}" for r in retrieved
            ),
        }


# ---------------------------------------------------------------------------
# 4. Responder Agent
# ---------------------------------------------------------------------------

class ResponderAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, ticket: str, category: str, kb_context: str, plan: str = "Professional") -> dict:
        draft = await self.llm.draft_response(ticket, category, kb_context, plan)
        return {
            "agent": "responder",
            "draft": draft,
        }


# ---------------------------------------------------------------------------
# 5. Grounding Checker
# ---------------------------------------------------------------------------

class GroundingChecker:
    """
    Checks each sentence of the response against retrieved KB content.
    Sentences below the grounding threshold are flagged as ungrounded.

    Thresholds are backend-aware:
      - sbert   : 0.18  (semantic cosine similarity, scale −1..1)
      - tfidf   : 0.01  (sparse cosine of short vs long texts is naturally low)
      - keyword : 0.08  (Jaccard overlap)
    """
    THRESHOLDS = {"sbert": 0.18, "tfidf": 0.01, "keyword": 0.08}

    def _threshold(self) -> float:
        return self.THRESHOLDS.get(emb.backend_name(), 0.05)

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 20]

    def run(self, draft: str, kb_context: str, retrieved_articles: list) -> dict:
        sentences = self._split_sentences(draft)
        if not sentences:
            return {
                "agent": "grounding_checker",
                "grounding_score": 1.0,
                "sentences": [],
                "ungrounded": [],
                "verified_response": draft,
                "claims_checked": 0,
                "claims_grounded": 0,
            }

        kb_chunks = [a["content"] for a in retrieved_articles] if retrieved_articles else [kb_context]
        threshold = self._threshold()

        sentence_embs = emb.encode(sentences)
        chunk_embs = emb.encode(kb_chunks)

        sentence_results = []
        ungrounded = []

        for i, sentence in enumerate(sentences):
            if sentence_embs is not None and chunk_embs is not None:
                sims = sentence_embs[i] @ chunk_embs.T
                best_score = float(np.max(sims))
                best_chunk_idx = int(np.argmax(sims))
                method = emb.backend_name()
            else:
                scores = [emb._keyword_overlap(sentence, c) for c in kb_chunks]
                best_score = max(scores) if scores else 0.0
                best_chunk_idx = int(np.argmax(scores)) if scores else 0
                method = "keyword"

            is_grounded = best_score >= threshold
            entry = {
                "sentence": sentence,
                "grounded": is_grounded,
                "score": round(best_score, 3),
                "source": retrieved_articles[best_chunk_idx]["id"] if retrieved_articles else "kb",
                "method": method,
            }
            sentence_results.append(entry)
            if not is_grounded:
                ungrounded.append(sentence)

        grounded_count = sum(1 for s in sentence_results if s["grounded"])
        grounding_score = grounded_count / len(sentences) if sentences else 1.0

        # Build verified response — keep grounded sentences only
        verified = " ".join(s["sentence"] for s in sentence_results if s["grounded"])
        if not verified.strip():
            verified = draft  # Safety: never send an empty response

        return {
            "agent": "grounding_checker",
            "grounding_score": round(grounding_score, 3),
            "sentences": sentence_results,
            "ungrounded": ungrounded,
            "verified_response": verified,
            "claims_checked": len(sentences),
            "claims_grounded": grounded_count,
        }


# ---------------------------------------------------------------------------
# 6. Confidence Scorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Combines classifier confidence, grounding score, and novelty penalty
    into a final routing decision.

    Weights:
      - classifier confidence: 40%
      - grounding score:       40%
      - novelty penalty:       20%

    Routing thresholds:
      >= 0.70 → AUTO_APPROVE
      0.50–0.69 → LIGHT_REVIEW
      < 0.50 → FULL_ESCALATION
    """
    WEIGHTS = {"classifier": 0.40, "grounding": 0.40, "novelty": 0.20}

    AUTO_APPROVE = 0.70
    LIGHT_REVIEW = 0.50

    def run(
        self,
        classifier_confidence: float,
        grounding_score: float,
        is_novel: bool,
        novelty_similarity: float,
        priority: str,
    ) -> dict:
        novelty_score = novelty_similarity if not is_novel else max(0.1, novelty_similarity)

        raw = (
            self.WEIGHTS["classifier"] * classifier_confidence
            + self.WEIGHTS["grounding"] * grounding_score
            + self.WEIGHTS["novelty"] * novelty_score
        )

        # Downgrade confidence for novel tickets regardless of other signals
        if is_novel:
            raw = raw * 0.75

        final = round(min(max(raw, 0.0), 1.0), 3)

        if final >= self.AUTO_APPROVE and priority not in ("critical",):
            routing = "AUTO_APPROVE"
            routing_label = "Auto-approve"
            routing_reason = (
                f"All signals healthy (classifier: {classifier_confidence:.0%}, "
                f"grounding: {grounding_score:.0%}, novelty: {novelty_score:.0%}). "
                "Response queued for delivery."
            )
        elif final >= self.LIGHT_REVIEW and priority not in ("critical",):
            routing = "LIGHT_REVIEW"
            routing_label = "Light review"
            routing_reason = (
                "Moderate confidence — one or more signals below threshold. "
                "Queued for quick human review before sending."
            )
        else:
            routing = "FULL_ESCALATION"
            routing_label = "Full escalation"
            routing_reason = (
                "Low confidence or critical/novel ticket. "
                "Routed to senior support agent for manual review and response."
            )

        # Priority override: critical always escalates
        if priority == "critical" and routing == "AUTO_APPROVE":
            routing = "LIGHT_REVIEW"
            routing_label = "Light review (priority override)"
            routing_reason = "Ticket marked CRITICAL — human review required regardless of confidence score."

        return {
            "agent": "confidence_scorer",
            "scores": {
                "classifier": classifier_confidence,
                "grounding": grounding_score,
                "novelty": round(novelty_score, 3),
            },
            "final_confidence": final,
            "routing": routing,
            "routing_label": routing_label,
            "routing_reason": routing_reason,
            "is_novel_ticket": is_novel,
        }
