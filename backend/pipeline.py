"""
Pipeline orchestrator.

Runs all six agents in sequence and yields real-time status updates
(for WebSocket streaming). Each yield is a JSON-serialisable dict.
"""

import asyncio
import time
from typing import AsyncGenerator

from llm import LLMClient
from agents import (
    NoveltyDetector,
    ClassifierAgent,
    ResearcherAgent,
    ResponderAgent,
    GroundingChecker,
    ConfidenceScorer,
)

_llm = LLMClient()
_novelty = NoveltyDetector()
_classifier = ClassifierAgent(_llm)
_researcher = ResearcherAgent()
_responder = ResponderAgent(_llm)
_grounding = GroundingChecker()
_scorer = ConfidenceScorer()


async def run_pipeline(ticket_text: str, plan: str = "Professional") -> AsyncGenerator[dict, None]:
    """
    Async generator — yields one dict per pipeline event.
    Events: {type: "start"|"agent_start"|"agent_done"|"complete"|"error"}
    """
    try:
        yield {"type": "start", "message": "Pipeline started", "llm_mode": _llm.mode}
        await asyncio.sleep(0.05)

        # ----------------------------------------------------------------
        # Agent 1: Novelty Detector
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "novelty_detector", "label": "Novelty Detector"}
        t0 = time.monotonic()
        novelty_result = await asyncio.get_event_loop().run_in_executor(
            None, _novelty.run, ticket_text
        )
        novelty_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **novelty_result}
        await asyncio.sleep(0.05)

        # ----------------------------------------------------------------
        # Agent 2: Classifier
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "classifier", "label": "Classifier"}
        t0 = time.monotonic()
        classifier_result = await _classifier.run(ticket_text)
        classifier_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **classifier_result}
        await asyncio.sleep(0.05)

        category = classifier_result.get("category", "general")
        priority = classifier_result.get("priority", "medium")
        classifier_confidence = classifier_result.get("confidence", 0.5)

        # ----------------------------------------------------------------
        # Agent 3: Researcher (RAG)
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "researcher", "label": "Researcher (RAG)"}
        t0 = time.monotonic()
        researcher_result = await asyncio.get_event_loop().run_in_executor(
            None, _researcher.run, ticket_text, category
        )
        researcher_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **researcher_result}
        await asyncio.sleep(0.05)

        kb_context = researcher_result.get("kb_context", "")
        retrieved_articles = researcher_result.get("articles", [])

        # ----------------------------------------------------------------
        # Agent 4: Responder
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "responder", "label": "Responder"}
        t0 = time.monotonic()
        responder_result = await _responder.run(ticket_text, category, kb_context, plan)
        responder_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **responder_result}
        await asyncio.sleep(0.05)

        draft = responder_result.get("draft", "")

        # ----------------------------------------------------------------
        # Agent 5: Grounding Checker (Verification Layer 1)
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "grounding_checker", "label": "Grounding Checker"}
        t0 = time.monotonic()
        grounding_result = await asyncio.get_event_loop().run_in_executor(
            None, _grounding.run, draft, kb_context, retrieved_articles
        )
        grounding_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **grounding_result}
        await asyncio.sleep(0.05)

        grounding_score = grounding_result.get("grounding_score", 0.0)
        verified_response = grounding_result.get("verified_response", draft)

        # ----------------------------------------------------------------
        # Agent 6: Confidence Scorer (Verification Layer 2 — routing)
        # ----------------------------------------------------------------
        yield {"type": "agent_start", "agent": "confidence_scorer", "label": "Confidence Scorer"}
        t0 = time.monotonic()
        scorer_result = await asyncio.get_event_loop().run_in_executor(
            None,
            _scorer.run,
            classifier_confidence,
            grounding_score,
            novelty_result.get("is_novel", False),
            novelty_result.get("similarity_to_known", 0.5),
            priority,
        )
        scorer_result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
        yield {"type": "agent_done", **scorer_result}

        # ----------------------------------------------------------------
        # Final output
        # ----------------------------------------------------------------
        yield {
            "type": "complete",
            "verified_response": verified_response,
            "routing": scorer_result.get("routing"),
            "routing_label": scorer_result.get("routing_label"),
            "routing_reason": scorer_result.get("routing_reason"),
            "final_confidence": scorer_result.get("final_confidence"),
            "category": category,
            "priority": priority,
            "is_novel": novelty_result.get("is_novel", False),
            "grounding_score": grounding_score,
            "claims_checked": grounding_result.get("claims_checked", 0),
            "claims_grounded": grounding_result.get("claims_grounded", 0),
            "llm_mode": _llm.mode,
        }

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
