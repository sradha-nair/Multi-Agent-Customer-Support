# SupportFlow — Multi-Agent Customer Support Pipeline

A working prototype of the multi-agent verification architecture described in the accompanying research note. Built with FastAPI, sentence-transformers, and Ollama (with a full demo-mode fallback).

---

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. (Optional) Start Ollama for real LLM inference
#    If Ollama is not running, the pipeline runs in demo mode automatically.
ollama pull llama3.2

# 3. Run the server
uvicorn main:app --reload --port 8000

# 4. Open in browser
open http://localhost:8000
```

The UI serves from the root. No separate frontend build step is required.

---

## LLM & Model Strategy

| Component | Model | Runs locally? | API key needed? |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Yes | No |
| LLM inference | Ollama (`llama3.2`) | Yes | No |
| Fallback (no Ollama) | Rule-based DemoLLM | Yes | No |

Everything runs locally. No external API keys are required at any point. If Ollama is installed and running, it is used automatically. Otherwise the pipeline switches to `DemoLLM` — a deterministic, keyword-pattern simulator that produces realistic outputs for all six agents. The status indicator in the top-right corner shows which mode is active.

**Why Ollama + sentence-transformers?** Both are open-source, run entirely offline, and require no user accounts or API keys. `all-MiniLM-L6-v2` is 22MB and downloads once on first run. Ollama models are pulled with a single command. The demo fallback means the prototype is immediately runnable even without any model downloads.

---

## Pipeline Architecture


```
Input ticket
     │
     ▼
┌─────────────────────┐
│  1. Novelty Detector│  ← Pre-filter: is this ticket in-distribution?
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Classifier      │  ← Category + priority + confidence (LLM)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Researcher (RAG)│  ← Semantic retrieval from knowledge base
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Responder       │  ← Draft response (LLM, grounded in KB context)
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────────┐
│  5. Grounding Checker  ✓     │  ← VERIFICATION LAYER 1
│  sentence-level KB grounding │    Removes ungrounded claims
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  6. Confidence Scorer  ✓     │  ← VERIFICATION LAYER 2
│  (multi-signal routing)      │    Routes to auto-approve / review / escalate
└─────────┬────────────────────┘
          │
          ▼
   AUTO_APPROVE / LIGHT_REVIEW / FULL_ESCALATION
```

### What each agent does

**1. Novelty Detector**
Embeds the incoming ticket using `all-MiniLM-L6-v2` and computes cosine similarity to a corpus of known historical tickets. Tickets with similarity below 0.45 are flagged as "novel" — outside the known distribution. This implements the novelty-detection mechanism described in the research note's "underappreciated failure mode" section. Novel tickets receive a 25% confidence penalty regardless of what downstream agents report, and are never auto-approved.

**2. Classifier Agent**
Sends the ticket to the LLM with a structured prompt and extracts `{category, priority, confidence, reasoning}` from the JSON response. Categories: billing, access, api, performance, data, sla, account, technical, feature. Priority: critical, high, medium, low. The confidence value feeds directly into the final routing score.

**3. Researcher Agent (RAG)**
Embeds the ticket and retrieves the top-3 most semantically similar knowledge base articles using cosine similarity against pre-computed article embeddings. Also includes any category-matched articles not in the semantic top-3. The retrieved context is passed to the Responder. Relevance scores are displayed in the UI so the reasoning is transparent.

**4. Responder Agent**
Generates a draft response conditioned on: the ticket text, the classified category, the customer plan, and the retrieved KB context. The prompt explicitly instructs the LLM to base claims only on the provided context. In demo mode, category-specific templates are used.

**5. Grounding Checker** *(Verification Layer 1)*
Implements the retrieval-augmented guardrail described in the research note. Splits the draft response into individual sentences, embeds each one, and computes its max cosine similarity to the retrieved KB article embeddings. Sentences below the threshold (0.18) are flagged as ungrounded and removed from the verified response. Outputs a grounding score (fraction of sentences that passed) and a sentence-by-sentence breakdown visible in the UI.

**6. Confidence Scorer** *(Verification Layer 2)*
Combines three signals into a final confidence score:
- Classifier confidence (40% weight)
- Grounding score (40% weight)
- Novelty score (20% weight — penalised for novel tickets)

Routing thresholds: ≥0.70 → AUTO_APPROVE, 0.50–0.69 → LIGHT_REVIEW, <0.50 → FULL_ESCALATION. Critical-priority tickets are never auto-approved, regardless of confidence.

---

## Verification Mechanism

The prototype implements the research note's recommended three-layer architecture:

**Layer 1 — Retrieval-augmented grounding (Agents 3 + 5)**
Every response claim is verified against retrieved KB articles before the response is sent. Ungrounded sentences are removed. This is the only deterministic check — it makes fabrication of non-KB content structurally impossible for covered claims.

**Layer 2 — Confidence-gated routing (Agent 6)**
Multi-signal confidence scoring determines whether a ticket goes to auto-approve, light review, or full escalation. Novel tickets are penalised regardless of how confident the classifier is.

**Novelty detection as a pre-filter (Agent 1)**
The research note's "underappreciated failure mode" is distribution shift — errors that look like normal operation because confidence stays high. This agent addresses it by treating unfamiliarity as a first-class signal, applied before any LLM inference runs.

---

## Data Sources

**Knowledge Base** (`backend/data/kb_articles.json`)
12 synthetic articles covering: billing & pricing, refund policy, SLA guarantees, API authentication & rate limits, SSO configuration, account management, data export, webhook troubleshooting, performance optimisation, onboarding, security & compliance, and third-party integrations.

Written to be representative of a real B2B SaaS product knowledge base. Each article is specific enough to ground factual claims (prices, timeframes, URLs, thresholds) without being so long that semantic retrieval degrades.

**Sample Tickets** (`backend/data/sample_tickets.json`)
10 synthetic tickets covering realistic B2B support scenarios: SSO lockout with time pressure, duplicate billing charges, API rate-limit investigation, admin ownership transfer, dashboard performance degradation, webhook failures, CSV export feature request, refund request, SLA breach claim, and a Salesforce integration failure.

These also serve as the "known distribution" corpus for the novelty detector.

---

## Project Structure

```
├── backend/
│   ├── main.py            FastAPI app + WebSocket endpoint
│   ├── pipeline.py        Pipeline orchestrator (streams agent updates)
│   ├── agents.py          All 6 agent implementations
│   ├── llm.py             LLM abstraction (Ollama + DemoLLM fallback)
│   ├── embeddings.py      sentence-transformers utilities + keyword fallback
│   ├── requirements.txt
│   └── data/
│       ├── kb_articles.json     12 KB articles
│       └── sample_tickets.json  10 sample tickets
└── frontend/
    ├── index.html         Single-page app
    ├── style.css          Pastel blue + white, responsive
    └── app.js             WebSocket client + real-time pipeline rendering
```

---

## One Unexpected Observation

The grounding checker removes sentences with cosine similarity below 0.18 — a deliberately permissive threshold. I expected it to catch hallucinated specifics (invented prices, fake policy details). What I did not expect: it most consistently flags *transitional and hedging sentences* — phrases like "While I can't commit to a specific timeline…" or "If this issue is time-sensitive, please reply…" — because these have low semantic overlap with any KB article.

These sentences are actually harmless and often good practice in support writing. But the grounding checker has no way to distinguish "ungrounded because fabricated" from "ungrounded because conversational." The removal makes the verified response more terse and factual, but occasionally more abrupt.

This is a real tradeoff that emerges directly from the research note's point about verification systems having their own failure modes. A grounding check is the only deterministic layer in the pipeline — but "deterministic" does not mean "correct." It means consistently wrong in the same direction. The fix would be to classify sentence types first and only apply grounding checks to factual claim sentences, not transitional or closing language. That would require a sentence-type classifier as a pre-pass — adding another agent, more latency, and its own failure modes.

The research note argues that reliability is not about making fewer errors; it's about knowing when you're in territory where you might be wrong. The grounding checker does know — it outputs scores and labels every sentence. The question is what you do with that knowledge, and whether removal is always the right response.
