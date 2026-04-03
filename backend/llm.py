"""
LLM Client abstraction.

Priority order (auto-detected at runtime):
  1. Ollama  — local open-source LLM (no key, best for local dev)
  2. Groq    — free cloud API, serverless-compatible (set GROQ_API_KEY env var)
  3. DemoLLM — rule-based fallback, zero dependencies, works everywhere

For Vercel deployment set GROQ_API_KEY in your Vercel environment variables.
Groq is free: https://console.groq.com
"""

import json
import os
import re
import httpx
import asyncio
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3.2"
GROQ_BASE     = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.1-8b-instant"   # fast, free, high quality


# ─────────────────────────── Ollama ───────────────────────────

async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def _ollama_generate(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


# ─────────────────────────── Groq ───────────────────────────

def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


async def _groq_generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GROQ_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {_groq_key()}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ─────────────────────────── Demo LLM ───────────────────────────

class DemoLLM:
    """
    Sophisticated rule-based simulator.
    Produces structured outputs that mirror real LLM responses
    so the verification layers (grounding, confidence) behave authentically.
    """

    CATEGORIES = {
        "billing": [
            "invoice", "charge", "payment", "subscription", "refund",
            "price", "billing", "cost", "charged", "receipt", "transaction",
            "credit card", "bank", "duplicate",
        ],
        "access": [
            "login", "password", "account", "access", "locked", "lockout",
            "authentication", "sso", "saml", "okta", "sign in", "locked out",
            "can't log", "cannot log",
        ],
        "api": [
            "api", "endpoint", "webhook", "integration", "request",
            "rate limit", "sdk", "429", "http", "token", "rate",
        ],
        "performance": [
            "slow", "latency", "timeout", "performance", "speed", "lag",
            "loading", "takes", "seconds", "delay",
        ],
        "data": [
            "export", "import", "data", "backup", "restore", "download",
            "csv", "json", "fields", "custom fields",
        ],
        "sla": [
            "sla", "uptime", "downtime", "outage", "unavailable", "breach",
            "credits", "service credit", "availability",
        ],
        "feature": [
            "feature", "request", "add", "support", "would like",
            "can you", "wish", "roadmap", "enhancement", "improve",
        ],
        "account": [
            "admin", "user", "role", "transfer", "owner", "team member",
            "permissions", "leaving", "offboarding",
        ],
        "technical": [
            "error", "bug", "crash", "not working", "broken", "issue",
            "problem", "fail", "sync", "salesforce", "integration",
        ],
    }

    PRIORITY_SIGNALS = {
        "critical": [
            "urgent", "critical", "asap", "immediately", "emergency",
            "production down", "demo", "client presentation",
            "3 hours", "2 hours", "1 hour",
        ],
        "high": [
            "affecting production", "customers affected", "breach",
            "locked out", "can't log in", "missing", "failed",
            "not working", "significant disruption",
        ],
        "medium": [
            "please", "investigate", "check", "resolve", "issue",
            "problem", "not syncing",
        ],
    }

    RESPONSE_TEMPLATES = {
        "billing": (
            "Thank you for reaching out about this billing matter.\n\n"
            "I have flagged this for investigation by our billing team. Duplicate charges or billing errors are "
            "processed within 5–7 business days back to the original payment method. "
            "Per our refund policy, confirmed billing errors are refunded automatically.\n\n"
            "You can expect an update within 1 business day. "
            "If you do not hear back, please reply to this ticket with your transaction reference numbers."
        ),
        "access": (
            "Thank you for contacting support. I understand this is time-sensitive.\n\n"
            "For SAML/SSO issues after an IdP update, the most common cause is a mismatch in the NameID format "
            "or assertion consumer service URL. Please verify in your IdP settings:\n"
            "1. NameID format is set to 'emailAddress'\n"
            "2. Assertion Consumer Service URL is: https://app.company.com/auth/saml/callback\n"
            "3. IdP metadata has been re-uploaded in Settings → Security → SSO\n\n"
            "As an immediate workaround, admins can access the platform at "
            "https://app.company.com/auth/local-recovery to bypass SSO while the configuration is corrected."
        ),
        "api": (
            "Thank you for reporting this. I've reviewed the API rate limit behaviour for your plan.\n\n"
            "Rate limits are enforced per rolling 60-second window. "
            "Bursts that average under the per-minute limit can still trigger a 429 if they arrive in a short burst. "
            "The API response includes a Retry-After header with the exact wait time.\n\n"
            "Recommended approach: implement exponential backoff when you receive a 429, "
            "starting with the Retry-After value. "
            "Also check whether multiple processes are sharing the same API key — combined traffic counts "
            "against a single key's limit."
        ),
        "performance": (
            "Thank you for reporting this performance issue.\n\n"
            "Dashboard slowdowns with large date ranges and many widgets are a known optimisation area. "
            "Based on your description, I recommend:\n"
            "1. Reduce the default date range to 30 days\n"
            "2. Enable dashboard data caching in Settings → Performance → Enable Caching\n"
            "3. Check if scheduled reports are running during your working hours\n\n"
            "These changes typically reduce load time significantly for dashboards of your size. "
            "I have flagged your account to our engineering team for a usage analysis and will follow up within 2 business days."
        ),
        "data": (
            "Thank you for reaching out about the data export. I understand the need for a format "
            "compatible with your workflow.\n\n"
            "Currently, the CSV export includes standard system fields only — custom fields are included "
            "in the JSON export but not the CSV. "
            "This is a known limitation and custom field support in CSV exports is on our product roadmap.\n\n"
            "In the meantime, you can export as JSON and use Power Query in Excel to flatten custom fields "
            "into a spreadsheet. I can share a template if that would help.\n\n"
            "I have added your request to the feature tracker for this specific enhancement."
        ),
        "sla": (
            "Thank you for contacting us regarding the service disruption.\n\n"
            "I can confirm the incident affected platform availability during the reported window. "
            "This exceeded our SLA uptime threshold for your plan.\n\n"
            "Per our SLA policy, you are entitled to a service credit which will appear as a deduction "
            "on your next invoice. I will process this to your account.\n\n"
            "A full incident post-mortem report will be sent to your account within 5 business days. "
            "If this incident caused you to breach an obligation to your own client, "
            "please reply with details and your account manager will be in touch."
        ),
        "account": (
            "Thank you for letting us know about the upcoming admin transition.\n\n"
            "Admin ownership transfers require verification of company ownership. The process is:\n"
            "1. The outgoing admin should initiate the transfer from Settings → Team → Transfer Ownership\n"
            "2. If unavailable, we require written confirmation on company letterhead signed by a director\n"
            "3. Send documentation to support@company.com with subject: 'Admin Transfer — [Company Name]'\n\n"
            "We can process emergency transfers within 4 business hours with proper documentation. "
            "The new admin must already have an active account before the transfer."
        ),
        "technical": (
            "Thank you for reporting this integration issue.\n\n"
            "For sync failures following a platform update, the most effective first step is to "
            "revoke and re-authorise the OAuth connection. "
            "The 'upstream timeout' error suggests the sync service is having difficulty connecting.\n\n"
            "Please check:\n"
            "1. In Salesforce → Setup → Connected Apps, confirm our app is still authorised\n"
            "2. Verify no Salesforce governor limits have been reached\n"
            "3. Check whether your Salesforce instance URL has changed\n\n"
            "I have escalated this to our integrations engineering team with priority status. "
            "You will receive a direct response within 2 hours."
        ),
        "feature": (
            "Thank you for your feature request.\n\n"
            "I've logged this as a product enhancement request in our system. "
            "Our product team reviews all submitted requests and prioritises them based on customer demand "
            "and strategic fit.\n\n"
            "While I can't commit to a specific timeline, I've tagged this request under your account "
            "so that if and when the feature ships, you'll be notified directly."
        ),
        "default": (
            "Thank you for reaching out to our support team.\n\n"
            "I've reviewed your request and am looking into this for you. "
            "Our team will investigate and provide a detailed response within 1 business day.\n\n"
            "If this issue is time-sensitive, please reply with the word URGENT and a brief description "
            "of the business impact and we will escalate immediately."
        ),
    }

    def _score_categories(self, text: str) -> dict:
        text_lower = text.lower()
        return {
            cat: sum(1 for kw in keywords if kw in text_lower)
            for cat, keywords in self.CATEGORIES.items()
        }

    def _detect_priority(self, text: str) -> str:
        text_lower = text.lower()
        for priority, signals in self.PRIORITY_SIGNALS.items():
            if any(s in text_lower for s in signals):
                return priority
        return "low"

    def classify(self, ticket: str) -> dict:
        scores = self._score_categories(ticket)
        best = max(scores, key=scores.get)
        best_score = scores[best]
        category = best if best_score > 0 else "general"
        priority = self._detect_priority(ticket)
        if best_score >= 3:
            confidence = 0.88
        elif best_score == 2:
            confidence = 0.76
        elif best_score == 1:
            confidence = 0.62
        else:
            confidence = 0.40
        return {
            "category": category,
            "priority": priority,
            "confidence": round(confidence, 2),
            "reasoning": (
                f"Matched {best_score} keyword signal(s) for '{category}' category. "
                f"Priority '{priority}' detected from urgency signals in the ticket."
            ),
        }

    def draft_response(self, ticket: str, category: str, kb_articles: list, plan: str = "Professional") -> str:
        template = self.RESPONSE_TEMPLATES.get(category, self.RESPONSE_TEMPLATES["default"])
        return template.format(plan=plan)


# ─────────────────────────── LLM Client ───────────────────────────

class LLMClient:
    """
    Tries backends in priority order:
      Ollama (local) → Groq (cloud, free API key) → DemoLLM (always works)
    """

    def __init__(self):
        self._ollama_ok: Optional[bool] = None
        self._demo = DemoLLM()

    async def _backend(self) -> str:
        """Returns 'ollama', 'groq', or 'demo'."""
        if self._ollama_ok is None:
            self._ollama_ok = await _ollama_available()
        if self._ollama_ok:
            return "ollama"
        if _groq_key():
            return "groq"
        return "demo"

    async def classify(self, ticket: str) -> dict:
        backend = await self._backend()

        if backend in ("ollama", "groq"):
            prompt = (
                "You are a customer support classifier for a B2B software company.\n"
                "Classify the following support ticket.\n\n"
                f"Ticket:\n{ticket}\n\n"
                "Respond ONLY with valid JSON — no markdown, no extra text:\n"
                '{"category": "<billing|access|api|performance|data|sla|account|technical|feature|general>", '
                '"priority": "<critical|high|medium|low>", '
                '"confidence": <0.0-1.0>, '
                '"reasoning": "<one sentence>"}'
            )
            try:
                if backend == "ollama":
                    raw = await _ollama_generate(prompt)
                else:
                    raw = await _groq_generate(prompt)
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass

        return self._demo.classify(ticket)

    async def draft_response(self, ticket: str, category: str, kb_context: str, plan: str = "Professional") -> str:
        backend = await self._backend()

        if backend in ("ollama", "groq"):
            prompt = (
                "You are a helpful B2B software support agent.\n"
                "Draft a professional, accurate response to the following support ticket.\n"
                "Base your response ONLY on the provided knowledge base context.\n"
                "Do not invent policies, prices, or features not mentioned in the context.\n\n"
                f"Category: {category}\nCustomer Plan: {plan}\n\n"
                f"Knowledge Base Context:\n{kb_context}\n\n"
                f"Ticket:\n{ticket}\n\n"
                "Write the support response (2–4 paragraphs, professional tone):"
            )
            try:
                if backend == "ollama":
                    return await _ollama_generate(prompt)
                else:
                    return await _groq_generate(prompt)
            except Exception:
                pass

        return self._demo.draft_response(ticket, category, [], plan)

    @property
    def mode(self) -> str:
        if self._ollama_ok:
            return f"ollama:{OLLAMA_MODEL}"
        if _groq_key():
            return f"groq:{GROQ_MODEL}"
        return "demo"
