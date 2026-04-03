"""
LLM Client abstraction.

Primary: Ollama (local, open-source, no API key required).
Fallback: DemoLLM — rule-based simulator for when Ollama is not available.
"""

import json
import re
import httpx
import asyncio
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


# ---------------------------------------------------------------------------
# Demo LLM (no external dependency)
# ---------------------------------------------------------------------------

class DemoLLM:
    """
    Realistic rule-based LLM simulator.
    Produces structured outputs that mirror what a real LLM would return for
    the classification, response-drafting, and grounding tasks.
    """

    CATEGORIES = {
        "billing": [
            "invoice", "charge", "payment", "subscription", "refund",
            "price", "billing", "cost", "charged", "receipt", "transaction",
            "credit card", "bank",
        ],
        "access": [
            "login", "password", "account", "access", "locked", "lockout",
            "authentication", "sso", "saml", "okta", "sign in", "locked out",
            "can't log", "cannot log",
        ],
        "api": [
            "api", "endpoint", "webhook", "integration", "request", "rate limit",
            "sdk", "429", "http", "token", "rate",
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
            "feature", "request", "add", "support", "would like", "can you",
            "wish", "roadmap", "enhancement", "improve",
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
            "production down", "demo", "client presentation", "3 hours",
            "2 hours", "1 hour",
        ],
        "high": [
            "affecting production", "customers affected", "breach",
            "locked out", "can't log in", "missing", "failed", "not working",
            "significant disruption",
        ],
        "medium": [
            "please", "investigate", "check", "resolve", "issue",
            "problem", "not syncing",
        ],
    }

    RESPONSE_TEMPLATES = {
        "billing": (
            "Thank you for reaching out about this billing matter. I can see your account is on the {plan} plan.\n\n"
            "I have flagged the duplicate charge for investigation by our billing team. Duplicate charges are processed within 5–7 business days back to the original payment method. "
            "Based on our refund policy, if this is confirmed as a duplicate transaction, a full refund will be issued automatically.\n\n"
            "I can confirm that our refund policy covers billing errors of this nature regardless of when they occurred. "
            "You can expect an update within 1 business day. If you do not hear back, please reply to this ticket with your transaction reference numbers."
        ),
        "access": (
            "Thank you for contacting support. I understand this is time-sensitive.\n\n"
            "For SAML/SSO configuration issues after an IdP update, the most common cause is a mismatch in the NameID format or assertion consumer service URL. "
            "Please verify the following in your Okta SAML app settings:\n"
            "1. NameID format is set to 'emailAddress'\n"
            "2. Assertion Consumer Service URL is: https://app.company.com/auth/saml/callback\n"
            "3. Your IdP metadata has been re-uploaded in Settings → Security → SSO\n\n"
            "As an immediate workaround, admin users can access the platform at https://app.company.com/auth/local-recovery to bypass SSO while the configuration is corrected.\n\n"
            "If you need hands-on assistance, our technical team can join a screen-share session — please reply with your availability."
        ),
        "api": (
            "Thank you for reporting this. I've reviewed the API rate limit behaviour for Starter plan accounts.\n\n"
            "The Starter plan allows 60 requests per minute and 10,000 per day. However, rate limits are enforced per rolling 60-second window, "
            "which can sometimes cause bursts that technically average under 60/min to trigger a 429. "
            "The API response includes a Retry-After header with the exact wait time before retrying.\n\n"
            "Recommended approach: implement exponential backoff when you receive a 429, starting with the Retry-After value. "
            "Additionally, check whether multiple processes are sharing the same API key — combined traffic counts against a single key's limit.\n\n"
            "If you are consistently hitting limits, upgrading to Professional provides 300 requests/minute — please let me know if you'd like to discuss this."
        ),
        "performance": (
            "Thank you for reporting this performance issue. Dashboard slowdowns with large date ranges and multiple widgets are a known optimisation area.\n\n"
            "Based on your description (12 widgets, 6-month date range), I recommend:\n"
            "1. Reduce the default date range to 30 days — you can use date filters on individual widgets for longer analysis\n"
            "2. Enable dashboard data caching in Settings → Performance → Enable Caching\n"
            "3. Check if any scheduled reports are running during your working hours (Settings → Reports → Scheduled)\n\n"
            "These changes typically reduce load time from 20+ seconds to under 5 seconds for dashboards of your size.\n\n"
            "I have also flagged your account ID (ACC-10442) to our engineering team for a usage analysis. "
            "We will review your query patterns and follow up within 2 business days."
        ),
        "data": (
            "Thank you for reaching out about the CSV export. I understand the need to include custom fields in a format compatible with your Excel workflow.\n\n"
            "Currently, the CSV export includes standard system fields only — custom fields are included in the JSON export but not the CSV. "
            "This is a known limitation and custom field support in CSV exports is on our product roadmap, though I do not have a confirmed release date to share.\n\n"
            "In the meantime, a workaround used by several customers: export as JSON, then use a simple Python script or Power Query in Excel to flatten the custom fields into a spreadsheet. "
            "I can share a template script if that would help.\n\n"
            "I have added your request to the feature tracker for this specific enhancement. We notify customers when requested features ship."
        ),
        "sla": (
            "Thank you for contacting us regarding the service disruption on March 26th.\n\n"
            "We can confirm the incident affected platform availability from 09:15 to 13:20 UTC — a total of 4 hours and 5 minutes. "
            "This exceeded our Enterprise plan SLA threshold of 99.9% monthly uptime (≤43.8 minutes per month).\n\n"
            "Per our SLA policy, you are entitled to a service credit of 25% of your monthly subscription fee for uptime between 98.0–98.9%. "
            "I will process this credit to your account, which will appear as a deduction on your next invoice.\n\n"
            "A full incident post-mortem report is being prepared and will be sent to your account within 5 business days. "
            "If this incident caused you to breach an obligation to your own client, please reply with details — your account manager will be in touch to discuss further."
        ),
        "account": (
            "Thank you for letting us know about the upcoming admin transition.\n\n"
            "Admin ownership transfers require verification of company ownership. The process is:\n"
            "1. The outgoing admin (James Park) should initiate the transfer from Settings → Team → Transfer Ownership before his last day\n"
            "2. If he is unavailable, we require written confirmation on company letterhead signed by a director, or your Companies House registration number\n"
            "3. Send the documentation to support@company.com with subject: 'Admin Transfer - Nova Corp'\n\n"
            "We can process emergency transfers within 4 business hours with proper documentation. "
            "I recommend starting this process immediately given the timeline. Please note that the new admin (Priya Sharma) must already have an active account before the transfer."
        ),
        "technical": (
            "Thank you for reporting this integration issue.\n\n"
            "For Salesforce sync failures following a platform update, the most effective first step is to revoke and re-authorise the OAuth connection — I can see you have already tried this. "
            "The 'upstream timeout' error suggests the sync service is having difficulty connecting to your Salesforce instance.\n\n"
            "Please check the following:\n"
            "1. In Salesforce → Setup → Connected Apps, confirm our app is still authorised\n"
            "2. Verify no Salesforce governor limits have been reached (Salesforce → Setup → System Overview)\n"
            "3. Check whether your Salesforce instance URL has changed (Setup → My Domain)\n\n"
            "I have escalated this to our integrations engineering team with priority status. "
            "You will receive a direct response from our technical team within 2 hours. Salesforce sync logs are also available in Settings → Integrations → Salesforce → Logs."
        ),
        "feature": (
            "Thank you for your feature request.\n\n"
            "I've logged this as a product enhancement request in our system. Our product team reviews all submitted requests and prioritises them based on customer demand and strategic fit.\n\n"
            "While I can't commit to a specific timeline, I've tagged this request under your account so that if and when the feature ships, you'll be notified directly.\n\n"
            "In the meantime, if there's a workaround that might help with your current workflow, I'd be happy to explore that with you."
        ),
        "default": (
            "Thank you for reaching out to our support team.\n\n"
            "I've reviewed your request and am looking into this for you. "
            "Our team will investigate and provide a detailed response within 1 business day.\n\n"
            "If this issue is time-sensitive, please reply with the word URGENT and a brief description of the business impact, "
            "and we will escalate immediately."
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
        # Confidence: higher if a clear category emerges
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._ollama_ok: Optional[bool] = None
        self._demo = DemoLLM()

    async def _use_ollama(self) -> bool:
        if self._ollama_ok is None:
            self._ollama_ok = await _ollama_available()
        return self._ollama_ok

    async def classify(self, ticket: str) -> dict:
        if await self._use_ollama():
            prompt = (
                "You are a customer support classifier for a B2B software company.\n"
                "Classify the following support ticket.\n\n"
                f"Ticket:\n{ticket}\n\n"
                "Respond ONLY with valid JSON in this exact format:\n"
                '{"category": "<billing|access|api|performance|data|sla|account|technical|feature|general>", '
                '"priority": "<critical|high|medium|low>", '
                '"confidence": <0.0-1.0>, '
                '"reasoning": "<one sentence>"}'
            )
            try:
                raw = await _ollama_generate(prompt, self.model)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass
        return self._demo.classify(ticket)

    async def draft_response(self, ticket: str, category: str, kb_context: str, plan: str = "Professional") -> str:
        if await self._use_ollama():
            prompt = (
                "You are a helpful B2B software support agent.\n"
                "Draft a professional, accurate response to the following support ticket.\n"
                "Base your response ONLY on the provided knowledge base context.\n"
                "Do not invent policies, prices, or features not in the context.\n\n"
                f"Category: {category}\nCustomer Plan: {plan}\n\n"
                f"Knowledge Base Context:\n{kb_context}\n\n"
                f"Ticket:\n{ticket}\n\n"
                "Write the support response (2–4 paragraphs, professional tone):"
            )
            try:
                return await _ollama_generate(prompt, self.model)
            except Exception:
                pass

        kb_articles = []  # demo mode doesn't use kb for template selection
        return self._demo.draft_response(ticket, category, kb_articles, plan)

    @property
    def mode(self) -> str:
        if self._ollama_ok:
            return f"ollama:{self.model}"
        return "demo"
