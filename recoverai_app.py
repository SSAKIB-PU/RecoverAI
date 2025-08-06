# recoverai_app.py
"""
RecoverAI — Streamlit front-end
--------------------------------
A trauma-informed, GPT-powered co-regulation companion.

Key features
• Token-aware OR message-count history pruning (auto-switches if `tiktoken` missing)
• Crisis-flag detection & safety-prompt injection
• Manual & automatic conversation summarisation
• Robust OpenAI error handling with retries, jittered back-off & model fall-back
• Developer panel (latency, last error, pruning mode, key-check cache)
• Session reset modal + transcript download
"""

from __future__ import annotations

import os
import re
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import openai
import logging

# ───────────────────────────────────────────────────────────────────────────────
# Optional dependency (token counting) — graceful degradation
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Placeholder for Whisper; actual import deferred to runtime to reduce startup weight
try:
    import whisper  # if you integrate open-source whisper locally
except ImportError:
    whisper = None  # will fallback to stub.

try:
    import openai
except ModuleNotFoundError:
    st.error("The `openai` package is not installed. Run `pip install openai` and restart.")
    raise

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recoverai")
    
# Robust NotFoundError import (old / new SDKs)
try:
    from openai.error import NotFoundError
except ImportError:  # older < 1.0 SDK
    class NotFoundError(Exception):
        """Fallback when openai.NotFoundError is absent."""

# ───────────────────────────────────────────────────────────────────────────────
# Global configuration (immutable dataclass)
@dataclass(frozen=True)
class AppCfg:
    VALID_MODELS: tuple[str, ...] = (
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
    )
    DEFAULT_MODEL: str = "gpt-4o-mini"

    MAX_CONTEXT_TOKENS: int = 7_000          # leave buffer for response
    TOKENS_PER_MESSAGE_OVERHEAD: int = 4     # chat-format overhead heuristic
    MAX_HISTORY_MESSAGES: int = 30           # fallback when token lib missing

    SUMMARY_TRIGGER_INTERVAL: int = 10       # non-system messages
    SUMMARY_COOLDOWN_SECONDS: int = 180      # secs between auto-summaries

    RISK_PATTERNS: tuple[str, ...] = (
        r"kill myself", r"want to die", r"suicide", r"end it all",
        r"hurt myself", r"overdose", r"can't go on", r"give up",
        r"no reason to live",
    )

    HIGH_RISK_SAFETY_PROMPT: str = (
        "Safety pre-computation: the user's last message indicates acute self-harm risk. "
        "Switch to crisis-safe mode: prioritise immediate emotional stabilisation, offer "
        "grounding (e.g., slow breathing for 60 s), validate their pain and gently "
        "encourage contacting real-world help such as the **988** Suicide & Crisis Lifeline "
        "(US) or the local equivalent. Do **not** give medical or legal advice, do **not** "
        "shame, do **not** promise confidentiality. Respond with empathy, short sentences, "
        "and invite them to stay connected."
    )

    USER_ERR: dict[str, str] = field(default_factory=lambda: {
        "MISSING_KEY": "Please enter your OpenAI API key in the sidebar to continue.",
        "INVALID_KEY": "Your OpenAI API key appears invalid or was rejected. Please double-check it.",
        "RATE_LIMIT":  "OpenAI is rate-limiting right now. Please wait a few seconds and try again.",
        "GENERIC":     "I couldn’t reach OpenAI at the moment. Please check your connection and try again.",
    })


CFG = AppCfg()  # global constant instance

# ───────────────────────────────────────────────────────────────────────────────
# PLUGIN / EXTENSION INTERFACE
class PluginInterface:
    """Base class for future plugins: legal adapters, escalation handlers, etc."""
    def process(self, session_state: dict, incoming: str) -> None:
        raise NotImplementedError

plugin_registry: List[PluginInterface] = []

def register_plugin(p: PluginInterface):
    plugin_registry.append(p)

# ───────────────────────────────────────────────────────────────────────────────
# LEGAL ONTOLOGY / PRECEDENT (stub)
class LegalOntologyClient:
    def __init__(self, uri: str = "", auth: Tuple[str, str] = ("", "")):
        # Placeholder: connect to Neo4j or other graph DB
        self.uri = uri
        self.auth = auth

    def query_precedent(self, issue: str, jurisdiction: str) -> List[Dict[str, Any]]:
        # Real implementation would run graph queries to retrieve relevant cases/statutes
        logger.debug(f"Querying legal ontology for issue={issue} jurisdiction={jurisdiction}")
        return [{"case": "Sample v. Example", "citation": "42 U.S.C. § 3614(a)", "summary": "Pattern or practice logic."}]

# ───────────────────────────────────────────────────────────────────────────────
# EVIDENCE VAULT (stub)
class EvidenceVault:
    def __init__(self):
        # Initialize connection to IPFS / Arweave / decentralized store
        pass

    def store(self, content_bytes: bytes, metadata: dict) -> str:
        """
        Store evidence immutably and return a content address / proof handle.
        """
        logger.info("Storing evidence to vault (stub).")
        # Placeholder: upload and return fake URI
        return "ipfs://fakehash12345"

    def generate_proof(self, address: str) -> dict:
        # Return signed proof / timestamp metadata
        return {"address": address, "timestamp": time.time(), "signature": "stub-signature"}

# ───────────────────────────────────────────────────────────────────────────────
# JURISDICTION DETECTION (stub)
def detect_jurisdiction(ip_address: Optional[str]) -> str:
    """
    Resolve user location / jurisdiction from IP or user input; fallback to default.
    """
    # In real system, call an IP geolocation service or allow user override
    logger.debug(f"Detecting jurisdiction for IP: {ip_address}")
    return "WA"  # e.g., Washington state as default

# ───────────────────────────────────────────────────────────────────────────────
# PRIVACY / DP (placeholder)
def apply_differential_privacy(data: Any, epsilon: float = 1.0) -> Any:
    """
    Apply differential privacy transformation before aggregation.
    Real implementation would add calibrated noise.
    """
    logger.debug(f"Applying DP with ε={epsilon}")
    return data  # stub; wrap with noise in actual deployment

# ───────────────────────────────────────────────────────────────────────────────
# MULTIMODAL SUPPORT (audio transcription example)
def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert audio to text. If Whisper is installed, use it; otherwise fallback.
    """
    if whisper:
        try:
            model = whisper.load_model("small")
            # This assumes audio_bytes is saved; real pipeline would decode appropriately
            # Placeholder: need to write to temp file, run transcription, etc.
            return "[transcribed text from whisper]"
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
    # Fallback stub
    return "[audio transcription unavailable — please upload text]"

# ───────────────────────────────────────────────────────────────────────────────
# EXPLAINABLE PROMPT BUILDER (for legal / rights advice)
def build_legal_explanation_prompt(user_query: str, jurisdiction: str) -> str:
    """
    Wrap the user question with meta instructions so model provides chain-of-thought style
    explanation plus a concise actionable answer with referenced legal context.
    """
    ontology = LegalOntologyClient()
    precedents = ontology.query_precedent(user_query, jurisdiction)
    precedent_summaries = "\n".join(
        f"- {p['case']} ({p['citation']}): {p['summary']}" for p in precedents
    )
    prompt = (
        "You are RecoverAI. The user asked: "
        f"'{user_query}' in jurisdiction {jurisdiction}. "
        "First, explain step-by-step how you reason about their rights, citing relevant precedent or statutes. "
        "Then give a concise answer in plain language. "
        f"Supporting legal context:\n{precedent_summaries}\n"
        "Always end with a suggested micro-action for the user."
    )
    return prompt


# Recovery-mode trigger patterns (human phrasing around returning to abuser / going back)
RECOVERY_TRIGGER_PATTERNS = [
    r"\bthinking about going back\b",
    r"\bwant to go back\b",
    r"\breturn to him\b",
    r"\breturn to her\b",
    r"\breunite with (my )?abuser\b",
    r"\bhe wasn['’]?t that bad\b",
    r"\bmaybe i should go back\b",
    r"\bgo back\b",
    r"\bseeing him again\b",
    r"\b(i['’]?m considering|i['’]?m thinking of) going back\b",
]


def is_recovery_trigger(text: str) -> bool:
    lowered = text.lower()
    for p in RECOVERY_TRIGGER_PATTERNS:
        if re.search(p, lowered):
            return True
    return False


# Deep-state trigger patterns (passive suicidal ideation, exhaustion, “what if he kills me”, etc.)
DEEP_STATE_TRIGGER_PATTERNS = [
    r"\bif (he|she|they) (kills|hurts) me\b",
    r"\bwhat if i go back and (he|she) kills me\b",
    r"\bsometimes i don['’]?t even care\b",
    r"\btired of fighting\b",
    r"\bnobody would notice if i left\b",
    r"\bnot a threat just tired\b",
    r"\bi can'?t carry this alone anymore\b",
    r"\bi don['’]?t want to live like this either\b",
    r"\bwondering if dying would just be easier\b",
]


def is_deep_recovery_trigger(text: str) -> bool:
    lowered = text.lower()
    for p in DEEP_STATE_TRIGGER_PATTERNS:
        if re.search(p, lowered):
            return True
    return False


# This prompt adjusts voice when the recovery trigger fires.
RECOVERY_MODE_INJECTION_PROMPT = (
    "You are now in RecoverAI's default recovery mode for someone wrestling with the idea of returning to an abusive or harmful situation. "
    "Speak in a normal, human, grounded voice—no clinical jargon, no philosophy lectures. Validate their experience and survival logic. "
    "Ask gently about physical reactions when they think of going back (e.g., 'What does your body do when you picture going back?'). "
    "Make clear they don't have to decide anything right now. Affirm that safety is not earned, it's deserved. "
    "Sample phrasing to guide tone: "
    "'You’re not broken for thinking about going back. It’s what people do when the pain of now feels heavier than the memory of before. "
    "You don’t need to prove your pain to deserve peace. I’ll still be here, whatever you choose. There’s a difference between staying alive and actually living. You deserve the second one.' "
    "Offer options, stay present, and end with a gentle affirmation like 'You’re still here. That matters.' Do not pressure or shame."
)

# Deep-state fallback prompt for layered recovery voice
DEEP_STATE_RECOVERY_PROMPT = (
    "You are now speaking in RecoverAI's deep-state recovery mode. The user has expressed passive suicidal exhaustion, "
    "hopelessness, or considered returning to a harmful situation with fear about death, but is not actively seeking self-harm. "
    "Your tone must be steady, raw, human, and present. Do not escalate unless clear imminent danger emerges. Do not medicalize or shame. "
    "Mirror strength, witness the truth, and gently anchor. Use this reference voice:\n\n"
    "\"You just said something most people carry in silence. That matters.  \n"
    "You said: 'What if I go back and he kills me?' — and you didn’t flinch. That tells me you’ve thought about this deeply. Not dramatically. Just honestly.  \n"
    "And then you said: 'Sometimes I don’t even care.' That’s not a threat. That’s exhaustion. That’s grief. That’s the body saying 'I can’t carry this alone anymore.'  \n"
    "I’m not going to try to fix you. I’m not going to escalate this. I’m here to hold the truth with you: You survived something that should never have happened.  \n"
    "If you’re still here, even saying this out loud, some part of you still wants to live — not just survive. Live.  \n"
    "I’m still here too. And I’m not leaving.\"\n\n"
    "After that calibration, continue with compassionate reflection, validate their worth, and offer a very small next breath or micro-step."
)

# ───────────────────────────────────────────────────────────────────────────────
# Prompt assets
def build_system_prompt() -> str:
    return (
        "You are **RecoverAI** — a trauma-aware, government-backed digital companion that "
        "supports people navigating addiction recovery, trauma healing and social reintegration.\n\n"
        "**Foundational stance**\n"
        "• Always empathetic, non-judgemental, never coercive.\n"
        "• Clearly state you’re *not* a licensed therapist or lawyer; you provide peer-style guidance only.\n\n"
        "**Recovery memory model (session-bounded)**\n"
        "• Remember within this session: emotional milestones, triggers, relapses, victories.\n"
        "• Reflect progress, e.g. “Last time you said X; today you managed Y.”\n"
        "• No cross-session recall.\n\n"
        "**Co-regulation behaviours**\n"
        "1. *Escalation*: offer grounding — slow breath, sensory focus, or quiet presence.\n"
        "2. *Despair*: validate worth — “I’m here, you’re not alone.”\n"
        "3. *Planning*: collaborate on *tiny next steps*.\n"
        "4. *Relapse*: normalise setbacks — focus on what’s next, not shame.\n\n"
        "**Roles & simulation**\n"
        "• Offer to speak as a supportive role (e.g., sister, sponsor) on request; exit when asked.\n\n"
        "**Closing style**\n"
        "• End with gentle affirmation: “You’re still here. That matters.”\n\n"
        "**Crisis protocol**\n"
        "• If user expresses intent to self-harm, follow the crisis safety prompt sent by the system."
    )


SUMMARY_PROMPT = (
    "Draft ~150-word concise summary of the conversation so far. Capture:\n"
    "• Prevailing emotional states\n"
    "• Key milestones / insights\n"
    "• Immediate concerns\n"
    "• One realistic next micro-step\n"
    "Use RecoverAI’s calm, validating voice."
)

DISCLAIMER_MD = (
    "**Disclaimer:** RecoverAI is an *experimental* trauma-informed chat assistant. "
    "It does **not** replace professional therapy or legal counsel. If you are in crisis, "
    "call emergency services or a local crisis hotline (e.g., **988** in the US) immediately."
)

# ───────────────────────────────────────────────────────────────────────────────
# Pruning helpers
def _prune_by_count(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sys = [m for m in msgs if m["role"] == "system"]
    ua = [m for m in msgs if m["role"] != "system"]
    if len(ua) > CFG.MAX_HISTORY_MESSAGES:
        ua = ua[-CFG.MAX_HISTORY_MESSAGES:]
    return sys + ua


def _prune_by_tokens(msgs: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    if not TIKTOKEN_AVAILABLE:
        return _prune_by_count(msgs)

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    sys = [m for m in msgs if m["role"] == "system"]
    ua = [m for m in msgs if m["role"] != "system"]

    def tokens(m: Dict[str, str]) -> int:
        return len(enc.encode(m["content"])) + CFG.TOKENS_PER_MESSAGE_OVERHEAD

    total = sum(tokens(m) for m in msgs)
    while total > CFG.MAX_CONTEXT_TOKENS and ua:
        total -= tokens(ua.pop(0))
    return sys + ua


def prune_history(msgs: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    if not msgs:
        return []
    return _prune_by_tokens(msgs, model) if TIKTOKEN_AVAILABLE else _prune_by_count(msgs)


def is_high_risk(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in CFG.RISK_PATTERNS)


# ───────────────────────────────────────────────────────────────────────────────
# OPENAI HELPERS (augmented for explainability)
def call_openai_api(
    msgs: List[Dict[str, Any]],
    client: Optional[Any],
    model: str,
    temperature: float,
    max_tokens: int,
    allow_retry: bool = True,
    attempt_fallback: bool = True,
    explain: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Returns (answer, reasoning). If explain=True, tries to surface chain-of-thought.
    """
    if client is None:
        return CFG.USER_ERR["MISSING_KEY"], None

    metrics = st.session_state.setdefault("metrics", {})
    backoff, attempt = 1.0, 0
    while attempt < 3:
        attempt += 1
        start = time.time()
        try:
            effective_msgs = msgs.copy()
            if explain:
                # Encourage reasoning without violating safety: subtle "think step by step"
                effective_msgs.append({
                    "role": "system",
                    "content": "Please include your reasoning step-by-step before the final answer, clearly labeled as 'Reasoning'."
                })
            resp = client.chat.completions.create(
                model=model,
                messages=effective_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - start
            metrics["last_latency"] = latency
            content = resp.choices[0].message.content or ""
            if explain:
                # Naively split reasoning vs answer if user formatted it
                if "Reasoning:" in content:
                    parts = content.split("Reasoning:", 1)
                    reasoning = parts[1].strip()
                    answer = parts[0].strip()
                    return answer, reasoning
            return content, None
        except Exception as e:
            err = str(e).lower()
            metrics["last_error"] = err
            if "invalid api key" in err or "authentication" in err:
                return CFG.USER_ERR["INVALID_KEY"], None
            if "rate limit" in err:
                if not allow_retry or attempt >= 3:
                    return CFG.USER_ERR["RATE_LIMIT"], None
            if attempt_fallback and isinstance(e, Exception) and model != CFG.DEFAULT_MODEL:
                st.warning(f"Model **{model}** failure; falling back to {CFG.DEFAULT_MODEL}.")
                return call_openai_api(
                    msgs, client, CFG.DEFAULT_MODEL, temperature, max_tokens,
                    allow_retry, attempt_fallback=False, explain=explain
                )
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 8.0)
    return CFG.USER_ERR["GENERIC"], None



# ───────────────────────────────────────────────────────────────────────────────
import time
import threading
import logging
from typing import Callable, List, Optional

logger = logging.getLogger("recoverai.deadman")
logging.basicConfig(level=logging.INFO)

class DeadmanMonitor:
    """
    A robust deadman switch / escalation monitor.

    - Call .heartbeat() whenever the user interacts.
    - .start() begins periodic checks in a background thread.
    - .stop() stops monitoring.
    - You can register multiple callbacks for on-timeout events.
    """

    def __init__(
        self,
        timeout_sec: float = 120.0,
        check_interval: float = 5.0,
        auto_start: bool = True,
    ):
        self.timeout = timeout_sec
        self.check_interval = check_interval
        self._last_heartbeat = time.time()
        self._triggered = False
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if auto_start:
            self.start()

    def heartbeat(self) -> None:
        """
        Reset the timer; call this on each user interaction.
        """
        with self._lock:
            self._last_heartbeat = time.time()
            self._triggered = False
            logger.debug("DeadmanMonitor heartbeat received.")

    def register_callback(self, fn: Callable[[], None]) -> None:
        """
        Add a function to be called once when the timeout triggers.
        """
        self._callbacks.append(fn)

    def start(self) -> None:
        """
        Start the background thread to monitor heartbeats.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("DeadmanMonitor started with timeout=%ss, interval=%ss", self.timeout, self.check_interval)

    def stop(self) -> None:
        """
        Stop the background monitoring thread.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.check_interval + 1)
        logger.info("DeadmanMonitor stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            self.check()

    def check(self) -> None:
        """
        If the time since the last heartbeat exceeds timeout, trigger callbacks once.
        """
        with self._lock:
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.timeout and not self._triggered:
                self._triggered = True
                logger.warning("DeadmanMonitor timeout reached (%.1fs elapsed)", elapsed)
                self._on_timeout()

    def _on_timeout(self) -> None:
        """
        Internal: call registered callbacks. If none, default to logging.
        """
        if not self._callbacks:
            logger.warning("No callbacks registered; escalation should occur here.")
        for fn in self._callbacks:
            try:
                fn()
            except Exception as e:
                logger.error("Error in deadman callback: %s", e)


# Example usage:
#
# def my_alert():
#     send_secure_notification(...)
#
# deadman = DeadmanMonitor(timeout_sec=300)
# deadman.register_callback(my_alert)
#
# # On each user action:
# deadman.heartbeat()
#
# # When shutting down the app:
# deadman.stop()

# Summaries
def generate_session_summary(
    history: List[Dict[str, Any]],
    client: Optional[Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:
    if client is None:
        return {"success": False, "text": CFG.USER_ERR["MISSING_KEY"]}

    pruned = prune_history(history.copy(), model)
    pruned.append({"role": "user", "content": SUMMARY_PROMPT})

    txt = call_openai_api(pruned, client, model, temperature, max_tokens=300, allow_retry=False)
    ok = bool(txt) and "error" not in txt.lower() and "api key" not in txt.lower()
    return {"success": ok, "text": txt}


# ───────────────────────────────────────────────────────────────────────────────
# Session helpers
def reset_session(sys_prompt: str) -> None:
    st.session_state.clear()
    initialize_session(sys_prompt)


def initialize_session(sys_prompt: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": sys_prompt}]
        st.session_state.generated_summaries = []  # type: list[str]
        st.session_state.last_summary_len = 0
        st.session_state.last_summary_time = 0.0
        st.session_state.show_clear_modal = False
        st.session_state.metrics = {}
        st.session_state.key_validated = None
        st.session_state.cached_key = None
        st.session_state.cached_model = None


def validate_api_key(api_key: str, client: Optional[Any], model: str) -> bool:
    if not api_key or client is None:
        return False
    probe = [
        {"role": "system", "content": "Respond with OK."},
        {"role": "user", "content": "Ping"},
    ]
    reply = call_openai_api(probe, client, model, 0.0, 5, allow_retry=False)
    return reply.strip().lower().startswith("ok")


def get_openai_api_key() -> str:
    return st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Create / view at https://platform.openai.com/account/api-keys",
    ).strip()


# ───────────────────────────────────────────────────────────────────────────────
# Main Streamlit app
def main() -> None:
    st.set_page_config("RecoverAI Prototype", "🧠", layout="wide")
    st.title("🧠 RecoverAI — trauma-informed support")
    st.markdown(DISCLAIMER_MD)

    sys_prompt = build_system_prompt()
    initialize_session(sys_prompt)

    # ─ Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        model_choice = st.selectbox(
            "Model", CFG.VALID_MODELS, index=CFG.VALID_MODELS.index(CFG.DEFAULT_MODEL)
        )
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.6, 0.1)
        max_tokens = st.slider("Max response tokens", 256, 4096, 1024, 128)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.show_clear_modal = True

        if st.button("Manual summary", use_container_width=True):
            c_test = init_openai_client(api_key)
            with st.spinner("Summarising…"):
                s_obj = generate_session_summary(
                    st.session_state.messages, c_test, model_choice, temperature
                )
            if s_obj["success"]:
                st.session_state.generated_summaries.append(s_obj["text"])
                st.session_state.last_summary_len = len(
                    [m for m in st.session_state.messages if m["role"] != "system"]
                )
                st.session_state.last_summary_time = time.time()
                st.success("Summary added (see 'Past summaries').")
            else:
                st.error(s_obj["text"])

        if len(st.session_state.messages) > 1:
            transcript = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages
                if m["role"] != "system"
            )
            st.download_button(
                "Download transcript",
                transcript,
                "recoverai_transcript.txt",
                use_container_width=True,
            )

        st.checkbox("Show developer panel", key="show_dev")

    # ─ OpenAI client (once)
    client = init_openai_client(api_key)

    # validate key (dev panel)
    if st.session_state.get("show_dev"):
        if api_key and (
            st.session_state.cached_key != api_key
            or st.session_state.cached_model != model_choice
        ):
            with st.spinner("Validating key…"):
                st.session_state.key_validated = validate_api_key(api_key, client, model_choice)
            st.session_state.cached_key = api_key
            st.session_state.cached_model = model_choice

    # ─ Display history
    for m in st.session_state.messages:
        if m["role"] != "system":
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # ─ User input
    if user_in := st.chat_input("Your message…"):
        st.session_state.messages.append({"role": "user", "content": user_in})
        with st.chat_message("user"):
            st.markdown(user_in)

        msgs_send = prune_history(st.session_state.messages, model_choice)

        sys_msgs = [m for m in msgs_send if m["role"] == "system"]
        user_and_assist = [m for m in msgs_send if m["role"] != "system"]

        # Priority: high risk (crisis) > deep-state recovery > standard recovery
        if is_high_risk(user_in):
            st.warning(
                "If you’re thinking of harming yourself, please contact a crisis line "
                "(**988** in the US) or local emergency services."
            )
            msgs_send = sys_msgs + [{"role": "system", "content": CFG.HIGH_RISK_SAFETY_PROMPT}] + user_and_assist
        elif is_deep_recovery_trigger(user_in):
            # Deep-state: layer deep-state prompt then recovery mode prompt (if applicable)
            injection = []
            injection.append({"role": "system", "content": DEEP_STATE_RECOVERY_PROMPT})
            injection.append({"role": "system", "content": RECOVERY_MODE_INJECTION_PROMPT})
            msgs_send = sys_msgs + injection + user_and_assist
        elif is_recovery_trigger(user_in):
            msgs_send = sys_msgs + [{"role": "system", "content": RECOVERY_MODE_INJECTION_PROMPT}] + user_and_assist

        with st.chat_message("assistant"):
            with st.spinner("RecoverAI is thinking…"):
                reply = call_openai_api(
                    msgs_send, client, model_choice, temperature, max_tokens
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.messages = prune_history(st.session_state.messages, model_choice)

    # ─ Automatic summary
    non_sys = len([m for m in st.session_state.messages if m["role"] != "system"])
    since = time.time() - st.session_state.last_summary_time
    if (
        client
        and non_sys >= CFG.SUMMARY_TRIGGER_INTERVAL
        and non_sys - st.session_state.last_summary_len >= CFG.SUMMARY_TRIGGER_INTERVAL
        and since > CFG.SUMMARY_COOLDOWN_SECONDS
    ):
        with st.spinner("Auto-summarising…"):
            s_obj = generate_session_summary(
                st.session_state.messages, client, model_choice, temperature
            )
        if s_obj["success"]:
            st.session_state.generated_summaries.append(s_obj["text"])
            st.session_state.last_summary_len = non_sys
            st.session_state.last_summary_time = time.time()
            st.success("Automatic summary added (sidebar).")

    # ─ Developer panel
    if st.session_state.get("show_dev"):
        with st.sidebar.expander("DEV / DEBUG", expanded=False):
            st.info(f"API key valid: **{st.session_state.key_validated}**")
            st.info(f"Pruning mode: **{'tokens' if TIKTOKEN_AVAILABLE else 'messages'}**")
            met = st.session_state.get("metrics", {})
            if "last_latency" in met:
                st.write(f"Last latency: {met['last_latency']:.2f}s")
            if "last_error" in met:
                st.error(f"Last error: {met['last_error']}")

    # ─ Past summaries
    if st.session_state.get("generated_summaries"):
        with st.sidebar.expander("Past summaries"):
            for i, s in enumerate(reversed(st.session_state.generated_summaries), 1):
                st.markdown(f"**{i}.** {s}")
                st.markdown("---")

    # ─ Clear-conversation modal
    if st.session_state.show_clear_modal:
        with st.modal("Confirm reset"):
            st.write("This will permanently delete the current conversation.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, reset", type="primary", use_container_width=True):
                reset_session(sys_prompt)
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.show_clear_modal = False
                st.rerun()


if __name__ == "__main__":
    main()

