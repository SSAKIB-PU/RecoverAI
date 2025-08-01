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

import os, re, time, random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import streamlit as st
import openai

# ───────────────────────────────────────────────────────────────────────────────
# Optional dependency (token counting) — graceful degradation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Robust NotFoundError import (old / new SDKs)
try:
    from openai.error import NotFoundError
except ImportError:
    try:
        from openai import NotFoundError
    except ImportError:  # very old SDK
        class NotFoundError(Exception):  # noqa: D401
            """Fallback placeholder when SDK doesn't expose NotFoundError."""


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
    "• Prevailing emotional states\n• Key milestones / insights\n• Immediate concerns\n"
    "• One realistic next micro-step\nUse RecoverAI’s calm, validating voice."
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
    ua  = [m for m in msgs if m["role"] != "system"]
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
    ua  = [m for m in msgs if m["role"] != "system"]

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
# OpenAI helpers
def init_openai_client(api_key: str | None) -> Optional[Any]:
    if not api_key:
        return None
    try:                        # ≥ 1.3
        return openai.OpenAI(api_key=api_key)
    except TypeError:           # ≤ 1.2
        openai.api_key = api_key
        return openai.OpenAI()


def call_openai_api(
    msgs: List[Dict[str, Any]],
    client: Optional[Any],
    model: str,
    temperature: float,
    max_tokens: int,
    allow_retry: bool = True,
    attempt_fallback: bool = True,
) -> str:
    if client is None:
        return CFG.USER_ERR["MISSING_KEY"]

    m = st.session_state.setdefault("metrics", {})
    backoff, attempt = 1.0, 0

    while attempt < 3:
        attempt += 1
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            m["last_latency"] = time.time() - start
            return resp.choices[0].message.content or "I received an empty response. Please retry."
        except NotFoundError:
            if attempt_fallback and model != CFG.DEFAULT_MODEL:
                st.warning(f"Model **{model}** unavailable; falling back to **{CFG.DEFAULT_MODEL}**.")
                return call_openai_api(msgs, client, CFG.DEFAULT_MODEL, temperature, max_tokens,
                                       allow_retry, attempt_fallback=False)
            m["last_error"] = f"Model {model!r} not found."
            return m["last_error"]
        except Exception as e:
            err = str(e).lower()
            m["last_error"] = err
            if "invalid api key" in err or "authentication" in err:
                return CFG.USER_ERR["INVALID_KEY"]
            if "rate limit" in err:
                if not allow_retry or attempt >= 3:
                    return CFG.USER_ERR["RATE_LIMIT"]
            elif not allow_retry:
                break
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 8.0)
    return CFG.USER_ERR["GENERIC"]

# ───────────────────────────────────────────────────────────────────────────────
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
    ok  = bool(txt) and "error" not in txt.lower() and "api key" not in txt.lower()
    return {"success": ok, "text": txt}

# ───────────────────────────────────────────────────────────────────────────────
# Session helpers
def reset_session(sys_prompt: str) -> None:
    st.session_state.clear()
    initialize_session(sys_prompt)


def initialize_session(sys_prompt: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages             = [{"role": "system", "content": sys_prompt}]
        st.session_state.generated_summaries  = []      # type: list[str]
        st.session_state.last_summary_len     = 0
        st.session_state.last_summary_time    = 0.0
        st.session_state.show_clear_modal     = False
        st.session_state.metrics              = {}
        st.session_state.key_validated        = None
        st.session_state.cached_key           = None
        st.session_state.cached_model         = None


def validate_api_key(api_key: str, client: Optional[Any], model: str) -> bool:
    if not api_key or client is None:
        return False
    probe = [
        {"role": "system", "content": "Respond with OK."},
        {"role": "user",   "content": "Ping"},
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
        model_choice = st.selectbox("Model", CFG.VALID_MODELS,
                                    index=CFG.VALID_MODELS.index(CFG.DEFAULT_MODEL))
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.6, 0.1)
        max_tokens  = st.slider("Max response tokens", 256, 4096, 1024, 128)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.show_clear_modal = True

        if st.button("Manual summary", use_container_width=True):
            c_test = init_openai_client(api_key)
            with st.spinner("Summarising…"):
                s_obj = generate_session_summary(st.session_state.messages, c_test, model_choice, temperature)
            if s_obj["success"]:
                st.session_state.generated_summaries.append(s_obj["text"])
                st.session_state.last_summary_len  = len([m for m in st.session_state.messages if m["role"] != "system"])
                st.session_state.last_summary_time = time.time()
                st.success("Summary added (see 'Past summaries').")
            else:
                st.error(s_obj["text"])

        if len(st.session_state.messages) > 1:
            transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}"
                                     for m in st.session_state.messages if m["role"] != "system")
            st.download_button("Download transcript", transcript,
                               "recoverai_transcript.txt", use_container_width=True)

        st.checkbox("Show developer panel", key="show_dev")

    # ─ OpenAI client (once)
    client = init_openai_client(api_key)

    # validate key (dev panel)
    if st.session_state.get("show_dev"):
        if api_key and (st.session_state.cached_key != api_key
                        or st.session_state.cached_model != model_choice):
            with st.spinner("Validating key…"):
                st.session_state.key_validated = validate_api_key(api_key, client, model_choice)
            st.session_state.cached_key   = api_key
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

        if is_high_risk(user_in):
            st.warning("If you’re thinking of harming yourself, please contact a crisis line "
                       "(**988** in the US) or local emergency services.")
            sys = [m for m in msgs_send if m["role"] == "system"]
            ua  = [m for m in msgs_send if m["role"] != "system"]
            msgs_send = sys + [{"role": "system", "content": CFG.HIGH_RISK_SAFETY_PROMPT}] + ua

        with st.chat_message("assistant"):
            with st.spinner("RecoverAI is thinking…"):
                reply = call_openai_api(msgs_send, client, model_choice,
                                        temperature, max_tokens)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.messages = prune_history(st.session_state.messages, model_choice)

    # ─ Automatic summary
    non_sys = len([m for m in st.session_state.messages if m["role"] != "system"])
    since   = time.time() - st.session_state.last_summary_time
    if (client and non_sys >= CFG.SUMMARY_TRIGGER_INTERVAL
            and non_sys - st.session_state.last_summary_len >= CFG.SUMMARY_TRIGGER_INTERVAL
            and since > CFG.SUMMARY_COOLDOWN_SECONDS):
        with st.spinner("Auto-summarising…"):
            s_obj = generate_session_summary(st.session_state.messages, client, model_choice, temperature)
        if s_obj["success"]:
            st.session_state.generated_summaries.append(s_obj["text"])
            st.session_state.last_summary_len  = non_sys
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
