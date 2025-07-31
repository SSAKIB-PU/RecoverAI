# recoverai_app.py
"""
RecoverAI — Streamlit front-end
--------------------------------
A trauma-informed, GPT-powered co-regulation companion.

Key features
•  Token-aware or message-count history pruning (auto-switches if `tiktoken` missing)
•  Crisis-flag detection and safety prompt injection
•  Manual & automatic conversation summarisation (structured return object)
•  Robust OpenAI error handling with retries, jittered back-off and model fallback
•  Developer panel (latency, last error, pruning mode, key check cache)
•  Session reset modal + transcript download
"""

from __future__ import annotations

import os, time, re, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st
import openai

# ────────────────────────────────────────────────────────────────────────────────
# Optional dependency (token counting) ─ graceful degradation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Robust import across OpenAI SDK branches
try:
    from openai.error import NotFoundError
except ImportError:
    try:
        from openai import NotFoundError
    except ImportError:
        class NotFoundError(Exception):
            pass

# ────────────────────────────────────────────────────────────────────────────────
# Global configuration
@dataclass(frozen=True)
class AppCfg:
    VALID_MODELS: tuple[str, ...] = ("gpt-4o-mini", "gpt-4o", "gpt-4-turbo")
    DEFAULT_MODEL: str = "gpt-4o-mini"
    MAX_CONTEXT_TOKENS: int = 7_000  # leave buffer for response
    TOKENS_PER_MESSAGE_OVERHEAD: int = 4  # chat-format overhead heuristic
    MAX_HISTORY_MESSAGES: int = 30  # fallback when token lib missing
    SUMMARY_TRIGGER_INTERVAL: int = 10  # non-system messages
    SUMMARY_COOLDOWN_SECONDS: int = 180  # secs between auto summaries

    RISK_PATTERNS: tuple[str, ...] = (
        r"kill myself", r"want to die", r"suicide", r"end it all",
        r"hurt myself", r"overdose", r"can't go on", r"give up",
        r"no reason to live",
    )

    HIGH_RISK_SAFETY_PROMPT: str = (
        "Safety pre-computation: the user's last message indicates acute self-harm risk. "
        "Switch to crisis-safe mode: prioritise immediate emotional stabilisation, offer "
        "grounding (e.g., slow breathing for 60 s), validate their pain and gently encourage "
        "contacting real-world help such as the 988 Suicide & Crisis Lifeline (US) or the "
        "local equivalent. Do not give medical or legal advice, do not shame, do not promise "
        "confidentiality. Respond with empathy, short sentences, and invite them to stay connected."
    )

    USER_ERR: dict[str, str] = {
        "MISSING_KEY": "Please enter your OpenAI API key in the sidebar to continue.",
        "INVALID_KEY": "Your OpenAI API key appears invalid or was rejected. Please double-check it.",
        "RATE_LIMIT":  "OpenAI is rate-limiting right now. Please wait a few seconds and try again.",
        "GENERIC":     "I couldn’t reach OpenAI at the moment. Please check your connection and try again.",
    }

CFG = AppCfg()

# ────────────────────────────────────────────────────────────────────────────────
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
        "• If user expresses intent to self-harm or overdose, follow the crisis safety prompt you will receive from the system role."
    )

SUMMARY_PROMPT = (
    "Draft a ~150-word concise summary of the conversation so far. Capture:\n"
    "• Prevailing emotional states\n"
    "• Key milestones / insights\n"
    "• Immediate concerns\n"
    "• One realistic next micro-step\n"
    "Use RecoverAI’s calm, validating voice."
)

DISCLAIMER_MD = (
    "**Disclaimer:** RecoverAI is an *experimental* trauma-informed chat assistant. "
    "It does **not** replace professional therapy or legal counsel. If you are in crisis, "
    "please call emergency services or a local crisis hotline (e.g., **988** in the US) immediately."
)

# ────────────────────────────────────────────────────────────────────────────────
# Pruning helpers
def _prune_by_message_count(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_msgs = [m for m in messages if m["role"] == "system"]
    ua_msgs = [m for m in messages if m["role"] != "system"]
    if len(ua_msgs) > CFG.MAX_HISTORY_MESSAGES:
        ua_msgs = ua_msgs[-CFG.MAX_HISTORY_MESSAGES:]
    return system_msgs + ua_msgs

def token_safe_prune(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    if not TIKTOKEN_AVAILABLE:
        return _prune_by_message_count(messages)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    system_msgs = [m for m in messages if m["role"] == "system"]
    ua_msgs = [m for m in messages if m["role"] != "system"]
    def msg_tokens(m: Dict[str, str]) -> int:
        return len(enc.encode(m["content"])) + CFG.TOKENS_PER_MESSAGE_OVERHEAD
    total = sum(msg_tokens(m) for m in messages)
    while total > CFG.MAX_CONTEXT_TOKENS and ua_msgs:
        removed = ua_msgs.pop(0)
        total -= msg_tokens(removed)
    return system_msgs + ua_msgs

def prune_history(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    if not messages:
        return []
    return token_safe_prune(messages, model) if TIKTOKEN_AVAILABLE else _prune_by_message_count(messages)

def is_high_risk(text: str) -> bool:
    return any(re.search(pat, text.lower()) for pat in CFG.RISK_PATTERNS)

# ────────────────────────────────────────────────────────────────────────────────
# OpenAI interaction
def init_openai_client(api_key: str | None) -> Optional[Any]:
    if not api_key:
        return None
    try:
        return openai.OpenAI(api_key=api_key)  # newer SDK
    except TypeError:
        openai.api_key = api_key               # legacy style
        return openai.OpenAI()

def call_openai_api(
    messages: List[Dict[str, Any]],
    client: Optional[Any],
    model: str,
    temperature: float,
    max_tokens: int,
    allow_retry: bool = True,
    attempt_fallback: bool = True,
) -> str:
    if client is None:
        return CFG.USER_ERR["MISSING_KEY"]
    metrics = st.session_state.setdefault("metrics", {})
    backoff, attempt = 1.0, 0
    while attempt < 3:
        attempt += 1
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            metrics["last_latency"] = time.time() - start
            return resp.choices[0].message.content or "I received an empty response. Please retry."
        except NotFoundError:
            if attempt_fallback and model != CFG.DEFAULT_MODEL:
                st.warning(f"Model **{model}** unavailable; falling back to **{CFG.DEFAULT_MODEL}**.")
                return call_openai_api(messages, client, CFG.DEFAULT_MODEL, temperature, max_tokens,
                                       allow_retry, attempt_fallback=False)
            metrics["last_error"] = f"Model {model!r} not found."
            return metrics["last_error"]
        except Exception as e:
            err_l = str(e).lower()
            metrics["last_error"] = err_l
            if "invalid api key" in err_l or "authentication" in err_l:
                return CFG.USER_ERR["INVALID_KEY"]
            if "rate limit" in err_l:
                if not allow_retry or attempt >= 3:
                    return CFG.USER_ERR["RATE_LIMIT"]
            elif not allow_retry:
                break
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 8.0)
    return CFG.USER_ERR["GENERIC"]

# ────────────────────────────────────────────────────────────────────────────────
# Summarisation
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
    text = call_openai_api(pruned, client, model, temperature, max_tokens=300, allow_retry=False)
    success = bool(text) and "error" not in text.lower() and "please enter your openai api key" not in text.lower()
    return {"success": success, "text": text}

# ────────────────────────────────────────────────────────────────────────────────
# Session + UI helpers
def reset_session(system_prompt: str) -> None:
    st.session_state.clear()
    initialize_session(system_prompt)

def initialize_session(system_prompt: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.session_state.generated_summaries: list[str] = []
        st.session_state.last_summary_len = 0
        st.session_state.last_summary_time = 0
        st.session_state.show_clear_modal = False
        st.session_state.metrics = {}
        st.session_state.key_validated = None
        st.session_state.cached_key = None
        st.session_state.cached_model = None

def validate_api_key(api_key: str, client: Optional[Any], model: str) -> bool:
    if not api_key or client is None:
        return False
    probe = [{"role": "system", "content": "Respond with OK."}, {"role": "user", "content": "Ping"}]
    reply = call_openai_api(probe, client, model, temperature=0.0, max_tokens=5, allow_retry=False)
    return reply.strip().lower().startswith("ok")

def get_openai_api_key() -> str:
    return st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Find it at https://platform.openai.com/account/api-keys",
    ).strip()

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit main
def main() -> None:
    st.set_page_config("RecoverAI Prototype", "🧠", layout="wide")
    st.title("🧠 RecoverAI — trauma-informed support")
    st.markdown(DISCLAIMER_MD)

    system_prompt_text = build_system_prompt()
    initialize_session(system_prompt_text)

    with st.sidebar:
        st.header("Configuration")
        api_key = get_openai_api_key()
        model_choice = st.selectbox("Model", CFG.VALID_MODELS, index=CFG.VALID_MODELS.index(CFG.DEFAULT_MODEL))
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.6, 0.1)
        max_tokens = st.slider("Max response tokens", 256, 4096, 1024, 128)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.show_clear_modal = True

        if st.button("Manual summary", use_container_width=True):
            client_test = init_openai_client(api_key)
            with st.spinner("Summarising…"):
                summary_obj = generate_session_summary(st.session_state.messages, client_test, model_choice, temperature)
            if summary_obj["success"]:
                st.session_state.generated_summaries.append(summary_obj["text"])
                st.session_state.last_summary_len = len([m for m in st.session_state.messages if m["role"] != "system"])
                st.session_state.last_summary_time = time.time()
                st.success("Summary added in sidebar.")
            else:
                st.error(summary_obj["text"])

        if len(st.session_state.messages) > 1:
            transcript = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages if m["role"] != "system"
            )
            st.download_button("Download transcript", transcript, "recoverai_transcript.txt", use_container_width=True)

        st.checkbox("Show developer panel", key="show_dev")

    client = init_openai_client(api_key)

    if st.session_state.get("show_dev"):
        if api_key and (st.session_state.cached_key != api_key or st.session_state.cached_model != model_choice):
            with st.spinner("Validating key…"):
                st.session_state.key_validated = validate_api_key(api_key, client, model_choice)
            st.session_state.cached_key, st.session_state.cached_model = api_key, model_choice

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("Your message…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        messages_to_send = prune_history(st.session_state.messages, model_choice)

        risk_flag = is_high_risk(user_input)
        if risk_flag:
            st.warning("If you’re thinking of harming yourself, please contact a crisis line (988 in the US) or local emergency services.")
            sys_part = [m for m in messages_to_send if m["role"] == "system"]
            ua_part = [m for m in messages_to_send if m["role"] != "system"]
            messages_to_send = sys_part + [{"role": "system", "content": CFG.HIGH_RISK_SAFETY_PROMPT}] + ua_part

        with st.chat_message("assistant"):
            with st.spinner("RecoverAI is thinking…"):
                assistant_reply = call_openai_api(
                    messages_to_send, client, model_choice, temperature, max_tokens
                )
            st.markdown(assistant_reply)

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.session_state.messages = prune_history(st.session_state.messages, model_choice)

    non_sys = len([m for m in st.session_state.messages if m["role"] != "system"])
    since_last = time.time() - st.session_state.last_summary_time
    if (client and non_sys >= CFG.SUMMARY_TRIGGER_INTERVAL and
        non_sys - st.session_state.last_summary_len >= CFG.SUMMARY_TRIGGER_INTERVAL and
        since_last > CFG.SUMMARY_COOLDOWN_SECONDS):
        with st.spinner("Auto-summarising…"):
            summ_obj = generate_session_summary(st.session_state.messages, client, model_choice, temperature)
        if summ_obj["success"]:
            st.session_state.generated_summaries.append(summ_obj["text"])
            st.session_state.last_summary_len = non_sys
            st.session_state.last_summary_time = time.time()
            st.success("Automatic summary added (sidebar).")

    if st.session_state.get("show_dev"):
        with st.sidebar.expander("DEV / DEBUG", expanded=False):
            st.info(f"API key valid: **{st.session_state.key_validated}**")
            st.info(f"Pruning mode: **{'tokens' if TIKTOKEN_AVAILABLE else 'messages'}**")
            met = st.session_state.get("metrics", {})
            if "last_latency" in met:
                st.write(f"Last latency: {met['last_latency']:.2f}s")
            if "last_error" in met:
                st.error(f"Last error: {met['last_error']}")

    if st.session_state.get("generated_summaries"):
        with st.sidebar.expander("Past summaries"):
            for i, s in enumerate(reversed(st.session_state.generated_summaries), 1):
                st.markdown(f"**{i}.** {s}")
                st.markdown("---")

    if st.session_state.show_clear_modal:
        with st.modal("Confirm reset"):
            st.write("This will permanently delete the current conversation.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, reset", type="primary", use_container_width=True):
                reset_session(system_prompt_text)
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.show_clear_modal = False
                st.rerun()

if __name__ == "__main__":
    main()


