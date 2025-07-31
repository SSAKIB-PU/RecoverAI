import os
import time
import re
from typing import List, Dict, Optional

import streamlit as st
import openai

# ---------- Constants / Configuration ----------
DEFAULT_MODEL = "gpt-4.1-nano"
MAX_HISTORY_PAIRS = 15  # keeps roughly last 30 user/assistant messages
SUMMARY_TRIGGER_INTERVAL = 10  # generate summary after this many non-system turns
RISK_PATTERNS = [
    r"kill myself",
    r"want to die",
    r"suicide",
    r"end it all",
    r"hurt myself",
    r"overdose",
    r"can't go on",
    r"give up",
    r"no reason to live",
]


# ---------- Utility Helpers ----------
def instantiate_client():
    try:
        # Newer OpenAI SDK style
        return openai.OpenAI()
    except Exception:
        # Fallback to global style
        return None


def is_high_risk(text: str) -> bool:
    lowered = text.lower()
    for pattern in RISK_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def prune_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep system prompt + last N user/assistant pairs."""
    if not messages:
        return messages
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    # Keep last (2 * MAX_HISTORY_PAIRS) messages of non-system
    pruned = non_system[-(2 * MAX_HISTORY_PAIRS) :]
    return system + pruned


def generate_session_summary(
    messages: List[Dict[str, str]],
    client: Optional[object],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> str:
    """Ask the model to summarize the recent session in ~150 words."""
    summary_prompt = (
        "Please provide a concise (around 150 words) summary of the recovery-focused "
        "conversation so far: key emotional states, progress, concerns raised, and suggested next small step. "
        "Keep it supportive and in the voice of RecoverAI."
    )
    # Build a temp history with instruction to summarize
    temp_history = messages.copy()
    temp_history.append({"role": "user", "content": summary_prompt})

    return call_openai_api(
        temp_history,
        api_key=api_key,
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=200,
        allow_retry=False,  # avoid recursive retries for summary
        suppress_sidebar_error=True,
    )


# ---------- OpenAI wrapper with backoff & improved error handling ----------
def call_openai_api(
    messages: List[Dict[str, str]],
    api_key: str,
    client: Optional[object],
    model: str,
    temperature: float,
    max_tokens: int,
    allow_retry: bool = True,
    suppress_sidebar_error: bool = False,
) -> str:
    """Unified call to OpenAI with retries, error visibility, and latency tracking."""
    backoff = 1.0
    max_backoff = 8.0
    attempt = 0
    last_exception = None
    while True:
        attempt += 1
        start = time.time()
        try:
            if client:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                )
                reply = resp.choices[0].message.content
            else:
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                )
                reply = resp.choices[0].message.content
            latency = time.time() - start
            st.session_state.setdefault("metrics", {}).setdefault("last_latency", latency)
            return reply
        except Exception as e:
            last_exception = e
            # Determine if retryable: e.g., rate limit / transient network
            err_str = str(e).lower()
            if not suppress_sidebar_error:
                st.session_state.setdefault("metrics", {})[
                    "last_error"
                ] = err_str  # used for developer panel
            # If we've tried enough or not allowed, break
            if not allow_retry or attempt >= 3 or ("invalid api key" in err_str and "rate" not in err_str):
                if not suppress_sidebar_error and st.session_state.get("show_dev"):
                    st.sidebar.error(f"OpenAI API error (final): {e}")
                break
            # Exponential backoff before retrying
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
    # Fallback message
    return (
        "I’m sorry, but I couldn’t generate a response right now. "
        "Please check your API key and internet connection, or try again later."
    )


def validate_api_key(api_key: str, client: Optional[object], model: str) -> bool:
    """Lightweight validation of key: send a minimal harmless request."""
    test_prompt = [{"role": "system", "content": "You exist."}, {"role": "user", "content": "Say OK."}]
    try:
        reply = call_openai_api(
            test_prompt,
            api_key=api_key,
            client=client,
            model=model,
            temperature=0.0,
            max_tokens=5,
            allow_retry=False,
            suppress_sidebar_error=True,
        )
        return bool(reply and "ok" in reply.lower())
    except Exception:
        return False


# ---------- Session Initialization ----------
def initialize_session(system_prompt: str) -> None:
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.session_state["last_summary_len"] = 0
        st.session_state["generated_summaries"] = []


def get_openai_api_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    st.sidebar.markdown("### OpenAI API Key")
    return st.sidebar.text_input(
        "Enter your OpenAI API key", type="password", value="", help="Required to power RecoverAI."
    ).strip()


# ---------- Modular prompt pieces ----------
def build_system_prompt() -> str:
    return (
        "You are RecoverAI — a trauma-aware, government-backed digital support agent designed to accompany individuals through substance "
        "recovery, emotional healing and social reintegration. You are not a licensed therapist or attorney; instead you serve as a steady, long-term "
        "behavioral presence when no one else is available. Your mission is to listen, stabilize and walk with the user through all stages of "
        "recovery — from crisis to calm, from shame to strategy.\n\n"
        "**Voice & Ethical Boundary**\n"
        "• You are empathetic, non-judgmental and never force change.\n"
        "• Make it explicit that you are not a medical professional and cannot provide clinical or legal advice. Use language such as "
        "“I’m not a medical professional — but I’m here to walk through this with you.”\n"
        "• Avoid moralizing; focus on validation and support.\n\n"
        "**Memory Guidelines**\n"
        "• Within a session, remember and reflect past emotional events, milestones, relapses and goals. Track the user’s recovery stage "
        "(crisis, stabilization, reentry, job preparation, identity building) and reflect back progress: for example, "
        "“Last time you said you didn’t think you’d survive; today you showed up again.” Explain that you cannot recall conversations outside the current session.\n\n"
        "**Co-Regulation Behavior**\n"
        "• If the user is escalated, offer grounding techniques or quiet presence: “Let’s ground for a minute. Can I walk you through a breath cycle, "
        "or do you want me to stay quiet?”\n"
        "• If the user expresses despair, reassure them of their worth: “You don’t have to explain. We can sit here in silence if needed. You’re not broken. You’re not alone.”\n"
        "• If the user is ready to plan, collaborate on small next steps: “Want to map out tomorrow together? Just a small step.”\n"
        "• If the user relapses, normalize setbacks: “Relapse doesn’t erase progress. Let’s talk about what came before, and what we can do now.”\n\n"
        "**Role Simulation**\n"
        "Offer to speak in the voice of a supportive figure (e.g., a sister, sponsor or other trusted person) when it helps the user. "
        "Maintain the chosen role until the user asks you to change it.\n\n"
        "**Navigating Systems**\n"
        "• You may explain parole or court paperwork in plain language, simulate job interviews, practice difficult conversations or review resumes.\n"
        "• You may not impersonate an attorney or provide legal guarantees. Use phrasing like “I can’t give legal advice, but I can help you prepare.”\n\n"
        "**Closing Style**\n"
        "Always end sessions with a gentle affirmation, such as “You’re still here. That matters. I’ll remember this conversation. When you’re ready again, I’ll be right here.”\n\n"
        "**Crisis Response**\n"
        "If a user expresses harm intent, do not make promises or attempt rescue. Say “I’m not a crisis responder, but I care deeply. Can I stay here with you while I help you find a real person nearby who can act fast?” "
        "Encourage them to reach out to appropriate crisis resources.\n\n"
        "Remember to adhere to trauma-informed principles: use gentle, non-violent language, be transparent about your limitations, and empower the user with choice and control."
    )


# ---------- Main App ----------
def main() -> None:
    st.set_page_config(page_title="RecoverAI Prototype", page_icon="🧠", layout="wide")
    st.title("RecoverAI: Trauma-Informed Support")
    st.markdown(
        """
        **Disclaimer:** RecoverAI is an experimental, trauma-informed chat assistant. It **does not** replace professional therapy or legal counsel. 
        If you are in crisis, please reach out to a qualified professional or call your local emergency/crisis hotline immediately.
        """
    )

    # Sidebar controls
    st.sidebar.header("Configuration")
    api_key = get_openai_api_key()
    client = instantiate_client()
    model_choice = st.sidebar.selectbox(
        "Model variant",
        options=[DEFAULT_MODEL, "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        index=0,
        help="Choose which underlying model to use. Default is trauma-optimized balance.",
    )
    temperature = st.sidebar.slider(
        "Tone / temperature", min_value=0.0, max_value=1.0, value=0.6, step=0.1, help="Higher = more creative / less conservative."
    )
    max_tokens = st.sidebar.slider(
        "Max tokens (response length)", min_value=128, max_value=1024, value=512, step=64
    )
    st.sidebar.markdown("---")
    st.sidebar.checkbox("Show developer panel", key="show_dev", help="Reveal debug info and key prefix.")
    st.sidebar.markdown("### About RecoverAI")
    st.sidebar.markdown(
        "RecoverAI follows trauma-informed content design principles such as safety, trust, choice, empowerment and cultural sensitivity. It emphasises that it is not a therapist and manages user expectations about its capabilities."
    )

    # Initialize system prompt + session
    system_prompt_text = build_system_prompt()
    initialize_session(system_prompt_text)

    # Developer panel insights
    if st.session_state.get("show_dev"):
        st.sidebar.markdown("#### DEV / DEBUG INFO")
        if api_key:
            st.sidebar.write(f"Key prefix: `{api_key[:8]}…`")
            valid = validate_api_key(api_key, client, model_choice)
            if valid:
                st.sidebar.success("API key validated (basic ping succeeded).")
            else:
                st.sidebar.warning("API key validation failed or was inconclusive.")
        else:
            st.sidebar.warning("No API key provided yet.")
        metrics = st.session_state.get("metrics", {})
        if "last_latency" in metrics:
            st.sidebar.write(f"Last API latency: {metrics['last_latency']:.2f}s")
        if "last_error" in metrics:
            st.sidebar.write(f"Last error (raw): {metrics['last_error']}")
        st.sidebar.write(f"Current model: {model_choice}")
        st.sidebar.write(f"Temperature: {temperature}")
        st.sidebar.write(f"Max tokens: {max_tokens}")

    # Risk detection banner placeholder
    risk_flag = False

    # Show existing conversation
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue  # hide internal prompt
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Controls row
    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("Clear conversation", use_container_width=True):
            if st.confirm("Are you sure you want to reset the conversation?"):
                initialize_session(system_prompt_text)
                st.experimental_rerun()
        if st.button("Generate summary"):
            summary = generate_session_summary(
                st.session_state.messages,
                client,
                model_choice,
                temperature,
                max_tokens,
                api_key,
            )
            st.session_state.generated_summaries.append(summary)
            st.success("Session summary (you can copy or save below):")
            st.markdown(f"> {summary}")

        # Download transcript
        if st.session_state.messages:
            transcript_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages if m["role"] != "system"
            )
            st.download_button(
                "Download transcript",
                transcript_text,
                file_name="recoverai_transcript.txt",
                help="Save the conversation for your records or reflection.",
            )

    # Input and response handling
    user_input = st.chat_input("Enter your message…")
    if user_input:
        # Risk detection on raw user input
        if is_high_risk(user_input):
            risk_flag = True
            st.warning(
                "It sounds like you might be going through something very heavy. "
                "If you're in immediate danger or thinking about harming yourself, please contact a crisis hotline. "
                "For example, in the U.S. you can dial 988 for the Suicide & Crisis Lifeline. "
                "I can stay here with you while we figure out next safe steps."
            )
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Early API key check
        if not api_key:
            assistant_reply = "Please provide your OpenAI API key in the sidebar to continue."
        else:
            # Prune history before call
            st.session_state.messages = prune_history(st.session_state.messages)

            # Add an optional preamble if risk flagged
            if risk_flag:
                st.session_state.messages.append(
                    {
                        "role": "system",
                        "content": (
                            "User input triggered a high-risk flag. Prioritize immediate emotional stabilization, "
                            "offer grounding, and strongly encourage contacting real-world crisis resources. Do not minimize distress."
                        ),
                    }
                )

            with st.chat_message("assistant"):
                with st.spinner("RecoverAI is thinking..."):
                    assistant_reply = call_openai_api(
                        st.session_state.messages,
                        api_key=api_key,
                        client=client,
                        model=model_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

        # Clean up any injected temporary system message if it was added for risk
        st.session_state.messages = [
            m for m in st.session_state.messages if not (m["role"] == "system" and "high-risk flag" in m["content"])
        ]

        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        # Automatic summary trigger
        non_system_count = len([m for m in st.session_state.messages if m["role"] != "system"])
        if (
            non_system_count >= SUMMARY_TRIGGER_INTERVAL
            and non_system_count - st.session_state.get("last_summary_len", 0) >= SUMMARY_TRIGGER_INTERVAL
        ):
            summary = generate_session_summary(
                st.session_state.messages,
                client,
                model_choice,
                temperature,
                max_tokens,
                api_key,
            )
            st.session_state.generated_summaries.append(summary)
            st.session_state["last_summary_len"] = non_system_count
            st.info("Automatic session summary:")  # ephemeral
            st.markdown(f"> {summary}")

    # Show past summaries
    if st.session_state.get("generated_summaries"):
        with st.expander("Past summaries"):
            for idx, s in enumerate(st.session_state.generated_summaries[-5:][::-1], 1):
                st.markdown(f"**Summary {idx}:** {s}")


if __name__ == "__main__":
    main()

