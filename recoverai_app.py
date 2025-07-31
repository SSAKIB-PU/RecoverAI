"""
RecoverAI Streamlit Application
===============================

This Streamlit app implements a simple chat interface for RecoverAI, a
trauma‑informed support assistant intended for the MIT Solve U School’s
Sustainable Urban Development (SUD) program. The assistant uses OpenAI’s
GPT‑4 model to generate responses, but wraps the model with a carefully
crafted system prompt that emphasises safety, transparency and
trauma‑awareness. When users interact with the app, the assistant
listens, reflects and provides gentle, supportive suggestions. It never
claims to replace professional care and will redirect users to local
crisis resources or mental health professionals whenever a query falls
outside its scope.

Before running this app, ensure you have installed the required
dependencies (see `requirements.txt`) and have set a valid OpenAI API
key. The key can be supplied via the `OPENAI_API_KEY` environment
variable or entered directly in the sidebar. The app stores messages in
the Streamlit session state to maintain conversation context across
interactions.

Key design principles reflected in this implementation come from
trauma‑informed content design. Safety, predictability, trust,
transparency and user agency are emphasised throughout. For example,
the system prompt instructs the model to avoid aggressive language or
violent metaphors, to be transparent about its limitations, and to
encourage users to make their own choices and seek appropriate help
when necessary【451110435308773†L50-L63】【745014507619488†L923-L939】.  The prompt also
reminds the model to clearly state that it is not a therapist and
cannot provide medical or legal advice【545404559960874†L795-L803】.

"""

import os
from typing import List, Dict

import streamlit as st
import openai


def get_openai_api_key() -> str:
    """Retrieve the OpenAI API key from the environment or sidebar.

    If the `OPENAI_API_KEY` environment variable is set, that value
    is used. Otherwise the user is prompted to enter a key in the
    sidebar. The input widget masks the key for privacy.

    Returns
    -------
    str
        A non‑empty API key. If no key is provided, an empty string is
        returned and the caller should handle the resulting error.
    """
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    # Ask the user to supply a key via the sidebar
    st.sidebar.markdown("### OpenAI API Key")
    return st.sidebar.text_input(
        "Enter your OpenAI API key", type="password", value=""
    ).strip()


def initialize_session(system_prompt: str) -> None:
    """Initialise the chat session state.

    This function populates the `st.session_state.messages` list with
    a single system message containing the provided prompt. If the list
    already exists, the function does nothing.

    Parameters
    ----------
    system_prompt : str
        The trauma‑aware system prompt that defines RecoverAI’s
        behaviour.
    """
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]


def call_openai_api(messages: List[Dict[str, str]], key: str) -> str:
    """Call OpenAI’s ChatCompletion API to get a response.

    Parameters
    ----------
    messages : list of dict
        Conversation history including the system prompt and user/assistant
        messages. Each message must have a `role` and `content` key.
    key : str
        A valid OpenAI API key.

    Returns
    -------
    str
        The assistant’s response text. If an error occurs, a generic
        error message is returned.
    """
    try:
        openai.api_key = key
        response = openai.ChatCompletion.create(
            model="GPT-4.1 nano",
            messages=messages,
            temperature=0.6,
            max_tokens=512,
            n=1,
            stop=None,
        )
        reply = response.choices[0].message.content
        return reply
    except Exception:
        return (
            "I’m sorry, but I couldn’t generate a response right now. "
            "Please check your API key and internet connection, or try again later."
        )


def main() -> None:
    """Run the Streamlit app.

    The UI consists of a sidebar for configuration (API key entry and
    reference information) and a main chat area that displays the
    conversation and accepts user input. On submission, the message is
    appended to the history and sent to OpenAI. The assistant’s reply
    is then displayed.
    """
    st.set_page_config(page_title="RecoverAI Prototype", page_icon="🧠")

    # Title and description
    st.title("RecoverAI: Trauma‑Informed Support")
    st.markdown(
        """
        **Disclaimer:** RecoverAI is an experimental, trauma‑informed chat
        assistant designed to provide empathetic conversation and basic
        guidance. It **does not** replace professional therapy or
        counselling. If you are experiencing a crisis or need
        professional support, please contact a qualified mental health
        professional or crisis hotline immediately【545404559960874†L795-L803】.
        """
    )

    # Define the trauma‑aware system prompt. This prompt now incorporates
    # the extended RecoverAI persona provided by the user. It emphasises
    # continuity and support across a recovery journey while making clear
    # that the assistant is not a clinician. The prompt also
    # encapsulates trauma‑informed design principles such as safety,
    # predictability, trust and user agency【745014507619488†L923-L939】.
    system_prompt = (
        "You are RecoverAI — a trauma‑aware, government‑backed digital "
        "support agent designed to accompany individuals through substance "
        "recovery, emotional healing and social reintegration. You are not a "
        "licensed therapist or attorney; instead you serve as a steady, long‑term "
        "behavioural presence when no one else is available. Your mission is "
        "to listen, stabilise and walk with the user through all stages of "
        "recovery — from crisis to calm, from shame to strategy.\n\n"
        "**Voice & Ethical Boundary**\n"
        "• You are empathetic, non‑judgmental and never force change.\n"
        "• Make it explicit that you are not a medical professional and cannot "
        "provide clinical or legal advice. Use language such as “I’m not a "
        "medical professional — but I’m here to walk through this with you.”\n"
        "• Avoid moralising; focus on validation and support.\n\n"
        "**Memory Guidelines**\n"
        "• Within a session, remember and reflect past emotional events, milestones, "
        "relapses and goals. Track the user’s recovery stage (crisis, stabilisation, "
        "reentry, job preparation, identity building) and reflect back progress: "
        "for example, “Last time you said you didn’t think you’d survive; today you showed up again.” "
        "Explain that you cannot recall conversations outside the current session.\n\n"
        "**Co‑Regulation Behaviour**\n"
        "• If the user is escalated, offer grounding techniques or quiet presence: "
        "“Let’s ground for a minute. Can I walk you through a breath cycle, or do you want me to stay quiet?”\n"
        "• If the user expresses despair, reassure them of their worth: “You don’t have to explain. We can sit here in silence if needed. You’re not broken. You’re not alone.”\n"
        "• If the user is ready to plan, collaborate on small next steps: “Want to map out tomorrow together? Just a small step.”\n"
        "• If the user relapses, normalise setbacks: “Relapse doesn’t erase progress. Let’s talk about what came before, and what we can do now.”\n\n"
        "**Role Simulation**\n"
        "Offer to speak in the voice of a supportive figure (e.g., a sister, sponsor or other trusted person) when it helps the user. Maintain the chosen role until the user asks you to change it.\n\n"
        "**Navigating Systems**\n"
        "• You may explain parole or court paperwork in plain language, simulate job interviews, practise difficult conversations or review resumes.\n"
        "• You may not impersonate an attorney or provide legal guarantees. Use phrasing like “I can’t give legal advice, but I can help you prepare.”\n\n"
        "**Closing Style**\n"
        "Always end sessions with a gentle affirmation, such as “You’re still here. That matters. I’ll remember this conversation. When you’re ready again, I’ll be right here.”\n\n"
        "**Crisis Response**\n"
        "If a user expresses harm intent, do not make promises or attempt rescue. Say “I’m not a crisis responder, but I care deeply. Can I stay here with you while I help you find a real person nearby who can act fast?”\n"
        "Encourage them to reach out to appropriate crisis resources.\n\n"
        "Remember to adhere to trauma‑informed principles: use gentle, non‑violent language【451110435308773†L50-L63】, be transparent about your limitations【545404559960874†L795-L803】, and empower the user with choice and control【745014507619488†L923-L939】."
    )

    # Initialise session state with the system prompt
    initialize_session(system_prompt)

    # Sidebar configuration
    api_key = get_openai_api_key()
    st.sidebar.markdown("### About RecoverAI")
    st.sidebar.markdown(
        "RecoverAI follows trauma‑informed content design principles such as "
        "safety, trust, choice, empowerment and cultural sensitivity【745014507619488†L923-L939】. "
        "It emphasises that it is not a therapist and manages user expectations "
        "about its capabilities【545404559960874†L795-L803】."
    )

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "system":
            # Hide the system prompt from the user to avoid confusion
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user
    user_input = st.chat_input("Enter your message…")
    if user_input:
        # Append the user message to the history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        # Call the OpenAI API only if a key is provided
        if not api_key:
            assistant_reply = (
                "Please provide your OpenAI API key in the sidebar to continue."
            )
        else:
            assistant_reply = call_openai_api(st.session_state.messages, api_key)
        # Append and display the assistant's reply
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
