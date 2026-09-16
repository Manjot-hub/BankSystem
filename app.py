"""
Streamlit UI for Banking Chatbot

Connects directly to the FastAPI single-agent endpoint (/agent/chat).
Configuration and secrets (API Auth Token, Backend URL) are securely 
loaded from environment variables / st.secrets without exposing them on the UI.
"""
import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Secure Configuration Helper
def get_config(key: str, default: str) -> str:
    """Retrieve secret from st.secrets first, then os.getenv, then fallback to default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# Retrieve Backend URL & Auth Token securely
BACKEND_URL = get_config("BACKEND_URL", "http://localhost:8000")
AUTH_TOKEN = get_config("AUTH_TOKEN", "dev-secret-token")

# Page Configuration
st.set_page_config(
    page_title="AI Banking Support Assistant",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 AI Banking Support Assistant")
st.markdown("Ask about bank policies, check account balances, or view recent transactions.")

# Sidebar Setup (Exposes only non-sensitive user parameters)
st.sidebar.header("Session Settings")

# Track active user in session state to handle user switching securely
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = "usr_000001"

selected_user_id = st.sidebar.text_input("User ID", st.session_state.current_user_id)

# Reset session_id and history automatically if the user switches User ID
if selected_user_id != st.session_state.current_user_id:
    st.session_state.current_user_id = selected_user_id
    st.session_state.messages = []
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Session Info in Sidebar
st.sidebar.divider()
st.sidebar.caption(f"**Session ID:** `{st.session_state.session_id}`")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Handling
user_query = st.chat_input("How can I help you today?")

if user_query:
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare secure API headers
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "message": user_query,
        "user_id": st.session_state.current_user_id,
        "session_id": st.session_state.session_id,
    }

    with st.chat_message("assistant"):
        with st.spinner("Agent processing..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agent/chat",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "No response content received.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Render Tool Calls debug info in an expandable block
                    if data.get("tool_calls"):
                        with st.expander("🛠️ Tool Execution Details"):
                            st.json(data["tool_calls"])
                elif response.status_code == 401:
                    st.error("Authentication Failed: Invalid server configuration or unauthorized token.")
                elif response.status_code == 429:
                    st.warning("Rate limit exceeded. Please wait a moment.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend server: {str(e)}")