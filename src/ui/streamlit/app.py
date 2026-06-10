import os, re, json, html
from datetime import datetime
from textwrap import dedent

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="credit-card-spend-summarizer bot",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. API CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")

UPLOAD_ENDPOINT = f"{API_BASE_URL}/multimodelrag/embed/multimodel/document"
QUERY_STREAM_ENDPOINT = f"{API_BASE_URL}/multimodelrag/query/stream"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #f0f2f5;
  }

  [data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1000px;
  }

  .app-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    margin: 0;
    padding: 0;
  }

  .app-title span {
    color: #2563eb;
  }

  .app-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: 0.3rem;
    font-weight: 400;
  }

  .title-wrapper {
    text-align: center;
    padding: 1.2rem 0 1.8rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.6rem;
  }

  .badge {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    margin-top: 0.6rem;
    border: 1px solid #6ee7b7;
  }

  .file-meta-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }

  .file-meta-icon { font-size: 1.8rem; }
  .file-meta-name { font-weight: 600; color: #111827; font-size: 0.95rem; }
  .file-meta-detail { color: #6b7280; font-size: 0.82rem; margin-top: 2px; }

  .ingestion-status {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    color: #065f46;
    font-weight: 500;
    font-size: 0.88rem;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .ingestion-error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    color: #991b1b;
    font-weight: 500;
    font-size: 0.88rem;
    margin-top: 0.8rem;
  }

  .chat-header-title {
    font-weight: 600;
    color: #111827;
    font-size: 0.97rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .chat-header-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
  }

  .chat-window {
    background: #e5ddd5;
    border-radius: 12px;
    padding: 1rem 0.8rem;
    min-height: 320px;
    max-height: 500px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    background-image: url("data:image/svg+xml,%3Csvg width='52' height='26' viewBox='0 0 52 26' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c8b9a7' fill-opacity='0.18'%3E%3Cpath d='M10 10c0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6h2c0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4v2c-3.314 0-6-2.686-6-6 0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6zm25.464-1.95l8.486 8.486-1.414 1.414-8.486-8.486 1.414-1.414z' /%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }

  .message-row-bot {
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    gap: 0.45rem;
    max-width: 75%;
    align-self: flex-start;
  }

  .message-row-user {
    display: flex;
    flex-direction: row-reverse;
    align-items: flex-end;
    gap: 0.45rem;
    max-width: 75%;
    align-self: flex-end;
  }

  .message-bubble-bot {
    background: #ffffff;
    color: #111827;
    border-radius: 18px 18px 18px 4px;
    padding: 0.65rem 0.95rem;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    position: relative;
    word-break: break-word;
  }

  .message-bubble-user {
    background: #dcf8c6;
    color: #111827;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 0.95rem;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    position: relative;
    word-break: break-word;
  }

  .message-time { font-size: 0.68rem; color: #9ca3af; margin-top: 3px; }
  .message-time-user { text-align: right; }
  .message-time-bot { text-align: left; }
  .message-icon-bot, .message-icon-user { font-size: 1.25rem; flex-shrink: 0; padding-bottom: 2px; }

  .unlock-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #1d4ed8;
    font-size: 0.88rem;
    font-weight: 500;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .proto-banner {
    text-align: center;
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 1.5rem;
    padding: 0.5rem;
    border-top: 1px solid #e5e7eb;
  }

  details summary {
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
  }

  [data-testid="stButton"] button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
  }

  .chat-window::-webkit-scrollbar { width: 5px; }
  .chat-window::-webkit-scrollbar-track { background: transparent; }
  .chat-window::-webkit-scrollbar-thumb { background: #c4b5a5; border-radius: 99px; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "file_uploaded": False,
        "file_ingested": False,
        "uploaded_file_info": None,
        "uploaded_file_key": None,
        "messages": [],
        "history_refreshed": False,
        "new_chat_started": False,
        "notification": None,
        "last_upload_response": None,
        "last_query_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _welcome_message() -> dict:
    return {
        "role": "bot",
        "content": (
            "Hello! 👋 I'm ready to summarize your credit card spend file. "
            "What would you like to know? You can ask about top categories, "
            "monthly trends, merchant breakdowns, and more."
        ),
        "time": _now(),
    }


def initialize_welcome_message():
    if not st.session_state.messages:
        st.session_state.messages = [_welcome_message()]


def start_new_chat():
    st.session_state.messages = [_welcome_message()]
    st.session_state.new_chat_started = True
    st.session_state.history_refreshed = False
    st.session_state.notification = (
        "success",
        "✅ Started a new chat for the uploaded file.",
    )


def get_file_icon(file_name: str) -> str:
    ext = file_name.split(".")[-1].lower()
    return {"csv": "📊", "xlsx": "📊", "xls": "📊", "pdf": "📄"}.get(ext, "📁")


def render_file_meta_card(info: dict):
    if not info:
        return

    safe_name = html.escape(info.get("name", "unknown"))
    safe_type = html.escape(info.get("type", "unknown"))
    safe_size = html.escape(info.get("size", "unknown"))
    ext_icon = get_file_icon(info.get("name", ""))

    st.markdown(
        f"""
        <div class="file-meta-card">
          <div class="file-meta-icon">{ext_icon}</div>
          <div>
            <div class="file-meta-name">{safe_name}</div>
            <div class="file-meta-detail">Type: {safe_type} &nbsp;|&nbsp; Size: {safe_size}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ingestion_success():
    st.markdown(
        """
        <div class="ingestion-status">
          ✅ &nbsp; <strong>Ingestion status:</strong>&nbsp; Ready for conversation
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ingestion_error(message: str):
    safe_message = html.escape(message)
    st.markdown(
        f"""
        <div class="ingestion-error">
          ❌ <strong>Ingestion failed:</strong> {safe_message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_mock_history() -> list:
    """Placeholder history function. Replace with backend history endpoint when available."""
    pairs = [
        (
            "What are my top spending categories this month?",
            "Your top 3 categories are: 🍔 Food & Dining, 🛒 Groceries, and ✈️ Travel.",
        ),
        (
            "How much did I spend in total last month?",
            "Your total spend last month was calculated from the uploaded statement.",
        ),
        (
            "Which merchant did I spend the most on?",
            "I can identify the highest-spend merchant from your uploaded transactions.",
        ),
    ]

    history = []
    for question, answer in pairs:
        history.append({"role": "user", "content": question, "time": "earlier"})
        history.append({"role": "bot", "content": answer, "time": "earlier"})

    return history[-25:]


def render_message(role: str, content: str, time_str: str = ""):
    """Render a single WhatsApp-style chat bubble via safe HTML."""
    time_str = html.escape(time_str or _now())
    safe_content = html.escape(str(content)).replace("\n", "<br>")

    if role == "bot":
        return dedent(f"""
        <div class="message-row-bot">
          <div class="message-icon-bot">🤖</div>
          <div>
            <div class="message-bubble-bot">{safe_content}</div>
            <div class="message-time message-time-bot">{time_str}</div>
          </div>
        </div>
        """).strip()

    return dedent(f"""
    <div class="message-row-user">
      <div class="message-icon-user">👤</div>
      <div>
        <div class="message-bubble-user">{safe_content}</div>
        <div class="message-time message-time-user">{time_str} ✓✓</div>
      </div>
    </div>
    """).strip()


def render_chat_window():
    """Builds the full chat window HTML from session messages."""
    parts = ['<div class="chat-window">']

    for msg in st.session_state.messages:
        parts.append(
            render_message(
                role=msg.get("role", "bot"),
                content=msg.get("content", ""),
                time_str=msg.get("time", ""),
            )
        )

    parts.append("</div>")

    st.markdown("\n".join(parts), unsafe_allow_html=True)


def upload_file_to_backend(uploaded_file):
    """
    Uploads the selected file to FastAPI ingestion endpoint.

    Backend route expected:
    POST /multimodelrag/embed/multimodel/document
    form-data key: file
    """
    try:
        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }

        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            timeout=240,
        )
        response.raise_for_status()

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {"message": response.text}

        return True, response_payload

    except requests.exceptions.Timeout:
        return False, {
            "error": "Upload request timed out. The backend may still be processing, or the file may be too large."
        }
    except requests.exceptions.ConnectionError:
        return False, {
            "error": f"Could not connect to backend at {API_BASE_URL}. Please verify FastAPI is running."
        }
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if e.response is not None else str(e)
        return False, {"error": f"Backend returned HTTP error: {error_text}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def extract_actual_answer(raw_response: str) -> str:
    """
    Extracts only the actual bot answer from backend response.

    Handles:
    - JSON string responses
    - SSE data lines
    - HTML-escaped JSON
    - fallback plain text
    """

    if raw_response is None:
        return ""

    text = str(raw_response).strip()

    # Decode HTML entities like &quot;, &amp;quot;, etc.
    text = html.unescape(text)

    # If backend accidentally sends SSE prefix
    if text.startswith("data:"):
        text = text.replace("data:", "", 1).strip()

    # Try parsing JSON directly
    try:
        payload = json.loads(text)

        if isinstance(payload, dict):
            for key in ["answer", "response", "content", "message", "result"]:
                if key in payload and payload[key]:
                    return str(payload[key]).strip()

        if isinstance(payload, str):
            return payload.strip()

    except Exception:
        pass

    # If JSON is embedded somewhere inside text, try extracting it
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))

            if isinstance(payload, dict):
                for key in ["answer", "response", "content", "message", "result"]:
                    if key in payload and payload[key]:
                        return str(payload[key]).strip()

        except Exception:
            pass

    # If HTML tags are present, remove them as fallback
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    print(f" bot generated answer ---------------- {text.strip()}")

    return text.strip()


def build_chat_history(max_messages: int = 10) -> list:
    """
    Builds recent conversation history for contextual follow-up questions.

    Streamlit UI roles:
    - "user" becomes "user"
    - "bot" becomes "assistant"

    The current user question should NOT be included here.
    So call this function BEFORE appending the latest user input.
    """

    history = []

    recent_messages = st.session_state.messages[-max_messages:]

    for msg in recent_messages:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()

        if not content:
            continue

        # Skip initial welcome message because it is not useful as conversation context
        if role == "bot" and content.startswith("Hello! 👋 I'm ready"):
            continue

        if role == "user":
            history.append({
                "role": "user",
                "content": content
            })

        elif role == "bot":
            history.append({
                "role": "assistant",
                "content": content
            })

    return history



def query_backend_stream(query: str, chat_history: list = None):
    """
    Calls FastAPI streaming query endpoint and returns the accumulated answer.

    Backend route expected:
    POST /multimodelrag/query/stream

    JSON body sent:
    {
        "query": "current user question",
        "chat_history": [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"}
        ]
    }
    """
    try:
        payload = {
            "query": query,
            "chat_history": chat_history or []
        }

        with requests.post(
            QUERY_STREAM_ENDPOINT,
            json=payload,
            stream=True,
            timeout=180,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            full_answer_parts = []

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                chunk = line.strip()

                if chunk.startswith("data:"):
                    chunk = chunk.replace("data:", "", 1).strip()

                if chunk in {"[DONE]", "DONE"}:
                    break

                full_answer_parts.append(chunk)

            full_answer = "".join(full_answer_parts).strip()
            print(f"full answer ---------------- {full_answer}")
            if not full_answer:
                full_answer = "I could not generate a response from the backend."

            actual_answer = extract_actual_answer(full_answer)

            return True, actual_answer

    except requests.exceptions.Timeout:
        return False, "Query request timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return (
            False,
            f"Could not connect to backend at {API_BASE_URL}. Please verify FastAPI is running.",
        )

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if e.response is not None else str(e)
        return False, f"Backend returned HTTP error: {error_text}"

    except requests.exceptions.RequestException as e:
        return False, f"Backend query failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIONAL SIDEBAR DEBUG INFO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Backend")
    st.caption(f"Base URL: `{API_BASE_URL}`")
    st.caption(f"Upload: `{UPLOAD_ENDPOINT}`")
    st.caption(f"Query: `{QUERY_STREAM_ENDPOINT}`")

    if st.session_state.last_upload_response is not None:
        with st.expander("Last upload response"):
            st.json(st.session_state.last_upload_response)

    if st.session_state.last_query_error:
        with st.expander("Last query error"):
            st.error(st.session_state.last_query_error)


# ─────────────────────────────────────────────────────────────────────────────
# 7. APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="title-wrapper">
  <div class="app-title">💳 credit-card-spend-<span>summarizer bot</span></div>
  <div class="app-subtitle">Upload your credit card spend file and chat with the summarizer bot.</div>
  <div class="badge">FASTAPI BACKEND CONNECTED</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8. NOTIFICATION DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.notification:
    ntype, ntext = st.session_state.notification

    if ntype == "success":
        st.success(ntext)
    elif ntype == "info":
        st.info(ntext)
    elif ntype == "error":
        st.error(ntext)
    else:
        st.write(ntext)

    st.session_state.notification = None


# ─────────────────────────────────────────────────────────────────────────────
# 9. FILE UPLOAD / INGESTION
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📂  1. Upload file for ingestion", expanded=True):
    st.markdown(
        "<p style='color:#6b7280;font-size:0.88rem;margin-bottom:0.6rem;'>"
        "Upload your credit card statement or spend export file to begin.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "pdf"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_size_kb = round(uploaded.size / 1024, 1)
        current_file_key = f"{uploaded.name}_{uploaded.size}_{uploaded.type}"

        st.session_state.uploaded_file_info = {
            "name": uploaded.name,
            "type": uploaded.type or "unknown",
            "size": f"{file_size_kb} KB",
        }

        render_file_meta_card(st.session_state.uploaded_file_info)

        # Upload to backend only when this file is new or changed.
        if st.session_state.uploaded_file_key != current_file_key:
            with st.spinner("Uploading and ingesting file into backend..."):
                success, result = upload_file_to_backend(uploaded)

            st.session_state.last_upload_response = result

            if success:
                st.session_state.file_uploaded = True
                st.session_state.file_ingested = True
                st.session_state.uploaded_file_key = current_file_key
                initialize_welcome_message()
                st.session_state.notification = (
                    "success",
                    "✅ File uploaded and ingested successfully. You can start chatting now.",
                )
                st.rerun()
            else:
                st.session_state.file_uploaded = False
                st.session_state.file_ingested = False
                st.session_state.uploaded_file_key = None
                error_message = result.get("error", str(result))
                render_ingestion_error(error_message)

        elif st.session_state.file_ingested:
            render_ingestion_success()

    else:
        # If Streamlit reruns and the uploader does not hold the file,
        # show persisted metadata if already uploaded in this session.
        if st.session_state.file_uploaded and st.session_state.uploaded_file_info:
            render_file_meta_card(st.session_state.uploaded_file_info)
            if st.session_state.file_ingested:
                render_ingestion_success()


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONDITIONAL PROMPT — NO FILE YET
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.file_uploaded:
    st.markdown(
        """
        <div class="unlock-info">
          🔒 &nbsp; Please upload a file to unlock the conversation window.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. CONVERSATION WINDOW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.file_uploaded and st.session_state.file_ingested:
    with st.expander("💬  2. Conversation window", expanded=True):
        col_title, col_new, col_refresh = st.columns([0.58, 0.21, 0.21])

        with col_title:
            st.markdown(
                """
                <div class="chat-header-title">
                  <span class="chat-header-dot"></span>
                  Chat with spend summarizer bot
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_new:
            if st.button(
                "➕ New chat",
                use_container_width=True,
                help="Starts a fresh conversation for the currently uploaded file.",
            ):
                start_new_chat()
                st.rerun()

        with col_refresh:
            if st.button(
                "🔄 Refresh history",
                use_container_width=True,
                help="Loads the last 25 messages from chat history.",
            ):
                # Replace this with backend history fetch when available.
                st.session_state.messages = generate_mock_history()
                st.session_state.history_refreshed = True
                st.session_state.notification = (
                    "info",
                    "🔄 Last 25 mock messages refreshed.",
                )
                st.rerun()

        st.markdown(
            "<hr style='margin:0.4rem 0 0.7rem 0; border-color:#e5e7eb;'>",
            unsafe_allow_html=True,
        )

        render_chat_window()
        
        user_input = st.chat_input("Ask something about your spend…")

        if user_input:
            # Important:
            # Capture previous conversation BEFORE appending the current user question.
            previous_chat_history = build_chat_history(max_messages=10)

            # Now add the current user message to the UI session state.
            st.session_state.messages.append(
                {"role": "user", "content": user_input, "time": _now()}
            )

            with st.spinner("Bot is analyzing your spend data..."):
                success, answer = query_backend_stream(
                    query=user_input,
                    chat_history=previous_chat_history
                )

            if success:
                st.session_state.last_query_error = None
                bot_answer = answer
            else:
                st.session_state.last_query_error = answer
                bot_answer = f"❌ {answer}"

            st.session_state.messages.append(
                {"role": "bot", "content": bot_answer, "time": _now()}
            )

            st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# 12. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="proto-banner">
  💳 Credit card spend summarizer — Backend connected via FastAPI &nbsp;|&nbsp;
  Built with Streamlit
</div>
""",
    unsafe_allow_html=True,
)
