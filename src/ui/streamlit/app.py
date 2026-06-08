import streamlit as st
from datetime import datetime

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
# 2. CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  /* ── Global ── */
  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #f0f2f5;
  }
  [data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1000px;
  }

  /* ── App Title ── */
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
    background: #fef3c7;
    color: #92400e;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    margin-top: 0.6rem;
    border: 1px solid #fcd34d;
  }

  /* ── File Meta Card ── */
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

  /* ── Chat Header ── */
  .chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 0.8rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 0.5rem;
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
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
  }

  /* ── Chat Window ── */
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

  /* ── Message Rows ── */
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

  /* ── Bubbles ── */
  .message-bubble-bot {
    background: #ffffff;
    color: #111827;
    border-radius: 18px 18px 18px 4px;
    padding: 0.65rem 0.95rem;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    position: relative;
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
  }
  .message-time {
    font-size: 0.68rem;
    color: #9ca3af;
    text-align: right;
    margin-top: 3px;
  }
  .message-time-user { text-align: right; }
  .message-time-bot  { text-align: left;  }

  /* ── Icons ── */
  .message-icon-bot, .message-icon-user {
    font-size: 1.25rem;
    flex-shrink: 0;
    padding-bottom: 2px;
  }

  /* ── Info box ── */
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

  /* ── Prototype Banner ── */
  .proto-banner {
    text-align: center;
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 1.5rem;
    padding: 0.5rem;
    border-top: 1px solid #e5e7eb;
  }

  /* Override Streamlit expander header */
  details summary {
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
  }

  /* Streamlit button tweaks */
  [data-testid="stButton"] button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
  }

  /* ── Scrollbar for chat ── */
  .chat-window::-webkit-scrollbar { width: 5px; }
  .chat-window::-webkit-scrollbar-track { background: transparent; }
  .chat-window::-webkit-scrollbar-thumb { background: #c4b5a5; border-radius: 99px; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "file_uploaded": False,
        "uploaded_file_info": None,
        "messages": [],
        "history_refreshed": False,
        "new_chat_started": False,
        "notification": None,  # ("type", "text")
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _welcome_message() -> dict:
    return {
        "role": "bot",
        "content": "Hello! 👋 I'm ready to summarize your credit card spend file. "
        "What would you like to know? You can ask about top categories, "
        "monthly trends, merchant breakdowns, and more.",
        "time": _now(),
    }


def initialize_welcome_message():
    if not st.session_state.messages:
        st.session_state.messages = [_welcome_message()]


def start_new_chat():
    # TODO: Create a backend conversation/session ID here when backend is connected.
    st.session_state.messages = [_welcome_message()]
    st.session_state.new_chat_started = True
    st.session_state.history_refreshed = False
    st.session_state.notification = (
        "success",
        "✅ Started a new chat for the uploaded file.",
    )


def generate_mock_history() -> list:
    """Returns 25 alternating mock messages simulating prior chat history."""
    pairs = [
        (
            "What are my top spending categories this month?",
            "Your top 3 categories are: 🍔 Food & Dining (₹8,240), 🛒 Groceries (₹5,670), and ✈️ Travel (₹4,120).",
        ),
        (
            "How much did I spend in total last month?",
            "Your total spend last month was ₹32,450 across 47 transactions.",
        ),
        (
            "Which merchant did I spend the most on?",
            "You spent the most at Amazon — ₹6,800 across 9 transactions.",
        ),
        (
            "Show me weekend vs weekday spending.",
            "Weekday spend: ₹21,300 (65.5%) | Weekend spend: ₹11,150 (34.5%).",
        ),
        (
            "Any unusual or large transactions?",
            "I noticed 2 transactions above ₹5,000: Flipkart (₹6,199) and MakeMyTrip (₹5,450).",
        ),
        (
            "What's my average daily spend?",
            "Your average daily spend is ₹1,048 over the past 31 days.",
        ),
        (
            "How does this month compare to last?",
            "This month you've spent 12% more than last month (₹32,450 vs ₹28,980).",
        ),
        (
            "Break down my food spending.",
            "Food breakdown: Swiggy ₹2,100 | Zomato ₹1,840 | Restaurants ₹4,300.",
        ),
        (
            "How many transactions were international?",
            "You had 3 international transactions totalling USD 142 (≈ ₹11,800).",
        ),
        (
            "When do I spend the most during the month?",
            "Spending tends to peak in the first week (salary week) and around the 20th.",
        ),
        (
            "Categorise my entertainment spend.",
            "Entertainment: Netflix ₹649 | Prime Video ₹299 | PVR Cinemas ₹1,200.",
        ),
        (
            "What's my highest single-day spend?",
            "Your highest single-day spend was ₹9,450 on the 5th (shopping + dining).",
        ),
    ]
    history = []
    for q, a in pairs[:12]:
        history.append({"role": "user", "content": q, "time": "earlier"})
        history.append({"role": "bot", "content": a, "time": "earlier"})
    # Pad to 25
    history.append(
        {
            "role": "bot",
            "content": "Is there anything else you'd like to explore in your spend data?",
            "time": "earlier",
        }
    )
    return history[:25]


def render_message(role: str, content: str, time_str: str = ""):
    """Render a single WhatsApp-style chat bubble via HTML."""
    time_str = time_str or _now()
    if role == "bot":
        html = f"""
        <div class="message-row-bot">
          <div class="message-icon-bot">🤖</div>
          <div>
            <div class="message-bubble-bot">{content}</div>
            <div class="message-time message-time-bot">{time_str}</div>
          </div>
        </div>"""
    else:
        html = f"""
        <div class="message-row-user">
          <div class="message-icon-user">👤</div>
          <div>
            <div class="message-bubble-user">{content}</div>
            <div class="message-time message-time-user">{time_str} ✓✓</div>
          </div>
        </div>"""
    return html


def render_chat_window():
    """Builds the full chat window HTML from session messages."""
    parts = ['<div class="chat-window">']
    for msg in st.session_state.messages:
        parts.append(render_message(msg["role"], msg["content"], msg.get("time", "")))
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="title-wrapper">
  <div class="app-title">💳 credit-card-spend-<span>summarizer bot</span></div>
  <div class="app-subtitle">Upload your credit card spend file and chat with the summarizer bot.</div>
  <div class="badge">UI PROTOTYPE — backend not connected</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. NOTIFICATION DISPLAY (transient, shown once)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.notification:
    ntype, ntext = st.session_state.notification
    if ntype == "success":
        st.success(ntext)
    elif ntype == "info":
        st.info(ntext)
    st.session_state.notification = None  # clear after display


# ─────────────────────────────────────────────────────────────────────────────
# 7. ACCORDION 1 — FILE UPLOAD / INGESTION
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
        # Persist file info in session state
        file_size_kb = round(uploaded.size / 1024, 1)
        st.session_state.uploaded_file_info = {
            "name": uploaded.name,
            "type": uploaded.type or "unknown",
            "size": f"{file_size_kb} KB",
        }
        if not st.session_state.file_uploaded:
            st.session_state.file_uploaded = True
            initialize_welcome_message()

        info = st.session_state.uploaded_file_info
        ext_icon = {"csv": "📊", "pdf": "📄", "xlsx": "📊", "xls": "📊"}.get(
            info["name"].split(".")[-1].lower(), "📁"
        )

        # ── File meta card ──
        st.markdown(
            f"""
        <div class="file-meta-card">
          <div class="file-meta-icon">{ext_icon}</div>
          <div>
            <div class="file-meta-name">{info['name']}</div>
            <div class="file-meta-detail">Type: {info['type']} &nbsp;|&nbsp; Size: {info['size']}</div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── Ingestion status ──
        st.markdown(
            """
        <div class="ingestion-status">
          ✅ &nbsp; <strong>Ingestion status:</strong>&nbsp; Ready for conversation
        </div>
        """,
            unsafe_allow_html=True,
        )

        # TODO: Trigger backend file ingestion pipeline here.
        # TODO: Parse CSV/XLSX/PDF and embed data into vector store.

    else:
        if st.session_state.file_uploaded and st.session_state.uploaded_file_info:
            # File was uploaded in a previous run; show persisted metadata
            info = st.session_state.uploaded_file_info
            ext_icon = {"csv": "📊", "pdf": "📄", "xlsx": "📊", "xls": "📊"}.get(
                info["name"].split(".")[-1].lower(), "📁"
            )
            st.markdown(
                f"""
            <div class="file-meta-card">
              <div class="file-meta-icon">{ext_icon}</div>
              <div>
                <div class="file-meta-name">{info['name']}</div>
                <div class="file-meta-detail">Type: {info['type']} &nbsp;|&nbsp; Size: {info['size']}</div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <div class="ingestion-status">
              ✅ &nbsp; <strong>Ingestion status:</strong>&nbsp; Ready for conversation
            </div>
            """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8. CONDITIONAL PROMPT — no file yet
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
# 9. ACCORDION 2 — CONVERSATION WINDOW (conditional)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.file_uploaded:
    with st.expander("💬  2. Conversation window", expanded=True):

        # ── Conversation header ──
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
                # TODO: Replace mock history with backend chat history fetch.
                st.session_state.messages = generate_mock_history()
                st.session_state.history_refreshed = True
                st.session_state.notification = (
                    "info",
                    "🔄 Last 25 messages refreshed from history.",
                )
                st.rerun()

        st.markdown(
            "<hr style='margin:0.4rem 0 0.7rem 0; border-color:#e5e7eb;'>",
            unsafe_allow_html=True,
        )

        # ── Chat window ──
        render_chat_window()

        # ── Chat input ──
        user_input = st.chat_input("Ask something about your spend…")

        if user_input:
            # Append user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                    "time": _now(),
                }
            )

            # TODO: Replace this mock response with actual summarization backend call.
            mock_response = (
                "⚙️ This is a UI-only placeholder response. "
                "Backend summarization logic can be connected here. "
                f"You asked: *{user_input}*"
            )
            st.session_state.messages.append(
                {
                    "role": "bot",
                    "content": mock_response,
                    "time": _now(),
                }
            )
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="proto-banner">
  💳 UI prototype only — backend summarization is not connected. &nbsp;|&nbsp;
  Built with Streamlit
</div>
""",
    unsafe_allow_html=True,
)
