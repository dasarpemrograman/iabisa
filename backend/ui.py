import json
import os

import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# CONFIGURATION & STYLING (Heuristic 4 & 8)
# ==============================================================================
st.set_page_config(
    page_title="Agentic BI Assistant",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalist design
st.markdown(
    """
<style>
    .stChatMessage { padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; background-color: #f9f9f9; }
    .stButton button { border-radius: 20px; }
    div[data-testid="stStatusWidget"] { border: 1px solid #ddd; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/generate-chart-stream")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# SIDEBAR: HELP & CONTROL (Heuristic 3 & 10)
# ==============================================================================
with st.sidebar:
    st.title("⚡ Agentic BI")

    st.markdown("### 📘 How to Use")
    st.info(
        "1. **Ask** a question about your data.\n"
        "2. **Watch** the agent plan & execute SQL.\n"
        "3. **Interact** with charts and maps."
    )

    st.markdown("### 🛠️ Capabilities")
    st.caption("The agent can handle:")
    st.markdown("- 📝 **Summaries**: General data questions")
    st.markdown("- 📊 **Analytics**: Trends & comparisons")
    st.markdown("- 🗺️ **Maps**: Provincial distribution")
    st.markdown("- 🔮 **Predictions**: Forecast future data")

    st.divider()

    # Heuristic 3: User Control (Reset)
    if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def handle_query(prompt_text):
    """Processes a query updates history, and calls the API."""
    # 1. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_text)

    # 2. Generate Response
    with st.chat_message("assistant", avatar="🤖"):
        # Heuristic 1: Visibility of System Status
        status_box = st.status(
            "🧠 **Agent Active:** Analyzing request...", expanded=True
        )

        final_content = None
        final_view = "text"
        final_data = None
        error_occurred = False

        try:
            # Prepare context for API
            api_history = [
                {"role": m["role"], "content": str(m["content"])}
                for m in st.session_state.messages[:-1]
            ]
            payload = {"message": prompt_text, "history": api_history}

            response = requests.post(API_URL, json=payload, stream=True)

            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if not decoded.startswith("data: "):
                        continue

                    event = json.loads(decoded.replace("data: ", ""))
                    evt_type = event.get("type")
                    content = event.get("content")
                    label = event.get("label")
                    state = event.get("state")

                    # --- UPDATE STATUS TRACKER ---
                    if evt_type == "status":
                        status_box.update(label=f"**{label}**", state=state)
                        if state == "error":
                            error_occurred = True
                            st.error(f"System Error: {content}")

                    elif evt_type == "log":
                        status_box.write(f"› {content}")

                    # Heuristic 8: Minimalist Design (Hide technical artifacts)
                    elif evt_type == "artifact":
                        with status_box:
                            with st.expander(
                                f"📄 {label or 'Details'}", expanded=False
                            ):
                                if event.get("view") == "sql":
                                    st.code(content, language="sql")
                                else:
                                    st.json(content)

                    elif evt_type == "final":
                        final_content = content
                        final_view = event.get("view")
                        if final_view in ["chart", "map"]:
                            final_data = content.get("data")

                        status_box.update(
                            label="✅ **Analysis Complete**",
                            state="complete",
                            expanded=False,
                        )

        except Exception as e:
            status_box.update(label="❌ **Connection Failed**", state="error")
            st.error(
                f"Could not reach the backend. Please check if Docker is running.\n\nDetails: {e}"
            )
            error_occurred = True

        # 3. Display Final Result (Heuristic 2 & 9)
        if not error_occurred and final_content:
            if final_view == "text":
                st.markdown(final_content)

            elif final_view == "chart":
                st.markdown(
                    f"### 📊 {final_content.get('component_name', 'Visualization')}"
                )
                st.info(
                    "ℹ️ *This is a React component definition. In a full implementation, it would render interactively.*"
                )
                st.code(final_content.get("react_code"), language="javascript")

                with st.expander("🔍 View Underlying Data"):
                    st.dataframe(pd.DataFrame(final_data), use_container_width=True)

            elif final_view == "map":
                st.markdown("### 🗺️ Geographic Analysis")
                st.success(f"**{len(final_data)} Regions Identified**")
                st.dataframe(pd.DataFrame(final_data), use_container_width=True)

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                    "view_type": final_view,
                    "data": final_data,
                }
            )


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

st.title("Agentic Business Intelligence")
st.markdown(
    "Ask questions in plain English. The agent will query the database and visualize results."
)
st.markdown("Ask questions in plain English. The agent will query the database and visualize results.")

# Heuristic 5 & 7: Error Prevention & Efficiency (Quick Actions)
# FIX: 'help' is now correctly passed as a keyword argument (help="...")
col1, col2, col3, col4 = st.columns(4)

if col1.button("📈 Trends", use_container_width=True, help="Show historical trends"):
    handle_query("Show me the trend of cases over the last 5 years")

if col2.button("🗺️ Maps", use_container_width=True, help="Show geographic data"):
    handle_query("Show a map of Faskes distribution by province")

if col3.button("🔮 Predict", use_container_width=True, help="Forecast future data"):
    handle_query("Predict the growth of participants for next year")

if col4.button("❓ Help", use_container_width=True, help="List available data"):
    handle_query("What tables and columns are available in the database?")

st.divider()

# Heuristic 6: Recognition rather than recall (Display History)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        view = msg.get("view_type", "text")
        content = msg["content"]

        if view == "text":
            st.markdown(content)
        elif view == "chart":
            st.markdown(f"### 📊 {content.get('component_name')}")
            with st.expander("Show Code"):
                st.code(content.get("react_code"), language="javascript")
        elif view == "map":
            st.markdown("### 🗺️ Geographic Analysis")
            if msg.get("data"):
                st.dataframe(pd.DataFrame(msg["data"]).head(), use_container_width=True)

# Chat Input
if prompt := st.chat_input("Type your question here..."):
    handle_query(prompt)
