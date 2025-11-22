import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --- CONFIGURATION ---
st.set_page_config(page_title="Agentic BI + RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
with st.sidebar:
    st.header("🔌 Connection")
    api_url = st.text_input("API URL", "http://localhost:8000")

    st.markdown("---")
    st.header("📚 RAG System")

    # Check RAG health
    try:
        health_resp = requests.get(f"{api_url}/rag/health", timeout=2)
        if health_resp.ok:
            health_data = health_resp.json()
            if health_data["rag_enabled"]:
                st.success(f"✅ Active: {health_data['documents_indexed']} docs")
            else:
                st.warning("⚠️ Not Ready")
        else:
            st.error("❌ Unavailable")
    except:
        st.info("🔌 Checking...")

    # Document upload section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Attach files to enhance AI",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Upload documents for RAG-enhanced answers",
    )

    if uploaded_files and st.button("📥 Index Documents", type="primary"):
        with st.spinner("Uploading and indexing..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
            ]
            try:
                upload_resp = requests.post(
                    f"{api_url}/rag/upload", files=files_payload, timeout=60
                )
                if upload_resp.ok:
                    result = upload_resp.json()
                    st.success(
                        f"✅ {result['indexed_count']} docs, {result['chunk_count']} chunks"
                    )
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {upload_resp.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # RAG Stats
    st.markdown("### 📊 Statistics")
    try:
        stats_resp = requests.get(f"{api_url}/rag/stats", timeout=2)
        if stats_resp.ok:
            stats = stats_resp.json()
            st.metric("Documents", stats["total_documents"])
            st.caption(f"Chunks: {stats['chunk_size']} tokens")
        else:
            st.caption("Stats unavailable")
    except:
        pass

    # View documents
    with st.expander("📄 View Documents"):
        try:
            docs_resp = requests.get(f"{api_url}/rag/documents", timeout=3)
            if docs_resp.ok:
                docs = docs_resp.json()
                if docs:
                    for doc in docs[:10]:
                        st.caption(f"• {doc['filename']} ({doc['size'] // 1024}KB)")
                else:
                    st.info("No documents")
            else:
                st.error("Failed to load")
        except:
            st.caption("Error loading docs")

    # Clear all
    st.markdown("---")
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.get("confirm_clear"):
            try:
                clear_resp = requests.post(f"{api_url}/rag/clear", timeout=10)
                if clear_resp.ok:
                    st.success("✅ Cleared")
                    st.session_state.confirm_clear = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Click again to confirm")

    # Clear chat
    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.caption("• Upload docs for context-aware answers")
    st.caption("• Ask about predictions, SQL, or chat")
    st.caption("• RAG auto-enhances responses")

st.title("🤖 Agentic BI + RAG")
st.caption("AI-powered analytics with document-enhanced context")

# --- HELPER FUNCTIONS ---


def parse_stream_line(line):
    """Parses a Server-Sent Event (SSE) line."""
    if line:
        decoded_line = line.decode("utf-8").strip()
        if decoded_line.startswith("data: "):
            try:
                return json.loads(decoded_line[6:])
            except json.JSONDecodeError:
                return None
    return None


def render_message(role, content, view_type="text", steps=None):
    """Renders a message block with specific logic for charts/maps."""
    with st.chat_message(role):
        # Render intermediate steps if they exist (Accordion style)
        if steps:
            with st.status("⚙️ Agent Workflow", expanded=False):
                for step in steps:
                    st.write(f"**{step['label']}**")
                    if step["content"]:
                        if step.get("view") == "sql":
                            st.code(step["content"], language="sql")
                        elif step.get("view") == "error":
                            st.error(step["content"])
                        else:
                            st.caption(str(step["content"]))

        # Render Final Content
        if view_type == "text":
            st.markdown(content)

        elif view_type == "chart":
            # The backend sends Recharts code/config. For Python UI, we plot the raw data using Plotly.
            st.subheader(content.get("title", "Data Visualization"))
            chart_data = content.get("data", [])
            if chart_data:
                df = pd.DataFrame(chart_data)
                st.dataframe(df.head(), use_container_width=True)

                # Auto-detect numeric columns for plotting
                num_cols = df.select_dtypes(include=["number"]).columns
                cat_cols = df.select_dtypes(include=["object", "string"]).columns

                if len(num_cols) > 0 and len(cat_cols) > 0:
                    # Simple heuristic for plotting
                    fig = px.bar(
                        df, x=cat_cols[0], y=num_cols[0], title="Generated Chart"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("See Raw React Code"):
                        st.code(content.get("react_code", ""), language="javascript")
                else:
                    st.info("Data received, but could not auto-plot. See table above.")

        elif view_type == "map":
            st.subheader("🗺️ Geographic Distribution")
            map_data = content.get("data", [])
            val_key = content.get("value_key")
            prov_key = content.get("province_key")

            if map_data:
                df = pd.DataFrame(map_data)
                st.dataframe(df, use_container_width=True)
                st.info(f"Mapping **{val_key}** by **{prov_key}**")

        elif view_type == "error":
            st.error(content)


# --- CHAT LOGIC ---

# 1. Display History
for msg in st.session_state.messages:
    render_message(
        msg["role"], msg["content"], msg.get("view", "text"), msg.get("steps")
    )

# 2. Handle Input
if prompt := st.chat_input(
    "Ask about data, upload docs for context, or request predictions..."
):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        step_container = st.status("🤔 Processing...", expanded=True)

        full_history = [
            {"role": m["role"], "content": str(m["content"])}
            for m in st.session_state.messages
            if isinstance(m["content"], str)
        ]
        payload = {"message": prompt, "history": full_history}

        collected_steps = []
        final_content = ""
        final_view = "text"

        try:
            with requests.post(
                f"{api_url}/generate-chart-stream", json=payload, stream=True
            ) as r:
                for line in r.iter_lines():
                    data = parse_stream_line(line)
                    if not data:
                        continue

                    event_type = data.get("type")
                    label = data.get("label")
                    content = data.get("content")
                    view = data.get("view")

                    # Handle Status Updates (Intermediate Steps)
                    if event_type in ["status", "log", "artifact"]:
                        step_container.write(f"**{label}**")
                        collected_steps.append(data)

                        if view == "sql":
                            step_container.code(content, language="sql")
                        if event_type == "status" and data.get("state") == "error":
                            step_container.update(
                                label="Error", state="error", expanded=True
                            )
                            st.error(content)

                    # Handle Final Result
                    elif event_type == "final":
                        final_content = content
                        final_view = view
                        step_container.update(
                            label="Complete!", state="complete", expanded=False
                        )

        except Exception as e:
            step_container.update(label="Connection Error", state="error")
            st.error(f"Could not connect to backend: {e}")
            final_content = "Error connecting to API."
            final_view = "error"

        # Render Final Output in the placeholder
        message_placeholder.empty()
        if final_view == "text":
            st.markdown(final_content)
        elif final_view == "chart":
            st.subheader(final_content.get("title", "Chart Result"))
            df = pd.DataFrame(final_content.get("data", []))
            if not df.empty:
                # Try to plot dynamically
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    fig = (
                        px.line(df, x=cols[1], y=cols[-1])
                        if "year" in cols[1].lower()
                        else px.bar(df, x=cols[0], y=cols[-1])
                    )
                    st.plotly_chart(fig)
                st.dataframe(df)
        elif final_view == "map":
            st.subheader("🗺️ Map Data")
            st.dataframe(pd.DataFrame(final_content.get("data", [])))

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_content,
                "view": final_view,
                "steps": collected_steps,
            }
        )

# Footer with quick examples
with st.expander("💡 Example Queries"):
    st.markdown("""
    **With RAG (after uploading docs):**
    - "What are the coverage limits in the uploaded policy?"
    - "Summarize the guidelines document"

    **Predictions:**
    - "Predict faskes growth for 2025"
    - "Show me penyakit trends for the next 3 years"

    **SQL/Data:**
    - "Show me top 10 provinces by claims"
    - "Create a chart of monthly registrations"
    """)
