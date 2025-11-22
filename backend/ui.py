import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --- CONFIGURATION ---
st.set_page_config(page_title="Agentic BI Test Console", layout="wide")
API_URL = os.getenv("API_URL", "http://localhost:8000/generate-chart-stream")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🎛️ Testing Console")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

def render_chart(payload):
    try:
        data = payload.get("data", [])
        if not data:
            st.warning("No data to visualize")
            return

        df = pd.DataFrame(data)
        title = payload.get("title", "Data Visualization")

        # Column Classification
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        # 1. IDENTIFY TIME COLUMN
        time_keywords = ["year", "date", "time", "tahun", "thn", "waktu", "time_col"]
        time_col = next(
            (c for c in cols if any(x in c.lower() for x in time_keywords)), None
        )

        # 2. FORECAST / LINE CHART
        if time_col and len(numeric_cols) > 0:
            # CRITICAL FIX: Exclude Time Column from Y-Axis candidates
            y_candidates = [c for c in numeric_cols if c != time_col]

            if not y_candidates:
                st.warning(f"Y-Axis Error. Numeric: {numeric_cols}, Time: {time_col}")
                st.dataframe(df)
                return

            y_col = y_candidates[0]
            entity_col = categorical_cols[0] if categorical_cols else None

            fig = px.line(
                df, x=time_col, y=y_col, color=entity_col, title=title, markers=True
            )

            # Forecast Divider
            if "type" in df.columns:
                history_df = df[df["type"] == "history"]
                if not history_df.empty:
                    fig.add_vline(
                        x=history_df[time_col].max(),
                        line_dash="dash",
                        annotation_text="Forecast",
                    )

            st.plotly_chart(fig, use_container_width=True)

        # 3. BAR CHART
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            fig = px.bar(
                df,
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=title,
                color=categorical_cols[0],
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.dataframe(df)

    except Exception as e:
        st.error(f"Chart Error: {e}")


st.title("🤖 Agentic BI: Unified Test UI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "text":
            st.markdown(msg["content"])
        elif msg.get("type") == "chart":
            render_chart(msg["content"])
        elif msg.get("type") == "sql":
            st.code(msg["content"], language="sql")

if prompt := st.chat_input("Predict trends, ask data..."):
    st.session_state.messages.append(
        {"role": "user", "type": "text", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)
        response_placeholder = st.empty()

        api_history = [
            {"role": m["role"], "content": str(m["content"])}
            for m in st.session_state.messages
        ]

        try:
            with requests.post(
                API_URL, json={"message": prompt, "history": api_history}, stream=True
            ) as r:
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8")[6:])
                        e_type, content = data.get("type"), data.get("content")

                        if e_type == "status":
                            status_container.update(
                                label=data.get("label"), state="running"
                            )
                        elif e_type == "log":
                            status_container.write(f"🧠 {content}")
                        elif e_type == "artifact":
                            status_container.code(content, language="sql")
                            st.session_state.messages.append(
                                {"role": "assistant", "type": "sql", "content": content}
                            )
                        elif e_type == "final":
                            status_container.update(
                                label="Done", state="complete", expanded=False
                            )
                            view = data.get("view")
                            if view == "text":
                                response_placeholder.markdown(content)
                            elif view == "chart":
                                render_chart(content)
                            st.session_state.messages.append(
                                {"role": "assistant", "type": view, "content": content}
                            )
                            st.session_state.messages.append({"role": "assistant", "type": view, "content": content})

        except Exception as e:
            st.error(f"Error: {e}")
