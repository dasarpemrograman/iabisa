import json
import os  # <--- Add this import

import pandas as pd
import requests
import streamlit as st

# Use environment variable if available, otherwise fallback to localhost (for local non-docker runs)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/generate-chart-stream")

st.set_page_config(page_title="Thin Client BI", layout="wide")
st.title("⚡ Thin-Client Agentic BI")

# ... rest of the file remains the same ...
query = st.text_area("Ask a question:", "Show top 5 selling products")
run_btn = st.button("🚀 Run", type="primary")

if run_btn and query:
    # This container handles the "Visibility of System Status"
    status_container = st.status("🚀 Starting Agent System...", expanded=True)

    final_result = None

    try:
        response = requests.post(API_URL, json={"query": query}, stream=True)

        for line in response.iter_lines():
            if line:
                # 1. Parse the Server Instruction
                data = json.loads(line.decode("utf-8").replace("data: ", ""))

                msg_type = data.get("type")
                label = data.get("label")
                content = data.get("content")
                view = data.get("view")
                state = data.get("state")

                # 2. Render based on Instruction
                with status_container:
                    # A. Update the Main Status Label (The "Header")
                    if msg_type == "status":
                        status_container.update(label=label, state=state)

                    # B. Show simple log lines
                    elif msg_type == "log":
                        st.write(f"› {content}")

                    # C. Show rich artifacts (Code, JSON, Tables)
                    elif msg_type == "artifact":
                        with st.expander(label or "Details", expanded=False):
                            if view == "sql":
                                st.code(content, language="sql")
                            elif view == "json":
                                st.json(content)
                            elif view == "javascript":
                                st.code(content, language="javascript")

                    # D. Handle Final Result
                    elif msg_type == "final":
                        final_result = content
                        status_container.update(
                            label="✅ Done!", state="complete", expanded=False
                        )

    except Exception as e:
        st.error(f"Connection Error: {e}")

    # 3. Render Final Output (Outside the status stream)
    if final_result:
        st.divider()
        c1, c2 = st.columns([1.5, 1])

        with c1:
            st.subheader(f"<{final_result['component_name']} />")
            st.code(final_result["react_code"], language="javascript")

        with c2:
            st.subheader("Underlying Data")
            st.dataframe(pd.DataFrame(final_result["data"]), use_container_width=True)
            with st.expander("Executed SQL"):
                st.code(final_result["sql"], language="sql")
