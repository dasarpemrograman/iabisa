import json
import logging
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AgenticBI")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-flash-lite-latest")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY is missing.")
    sys.exit(1)
if not DATABASE_URL:
    logger.critical("DATABASE_URL is missing.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    llm_model = GoogleModel(model_name=LLM_MODEL_NAME)
    logger.info("Google Gemini Model initialized: %s", LLM_MODEL_NAME)
except Exception as exc:
    logger.critical("Failed to initialize Google Model: %s", exc)
    sys.exit(1)

app = FastAPI(title="Enterprise Agentic BI API (Composable)", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 1. DATABASE INSPECTOR
# ==============================================================================


class DatabaseInspector:
    _TEXTUAL_TYPES = ("TEXT", "CHAR", "VARCHAR", "CHARACTER VARYING", "STRING")
    _MAX_CATEGORY_VALUES = 25

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def _get_connection(self) -> psycopg.Connection:
        return psycopg.connect(self.db_url, row_factory=dict_row)

    def get_schema_summary(self) -> str:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"
                )
                tables = [row["table_name"] for row in cursor.fetchall()]
                report_lines = [f"DATABASE TYPE: PostgreSQL", "SCHEMA REPORT:"]

                for table in tables:
                    safe_table = (
                        f'"{table}"' if any(c.isupper() for c in table) else table
                    )
                    report_lines.append(f"\nTABLE: {safe_table}")
                    report_lines.append("COLUMNS:")

                    cursor.execute(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_name = %s AND table_schema = 'public'",
                        (table,),
                    )
                    columns = cursor.fetchall()
                    for col in columns:
                        col_name = col["column_name"]
                        col_type = col["data_type"] or ""
                        display_col = (
                            f'"{col_name}"'
                            if any(c.isupper() for c in col_name)
                            else col_name
                        )

                        # Sample values for textual columns
                        sample_info = ""
                        if any(t in col_type.upper() for t in self._TEXTUAL_TYPES):
                            try:
                                cursor.execute(
                                    f"SELECT COUNT(DISTINCT {display_col}) as cnt FROM {safe_table}"
                                )
                                cnt = cursor.fetchone()["cnt"]
                                if 0 < cnt < self._MAX_CATEGORY_VALUES:
                                    cursor.execute(
                                        f"SELECT DISTINCT {display_col} as val FROM {safe_table}"
                                    )
                                    vals = [str(r["val"]) for r in cursor.fetchall()]
                                    sample_info = f" (Categories: {vals})"
                            except Exception:
                                pass

                        report_lines.append(
                            f"  - {display_col} ({col_type}){sample_info}"
                        )

                return "\n".join(report_lines)
        except Exception as exc:
            logger.exception("Error inspecting schema")
            return f"Error: {exc}"

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                if cursor.description:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                return []
        except Exception as exc:
            return [{"error": f"SQL Execution Failed: {exc}"}]


inspector = DatabaseInspector(DATABASE_URL)
DYNAMIC_SCHEMA_CONTEXT = inspector.get_schema_summary()

# ==============================================================================
# 2. DATA MODELS & STATE
# ==============================================================================


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


# --- WORKFLOW STATE (The "Context" that flows through steps) ---
class WorkflowContext(BaseModel):
    request: ChatRequest
    intent: Optional[str] = None
    draft_sql: Optional[str] = None
    optimized_sql: Optional[str] = None
    sql_explanation: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    final_response: Optional[Any] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# Response Models
class IntentResponse(BaseModel):
    intent: Literal[
        "general_chat", "sql_text_only", "sql_chart", "sql_map", "prediction"
    ]
    reasoning: str


class SQLResponse(BaseModel):
    sql_query: str
    explanation: str


class BIReviewResponse(BaseModel):
    optimized_sql: str
    is_safe: bool


class RechartsCodeResponse(BaseModel):
    code: str
    component_name: str


class MapDataResponse(BaseModel):
    province_column: str
    value_column: str


class PredictionParams(BaseModel):
    dataset: Optional[str] = None
    year: Optional[int] = None
    provinces: Optional[List[str]] = None


# ==============================================================================
# 3. AGENTS
# ==============================================================================

router_agent = Agent(
    llm_model,
    output_type=IntentResponse,
    system_prompt=(
        "You are an Intent Classifier. Classify the user request.\n"
        "CATEGORIES:\n"
        "1. 'general_chat': Greetings, clarifications.\n"
        "2. 'sql_text_only': Data questions, lists, simple counts.\n"
        "3. 'sql_chart': Trends, comparisons, distributions.\n"
        "4. 'sql_map': Geographic distribution (provinces, maps).\n"
        "5. 'prediction': Forecasting future values.\n"
    ),
)

chat_agent = Agent(
    llm_model,
    system_prompt="You are a helpful BI Assistant. Answer politely. If asked about data, refer to tools.",
)

sql_agent = Agent(
    llm_model,
    output_type=SQLResponse,
    system_prompt=(
        f"You are a PostgreSQL Expert. SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
        "Rules: Use valid PostgreSQL. Use exact table/column names from schema (quote if needed)."
    ),
)

bi_agent = Agent(
    llm_model,
    output_type=BIReviewResponse,
    system_prompt=(
        f"You are a BI Analyst. Optimize SQL for read-only safety.\nSCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}"
    ),
)

summarizer_agent = Agent(
    llm_model,
    system_prompt="Summarize the data results concisely for the user.",
)

chart_agent = Agent(
    llm_model,
    output_type=RechartsCodeResponse,
    system_prompt="Generate Recharts config (JSON). Embed data in `const data = [...]`. Return valid JSON.",
)

map_agent = Agent(
    llm_model,
    output_type=MapDataResponse,
    system_prompt="Identify the 'Province' column and 'Value' column from the data sample.",
)

prediction_param_agent = Agent(
    llm_model,
    output_type=PredictionParams,
    system_prompt="Extract: dataset ('fkrtl', 'klinik_pratama', 'praktek_dokter'), year, provinces.",
)

# ==============================================================================
# 4. COMPOSABLE STEPS
# ==============================================================================


def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def create_event(
    type_: str,
    label: str = "",
    state: str = "running",
    content: Any = None,
    view: str = "text",
) -> str:
    payload = {
        "type": type_,
        "label": label,
        "state": state,
        "content": content,
        "view": view,
    }
    return f"data: {json.dumps(payload, default=json_serial)}\n\n"


def format_history(msgs: List[Message]) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in msgs[-5:]])


# --- STEP 1: ROUTING ---
async def step_identify_intent(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    yield create_event("status", "🧠 Analyzing Intent...", "running")
    history_text = format_history(ctx.request.history)
    full_prompt = f"History:\n{history_text}\n\nUser Input: {ctx.request.message}"

    result = await router_agent.run(full_prompt)
    ctx.intent = result.output.intent
    yield create_event("log", f"Intent detected: {ctx.intent}", "running")


# --- STEP 2: GENERAL CHAT ---
async def step_general_chat(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    history_text = format_history(ctx.request.history)
    result = await chat_agent.run(f"{history_text}\nUser: {ctx.request.message}")
    yield create_event("final", content=result.output, view="text", state="complete")


# --- STEP 3: PREDICTION ---
async def step_prediction(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    # (Simplified placeholder for brevity, integrates with prediction.py logic)
    yield create_event("status", "🔮 Configuring Prediction...", "running")
    params = await prediction_param_agent.run(ctx.request.message)
    yield create_event(
        "final",
        content=f"Prediction initiated for {params.output.dataset}",
        view="text",
        state="complete",
    )


# --- STEP 4: SQL GENERATION ---
async def step_draft_sql(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    yield create_event("status", "📝 Drafting Query...", "running")
    history_text = format_history(ctx.request.history)
    prompt = f"{history_text}\nUser Question: {ctx.request.message}"

    result = await sql_agent.run(prompt)
    ctx.draft_sql = result.output.sql_query
    ctx.sql_explanation = result.output.explanation
    yield create_event("artifact", "Draft SQL", content=ctx.draft_sql, view="sql")


# --- STEP 5: SQL REVIEW ---
async def step_optimize_sql(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    yield create_event("status", "🕵️ Optimizing Query...", "running")
    result = await bi_agent.run(f"Draft SQL: {ctx.draft_sql}")

    if not result.output.is_safe:
        ctx.error = "Query deemed unsafe."
        yield create_event("status", "❌ Unsafe Query Detected", "error")
        return

    ctx.optimized_sql = result.output.optimized_sql
    yield create_event(
        "artifact", "Optimized SQL", content=ctx.optimized_sql, view="sql"
    )


# --- STEP 6: DATA FETCH ---
async def step_execute_query(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    yield create_event("status", "🗄️ Fetching Data...", "running")

    if not ctx.optimized_sql:
        return

    data = inspector.execute_query(ctx.optimized_sql)

    # Check for DB errors
    if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
        ctx.error = data[0]["error"]
        yield create_event("status", "⚠️ Database Error", "error", content=ctx.error)
        return

    ctx.data = data
    yield create_event("log", f"Fetched {len(data)} rows.", "running")


# --- STEP 7: VISUALIZATION / FORMATTING ---
async def step_format_response(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    if ctx.error or not ctx.data:
        if not ctx.error:
            yield create_event("final", "No data found.", view="text", state="complete")
        return

    # A. TEXT SUMMARY
    if ctx.intent == "sql_text_only":
        yield create_event("status", "✍️ Summarizing...", "running")
        summary = await summarizer_agent.run(
            f"Question: {ctx.request.message}\nData: {ctx.data}"
        )
        yield create_event(
            "final", content=summary.output, view="text", state="complete"
        )

    # B. MAP VISUALIZATION
    elif ctx.intent == "sql_map":
        yield create_event("status", "🗺️ Formatting Map...", "running")
        # Helper agent to find columns
        map_info = await map_agent.run(f"Data Sample: {ctx.data[:5]}")

        payload = {
            "province_key": map_info.output.province_column,
            "value_key": map_info.output.value_column,
            "data": ctx.data,
        }
        yield create_event("final", content=payload, view="map", state="complete")

    # C. CHART VISUALIZATION
    elif ctx.intent == "sql_chart":
        yield create_event("status", "🎨 Designing Chart...", "running")
        chart_res = await chart_agent.run(
            f"Question: {ctx.request.message}\nData Sample: {ctx.data[:10]}"
        )

        payload = {
            "component_name": chart_res.output.component_name,
            "react_code": chart_res.output.code,
            "data": ctx.data,
        }
        yield create_event("final", content=payload, view="chart", state="complete")


# ==============================================================================
# 5. ORCHESTRATOR
# ==============================================================================


async def run_pipeline(request: ChatRequest):
    """
    The Orchestrator:
    1. Creates Context
    2. Identifies Intent
    3. Composes the correct list of steps (Recipe)
    4. Executes them sequentially
    """
    ctx = WorkflowContext(request=request)

    try:
        # 1. Identify Intent
        async for event in step_identify_intent(ctx):
            yield event

        # 2. Select Recipe based on Intent
        pipeline_steps = []

        if ctx.intent == "general_chat":
            pipeline_steps = [step_general_chat]

        elif ctx.intent == "prediction":
            pipeline_steps = [step_prediction]

        elif ctx.intent in ["sql_text_only", "sql_chart", "sql_map"]:
            # The "Standard BI Pipeline"
            pipeline_steps = [
                step_draft_sql,
                step_optimize_sql,
                step_execute_query,
                step_format_response,
            ]

        # 3. Execute Recipe
        for step_fn in pipeline_steps:
            if ctx.error:
                break  # Stop pipeline on error

            async for event in step_fn(ctx):
                yield event

    except Exception as e:
        logger.exception("Pipeline crashed")
        yield create_event("status", "❌ System Error", "error", content=str(e))


@app.post("/generate-chart-stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(run_pipeline(request), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
