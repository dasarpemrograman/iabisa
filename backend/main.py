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
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from automl_service import AutoMLService

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

if not GEMINI_API_KEY or not DATABASE_URL:
    logger.critical("Missing API Key or Database URL.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm_model = GoogleModel(model_name=LLM_MODEL_NAME)

app = FastAPI(title="Enterprise Agentic BI API (AutoML)", version="3.2.0")

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
    # Increased to ensure we capture all provinces (usually ~34-38 distinct values)
    _MAX_CATEGORY_VALUES = 100

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
                                cnt_res = cursor.fetchone()
                                cnt = cnt_res["cnt"] if cnt_res else 0

                                if 0 < cnt < self._MAX_CATEGORY_VALUES:
                                    cursor.execute(
                                        f"SELECT DISTINCT {display_col} as val FROM {safe_table} LIMIT {self._MAX_CATEGORY_VALUES}"
                                    )
                                    vals = [str(r["val"]) for r in cursor.fetchall()]
                                    sample_info = f" (Values: {vals})"
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


# --- Pydantic Models ---
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


class WorkflowContext(BaseModel):
    request: ChatRequest
    intent: Optional[str] = None
    draft_sql: Optional[str] = None
    optimized_sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# Output Models
class IntentResponse(BaseModel):
    intent: Literal[
        "general_chat", "sql_text_only", "sql_chart", "sql_map", "prediction"
    ]
    reasoning: str


class ForecastParams(BaseModel):
    years_to_predict: int = Field(
        default=5, description="Number of future years to predict"
    )


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


# ==============================================================================
# 2. AGENTS
# ==============================================================================

router_agent = Agent(
    llm_model,
    output_type=IntentResponse,
    system_prompt="Classify intent: 'prediction' (future/forecast/will happen), 'sql_map' (geographic), 'sql_chart' (trends/comparison), 'sql_text_only' (data lookup), 'general_chat'.",
)

param_agent = Agent(
    llm_model,
    output_type=ForecastParams,
    system_prompt="Extract the number of years to predict. If not specified, default to 5.",
)

# --- OMNIPOTENT SQL AGENT ---
sql_agent = Agent(
    llm_model,
    output_type=SQLResponse,
    system_prompt=(
        f"You are an Omniscient PostgreSQL Expert. SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
        "STRICT RULES:\n"
        "1. **VALID SYNTAX**: Use valid PostgreSQL.\n"
        '2. **QUOTING**: Double-quote ALL identifiers (tables and columns) to preserve case (e.g. "Tahun", "Jumlah_Klinik").\n'
        "3. **CASE-INSENSITIVITY**: Use `ILIKE` for string comparisons (e.g. WHERE \"Provinsi\" ILIKE '%Jawa Barat%').\n"
        "4. **DATA VALUES**: Check the SCHEMA REPORT for exact values. If searching for 'West Java', look at the schema to see it is 'JAWA BARAT'.\n"
        "5. **AGGREGATION**: If asked for 'total', use SUM(). If 'number of rows', use COUNT(*).\n"
    ),
)

bi_agent = Agent(
    llm_model,
    output_type=BIReviewResponse,
    system_prompt=f"Optimize SQL. Schema:\n{DYNAMIC_SCHEMA_CONTEXT}",
)
chart_agent = Agent(
    llm_model,
    output_type=RechartsCodeResponse,
    system_prompt="Generate Recharts JSON config (type, xAxisKey, series). No data values.",
)
map_agent = Agent(
    llm_model,
    output_type=MapDataResponse,
    system_prompt="Identify province and value columns.",
)
chat_agent = Agent(llm_model, system_prompt="Helpful BI Assistant.")
summarizer_agent = Agent(llm_model, system_prompt="Summarize data results.")

# ==============================================================================
# 3. STEPS
# ==============================================================================


def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def create_event(type_, label="", state="running", content=None, view="text"):
    return f"data: {json.dumps({'type': type_, 'label': label, 'state': state, 'content': content, 'view': view}, default=json_serial)}\n\n"


async def step_identify_intent(ctx: WorkflowContext):
    yield create_event("status", "🧠 Analyzing Intent...")
    res = await router_agent.run(
        f"History: {ctx.request.history}\nInput: {ctx.request.message}"
    )
    ctx.intent = res.output.intent
    yield create_event("log", f"Intent: {ctx.intent}")


async def step_prediction(ctx: WorkflowContext):
    yield create_event("status", "🔮 Configuring Forecast...")

    try:
        param_res = await param_agent.run(ctx.request.message)
        n_years = param_res.output.years_to_predict
    except:
        n_years = 5

    yield create_event("log", f"Forecasting horizon: {n_years} years.")

    try:
        ctx.data = await AutoMLService.run_forecast(
            user_query=ctx.request.message,
            llm_model=llm_model,
            inspector=inspector,
            forecast_years=n_years,
        )
    except Exception as e:
        logger.error(f"AutoML Error: {e}")
        yield create_event(
            "final", f"Prediction failed: {str(e)}", view="text", state="error"
        )
        return

    if not ctx.data:
        yield create_event(
            "final",
            "No forecast generated (insufficient data).",
            view="text",
            state="complete",
        )
        return

    yield create_event("status", "🎨 Visualizing...")
    chart_res = await chart_agent.run(
        f"User: {ctx.request.message}\nSample: {ctx.data[:3]}\nCreate a LineChart."
    )

    payload = {
        "component_name": chart_res.output.component_name,
        "react_code": chart_res.output.code,
        "data": ctx.data,
        "title": f"Forecast (+{n_years} Years)",
    }
    yield create_event("final", content=payload, view="chart", state="complete")


# Standard Steps
async def step_draft_sql(ctx: WorkflowContext):
    yield create_event("status", "📝 Drafting SQL...")
    res = await sql_agent.run(f"Input: {ctx.request.message}")
    ctx.draft_sql = res.output.sql_query
    yield create_event("artifact", "Draft SQL", content=ctx.draft_sql, view="sql")


async def step_optimize_sql(ctx: WorkflowContext):
    yield create_event("status", "🕵️ Optimizing...")
    res = await bi_agent.run(f"SQL: {ctx.draft_sql}")
    if not res.output.is_safe:
        ctx.error = "Unsafe SQL"
        return
    ctx.optimized_sql = res.output.optimized_sql
    yield create_event(
        "artifact", "Optimized SQL", content=ctx.optimized_sql, view="sql"
    )


async def step_execute_query(ctx: WorkflowContext):
    yield create_event("status", "🗄️ Fetching Data...")
    if not ctx.optimized_sql:
        return
    ctx.data = inspector.execute_query(ctx.optimized_sql)
    if isinstance(ctx.data, list) and ctx.data and "error" in ctx.data[0]:
        ctx.error = ctx.data[0]["error"]
    yield create_event("log", f"Rows: {len(ctx.data)}")


async def step_format_response(ctx: WorkflowContext):
    if ctx.error:
        yield create_event("status", "❌ Error", "error", content=ctx.error)
        return

    if not ctx.data:
        # Only show "No data" if it's truly empty, not just 0 rows
        yield create_event(
            "final", "No matching data found.", view="text", state="complete"
        )
        return

    if ctx.intent == "sql_chart":
        res = await chart_agent.run(
            f"Input: {ctx.request.message}\nSample: {ctx.data[:5]}"
        )
        payload = {
            "component_name": res.output.component_name,
            "react_code": res.output.code,
            "data": ctx.data,
        }
        yield create_event("final", content=payload, view="chart", state="complete")
    elif ctx.intent == "sql_map":
        res = await map_agent.run(f"Sample: {ctx.data[:5]}")
        payload = {
            "province_key": res.output.province_column,
            "value_key": res.output.value_column,
            "data": ctx.data,
        }
        yield create_event("final", content=payload, view="map", state="complete")
    else:
        res = await summarizer_agent.run(
            f"Input: {ctx.request.message}\nData: {ctx.data}"
        )
        yield create_event("final", content=res.output, view="text", state="complete")


async def step_general_chat(ctx: WorkflowContext):
    res = await chat_agent.run(
        f"History: {ctx.request.history}\nUser: {ctx.request.message}"
    )
    yield create_event("final", content=res.output, view="text", state="complete")


# ==============================================================================
# 4. PIPELINE
# ==============================================================================


async def run_pipeline(request: ChatRequest):
    ctx = WorkflowContext(request=request)
    try:
        async for e in step_identify_intent(ctx):
            yield e

        steps = []
        if ctx.intent == "general_chat":
            steps = [step_general_chat]
        elif ctx.intent == "prediction":
            steps = [step_prediction]
        elif ctx.intent in ["sql_text_only", "sql_chart", "sql_map"]:
            steps = [
                step_draft_sql,
                step_optimize_sql,
                step_execute_query,
                step_format_response,
            ]

        for s in steps:
            if ctx.error:
                break
            async for e in s(ctx):
                yield e

    except Exception as e:
        logger.exception("Crash")
        yield create_event("status", "❌ Error", "error", content=str(e))


@app.post("/generate-chart-stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(run_pipeline(request), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
