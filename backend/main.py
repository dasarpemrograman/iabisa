import json
import logging
import os
import sys
from datetime import date, datetime
from decimal import Decimal  # <--- FIX 1: Import Decimal
from typing import Any, Dict, List, Literal, Optional

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

# Try importing prediction logic (if available)
try:
    from routers.prediction import PredictionRequest, load_model_from_supabase

    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================

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

app = FastAPI(title="Enterprise Agentic BI API (Gemini)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. DATABASE INSPECTOR
# ==============================================================================


class DatabaseInspector:
    _TEXTUAL_TYPES = ("TEXT", "CHAR", "VARCHAR", "CHARACTER VARYING", "STRING")
    _MAX_CATEGORY_VALUES = 25
    _SAMPLE_LIMIT = 3

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
                    # Quote table names if mixed-case
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

                        # --- FIX 2: Force Quotes in Schema Context ---
                        # If the column has mixed casing, present it with quotes to the LLM
                        if any(c.isupper() for c in col_name):
                            display_col = f'"{col_name}"'
                        else:
                            display_col = col_name

                        sample_info = self._build_column_sample_info(
                            cursor, table, col_name, col_type
                        )
                        report_lines.append(
                            f"  - {display_col} ({col_type}){sample_info}"
                        )

                return "\n".join(report_lines)
        except Exception as exc:
            logger.exception("Error inspecting schema: %s", exc)
            return f"Error inspecting DB: {exc}"

    def _build_column_sample_info(self, cursor, table, col_name, col_type) -> str:
        is_text = any(t in col_type.upper() for t in self._TEXTUAL_TYPES)
        if not is_text:
            return ""
        try:
            # Safely quote identifiers for sampling queries
            quoted_col = f'"{col_name}"'
            quoted_table = f'"{table}"' if any(c.isupper() for c in table) else table

            cursor.execute(
                f"SELECT COUNT(DISTINCT {quoted_col}) as cnt FROM {quoted_table}"
            )
            distinct_count = cursor.fetchone()["cnt"]

            if 0 < distinct_count < self._MAX_CATEGORY_VALUES:
                cursor.execute(
                    f"SELECT DISTINCT {quoted_col} as val FROM {quoted_table}"
                )
                values = [str(row["val"]) for row in cursor.fetchall()]
                return f" (Categories: {values})"
            return ""
        except Exception:
            return ""

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("SQL execution failed: %s", exc)
            return [{"error": f"SQL Execution Failed: {exc}"}]


inspector = DatabaseInspector(DATABASE_URL)
# Load schema once at startup (or refresh periodically)
DYNAMIC_SCHEMA_CONTEXT = inspector.get_schema_summary()

# ==============================================================================
# 3. PYDANTIC MODELS & INTENT
# ==============================================================================


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


class IntentResponse(BaseModel):
    intent: Literal[
        "general_chat", "sql_text_only", "sql_chart", "sql_map", "prediction"
    ]
    reasoning: str = Field(description="Reasoning behind the intent classification")


class SQLResponse(BaseModel):
    sql_query: str
    explanation: str


class BIReviewResponse(BaseModel):
    optimized_sql: str
    changes_made: str
    is_safe: bool


class RechartsCodeResponse(BaseModel):
    code: str
    component_name: str


class MapDataResponse(BaseModel):
    province_column: str
    value_column: str
    data: List[Dict[str, Any]]


class PredictionParams(BaseModel):
    dataset: Optional[str] = None
    year: Optional[int] = None
    provinces: Optional[List[str]] = None


# ==============================================================================
# 4. AGENTS
# ==============================================================================


# --- A. ROUTER AGENT (UPDATED) ---
def _make_router_agent() -> Agent[None, IntentResponse]:
    return Agent(
        llm_model,
        output_type=IntentResponse,
        system_prompt=(
            "You are an Intent Classifier for an Enterprise BI System.\n"
            "Classify the user's request based on the conversation history.\n"
            "INTENT CATEGORIES:\n"
            "1. 'general_chat': Greetings, clarifications, or questions NOT requiring data access.\n"
            "2. 'sql_text_only': Questions requiring database data OR Schema Metadata (e.g. 'what tables are there?', 'list columns', 'Who is the manager?', 'Total count').\n"
            "3. 'sql_chart': Questions asking for trends, comparisons, or distributions suitable for charts.\n"
            "4. 'sql_map': Questions specifically asking for GEOGRAPHIC distribution or MAPS (e.g., 'Show map of provinces').\n"
            "5. 'prediction': Questions asking to FORECAST or PREDICT future values (e.g., 'Predict next year', 'Future trends').\n"
        ),
    )


# --- B. GENERAL CHAT AGENT ---
def _make_chat_agent() -> Agent[None, str]:
    return Agent(
        llm_model,
        system_prompt="You are a helpful BI Assistant. Answer general questions politely. If asked about data, refer them to the data tools.",
    )


# --- C. SQL PIPELINE AGENTS ---
def _make_sql_agent() -> Agent[None, SQLResponse]:
    return Agent(
        llm_model,
        output_type=SQLResponse,
        system_prompt=(
            "You are a PostgreSQL SQL Engine. \n"
            f"SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
            "RULES:\n"
            "1. Use valid PostgreSQL syntax.\n"
            "2. If asking for tables/schema, query 'information_schema'.\n"
            "3. Use the exact column names (including quotes) from the SCHEMA provided above."
        ),
    )


def _make_bi_reviewer_agent() -> Agent[None, BIReviewResponse]:
    return Agent(
        llm_model,
        output_type=BIReviewResponse,
        system_prompt=(
            "You are a BI Analyst. Optimize SQL for read-only safety and performance.\n"
            f"SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
            "Ensure queries to 'information_schema' are allowed."
        ),
    )


def _make_summarizer_agent() -> Agent[None, str]:
    return Agent(
        llm_model,
        system_prompt="You are a Data Analyst. Given the User Query and Data Results, provide a concise text answer. Do not expose internal SQL details.",
    )


def _make_chart_agent() -> Agent[None, RechartsCodeResponse]:
    return Agent(
        llm_model,
        output_type=RechartsCodeResponse,
        system_prompt=(
            "You are a Recharts Expert. Generate a React component.\n"
            "Rules: Embed data in `const data = [...]`. Use Recharts. Return valid JSON."
        ),
    )


def _make_map_formatter_agent() -> Agent[None, MapDataResponse]:
    return Agent(
        llm_model,
        output_type=MapDataResponse,
        system_prompt=(
            "You are a Map Data Formatter. Analyze the SQL results.\n"
            "Identify which column represents the 'Province' and which is the 'Value'.\n"
            "Return the structured data suitable for a map."
        ),
    )


# --- D. PREDICTION AGENT ---
def _make_prediction_param_agent() -> Agent[None, PredictionParams]:
    return Agent(
        llm_model,
        output_type=PredictionParams,
        system_prompt="Extract: dataset ('fkrtl', 'klinik_pratama', 'praktek_dokter'), target year, and provinces.",
    )


router_agent = _make_router_agent()
chat_agent = _make_chat_agent()
sql_agent = _make_sql_agent()
bi_agent = _make_bi_reviewer_agent()
summarizer_agent = _make_summarizer_agent()
chart_agent = _make_chart_agent()
map_agent = _make_map_formatter_agent()
prediction_param_agent = _make_prediction_param_agent()

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # --- FIX 3: Handle Decimal Serialization ---
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def create_event(
    event_type: str,
    label: str = "",
    state: str = "running",
    content: Any = None,
    view: str = "text",
) -> str:
    payload = {
        "type": event_type,
        "label": label,
        "state": state,
        "content": content,
        "view": view,
    }
    return f"data: {json.dumps(payload, default=json_serial)}\n\n"


def format_history(messages: List[Message]) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in messages[-5:]])


# ==============================================================================
# 6. MAIN STREAMING ENDPOINT
# ==============================================================================


@app.post("/generate-chart-stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        try:
            history_text = format_history(request.history)
            full_context = (
                f"History:\n{history_text}\n\nCurrent User Input: {request.message}"
            )

            logger.info("Processing request: %s", request.message)
            yield create_event("status", "🧠 Analyzing Intent...", "running")

            # --- STEP 1: ROUTING ---
            intent_res = await router_agent.run(full_context)
            intent = intent_res.output.intent
            yield create_event("log", f"Intent detected: {intent}", "running")

            # ==================================================================
            # BRANCH: GENERAL CHAT
            # ==================================================================
            if intent == "general_chat":
                chat_res = await chat_agent.run(full_context)
                yield create_event(
                    "final", content=chat_res.output, view="text", state="complete"
                )
                return

            # ==================================================================
            # BRANCH: PREDICTION
            # ==================================================================
            if intent == "prediction":
                if not PREDICTION_AVAILABLE:
                    yield create_event(
                        "status", "❌ Prediction Module Missing", "error"
                    )
                    return

                yield create_event("status", "🔮 Configuring Prediction...", "running")
                params_res = await prediction_param_agent.run(full_context)
                params = params_res.output

                yield create_event(
                    "log", f"Predicting {params.dataset} for {params.year}", "running"
                )
                # Placeholder for actual call integration
                response_text = f"Predictive analysis for {params.dataset or 'healthcare'} in {params.year or 'upcoming year'} initialized."
                yield create_event(
                    "final", content=response_text, view="text", state="complete"
                )
                return

            # ==================================================================
            # BRANCH: SQL (TEXT, CHART, MAP)
            # ==================================================================

            # 1. Draft SQL
            yield create_event("status", "📝 Drafting Query...", "running")
            sql_res = await sql_agent.run(full_context)
            draft_sql = sql_res.output.sql_query
            yield create_event("artifact", "Draft SQL", content=draft_sql, view="sql")

            # 2. Optimize SQL
            yield create_event("status", "🕵️ Optimizing Query...", "running")
            bi_res = await bi_agent.run(
                f"Query: {request.message}\nDraft SQL: {draft_sql}"
            )
            optimized_sql = bi_res.output.optimized_sql

            if not bi_res.output.is_safe:
                yield create_event("status", "❌ Unsafe Query", "error")
                return

            yield create_event(
                "artifact", "Optimized SQL", content=optimized_sql, view="sql"
            )

            # 3. Execute
            yield create_event("status", "🗄️ Fetching Data...", "running")
            db_data = inspector.execute_query(optimized_sql)

            if not db_data or (
                isinstance(db_data, list) and len(db_data) > 0 and "error" in db_data[0]
            ):
                err = db_data[0]["error"] if db_data else "No data returned"
                yield create_event("status", "⚠️ Database Error", "error")
                yield create_event("log", content=str(err))
                return

            yield create_event("log", f"Fetched {len(db_data)} rows.", "running")

            # 4. Format Output
            if intent == "sql_text_only":
                yield create_event("status", "✍️ Summarizing...", "running")
                summary_prompt = f"User Question: {request.message}\nData: {db_data}"
                summary_res = await summarizer_agent.run(summary_prompt)
                yield create_event(
                    "final", content=summary_res.output, view="text", state="complete"
                )

            elif intent == "sql_map":
                yield create_event("status", "🗺️ Formatting Map Data...", "running")
                map_res = await map_agent.run(f"Data: {db_data[:10]}")
                payload = {
                    "province_key": map_res.output.province_column,
                    "value_key": map_res.output.value_column,
                    "data": db_data,
                }
                yield create_event(
                    "final", content=payload, view="map", state="complete"
                )

            elif intent == "sql_chart":
                yield create_event("status", "🎨 Designing Chart...", "running")
                chart_res = await chart_agent.run(
                    f"User Query: {request.message}\nData Sample: {db_data[:20]}"
                )
                payload = {
                    "component_name": chart_res.output.component_name,
                    "react_code": chart_res.output.code,
                    "data": db_data,
                }
                yield create_event(
                    "final", content=payload, view="chart", state="complete"
                )

        except Exception as exc:
            logger.exception("Pipeline failed")
            yield create_event("status", "❌ System Error", "error", content=str(exc))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
