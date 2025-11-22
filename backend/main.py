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

from rag import get_rag_engine
from routers.prediction import PredictionService
from routers.rag import router as rag_router

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AgenticBI")

load_dotenv()

# Use direct connection for SQL queries (admin queries)
DATABASE_URL = os.getenv("DATABASE_URL")
# Pooler connection is used by prediction modules via DATABASE_URL env var
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

# Mount RAG router
app.include_router(rag_router)

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
            logger.warning("Error inspecting schema (using fallback): %s", exc)
            return (
                "DATABASE TYPE: PostgreSQL\n"
                "Note: Schema inspection unavailable. Connection to database may be limited."
            )

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                if cursor.description:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                return []
        except psycopg.OperationalError as exc:
            # Try using pooler connection as fallback
            pooler_url = os.getenv("DATABASE_URL")
            if pooler_url and pooler_url != self.db_url:
                logger.info("Direct connection failed, trying pooler connection...")
                try:
                    with psycopg.connect(pooler_url, row_factory=dict_row) as conn:
                        cursor = conn.cursor()
                        cursor.execute(sql)
                        if cursor.description:
                            rows = cursor.fetchall()
                            return [dict(row) for row in rows]
                        return []
                except Exception as pooler_exc:
                    return [
                        {
                            "error": f"SQL Execution Failed (both direct and pooler): {pooler_exc}"
                        }
                    ]
            return [{"error": f"SQL Execution Failed: {exc}"}]
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
    history: List[Message] = Field(default_factory=list)


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
    rag_context: Optional[str] = None
    rag_sources: Optional[List[Dict[str, Any]]] = None

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
    n_years: Optional[int] = None
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
    system_prompt=(
        "You are a helpful BI Assistant. Answer politely. "
        "If asked about data, refer to tools."
    ),
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
    system_prompt=(
        "You are a Data Visualization Expert. "
        "Return a JSON object with TWO fields: "
        '"component_name" (a valid React component name as string) and '
        '"code" (the full React component source code that uses Recharts). '
        "The code must NOT include the data itself, it should accept a `data` prop. "
        "Do NOT write `const data =` inside the component. "
        "Return ONLY valid JSON matching this schema: "
        '{ "component_name": "MyChartComponent", "code": "<react code here>" }.'
    ),
)

map_agent = Agent(
    llm_model,
    output_type=MapDataResponse,
    system_prompt="Identify the 'Province' column and 'Value' column from the data sample.",
)

prediction_param_agent = Agent(
    llm_model,
    output_type=PredictionParams,
    system_prompt=(
        "Extract prediction parameters from the user's request.\n"
        "Available datasets:\n"
        "- Faskes (Healthcare Facilities): 'fkrtl', 'klinik_pratama', 'praktek_dokter'\n"
        "- Peserta (Participants): 'geo', 'kelas', 'segmen'\n"
        "- Penyakit (Diseases): 'kasus_per_service', 'peserta_per_service'\n"
        "If user asks for 'peserta' generically, default to 'geo'.\n\n"
        "Extract:\n"
        "- dataset: exact name from list\n"
        "- year: starting year (int)\n"
        "- n_years: number of years to predict (int)\n"
        "  * If user asks for ONE specific year (e.g., 'predict 2020'), set n_years=1\n"
        "  * If user asks for multiple years or a range (e.g., '2020-2025'), calculate n_years\n"
        "  * If unclear, default to 1 for single year prediction\n"
        "- provinces: list of province names (list of strings)\n"
        "  * Use proper capitalization: 'Bali', 'DKI Jakarta', 'Jawa Barat'\n"
        "  * Handle variations: 'bali' -> 'Bali', 'jakarta' -> 'DKI Jakarta'\n"
        "  * Common provinces: Bali, DKI Jakarta, Jawa Barat, Jawa Tengah, Jawa Timur, Sumatera Utara, etc."
    ),
)

# ==============================================================================
# 4. COMPOSABLE STEPS
# ==============================================================================


def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, BaseModel):
        # Pydantic model – make it JSON-serializable
        return obj.model_dump()
    # Fallbacks for arbitrary objects that provide dict-like exports
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
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

    # Check if RAG context is available
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()
        if stats.get("total_documents", 0) > 0:
            yield create_event("status", "📚 Retrieving relevant context...", "running")
            rag_result = rag_engine.query(ctx.request.message, top_k=3)
            if rag_result["success"] and rag_result["context"]:
                ctx.rag_context = rag_result["context"]
                ctx.rag_sources = rag_result["sources"]
                yield create_event(
                    "log",
                    f"Retrieved {len(rag_result['sources'])} relevant sources",
                    "running",
                )
            else:
                ctx.rag_context = None
                ctx.rag_sources = []
        else:
            ctx.rag_context = None
            ctx.rag_sources = []
    except Exception as e:
        logger.warning(f"RAG retrieval skipped: {e}")
        ctx.rag_context = None
        ctx.rag_sources = []

    history_text = format_history(ctx.request.history)
    full_prompt = f"History:\n{history_text}\n\nUser Input: {ctx.request.message}"

    result = await router_agent.run(full_prompt)
    ctx.intent = result.output.intent
    yield create_event("log", f"Intent detected: {ctx.intent}", "running")


# --- STEP 2: GENERAL CHAT ---
async def step_general_chat(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    history_text = format_history(ctx.request.history)

    # Prefer answering with RAG context when available
    if ctx.rag_context:
        enhanced_prompt = f"""
Relevant Context from Documents:
{ctx.rag_context}

---
{history_text}
User: {ctx.request.message}

Please answer using the provided context when relevant. If you cite information, mention it's from the documents.
"""
        result = await chat_agent.run(enhanced_prompt)
        answer = result.output

        if ctx.rag_sources:
            sources_text = "\n\n📚 **Sources:**\n"
            for source in ctx.rag_sources[:3]:
                filename = source.get("metadata", {}).get("file_name", "Unknown")
                score = source.get("score", 0)
                sources_text += f"- {filename} (relevance: {score:.2%})\n"
            answer += sources_text
    else:
        result = await chat_agent.run(f"{history_text}\nUser: {ctx.request.message}")
        answer = result.output

    yield create_event("final", content=answer, view="text", state="complete")


# --- STEP 3: PREDICTION (Full Implementation) ---
async def step_prediction(ctx: WorkflowContext) -> AsyncGenerator[str, None]:
    yield create_event("status", "🔮 Configuring Prediction...", "running")

    # 1. Extract Parameters using the Agent
    params_result = await prediction_param_agent.run(ctx.request.message)
    params = params_result.output

    # Defaults if agent misses them
    start_year = params.year if params.year else datetime.now().year
    dataset = params.dataset if params.dataset else "fkrtl"
    # Use extracted n_years, default to 1 for single-year questions
    n_years = params.n_years if params.n_years else 1

    # Log extracted parameters for debugging
    logger.info(
        "Extracted prediction params - dataset: %s, year: %s, n_years: %s, provinces: %s",
        dataset,
        start_year,
        n_years,
        params.provinces,
    )

    yield create_event("status", f"🧠 Running Model ({dataset})...", "running")

    try:
        # 2. Call the Service directly (No HTTP overhead)
        pred_result = await PredictionService.predict_faskes(
            dataset=dataset,
            start_year=start_year,
            n_years=n_years,
            provinces=params.provinces,
        )

        # Make sure ctx.data is JSON-serializable (dicts, not models)
        predictions = getattr(pred_result, "predictions", None)
        if predictions is None:
            ctx.data = []
        else:
            ctx.data = [
                p.model_dump() if isinstance(p, BaseModel) else p for p in predictions
            ]

        if not ctx.data:
            yield create_event(
                "final",
                state="complete",
                content="No prediction data generated.",
                view="text",
            )
            return

        # 3. Generate Visualization using the Chart Agent
        yield create_event("status", "🎨 Visualizing Forecast...", "running")

        # We give the chart agent a sample so it knows schema (year, prediction, province)
        chart_prompt = (
            f"User asked: {ctx.request.message}\n"
            f"Dataset: {dataset}\n"
            f"Data Sample (showing columns): {ctx.data[:5]}\n"
            "Create a LineChart or BarChart. If multiple provinces, use different colors or lines."
        )

        chart_res = await chart_agent.run(chart_prompt)

        # Build title based on actual prediction range
        end_year = start_year + n_years - 1
        year_display = str(start_year) if n_years == 1 else f"{start_year}-{end_year}"

        payload = {
            "component_name": chart_res.output.component_name,
            "react_code": chart_res.output.code,
            "data": ctx.data,
            "title": f"Prediction: {dataset.replace('_', ' ').title()} ({year_display})",
        }

        # 4. Yield the Chart View
        yield create_event("final", content=payload, view="chart", state="complete")

    except Exception as e:
        logger.exception("Prediction step failed")
        ctx.error = str(e)
        yield create_event("status", "❌ Prediction Error", "error", content=str(e))


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
            yield create_event(
                "final",
                state="complete",
                content="No data found.",
                view="text",
            )
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
        pipeline_steps: List[Any] = []

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
