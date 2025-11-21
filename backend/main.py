import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

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

DB_FILE_PATH = os.getenv("DB_FILE_PATH", "Chinook_Sqlite.sqlite")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-flash-lite-latest")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY is missing from environment variables.")
    sys.exit(1)

# GoogleModel expects GOOGLE_API_KEY in the environment
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    llm_model = GoogleModel(model_name=LLM_MODEL_NAME)
    logger.info("Google Gemini Model initialized: %s", LLM_MODEL_NAME)
except Exception as exc:  # pragma: no cover - startup failure
    logger.critical("Failed to initialize Google Model: %s", exc)
    sys.exit(1)

app = FastAPI(title="Enterprise Agentic BI API (Gemini)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. DATABASE INSPECTOR
# ==============================================================================


class DatabaseInspector:
    """Introspects and queries a SQLite database in a schema-agnostic way."""

    _TEXTUAL_TYPES = ("TEXT", "CHAR", "NVARCHAR", "VARCHAR")
    _MAX_CATEGORY_VALUES = 25
    _SAMPLE_LIMIT = 3

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._log_db_presence()

    def _log_db_presence(self) -> None:
        if os.path.exists(self.db_path):
            logger.info("Using SQLite database at: %s", self.db_path)
        else:
            logger.warning(
                "Database file not found at initialization: %s", self.db_path
            )

    def _ensure_db_exists(self) -> None:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        self._ensure_db_exists()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_schema_summary(self) -> str:
        """
        Return a human-readable schema summary with basic category/sample info.
        Designed to be fed directly into the LLM as context.
        """
        if not os.path.exists(self.db_path):
            return f"ERROR: Database file '{self.db_path}' does not exist."

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [row["name"] for row in cursor.fetchall()]

                report_lines: List[str] = [
                    f"DATABASE SOURCE: {self.db_path}",
                    "SCHEMA REPORT:",
                ]

                for table in tables:
                    report_lines.append(f"\nTABLE: '{table}'")
                    report_lines.append("COLUMNS:")

                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()

                    for col in columns:
                        col_name = col["name"]
                        col_type = col["type"] or ""
                        sample_info = self._build_column_sample_info(
                            cursor=cursor,
                            table=table,
                            col_name=col_name,
                            col_type=col_type,
                        )
                        report_lines.append(f"  - {col_name} ({col_type}){sample_info}")

                return "\n".join(report_lines)
            finally:
                conn.close()
        except Exception as exc:
            logger.exception("Error while inspecting database schema: %s", exc)
            return f"Error inspecting DB: {exc}"

    def _build_column_sample_info(
        self,
        cursor: sqlite3.Cursor,
        table: str,
        col_name: str,
        col_type: str,
    ) -> str:
        """
        For textual columns, return either category list or small sample list.
        Failure to introspect samples is non-fatal.
        """
        is_text = any(t in col_type.upper() for t in self._TEXTUAL_TYPES)
        if not is_text:
            return ""

        try:
            cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table}")
            distinct_count = cursor.fetchone()[0]

            if 0 < distinct_count < self._MAX_CATEGORY_VALUES:
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table}")
                values = [str(row[0]) for row in cursor.fetchall()]
                return f" (Categories: {values})"

            cursor.execute(
                f"SELECT {col_name} FROM {table} "
                f"WHERE {col_name} IS NOT NULL LIMIT {self._SAMPLE_LIMIT}"
            )
            values = [str(row[0]) for row in cursor.fetchall()]
            return f" (Examples: {values})"
        except Exception as exc:
            logger.debug(
                "Failed to fetch sample values for %s.%s: %s", table, col_name, exc
            )
            return ""

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return rows as list[dict]."""
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
        except Exception as exc:
            logger.error("SQL execution failed: %s", exc)
            return [{"error": f"SQL Execution Failed: {exc}"}]


inspector = DatabaseInspector(DB_FILE_PATH)
logger.info("Inspecting database schema for dynamic context...")
DYNAMIC_SCHEMA_CONTEXT = inspector.get_schema_summary()
logger.info("Schema context loaded.")

# ==============================================================================
# 3. PYDANTIC MODELS
# ==============================================================================


class SQLResponse(BaseModel):
    sql_query: str
    explanation: str


class BIReviewResponse(BaseModel):
    optimized_sql: str = Field(
        ...,
        description="The improved, safer, and BI-optimized SQL.",
    )
    changes_made: str = Field(
        ...,
        description="Brief explanation of changes.",
    )
    is_safe: bool = Field(
        ...,
        description="Is the query safe to run?",
    )


class RechartsCodeResponse(BaseModel):
    code: str
    component_name: str


class EvaluationResult(BaseModel):
    is_valid: bool
    reasoning: str
    corrected_code: Optional[str] = None


class QueryRequest(BaseModel):
    query: str


# ==============================================================================
# 4. AGENT PIPELINE
# ==============================================================================


def _make_sql_agent() -> Agent[None, SQLResponse]:
    return Agent(
        llm_model,
        output_type=SQLResponse,
        retries=2,
        system_prompt=(
            "You are a generic SQL Engine capable of querying ANY SQLite database.\n"
            f"SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
            "INSTRUCTIONS:\n"
            "1. Map user intent to valid tables/columns.\n"
            "2. Return valid SQLite SQL in the `sql_query` field.\n"
        ),
    )


def _make_bi_reviewer_agent() -> Agent[None, BIReviewResponse]:
    return Agent(
        llm_model,
        output_type=BIReviewResponse,
        retries=2,
        system_prompt=(
            "You are a Senior BI Analyst and SQL Optimizer.\n"
            f"SCHEMA:\n{DYNAMIC_SCHEMA_CONTEXT}\n"
            "GOALS:\n"
            "1. Use GROUP BY for trends/distributions when appropriate.\n"
            "2. Add LIMIT 100 if results could be large.\n"
            "3. Use AS aliases for readability.\n"
            "4. Ensure the query is safe and read-only.\n"
        ),
    )


def _make_chart_agent() -> Agent[None, RechartsCodeResponse]:
    return Agent(
        llm_model,
        output_type=RechartsCodeResponse,
        retries=2,
        system_prompt=(
            "You are a Recharts Expert. Generate a React functional component.\n"
            "Requirements:\n"
            "1. Embed data strictly into `const data = [...]` inside the component.\n"
            "2. ESCAPE STRINGS: Ensure all strings in the data array are valid JS strings.\n            "
            "3. Use standard Recharts components.\n"
            "4. Do not fetch data from an API; use only the provided data sample.\n"
        ),
    )


def _make_evaluator_agent() -> Agent[None, EvaluationResult]:
    return Agent(
        llm_model,
        output_type=EvaluationResult,
        retries=2,
        system_prompt=(
            "You are a Code Reviewer. Check the provided React/Recharts code.\n"
            "Tasks:\n"
            "1. Check for syntax errors or obvious runtime issues.\n"
            "2. If errors exist and can be fixed, return corrected_code.\n"
            "3. ALWAYS return valid JSON for the EvaluationResult model.\n"
        ),
    )


sql_agent = _make_sql_agent()
bi_reviewer_agent = _make_bi_reviewer_agent()
chart_agent = _make_chart_agent()
evaluator_agent = _make_evaluator_agent()

# ==============================================================================
# 5. THIN CLIENT PROTOCOL (SSE) & ENDPOINT
# ==============================================================================


def create_event(
    event_type: str,
    label: str = "",
    state: str = "running",
    content: Any = None,
    view: str = "text",
) -> str:
    """
    Create a single Server-Sent Event line.

    The frontend expects:
    - `type`: semantic classification (status, log, artifact, final, ...)
    - `label`: short label for UI
    - `state`: running | complete | error
    - `content`: arbitrary JSON-serializable payload
    - `view`: rendering hint for the client (text, sql, json, etc.)
    """
    payload = {
        "type": event_type,
        "label": label,
        "state": state,
        "content": content,
        "view": view,
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/generate-chart-stream")
async def generate_chart_stream(request: QueryRequest) -> StreamingResponse:
    """
    Main Agentic BI pipeline exposed as an SSE endpoint.

    Stages:
    1. SQL drafting
    2. BI optimization
    3. Database execution
    4. Chart generation (React/Recharts)
    5. Code evaluation / auto-fix
    """

    async def event_generator():
        try:
            logger.info("Processing query: %s", request.query)

            # --- PHASE 1: DRAFTING ---
            yield create_event("status", "🧠 SQL Agent: Drafting...", "running")
            draft_res = await sql_agent.run(request.query)
            draft_sql = draft_res.output.sql_query
            yield create_event("artifact", "Draft SQL", content=draft_sql, view="sql")

            # --- PHASE 2: OPTIMIZATION ---
            yield create_event("status", "🕵️ BI Agent: Optimizing...", "running")
            review_prompt = (
                f"User query: {request.query}\n"
                f"Draft SQL:\n{draft_sql}\n"
                "Please improve and validate this query."
            )
            review_res = await bi_reviewer_agent.run(review_prompt)
            optimized_sql = review_res.output.optimized_sql

            yield create_event(
                "log",
                label="BI Changes",
                content=review_res.output.changes_made,
                view="text",
            )
            yield create_event(
                "artifact",
                "Optimized SQL",
                content=optimized_sql,
                view="sql",
            )

            if not review_res.output.is_safe:
                yield create_event("status", "❌ Query flagged as unsafe", "error")
                yield create_event(
                    "log",
                    content="BI reviewer marked the query as unsafe. Aborting execution.",
                )
                return

            # --- PHASE 3: EXECUTION ---
            yield create_event("status", "🗄️ Database: Fetching...", "running")
            db_data = inspector.execute_query(optimized_sql)

            if not db_data:
                yield create_event("status", "⚠️ No Data", "error")
                yield create_event("log", content="Query returned empty result.")
                return

            first_row = db_data[0]
            if isinstance(first_row, dict) and "error" in first_row:
                yield create_event("status", "❌ SQL Error", "error")
                yield create_event("log", content=first_row["error"])
                return

            yield create_event(
                "log",
                content=f"Fetched {len(db_data)} rows from the database.",
            )
            yield create_event(
                "artifact",
                "Data Preview",
                content=db_data[:5],
                view="json",
            )

            # --- PHASE 4: VISUALIZATION ---
            yield create_event(
                "status", "🎨 Chart Agent: Designing chart component...", "running"
            )
            chart_prompt = (
                f"User query: {request.query}\n"
                f"Data sample (first {min(len(db_data), 30)} rows):\n"
                f"{db_data[:30]}"
            )
            chart_res = await chart_agent.run(chart_prompt)
            raw_code = chart_res.output.code
            component_name = chart_res.output.component_name

            yield create_event(
                "log",
                content=f"Generated React component <{component_name} />.",
            )

            # --- PHASE 5: EVALUATION ---
            yield create_event("status", "🧐 Evaluator: Validating code...", "running")
            final_code = raw_code
            try:
                eval_res = await evaluator_agent.run(raw_code)
                if not eval_res.output.is_valid and eval_res.output.corrected_code:
                    final_code = eval_res.output.corrected_code
                    yield create_event("log", content="Applied syntax fixes.")
                else:
                    yield create_event("log", content="Syntax check passed.")
            except Exception as exc:
                logger.warning("Evaluator failed: %s", exc)
                yield create_event(
                    "log",
                    content="⚠️ Auto-fixer skipped due to evaluator error.",
                )

            # --- FINAL PAYLOAD ---
            yield create_event("status", "✅ Complete", "complete")
            final_payload = {
                "sql": optimized_sql,
                "data": db_data,
                "component_name": component_name,
                "react_code": final_code,
            }
            yield create_event("final", content=final_payload, view="json")

        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            yield create_event("status", "❌ System Error", "error")
            yield create_event("log", content=str(exc))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==============================================================================
# 6. ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logger.info("Starting Gemini-powered API on %s:%s", API_HOST, API_PORT)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
