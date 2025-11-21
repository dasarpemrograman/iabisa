FROM python:3.12-slim

# 1. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. CRITICAL FIX: Install venv in a separate directory (/venv)
#    so it doesn't get overwritten by 'COPY . .' later.
ENV UV_PROJECT_ENVIRONMENT="/venv"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# 4. Add the new venv path to PATH
ENV PATH="/venv/bin:$PATH"

# 5. Copy application code
COPY . .

EXPOSE 8000

# 6. Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
