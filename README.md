# IABISA - Intelligent Agentic Business Intelligence System for Healthcare Analytics

A comprehensive healthcare analytics platform combining AI-powered conversational interfaces with predictive modeling for Indonesian healthcare facilities (Faskes), diseases (Penyakit), and participants (Peserta).

---

## 📋 Table of Contents

- Overview
- Architecture
- Features
- Technology Stack
- Project Structure
- Installation
- Configuration
- Usage
- API Documentation
- Prediction Models
- Development
- Deployment
- Contributing

---

## 🎯 Overview

IABISA is an enterprise-grade Business Intelligence platform specifically designed for healthcare analytics in Indonesia. It combines:

- **Agentic AI System**: Natural language interface powered by Google Gemini for conversational analytics
- **Predictive Analytics**: Machine learning models (XGBoost, Prophet) for forecasting healthcare trends
- **Interactive Dashboards**: Drag-and-drop visualization builder with real-time data
- **Geographic Visualization**: Province-level mapping for healthcare distribution analysis

### Key Capabilities

1. **Healthcare Facilities Prediction** (Faskes)
   - FKRTL (Primary Healthcare Facilities)
   - Klinik Pratama (Primary Clinics)
   - Praktek Dokter (Doctor Practices)

2. **Disease Analysis** (Penyakit)
   - Case predictions by service type (RITL, RITP, RJTL, RJTP)
   - ICD-X code-based disease classification
   - Patient count forecasting

3. **Participant Analytics** (Peserta)
   - Geographic segmentation
   - Service class analysis
   - Demographic trends

---

## 🏗 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Dashboard   │  │   Chatbot    │  │     Maps     │     │
│  │   Builder    │  │   Interface  │  │ Visualization│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                    REST API / SSE Streaming
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Agentic AI Orchestrator                     │  │
│  │  • Intent Classification (Router Agent)               │  │
│  │  • SQL Generation & Optimization                      │  │
│  │  • Chart Code Generation                              │  │
│  │  • Prediction Parameter Extraction                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│  ┌────────────┬──────────────┴────────────┬─────────────┐  │
│  │   Faskes   │     Penyakit              │   Peserta   │  │
│  │ Prediction │    Prediction             │ Prediction  │  │
│  │  Module    │     Module                │   Module    │  │
│  └────────────┴───────────────────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌────────▼────────┐
│   PostgreSQL   │                          │ Supabase Storage│
│   Database     │                          │  (Model Repo)   │
│ (Supabase)     │                          │                 │
└────────────────┘                          └─────────────────┘
```

### Agent Workflow (Backend)

The system uses a **composable agent pipeline** with specialized agents:

1. **Router Agent**: Classifies intent (chat, SQL query, prediction, map)
2. **SQL Agent**: Generates PostgreSQL queries from natural language
3. **BI Review Agent**: Optimizes and validates SQL for security
4. **Chart Agent**: Generates Recharts configuration
5. **Map Agent**: Identifies geographic columns for visualization
6. **Prediction Parameter Agent**: Extracts forecasting parameters
7. **Summarizer Agent**: Creates human-readable summaries

---

## ✨ Features

### Frontend Features

#### 🎨 Dashboard Builder
- Drag-and-drop grid system with responsive breakpoints
- Pre-built chart templates (Line, Area, Bar, Pie)
- Real-time data binding
- Persistent layout storage via context
- Dark/light theme support

#### 💬 AI Chatbot Interface
- Natural language queries
- Streaming responses with step-by-step progress
- Multiple response types:
  - Text summaries
  - Interactive charts
  - Geographic maps
  - Prediction visualizations
- One-click "Add to Dashboard" for charts
- Conversation history

#### 🗺️ Geographic Visualization
- Indonesia province-level mapping
- Color-coded value representation
- Tooltip data display
- Fuzzy province name matching

### Backend Features

#### 🤖 Agentic AI System
- Multi-step reasoning with composable agents
- Context-aware query optimization
- Automatic intent classification
- Server-Sent Events (SSE) streaming
- Error recovery and retry logic

#### 📊 Prediction Engine
- **Multi-step forecasting**: Recursive predictions using historical data as features
- **Panel data modeling**: Entity-specific trends and seasonality
- **Feature engineering**: Lag features, rolling statistics, year-over-year growth
- **Hierarchical reconciliation**: Ensures consistency across prediction levels
- **Model caching**: Supabase-backed model storage with local cache

#### 🔒 Security
- SQL injection prevention via parameterized queries
- Read-only database access enforcement
- Query complexity analysis
- CORS configuration

---

## 🛠 Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript/JavaScript (JSX)
- **Styling**: Tailwind CSS
- **UI Components**: Material-UI, Lucide Icons
- **Charts**: Recharts
- **Drag & Drop**: dnd-kit
- **State Management**: React Context API
- **HTTP Client**: Fetch API

### Backend
- **Framework**: FastAPI
- **AI/LLM**: Pydantic-AI + Google Gemini (Flash Lite)
- **Database**: PostgreSQL (Supabase)
- **ORM**: psycopg3 (direct SQL)
- **ML Libraries**:
  - XGBoost 3.1.2
  - Prophet 1.2.1
  - scikit-learn 1.7.2
  - LightGBM 4.6.0
- **Storage**: Supabase Storage (model versioning)
- **Validation**: Pydantic

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Package Management**: 
  - Frontend: pnpm
  - Backend: uv (Astral)
- **Environment**: `.env` configuration
- **Deployment**: Docker-ready with multi-stage builds

---

## 📁 Project Structure

```
iabisa/
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.jsx                 # Main entry point
│   │   │   ├── loading.tsx              # Loading screen
│   │   │   ├── Area.jsx                 # Dashboard grid item
│   │   │   └── Grid.jsx                 # Grid layout manager
│   │   ├── components/
│   │   │   ├── dashboard-layout.jsx     # Split-pane layout
│   │   │   ├── dashboard-panel.tsx      # Dashboard container
│   │   │   ├── dashboard-content.jsx    # Chart templates
│   │   │   ├── chatbot-panel.jsx        # AI chat interface
│   │   │   ├── dynamic-chart.jsx        # Chart renderer
│   │   │   ├── map.jsx                  # Indonesia map
│   │   │   └── navbar.tsx               # Top navigation
│   │   ├── context/
│   │   │   └── dashboard-context.jsx    # Global state
│   │   ├── server/
│   │   │   └── db/
│   │   │       └── schema.ts            # Drizzle ORM schema
│   │   └── env.js                       # Environment validation
│   ├── Dockerfile
│   ├── package.json
│   └── next.config.js
│
├── backend/
│   ├── main.py                          # Agentic AI orchestrator
│   ├── api.py                           # Prediction API endpoints
│   ├── ui.py                            # Streamlit dev console
│   ├── supabase_storage.py              # Model storage client
│   │
│   ├── routers/
│   │   └── prediction.py                # Unified prediction router
│   │
│   ├── faskes/                          # Healthcare facilities
│   │   └── predict.py                   # XGBoost panel model
│   │
│   ├── penyakit/                        # Disease analytics
│   │   ├── predict.py                   # Disease forecasting
│   │   └── predict_new.py               # Updated model
│   │
│   ├── peserta/                         # Participant analytics
│   │   └── predict.py                   # Prophet + XGBoost hybrid
│   │
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── .python-version
│
├── compose.yaml                         # Docker Compose orchestration
└── .env.example                         # Environment template
```

---

## 🚀 Installation

### Prerequisites

- **Docker** >= 20.10 (recommended) or:
  - Node.js 20+ and pnpm
  - Python 3.12+
  - PostgreSQL 15+
- **Supabase account** (for database and model storage)
- **Google AI API key** (for Gemini)

### Quick Start (Docker)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd iabisa
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   cp backend/.env.example backend/.env
   cp frontend/.env.example frontend/.env
   ```

3. **Set required variables** (see Configuration)

4. **Start services**
   ```bash
   docker compose up -d
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

### Manual Installation

#### Backend

```bash
cd backend

# Install uv package manager
pip install uv

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build
pnpm start
```

---

## ⚙️ Configuration

### Backend Environment Variables (.env)

```bash
# Database (Supabase)
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Supabase Storage
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_BUCKET_NAME=healthkathon-models

# Google AI
GEMINI_API_KEY=your-gemini-api-key
LLM_MODEL_NAME=gemini-flash-lite-latest

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Frontend Environment Variables (.env)

```bash
# Database (for Drizzle ORM - optional)
DATABASE_URL=postgresql://user:pass@host:5432/db

# API Endpoint
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
```

### Supabase Setup

1. **Create a Supabase project** at https://supabase.com

2. **Create storage bucket**
   ```sql
   -- In Supabase SQL Editor
   insert into storage.buckets (id, name, public)
   values ('healthkathon-models', 'healthkathon-models', false);
   ```

3. **Upload trained models** (structure):
   ```
   healthkathon-models/
   ├── faskes/
   │   ├── model_fkrtl.pkl
   │   ├── model_klinik_pratama.pkl
   │   └── model_praktek_dokter.pkl
   ├── penyakit/
   │   ├── models_kasus_per_service.pkl
   │   ├── models_peserta_per_service.pkl
   │   ├── feature_names.pkl
   │   └── categorical_features.pkl
   └── peserta/
       ├── model_geo.pkl
       ├── model_kelas.pkl
       └── model_segmen.pkl
   ```

4. **Create database tables** (import your healthcare datasets)

---

## 📖 Usage

### Chatbot Queries

#### General Analytics
```
"Show me the top 10 provinces by healthcare facilities"
"How many patients were treated in 2023?"
"Compare FKRTL vs Klinik Pratama distribution"
```

#### Predictions
```
"Predict healthcare facilities in Bali for 2025"
"Forecast disease cases for RITL service next year"
"How many participants will there be in 2025?"
```

#### Visualizations
```
"Show me a map of healthcare facilities by province"
"Create a trend chart for patient growth"
"Visualize disease distribution across service types"
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Faskes Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "fkrtl",
    "start_year": 2025,
    "n_years": 3,
    "provinces": ["Bali", "DKI Jakarta"],
    "use_cache": true
  }'
```

#### Disease Prediction
```bash
curl -X POST http://localhost:8000/predict/penyakit \
  -H "Content-Type: application/json" \
  -d '{
    "service_type": "RITL",
    "top_n": 10,
    "months": 12,
    "use_cache": true
  }'
```

#### Participant Prediction
```bash
curl -X POST http://localhost:8000/predict/peserta \
  -H "Content-Type: application/json" \
  -d '{
    "segment_type": "geo",
    "months": 12,
    "use_cache": true
  }'
```

### Streamlit Dev Console

For testing the AI agents in isolation:

```bash
cd backend
streamlit run ui.py
```

Features:
- Direct API connection testing
- Step-by-step agent workflow visualization
- SQL query inspection
- Chart preview

---

## 🤖 Prediction Models

### Faskes (XGBoost Panel Model)

**Features**:
- Entity-specific statistics (mean, std)
- Time-series features (lag 1, lag 2)
- Rolling window statistics (2-year)
- Year-over-year growth
- Entity trend coefficients

**Training Data**: Historical facility counts by province (2014-2024)

**Output**: Predicted facility count by province and year

### Penyakit (Disease Prediction)

**Models**: Separate XGBoost models per service type (RITL, RITP, RJTL, RJTP)

**Features**:
- ICD-X category and subcategory
- Disease frequency
- Historical case logs (lag features)
- Service-specific trends
- Participant volume correlations

**Output**: Top N diseases with predicted case counts and participant counts

### Peserta (Participant Prediction)

**Models**: Hybrid approach
- **XGBoost**: For geographic and class segmentation
- **Prophet**: For time-series segments

**Features**:
- Geographic/demographic indicators
- Historical participant growth
- Seasonal patterns (Prophet)
- Hierarchical reconciliation across segments

**Output**: Predicted participant counts by segment and year

---

## 🧪 Development

### Running Tests

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
pnpm test
```

### Code Quality

```bash
# Frontend linting
cd frontend
pnpm lint

# Format code
pnpm format
```

### Database Migrations

```bash
cd frontend
pnpm db:push    # Push schema changes
pnpm db:studio  # Open Drizzle Studio
```

### Hot Reload

Both frontend and backend support hot reload in development mode:
- Frontend: Next.js Fast Refresh
- Backend: Uvicorn `--reload` flag

---

## 🐳 Deployment

### Docker Production Build

```bash
# Build images
docker compose -f compose.yaml build

# Run in production mode
docker compose -f compose.yaml up -d
```

### Environment Variables for Production

Update compose.yaml or use a `.env` file:

```yaml
services:
  backend:
    environment:
      - DATABASE_URL=${PROD_DATABASE_URL}
      - GEMINI_API_KEY=${PROD_GEMINI_KEY}
      - API_HOST=0.0.0.0
      - API_PORT=8000
  
  frontend:
    environment:
      - DATABASE_URL=${PROD_DATABASE_URL}
      - NEXT_PUBLIC_API_URL=https://your-api-domain.com
      - NODE_ENV=production
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🤝 Contributing

### Development Workflow

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes and test locally

3. Commit with descriptive messages
   ```bash
   git commit -m "feat: add province filtering to predictions"
   ```

4. Push and create a pull request

### Code Style

- **Frontend**: Follow Next.js/React best practices, use TypeScript where possible
- **Backend**: Follow PEP 8, use type hints
- **Commits**: Use conventional commit format

---

## 📄 License

This project is developed for the Healthkathon competition. Please refer to competition guidelines for usage restrictions.

---

## 👥 Team

**Developed by FWB Team**

For questions or support, please contact the development team.

---

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Next.js Documentation](https://nextjs.org/docs)
- [Pydantic-AI](https://ai.pydantic.dev)
- [Supabase Documentation](https://supabase.com/docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io)

---

**Built with ❤️ for Indonesian Healthcare Analytics**
