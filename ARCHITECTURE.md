# AI-ML Workflow Automation Platform - Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Browser)                                │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Vite)                            │ │
│  │                                                                      │ │
│  │  ├─ Uploader        ├─ Console (WebSocket)   ├─ Dashboard         │ │
│  │  ├─ Pipeline Editor ├─ Task List             ├─ Train Panel       │ │
│  │  └─ API Client      └─ Deploy Panel                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                HTTP/REST                     WebSocket
                    │                               │
┌───────────────────┴───────────────────────────────┴───────────────────────┐
│                         FastAPI Backend (Python)                           │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                        API Layer (app.py)                           │   │
│  │                                                                      │   │
│  │  ├─ /api/upload              ├─ /api/suggest/*                     │   │
│  │  ├─ /api/run_pipeline        ├─ /api/task/*                        │   │
│  │  ├─ /ws/logs                 └─ /api/logs/*                        │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Services                                │   │
│  │                                                                      │   │
│  │  ├─ Task Manager       ├─ Log Emitter         ├─ Log Store        │   │
│  │  ├─ Data Profiler      ├─ Suggestion Engine   ├─ Preprocessor     │   │
│  │  └─ WebSocket Broker                                               │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Infrastructure                          │   │
│  │                                                                      │   │
│  │  ├─ Database (SQLAlchemy)  ├─ Auth (JWT)     ├─ Config Manager   │   │
│  │  └─ Log Emitter/Store                                              │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                               │
            ┌───────┴──────┐               ┌────────┴────────┐
            │              │               │                 │
    ┌───────▼──────┐  ┌───▼────────┐  ┌──▼──────────┐  ┌──▼──────────┐
    │  PostgreSQL  │  │   Redis    │  │  Artifacts  │  │    Logs     │
    │   Database   │  │   Cache    │  │   Storage   │  │   Storage   │
    └──────────────┘  └────────────┘  └─────────────┘  └─────────────┘
```

## Component Details

### Frontend Layer
**Technology**: React 18 + Vite + Tailwind CSS

**Components**:
- `Uploader`: Drag-and-drop file upload with validation
- `PipelineEditor`: Interactive step selection and configuration
- `Console`: Real-time log streaming with WebSocket
- `Dashboard`: Metrics and activity overview
- `TaskList`: Task status tracking
- `TrainPanel`: Model selection and training control
- `DeployPanel`: Deployment platform selection

**Communication**:
- HTTP/REST for CRUD operations
- WebSocket for real-time log streaming
- React Query for data caching

### Backend Layer
**Technology**: FastAPI (Python 3.10+) + Async/Await

**API Endpoints**:
```
Dataset Management:
  POST   /api/upload              - Upload dataset
  GET    /api/datasets/{id}       - Get dataset info

AI Suggestions:
  GET    /api/suggest/pipeline    - Get preprocessing steps
  GET    /api/suggest/models      - Get model recommendations

Pipeline Execution:
  POST   /api/run_pipeline        - Execute preprocessing

Task Management:
  GET    /api/task/{id}/status    - Get task status
  GET    /api/tasks               - List tasks
  POST   /api/cancel/{id}         - Cancel task

Log Streaming:
  WS     /ws/logs                 - WebSocket streaming
  GET    /api/logs/{id}           - Get logs (JSON)
  GET    /api/logs/{id}.txt       - Download logs (text)

Models:
  GET    /api/models              - List models

System:
  GET    /                        - Root info
  GET    /health                  - Health check
  GET    /docs                    - OpenAPI docs
```

### Service Layer

**Task Manager** (`services/task_manager.py`):
- Create and track background tasks
- Update task status (pending, running, completed, failed)
- Emit structured logs

**Data Profiler** (`services/profiler.py`):
- Analyze dataset structure
- Generate column-level statistics
- Detect target column
- Identify data quality issues

**Suggestion Engine** (`services/suggestor.py`):
- Rule-based pipeline generation
- Model recommendations
- Confidence scoring
- Rationale generation

**Preprocessing Service** (`services/preprocess.py`):
- Execute pipeline steps
- Stream logs in real-time
- Handle imputation, encoding, scaling
- Save processed datasets

**WebSocket Broker** (`ws_log_broker.py`):
- Manage WebSocket connections
- Broadcast logs to clients
- Replay historical logs
- Handle reconnections

### Core Infrastructure

**Log Emitter** (`core/log_emitter.py`):
- Structured log entry creation
- WebSocket subscription management
- Async and sync log emission

**Log Store** (`core/log_store.py`):
- In-memory log caching (1000 entries/task)
- Persistent JSONL file storage
- Log retrieval and formatting

**Database** (`core/database.py`):
- SQLAlchemy ORM models
- Connection management
- Schema: Dataset, Task, Model, Deployment, DriftAlert

**Auth** (`core/auth.py`):
- JWT token generation
- Password hashing (bcrypt)
- Token verification

**Config** (`config.py`):
- Environment variable management
- Pydantic settings
- Database URLs, API keys, etc.

### Data Layer

**PostgreSQL**:
- Primary data storage
- Tables: datasets, tasks, models, deployments, drift_alerts
- Persistent storage for all entities

**Redis**:
- Session cache
- Task queue broker (Celery ready)
- Transient state management

**File Storage**:
- Artifacts: Processed datasets, model files
- Logs: JSONL format for persistence
- Uploads: Original dataset files

## Data Flow

### Dataset Upload Flow
```
1. User uploads file → Frontend Uploader
2. POST /api/upload → Backend API
3. Save file → File Storage
4. Load with pandas → Data Profiler
5. Generate profile → Response to Frontend
6. Display stats → Frontend Dashboard
```

### Pipeline Suggestion Flow
```
1. User requests suggestions → Frontend
2. GET /api/suggest/pipeline → Backend API
3. Load dataset → Pandas DataFrame
4. Analyze profile → Suggestion Engine
5. Generate steps with confidence → Rules + AI
6. Return suggestions → Frontend Pipeline Editor
7. Display interactive cards → User
```

### Pipeline Execution Flow
```
1. User selects steps & clicks Run → Frontend
2. POST /api/run_pipeline → Backend API
3. Create task → Task Manager
4. Start async execution → Background
5. Connect WebSocket → ws/logs
6. Execute steps → Preprocessing Service
   ├─ Step 1: Emit logs → Log Emitter
   ├─ Step 2: Emit logs → Log Emitter
   └─ Step N: Emit logs → Log Emitter
7. Forward logs → WebSocket Broker
8. Render logs → Frontend Console
9. Update task status → Database
10. Display completion → User
```

### Log Streaming Flow
```
1. Task starts → Backend
2. Service emits log → Log Emitter
3. Store in memory → Log Store
4. Write to file → JSONL file
5. Publish to queue → Log Emitter subscriptions
6. Forward to WebSocket → WebSocket Broker
7. Send to client → WebSocket connection
8. Render in console → Frontend Console component
```

## Deployment Architecture

### Docker Compose (Development)
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Frontend   │  │   Backend    │  │   Postgres   │  │
│  │   (Node)     │  │  (FastAPI)   │  │              │  │
│  │  Port 5173   │  │  Port 8000   │  │  Port 5432   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐                                        │
│  │    Redis     │                                        │
│  │  Port 6379   │                                        │
│  └──────────────┘                                        │
│                                                           │
│  Volumes: postgres_data, redis_data, ./artifacts         │
└─────────────────────────────────────────────────────────┘
```

### Production Deployment Options

**Hugging Face Spaces**:
```
Backend Docker → HF Container Registry → HF Space
Frontend → Static hosting on HF
Database → External PostgreSQL (e.g., Supabase)
```

**Google Cloud Platform**:
```
Backend → Cloud Run (serverless containers)
Frontend → Cloud Storage + CDN
Database → Cloud SQL (PostgreSQL)
Redis → Memorystore
```

**AWS**:
```
Backend → ECS Fargate (containers)
Frontend → S3 + CloudFront
Database → RDS PostgreSQL
Redis → ElastiCache
```

## Security Considerations

**Implemented**:
- ✅ Environment variable configuration
- ✅ JWT token support
- ✅ Password hashing (bcrypt)
- ✅ CORS middleware with origin restrictions
- ✅ Input validation (Pydantic)

**Production Recommendations**:
- Enable authentication on all endpoints
- Use HTTPS/WSS in production
- Implement rate limiting
- Add request validation
- Set up database backups
- Monitor for security alerts

## Scalability Notes

**Current Setup**: Single-node development

**Scaling Options**:
1. **Horizontal Scaling**: Deploy multiple backend instances behind load balancer
2. **Database**: Use connection pooling, read replicas
3. **Caching**: Redis cluster for distributed caching
4. **Task Queue**: Enable Celery with multiple workers
5. **WebSocket**: Use Redis pub/sub for multi-instance log broadcasting
6. **Storage**: S3 or equivalent for artifacts and logs

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 | UI components |
| | Vite | Build tool |
| | Tailwind CSS | Styling |
| | React Query | Data fetching |
| Backend | FastAPI | API framework |
| | Python 3.10+ | Programming language |
| | Uvicorn | ASGI server |
| | WebSockets | Real-time communication |
| Database | PostgreSQL | Persistent storage |
| | SQLAlchemy | ORM |
| Cache | Redis | In-memory cache |
| ML | Pandas | Data manipulation |
| | NumPy | Numerical computing |
| | scikit-learn | ML algorithms |
| | XGBoost | Gradient boosting |
| DevOps | Docker | Containerization |
| | Docker Compose | Multi-container orchestration |
| | GitHub Actions | CI/CD |

## File Structure

```
ai-ml-workflow-automation/
├── backend/
│   ├── app.py                    # Main FastAPI application
│   ├── config.py                 # Configuration management
│   ├── requirements.txt          # Python dependencies
│   ├── ws_log_broker.py         # WebSocket log broker
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication utilities
│   │   ├── database.py          # Database models and connection
│   │   ├── log_emitter.py       # Log emission system
│   │   └── log_store.py         # Log storage system
│   ├── services/
│   │   ├── __init__.py
│   │   ├── profiler.py          # Dataset profiling
│   │   ├── suggestor.py         # AI suggestion engine
│   │   ├── preprocess.py        # Preprocessing service
│   │   └── task_manager.py      # Task management
│   ├── tasks/
│   │   └── __init__.py          # Async task definitions
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py          # Test configuration
│       ├── test_api.py          # API tests
│       ├── test_profiler.py     # Profiler tests
│       └── test_suggestor.py    # Suggestion engine tests
├── frontend/
│   ├── index.html               # HTML entry point
│   ├── package.json             # NPM dependencies
│   ├── vite.config.js           # Vite configuration
│   ├── tailwind.config.js       # Tailwind configuration
│   ├── postcss.config.js        # PostCSS configuration
│   └── src/
│       ├── main.jsx             # React entry point
│       ├── App.jsx              # Main App component
│       ├── index.css            # Global styles
│       ├── api/
│       │   ├── client.js        # API client
│       │   └── websocket.js     # WebSocket client
│       └── components/
│           ├── Console/
│           │   └── Console.jsx
│           ├── Uploader/
│           │   └── Uploader.jsx
│           ├── PipelineEditor/
│           │   └── PipelineEditor.jsx
│           ├── Dashboard/
│           │   └── Dashboard.jsx
│           ├── TaskList/
│           │   └── TaskList.jsx
│           ├── TrainPanel/
│           │   └── TrainPanel.jsx
│           └── DeployPanel/
│               └── DeployPanel.jsx
├── infra/
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── Dockerfile.backend       # Backend container
│   ├── Dockerfile.frontend      # Frontend container
│   └── k8s/                     # Kubernetes manifests (placeholder)
├── scripts/
│   ├── run_demo.sh              # Quick start demo
│   ├── validate_setup.sh        # Setup validation
│   ├── deploy_to_hf.sh          # Hugging Face deployment
│   ├── deploy_to_gcp.sh         # GCP deployment
│   └── deploy_to_aws.sh         # AWS deployment
├── demo/
│   ├── sample_dataset.csv       # Demo dataset
│   └── demo-console.log         # Sample console output
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
└── PROJECT_STATUS.md            # Implementation status
```

---

**Architecture Version**: 1.0
**Last Updated**: 2025-12-26
**Status**: Production-Ready Core Platform
