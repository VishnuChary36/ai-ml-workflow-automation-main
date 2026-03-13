# AI-ML Workflow Automation Platform

A production-ready full-stack platform that implements the complete ML lifecycle with **live console streaming**, AI-powered suggestions, **storytelling dashboards**, and one-click deployment capabilities.

---

## 📋 Changelog & Recent Updates  

### ✅ Fully Implemented Features (December 2025)

#### 📊 NEW: Data Analytics Dashboard (Power BI-Style)

| Feature                    | Description                                                               | Status      |
| -------------------------- | ------------------------------------------------------------------------- | ----------- |
| **Executive Summary**      | AI-generated narrative explaining key insights from your data             | ✅ Complete |
| **KPI Cards**              | Key metrics including completeness, feature counts, and class balance     | ✅ Complete |
| **Data Quality Scorecard** | Column-level quality scores with missing value analysis                   | ✅ Complete |
| **Correlation Heatmap**    | Interactive correlation matrix with color-coded relationships             | ✅ Complete |
| **Distribution Charts**    | Histograms, bar charts, and pie charts for all features                   | ✅ Complete |
| **Insights Engine**        | Auto-generated insights about outliers, correlations, and class imbalance | ✅ Complete |
| **Recommendations**        | AI-powered recommendations for model selection and feature engineering    | ✅ Complete |
| **Storytelling Format**    | Dashboard designed for data analysts with narrative explanations          | ✅ Complete |

**New Dashboard Endpoints:**

```
GET    /api/dashboard/{task_id}         - Full analytics dashboard with all charts
GET    /api/dashboard/{task_id}/summary - Quick summary for initial load
GET    /api/processed-datasets          - List all processed datasets available
```

**Dashboard Access:**

- Navigate to `/dashboard` in the frontend after preprocessing completes
- Click "View Data Dashboard" button after pipeline execution
- Auto-detects target column or select from dropdown

#### 🔧 Backend Enhancements

| Feature                      | Description                                                                                                        | Status      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------- |
| **Model Training Service**   | Complete training pipeline with XGBoost, Random Forest, Logistic/Linear Regression, and Gradient Boosting support  | ✅ Complete |
| **Model Deployment Service** | Full deployment package generation for local, Docker, and cloud platforms                                          | ✅ Complete |
| **Visualization Service**    | Comprehensive model visualizations including confusion matrix, ROC curves, feature importance, and learning curves | ✅ Complete |
| **Explainability Service**   | SHAP and LIME-based model explanations with feature importance analysis                                            | ✅ Complete |
| **Drift Monitoring Service** | PSI, KL divergence, and ADWIN-based drift detection for production monitoring                                      | ✅ Complete |
| **Dashboard Service**        | Power BI-style storytelling dashboard with insights and recommendations                                            | ✅ Complete |
| **Live Prediction API**      | Real-time predictions with preprocessing pipeline preservation                                                     | ✅ Complete |

#### 🖥️ Frontend Components

| Component                    | Description                                                    | Status      |
| ---------------------------- | -------------------------------------------------------------- | ----------- |
| **Data Analytics Dashboard** | Power BI-style storytelling dashboard with insights            | ✅ Complete |
| **Visualization Dashboard**  | Interactive charts for model performance metrics with Recharts | ✅ Complete |
| **Explainability Panel**     | Visual feature importance and SHAP value displays              | ✅ Complete |
| **Deployment Panel**         | One-click deployment interface with progress streaming         | ✅ Complete |
| **Train Panel**              | Model selection and training configuration UI                  | ✅ Complete |

#### 🔌 New API Endpoints

```
POST   /api/train_model              - Train model with live progress streaming
POST   /api/deploy_model             - Deploy model to local/docker/cloud
POST   /api/predict/{model_id}       - Make live predictions with deployed model
GET    /api/predict/{model_id}/info  - Get prediction API documentation and examples
GET    /api/deployed-models          - List all deployed models
GET    /api/dashboard/{task_id}      - Get full data analytics dashboard
GET    /api/dashboard/{task_id}/summary - Quick dashboard summary
GET    /api/processed-datasets       - List processed datasets for dashboard
GET    /api/deployment/{id}          - Get deployment details
GET    /api/deployment/{id}/download - Download deployment package
GET    /api/model/{id}/deployments   - List deployments for a model
GET    /api/visualizations/{id}      - Get model visualizations (auto-generated)
GET    /api/explainability/{id}      - Get SHAP/LIME explanations
POST   /api/start_drift_monitoring   - Start drift monitoring for deployed model
```

#### 📁 New Service Files

- [services/dashboard.py](backend/services/dashboard.py) - Power BI-style analytics dashboard generation
- [services/training.py](backend/services/training.py) - Complete model training with metrics logging
- [services/deployment.py](backend/services/deployment.py) - Deployment package generation
- [services/visualization.py](backend/services/visualization.py) - Chart and graph generation
- [services/explainability.py](backend/services/explainability.py) - SHAP/LIME integration
- [services/drift_monitoring.py](backend/services/drift_monitoring.py) - Production drift detection

#### 🎨 New Frontend Components

- [Dashboard/DataDashboard.jsx](frontend/src/components/Dashboard/DataDashboard.jsx) - Power BI-style analytics dashboard
- [Visualization/](frontend/src/components/Visualization/) - Model performance visualization dashboard
- [Explainability/](frontend/src/components/Explainability/) - Feature importance and explanation views
- [Deployment/](frontend/src/components/Deployment/) - Deployment management interface

### 🏗️ Architecture Improvements

1. **On-Demand Generation**: Visualizations and explainability are generated on-demand if not cached
2. **Caching System**: Generated artifacts are saved to `./artifacts/` for faster subsequent access
3. **Async Task Execution**: All long-running tasks run in background with real-time WebSocket streaming
4. **Error Resilience**: Graceful fallbacks when model files or datasets are not found

---

### 🔐 NEW: Security & Infrastructure Enhancements (December 2025)

#### Authentication & Role-Based Access Control (RBAC)

| Feature                      | Description                                                            | Status      |
| ---------------------------- | ---------------------------------------------------------------------- | ----------- |
| **JWT Authentication**       | Secure token-based authentication with configurable expiry             | ✅ Complete |
| **API Key Support**          | Alternative authentication method for programmatic access              | ✅ Complete |
| **Role-Based Access**        | 5 roles: Admin, Data Scientist, ML Engineer, Viewer, API User          | ✅ Complete |
| **Fine-Grained Permissions** | 11 permissions covering datasets, models, deployments, and predictions | ✅ Complete |
| **Auth Middleware**          | Automatic route protection with role/permission enforcement            | ✅ Complete |
| **Protected Endpoints**      | `/api/predict/*`, `/api/deploy_model`, `/api/deployment/*` secured     | ✅ Complete |

**New Auth Endpoints:**

```
POST   /api/auth/login     - User login with username/password
POST   /api/auth/token     - Get JWT token
GET    /api/auth/me        - Get current user info
POST   /api/auth/api-key   - Generate API key (Admin only)
```

**RBAC Roles & Permissions:**
| Role | Permissions |
|------|-------------|
| Admin | Full access to all operations |
| Data Scientist | Dataset CRUD, model training, predictions |
| ML Engineer | Model operations, deployments, monitoring |
| Viewer | Read-only access to datasets and models |
| API User | Prediction endpoints only |

#### Dedicated Inference Service

| Feature                       | Description                                     | Status      |
| ----------------------------- | ----------------------------------------------- | ----------- |
| **Standalone FastAPI App**    | Separate service optimized for model serving    | ✅ Complete |
| **Health & Readiness Probes** | `/health` and `/ready` endpoints for Kubernetes | ✅ Complete |
| **Batch Predictions**         | `/predict/batch` for high-throughput inference  | ✅ Complete |
| **Model Store**               | Efficient model loading and caching             | ✅ Complete |
| **Metrics Collection**        | Request latency and throughput tracking         | ✅ Complete |
| **Docker Image**              | Multi-stage optimized Dockerfile for inference  | ✅ Complete |

**Inference Service Endpoints:**

```
POST   /predict           - Single prediction
POST   /predict/batch     - Batch predictions (up to 1000)
GET    /health            - Health check
GET    /ready             - Readiness probe (model loaded)
```

**New Files:**

- [backend/inference_app.py](backend/inference_app.py) - Standalone inference service
- [infra/Dockerfile.inference](infra/Dockerfile.inference) - Optimized Docker build
- [backend/requirements.inference.txt](backend/requirements.inference.txt) - Minimal dependencies

#### Input/Output Contracts (Pydantic Schemas)

| Schema                              | Purpose                                  | Status      |
| ----------------------------------- | ---------------------------------------- | ----------- |
| **PredictRequest/Response**         | Prediction API contracts with validation | ✅ Complete |
| **BatchPredictRequest/Response**    | Batch prediction contracts               | ✅ Complete |
| **LoginRequest/Response**           | Authentication contracts                 | ✅ Complete |
| **DeployModelRequest/Response**     | Deployment API contracts                 | ✅ Complete |
| **DriftMonitoringRequest/Response** | Monitoring contracts                     | ✅ Complete |
| **HealthResponse/ReadyResponse**    | Service status contracts                 | ✅ Complete |
| **Contract Tests**                  | 33 pytest tests validating all schemas   | ✅ Complete |

**New Schema Files:**

- [backend/schemas/auth.py](backend/schemas/auth.py) - Authentication schemas
- [backend/schemas/prediction.py](backend/schemas/prediction.py) - Prediction schemas
- [backend/schemas/deployment.py](backend/schemas/deployment.py) - Deployment schemas
- [backend/schemas/monitoring.py](backend/schemas/monitoring.py) - Monitoring schemas
- [backend/schemas/common.py](backend/schemas/common.py) - Shared response schemas
- [backend/tests/test_contracts.py](backend/tests/test_contracts.py) - Contract validation tests

#### Secrets Management

| Feature                   | Description                                                               | Status      |
| ------------------------- | ------------------------------------------------------------------------- | ----------- |
| **Multi-Backend Support** | AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, Azure Key Vault | ✅ Complete |
| **Environment Variables** | Fallback to env vars for local development                                | ✅ Complete |
| **Lazy Loading**          | Secrets fetched on-demand to avoid circular imports                       | ✅ Complete |
| **Caching**               | Secrets cached to reduce API calls                                        | ✅ Complete |
| **Migration Guide**       | Documentation for transitioning to cloud secrets                          | ✅ Complete |

**New Files:**

- [backend/core/secrets.py](backend/core/secrets.py) - Unified secrets management
- [infra/secret_manager_integration.md](infra/secret_manager_integration.md) - Migration guide

**Usage:**

```python
from core.secrets import secrets

# Automatically uses configured backend (AWS, GCP, Vault, or env)
db_password = secrets.get_secret("database/password")
api_key = secrets.get_secret("external-api/key")
```

#### CI/CD Pipeline with Safe Rollout

| Feature                     | Description                                           | Status      |
| --------------------------- | ----------------------------------------------------- | ----------- |
| **GitHub Actions Workflow** | Automated testing, building, and deployment           | ✅ Complete |
| **Multi-Stage Pipeline**    | Test → Build → Staging → Production                   | ✅ Complete |
| **Canary Deployments**      | Gradual rollout (10% → 50% → 100%) with health checks | ✅ Complete |
| **Blue-Green Deployments**  | Zero-downtime deployments with instant rollback       | ✅ Complete |
| **Semantic Versioning**     | Image tags: `model:vX.Y.Z+<sha>`                      | ✅ Complete |
| **Slack Notifications**     | Deployment status notifications                       | ✅ Complete |

**Pipeline Stages:**

```
1. Test      - Run pytest, linting, type checks
2. Build     - Build Docker images (backend, inference, frontend)
3. Staging   - Deploy to staging environment
4. Canary    - Gradual production rollout with monitoring
5. Blue-Green - Full production deployment with traffic shift
6. Notify    - Slack notification on success/failure
```

**New File:**

- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - Complete CI/CD pipeline

#### Rate Limiting & Abuse Protection

| Feature                       | Description                                 | Status      |
| ----------------------------- | ------------------------------------------- | ----------- |
| **Redis-Based Rate Limiting** | Sliding window algorithm with Redis backend | ✅ Complete |
| **In-Memory Fallback**        | Automatic fallback when Redis unavailable   | ✅ Complete |
| **Per-Route Limits**          | Configurable limits per API endpoint        | ✅ Complete |
| **Rate Limit Decorator**      | Easy-to-use decorator for custom limits     | ✅ Complete |
| **WAF Documentation**         | Cloudflare and AWS WAF configuration guides | ✅ Complete |

**Default Rate Limits:**
| Endpoint | Limit |
|----------|-------|
| `/api/predict` | 200 requests/minute |
| `/api/predict/batch` | 50 requests/minute |
| `/api/upload` | 10 requests/minute |
| `/api/train` | 5 requests/minute |
| Default | 100 requests/minute |

**New Files:**

- [backend/core/rate_limiter.py](backend/core/rate_limiter.py) - Rate limiting middleware
- [infra/waf_rules.md](infra/waf_rules.md) - WAF configuration documentation

---

### � NEW: End-to-End MLOps Pipeline (January 2026)

The platform now includes a fully compliant MLOps pipeline with production-grade capabilities. See [MLOPS.md](MLOPS.md) for complete documentation.

#### Model Tracking & Registry (`backend/mlops/`)

| Feature                   | Description                                                               | Status      |
| ------------------------- | ------------------------------------------------------------------------- | ----------- |
| **MLflow Integration**    | Experiment tracking, run management, artifact logging                     | ✅ Complete |
| **Model Registry**        | Version control with stage transitions (Staging/Production/Archived)      | ✅ Complete |
| **Experiment Management** | Run comparison, best model detection, reporting                           | ✅ Complete |
| **Feature Store**         | Feature set registration, online/offline serving, point-in-time retrieval | ✅ Complete |

#### Orchestrated Pipelines (`backend/pipelines/`)

| Feature                 | Description                                  | Status      |
| ----------------------- | -------------------------------------------- | ----------- |
| **Training Pipeline**   | End-to-end automated training workflow       | ✅ Complete |
| **Data Pipeline**       | Data loading, validation, preprocessing      | ✅ Complete |
| **Deployment Pipeline** | Model deployment with rollback support       | ✅ Complete |
| **Monitoring Pipeline** | Automated drift detection and alerting       | ✅ Complete |
| **Pipeline Scheduler**  | Cron/interval scheduling with job management | ✅ Complete |

#### Production Inference Service (`backend/inference/`)

| Feature                | Description                                                | Status      |
| ---------------------- | ---------------------------------------------------------- | ----------- |
| **Model Predictor**    | LRU caching, batch predictions, preprocessing              | ✅ Complete |
| **A/B Testing Router** | Traffic splitting, canary deployments, conversion tracking | ✅ Complete |
| **Prometheus Metrics** | Counter, Gauge, Histogram for observability                | ✅ Complete |
| **Inference Server**   | FastAPI with /health, /ready, /metrics, /predict endpoints | ✅ Complete |

#### Monitoring & Drift Detection (`backend/monitoring/`)

| Feature                    | Description                                            | Status      |
| -------------------------- | ------------------------------------------------------ | ----------- |
| **Drift Detection**        | PSI, KL/JS divergence, KS test, Chi-Square, ADWIN      | ✅ Complete |
| **Performance Monitoring** | Real-time metrics, degradation detection, trends       | ✅ Complete |
| **Alert Manager**          | Rule-based alerting with Email/Slack/Webhook/PagerDuty | ✅ Complete |
| **Monitoring Dashboard**   | Aggregated health scores and reports                   | ✅ Complete |

#### CI/CD Automation (`backend/cicd/`)

| Feature                 | Description                                          | Status      |
| ----------------------- | ---------------------------------------------------- | ----------- |
| **Model Validation**    | Schema, performance, latency, reproducibility checks | ✅ Complete |
| **Deployment Manager**  | Canary, blue-green, rolling deployment strategies    | ✅ Complete |
| **Rollback Manager**    | Safe rollback with version tracking                  | ✅ Complete |
| **MLOps GitHub Action** | Automated training, validation, deployment workflow  | ✅ Complete |

#### Security Enhancements (`backend/security/`)

| Feature              | Description                                     | Status      |
| -------------------- | ----------------------------------------------- | ----------- |
| **Audit Logging**    | Tamper-evident logging with chain verification  | ✅ Complete |
| **Input Validation** | SQL injection/XSS prevention, data sanitization | ✅ Complete |
| **Security Headers** | CSP, HSTS, X-Frame-Options middleware           | ✅ Complete |

**New MLOps Files:**

- [backend/mlops/](backend/mlops/) - Model tracking, registry, feature store
- [backend/pipelines/](backend/pipelines/) - Prefect tasks and flows
- [backend/inference/](backend/inference/) - Production inference service
- [backend/monitoring/](backend/monitoring/) - Drift detection and alerting
- [backend/cicd/](backend/cicd/) - Validation and deployment management
- [backend/security/](backend/security/) - Audit logging and input validation
- [.github/workflows/mlops.yml](.github/workflows/mlops.yml) - MLOps CI/CD pipeline
- [MLOPS.md](MLOPS.md) - Complete MLOps documentation

---

### �📊 Supported Model Types

| Algorithm                  | Classification | Regression |
| -------------------------- | :------------: | :--------: |
| Random Forest              |       ✅       |     ✅     |
| XGBoost                    |       ✅       |     ✅     |
| Logistic/Linear Regression |       ✅       |     ✅     |
| Gradient Boosting          |       ✅       |     ✅     |

### 📈 Visualization Types

- **Confusion Matrix** - Classification performance heatmap
- **ROC Curve** - Receiver Operating Characteristic analysis
- **Precision-Recall Curve** - For imbalanced datasets
- **Feature Importance** - Bar charts showing feature contributions
- **Learning Curves** - Training vs validation performance
- **Residual Plots** - Regression error analysis
- **Prediction Distribution** - Actual vs predicted comparisons

### 🔍 Explainability Methods

- **SHAP Values** - Feature impact on predictions (summary, bar, and force plots)
- **Permutation Importance** - Feature importance via shuffling
- **Feature Importance** - Built-in model feature rankings
- **Decision Paths** - Tree-based model decision visualization

---

## 🌟 Features

- **� End-to-End MLOps**: Complete pipeline with model registry, orchestration, inference, monitoring, and CI/CD
- **📊 Data Analytics Dashboard**: Power BI/Excel-style storytelling dashboard with AI-generated insights
- **🔐 Authentication & RBAC**: JWT/API key auth with role-based access control (5 roles, 11 permissions)
- **🚀 Dedicated Inference Service**: Standalone model serving with A/B testing and batch predictions
- **📈 Monitoring & Alerting**: Real-time drift detection with multi-channel alerts (Slack, Email, PagerDuty)
- **📋 Strict API Contracts**: Pydantic schemas with 33 contract tests ensuring API consistency
- **🔑 Secrets Management**: Multi-backend support (AWS, GCP, Vault, Azure) with caching
- **🔄 CI/CD Pipeline**: Canary and blue-green deployments with automated testing
- **⚡ Rate Limiting**: Redis-based per-route rate limiting with WAF integration
- **AI-Powered Pipeline Suggestions**: Automatically suggests preprocessing steps based on dataset profiling
- **Live Console Streaming**: Real-time log streaming via WebSocket with color-coded output
- **Interactive Pipeline Editor**: Visual pipeline builder with preview and editing capabilities
- **Background Task Management**: Robust async task execution with status tracking
- **Model Training**: Support for XGBoost, Random Forest, and other popular algorithms
- **One-Click Deployment**: Deploy models to Docker, Hugging Face, AWS, or GCP
- **Live Prediction API**: Ready-to-use prediction endpoints with code snippets
- **Drift Monitoring**: Continuous monitoring with PSI and ADWIN detection
- **Full API Documentation**: Auto-generated OpenAPI docs at `/docs`

## 🏗️ Architecture

### Backend (Python/FastAPI)

- **FastAPI** for high-performance async API
- **WebSocket** for real-time log streaming
- **SQLAlchemy** for database ORM
- **PostgreSQL** for persistent storage
- **Redis** for caching, task queue, and rate limiting
- **Celery** (optional) for distributed task execution
- **JWT + API Keys** for authentication
- **RBAC Middleware** for authorization
- **Pydantic v2** for request/response validation

### Inference Service (Standalone)

- **FastAPI** optimized for model serving
- **Model Store** for efficient model loading
- **Health/Readiness Probes** for Kubernetes
- **Batch Prediction** support
- **Metrics Collection** for monitoring

### Security & Infrastructure

- **Secrets Manager** abstraction (AWS, GCP, Vault, Azure)
- **Rate Limiting** middleware with Redis backend
- **WAF Integration** (Cloudflare, AWS WAF)
- **CI/CD Pipeline** with canary/blue-green deployments

### Frontend (React/Vite)

- **React 18** with functional components and hooks
- **Tailwind CSS** for styling
- **React Query** for data fetching and caching
- **WebSocket** client for live log streaming
- **Recharts** for training metrics visualization

### ML Stack

- pandas, numpy, scikit-learn
- XGBoost, LightGBM
- TensorFlow/Keras (optional)
- SHAP and LIME for explainability
- River for drift detection

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- Node.js 18+ (for local frontend development without Docker)
- Python 3.10+ (for local backend development without Docker)

### One-Command Setup (Recommended)

Clone the repository and run the included setup script — it checks prerequisites, copies the environment file, builds all Docker images, starts every service, and verifies the application is healthy:

```bash
git clone https://github.com/VishnuChary36/ai-ml-workflow-automation-main.git
cd ai-ml-workflow-automation-main

# Make the script executable (Linux/macOS)
chmod +x scripts/setup.sh

# Docker mode — starts the full stack automatically
./scripts/setup.sh

# --- OR --- local venv mode (no Docker required for the app itself)
./scripts/setup.sh --local
```

Once finished:

| Service    | URL                          |
| ---------- | ---------------------------- |
| Frontend   | http://localhost:5173        |
| Backend    | http://localhost:8000        |
| API Docs   | http://localhost:8000/docs   |

To stop: `docker compose -f infra/docker-compose.yml down`

---

### Manual Setup — Docker Compose

```bash
git clone https://github.com/VishnuChary36/ai-ml-workflow-automation-main.git
cd ai-ml-workflow-automation-main

cp .env.example .env
docker compose -f infra/docker-compose.yml up --build
```

### Manual Setup — Local Development

#### Backend

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (via Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=mlworkflow postgres:15
docker run -d -p 6379:6379 redis:7-alpine

# Run the backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### GitHub Actions CI

Every push and pull request triggers the **Application Setup Check** workflow (`.github/workflows/setup-check.yml`) which:

1. Installs all backend Python dependencies and verifies imports.
2. Installs all frontend Node.js dependencies and builds the production bundle.
3. Starts the full stack with Docker Compose and health-checks `GET /health`.

## 📖 User Flow

1. **Upload Dataset**: Drag & drop a CSV, Excel, or JSON file
2. **AI Profiling**: Backend automatically profiles the dataset and suggests preprocessing steps
3. **Edit Pipeline**: Review and modify suggested steps in the interactive pipeline editor
4. **Preview Console**: Click "Preview" on any step to see simulated console output
5. **Run Pipeline**: Execute the pipeline and watch real-time logs stream to the console
6. **Model Suggestion**: AI suggests optimal models based on the problem type
7. **Train Model**: Start training with live progress and metrics streaming
8. **View Visualizations**: Explore model performance through interactive charts (confusion matrix, ROC curves, feature importance)
9. **Explore Explainability**: Understand model decisions with SHAP values and feature importance analysis
10. **Deploy**: One-click deployment with live build logs and downloadable deployment packages
11. **Monitor**: Continuous drift monitoring with PSI and ADWIN detection for production models

## 🔌 API Endpoints

### Dataset Management

- `POST /api/upload` - Upload and profile dataset
- `GET /api/datasets/{dataset_id}` - Get dataset information

### AI Suggestions

- `GET /api/suggest/pipeline` - Get preprocessing pipeline suggestions
- `GET /api/suggest/models` - Get model recommendations

### Pipeline Execution

- `POST /api/run_pipeline` - Execute preprocessing pipeline

### Model Training

- `GET /api/suggest/models` - Get model recommendations
- `POST /api/train_model` - Train a model with specified configuration
- `GET /api/models` - List all trained models

### Model Visualization & Explainability

- `GET /api/visualizations/{model_id}` - Get model visualizations (confusion matrix, ROC, feature importance, etc.)
- `GET /api/explainability/{model_id}` - Get SHAP/LIME-based model explanations

### Model Deployment

- `POST /api/deploy_model` - Deploy a trained model to specified platform (local/docker/cloud)
- `GET /api/deployment/{deployment_id}` - Get deployment details
- `GET /api/deployment/{deployment_id}/download` - Download deployment package as zip
- `GET /api/model/{model_id}/deployments` - List all deployments for a model

### Drift Monitoring

- `POST /api/start_drift_monitoring` - Start drift monitoring for a deployed model

### Task Management

- `GET /api/task/{task_id}/status` - Get task status
- `GET /api/tasks` - List all tasks
- `POST /api/cancel/{task_id}` - Cancel a running task

### Log Streaming

- `WS /ws/logs?task_id={task_id}` - WebSocket for live log streaming
- `GET /api/logs/{task_id}` - Get logs as JSON
- `GET /api/logs/{task_id}.txt` - Download logs as text

### System

- `GET /` - Root info
- `GET /health` - Health check
- `GET /docs` - OpenAPI documentation (Swagger UI)

## 📊 Console Log Format

All logs follow a structured format:

```
LEVEL TIMESTAMP | SOURCE | MESSAGE
```

Example:

```
INFO 2025-12-26T17:01:03Z | preprocess.impute | Replaced 12 nulls in 'age' with mean 35.716
```

Log levels: `INFO`, `WARN`, `ERROR`, `DEBUG`

## 📁 Project Structure

```
ai-ml-workflow-automation/
├── backend/                    # FastAPI backend
│   ├── app.py                  # Main application with all API endpoints
│   ├── config.py               # Configuration management
│   ├── ws_log_broker.py        # WebSocket log broadcasting
│   ├── core/                   # Core utilities
│   │   ├── auth.py             # JWT authentication
│   │   ├── database.py         # SQLAlchemy models & DB setup
│   │   ├── log_emitter.py      # Structured log creation
│   │   └── log_store.py        # Log persistence (memory + file)
│   ├── services/               # Business logic services
│   │   ├── profiler.py         # Dataset profiling & statistics
│   │   ├── suggestor.py        # AI-powered pipeline & model suggestions
│   │   ├── preprocess.py       # Data preprocessing execution
│   │   ├── training.py         # Model training with metrics
│   │   ├── deployment.py       # Deployment package generation
│   │   ├── visualization.py    # Charts & graphs generation
│   │   ├── explainability.py   # SHAP/LIME explanations
│   │   ├── drift_monitoring.py # Production drift detection
│   │   └── task_manager.py     # Background task management
│   ├── tests/                  # Backend tests
│   ├── artifacts/              # Generated visualizations & models
│   ├── deployments/            # Deployment packages
│   ├── models/                 # Saved model files
│   └── uploads/                # Uploaded datasets
├── frontend/                   # React frontend
│   └── src/
│       ├── components/         # React components
│       │   ├── Console/        # Real-time log streaming
│       │   ├── Dashboard/      # Metrics overview
│       │   ├── Uploader/       # File upload interface
│       │   ├── PipelineEditor/ # Pipeline step configuration
│       │   ├── TrainPanel/     # Model training interface
│       │   ├── DeployPanel/    # Deployment configuration
│       │   ├── Deployment/     # Deployment management
│       │   ├── Visualization/  # Model performance charts
│       │   ├── Explainability/ # Feature importance views
│       │   └── TaskList/       # Task status tracking
│       └── api/                # API client and WebSocket
├── infra/                      # Infrastructure and deployment
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Dockerfile.backend      # Backend container
│   └── Dockerfile.frontend     # Frontend container
├── scripts/                    # Deployment and utility scripts
│   ├── run_demo.sh             # Quick start demo
│   ├── validate_setup.sh       # Setup validation
│   ├── deploy_to_hf.sh         # Hugging Face deployment
│   ├── deploy_to_gcp.sh        # Google Cloud deployment
│   └── deploy_to_aws.sh        # AWS ECS deployment
├── demo/                       # Demo datasets and logs
├── artifacts/                  # Generated artifacts storage
├── ARCHITECTURE.md             # System architecture documentation
├── PROJECT_STATUS.md           # Implementation status tracking
└── README.md                   # This file
```

## 🔐 Environment Variables

See `.env.example` for all available configuration options.

---

## 📊 Project Statistics

| Category                 | Count                                  |
| ------------------------ | -------------------------------------- |
| **Backend Python Files** | 15+ modules                            |
| **Frontend Components**  | 10 component directories               |
| **API Endpoints**        | 20+ endpoints                          |
| **Test Cases**           | 15+ tests                              |
| **Docker Services**      | 4 (backend, frontend, postgres, redis) |
| **Utility Scripts**      | 5 scripts                              |
| **Lines of Code**        | ~5,000+ (backend + frontend)           |

---

## ✅ Implementation Status

### Core Features - ✅ Complete

- [x] Dataset upload and automatic profiling
- [x] AI-powered preprocessing pipeline suggestions
- [x] Interactive pipeline editor with live preview
- [x] Real-time WebSocket log streaming
- [x] Background task management with status tracking
- [x] Model training with multiple algorithms
- [x] Model visualization generation
- [x] SHAP/LIME explainability analysis
- [x] Deployment package creation
- [x] Drift monitoring service

### Production-Ready Features

- [x] Async/await throughout the codebase
- [x] Database connection pooling (PostgreSQL)
- [x] Environment-based configuration
- [x] CORS middleware for cross-origin requests
- [x] Comprehensive error handling
- [x] Structured logging with timestamps
- [x] Type hints (Python) and PropTypes (React)
- [x] Docker containerization
- [x] CI/CD with GitHub Actions

### Security Features

- [x] JWT token support (auth.py)
- [x] Password hashing with bcrypt
- [x] Environment variable management
- [x] CORS origin restrictions
- ⚠️ Auth not enforced on endpoints (demo mode)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License
