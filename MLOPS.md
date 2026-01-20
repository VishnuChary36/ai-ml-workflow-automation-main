# MLOps Pipeline Documentation

This document provides comprehensive documentation for the MLOps pipeline components integrated into the AI-ML Workflow Automation Platform.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Tracking & Registry](#model-tracking--registry)
4. [Orchestrated Pipelines](#orchestrated-pipelines)
5. [Production Inference Service](#production-inference-service)
6. [Monitoring & Drift Detection](#monitoring--drift-detection)
7. [CI/CD Automation](#cicd-automation)
8. [Security](#security)
9. [Getting Started](#getting-started)
10. [API Reference](#api-reference)

---

## Overview

The MLOps pipeline transforms the AI-ML Workflow Automation Platform into a production-grade machine learning operations system. It provides:

- **Model Tracking & Registry**: Version control for ML models with MLflow integration
- **Orchestrated Pipelines**: Automated training and deployment workflows with Prefect
- **Production Inference**: Scalable prediction serving with A/B testing
- **Monitoring**: Real-time drift detection and performance tracking
- **CI/CD**: Automated testing, validation, and deployment
- **Security**: Comprehensive audit logging, input validation, and access control

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Platform Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Data       │    │   Training   │    │   Model      │    │ Inference  │ │
│  │   Pipeline   │───▶│   Pipeline   │───▶│   Registry   │───▶│  Service   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Monitoring & Alerting                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │    Drift    │  │ Performance │  │   Alerts    │  │  Dashboard  │  │   │
│  │  │  Detection  │  │  Tracking   │  │   Manager   │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          CI/CD Pipeline                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Validation  │  │   Testing   │  │ Deployment  │  │  Rollback   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Tracking & Registry

### Components

Located in `backend/mlops/`:

- **`tracking.py`**: MLflow integration for experiment tracking
- **`registry.py`**: Model versioning and lifecycle management
- **`experiment.py`**: Experiment comparison and reporting
- **`feature_store.py`**: Feature management and serving

### Usage

#### Experiment Tracking

```python
from mlops.tracking import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name="my-experiment")

# Start a run
with tracker.start_run(run_name="training-v1") as run:
    # Log parameters
    tracker.log_params({
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 100
    })

    # Train model...

    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.92,
        "f1_score": 0.89,
        "auc": 0.95
    })

    # Log model
    tracker.log_model(model, "model", input_example=X_train[:5])
```

#### Model Registry

```python
from mlops.registry import ModelRegistry

registry = ModelRegistry()

# Register a model
version = registry.register_model(
    model=trained_model,
    model_name="customer-churn-predictor",
    metrics={"accuracy": 0.92},
    parameters={"algorithm": "xgboost"},
    tags={"team": "data-science"}
)

# Transition to production
registry.transition_stage(
    model_name="customer-churn-predictor",
    version="1",
    stage="Production"
)

# Load production model
model = registry.load_model(
    model_name="customer-churn-predictor",
    stage="Production"
)
```

#### Feature Store

```python
from mlops.feature_store import FeatureStore

store = FeatureStore()

# Register feature set
store.register_feature_set(
    name="customer_features",
    description="Customer demographic features",
    features=customer_features_df,
    entity_column="customer_id",
    timestamp_column="created_at"
)

# Get features for prediction
features = store.get_online_features(
    feature_set="customer_features",
    entity_ids=["customer_123", "customer_456"]
)
```

---

## Orchestrated Pipelines

### Components

Located in `backend/pipelines/`:

- **`tasks.py`**: Prefect tasks for data processing and training
- **`flows.py`**: Pipeline definitions
- **`scheduler.py`**: Pipeline scheduling

### Available Pipelines

#### Training Pipeline

Complete end-to-end training workflow:

```python
from pipelines.flows import training_pipeline

# Run training pipeline
result = training_pipeline(
    data_source="s3://bucket/data.csv",
    model_type="xgboost",
    target_column="churn",
    experiment_name="churn-prediction"
)
```

#### Data Pipeline

```python
from pipelines.flows import data_pipeline

result = data_pipeline(
    data_source="postgresql://...",
    output_path="./processed/",
    validate=True
)
```

#### Deployment Pipeline

```python
from pipelines.flows import deployment_pipeline

result = deployment_pipeline(
    model_name="customer-churn",
    model_version="3",
    environment="production",
    strategy="canary",
    canary_percentage=10
)
```

### Scheduling

```python
from pipelines.scheduler import PipelineScheduler

scheduler = PipelineScheduler()

# Schedule daily retraining
scheduler.schedule_pipeline(
    pipeline_name="training_pipeline",
    schedule_type="cron",
    cron_expression="0 2 * * *",  # 2 AM daily
    parameters={"data_source": "s3://..."}
)

# Schedule hourly monitoring
scheduler.schedule_pipeline(
    pipeline_name="monitoring_pipeline",
    schedule_type="interval",
    interval_minutes=60
)
```

---

## Production Inference Service

### Components

Located in `backend/inference/`:

- **`predictor.py`**: Model loading and prediction
- **`ab_testing.py`**: A/B testing and canary deployments
- **`metrics.py`**: Prometheus-compatible metrics
- **`server.py`**: FastAPI inference server

### API Endpoints

| Endpoint         | Method | Description        |
| ---------------- | ------ | ------------------ |
| `/health`        | GET    | Health check       |
| `/ready`         | GET    | Readiness check    |
| `/metrics`       | GET    | Prometheus metrics |
| `/predict`       | POST   | Single prediction  |
| `/predict/batch` | POST   | Batch predictions  |
| `/model/info`    | GET    | Model information  |
| `/model/schema`  | GET    | Input schema       |

### Usage

#### Single Prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 50000, "tenure": 24}}'
```

#### Batch Prediction

```bash
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"age": 35, "income": 50000},
      {"age": 42, "income": 75000}
    ]
  }'
```

### A/B Testing

```python
from inference.ab_testing import ABTestingRouter

router = ABTestingRouter()

# Create experiment
router.create_experiment(
    experiment_name="model-improvement-test",
    variants=[
        {"name": "control", "model_id": "model-v1", "weight": 0.5},
        {"name": "treatment", "model_id": "model-v2", "weight": 0.5}
    ]
)

# Get prediction (routes to appropriate variant)
prediction = router.predict(
    experiment_name="model-improvement-test",
    features=input_data,
    user_id="user123"  # For consistent assignment
)

# Record conversion
router.record_conversion(
    experiment_name="model-improvement-test",
    user_id="user123"
)
```

---

## Monitoring & Drift Detection

### Components

Located in `backend/monitoring/`:

- **`drift.py`**: Statistical drift detection
- **`performance.py`**: Model performance tracking
- **`alerting.py`**: Alert management
- **`dashboard.py`**: Monitoring aggregation

### Drift Detection Methods

| Method        | Use Case                 | Threshold |
| ------------- | ------------------------ | --------- |
| PSI           | Distribution shift       | > 0.2     |
| KL Divergence | Probability distribution | > 0.1     |
| KS Test       | Continuous features      | p < 0.05  |
| Chi-Square    | Categorical features     | p < 0.05  |
| ADWIN         | Streaming data           | Adaptive  |

### Usage

```python
from monitoring.drift import DriftDetector

detector = DriftDetector()

# Set reference data
detector.set_reference(training_data)

# Check for drift
results = detector.detect_drift(
    current_data=production_data,
    methods=[DriftMethod.PSI, DriftMethod.KS_TEST],
    columns=["feature1", "feature2", "feature3"]
)

if results["drift_detected"]:
    print(f"Drift in columns: {results['columns_with_drift']}")
```

### Performance Monitoring

```python
from monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor(
    model_id="customer-churn-v1",
    problem_type="classification",
    baseline_metrics={"accuracy": 0.92, "f1_score": 0.89}
)

# Record predictions
monitor.record_prediction(prediction=1, actual=1)

# Check for degradation
report = monitor.check_degradation()
if report["degradation_detected"]:
    print(f"Degraded metrics: {report['degraded_metrics']}")
```

### Alerting

```python
from monitoring.alerting import AlertManager, AlertRule, AlertSeverity

manager = AlertManager()

# Add custom rule
manager.add_rule(AlertRule(
    rule_id="high_latency",
    name="High Inference Latency",
    condition=lambda d: d.get("p95_latency_ms", 0) > 200,
    severity=AlertSeverity.WARNING,
    cooldown_minutes=15
))

# Check rules
alerts = manager.check({
    "accuracy": 0.75,
    "p95_latency_ms": 250,
    "error_rate": 0.05
})
```

---

## CI/CD Automation

### GitHub Actions Workflows

#### MLOps Pipeline (`.github/workflows/mlops.yml`)

Triggers:

- Push to main/develop
- Pull requests
- Scheduled (daily)
- Manual dispatch

Jobs:

1. **validate-model**: Schema and performance validation
2. **train-model**: Automated retraining
3. **build-inference**: Docker image build
4. **deploy-staging**: Staging deployment
5. **deploy-production**: Production deployment (with approval)
6. **rollback**: Manual rollback

### Model Validation

```python
from cicd.validation import ModelValidator

validator = ModelValidator(
    min_accuracy=0.85,
    max_latency_ms=100
)

report = validator.validate(
    model=trained_model,
    model_id="model-v2",
    model_version="2",
    test_data=X_test,
    test_labels=y_test
)

if report.overall_status == ValidationStatus.PASSED:
    print("Model ready for deployment")
else:
    print(f"Validation failed: {report.checks}")
```

### Deployment Management

```python
from cicd.deployment import DeploymentManager, DeploymentConfig

manager = DeploymentManager()

# Create deployment
config = DeploymentConfig(
    model_id="customer-churn-v2",
    model_version="2",
    environment="production",
    strategy=DeploymentStrategy.CANARY,
    canary_percentage=10
)

deployment = manager.create_deployment(config)
manager.start_deployment(deployment.deployment_id)

# Rollback if needed
manager.rollback(environment="production")
```

---

## Security

### Components

Located in `backend/security/`:

- **`audit.py`**: Audit logging
- **`validation.py`**: Input sanitization
- **`headers.py`**: Security headers middleware

### Audit Logging

```python
from security.audit import audit_logger, AuditAction

# Log an action
audit_logger.log(
    action=AuditAction.MODEL_TRAIN,
    user_id="user123",
    username="john.doe",
    resource_type="model",
    resource_id="customer-churn-v2",
    details={"algorithm": "xgboost", "accuracy": 0.92}
)

# Query logs
events = audit_logger.query(
    action="model.train",
    start_date=datetime.now() - timedelta(days=7),
    limit=100
)
```

### Input Validation

```python
from security.validation import InputValidator, FieldSchema, DataType

validator = InputValidator()

schema = [
    FieldSchema("age", DataType.INTEGER, min_value=0, max_value=150),
    FieldSchema("email", DataType.STRING, pattern=r"^[\w.-]+@[\w.-]+\.\w+$"),
    FieldSchema("income", DataType.FLOAT, min_value=0),
]

validated = validator.validate(input_data, schema)
```

### Security Headers

```python
from security.headers import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- Docker (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/ai-ml-workflow-automation.git
cd ai-ml-workflow-automation

# Install dependencies
cd backend
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://user:pass@localhost:5432/mlworkflow
export REDIS_URL=redis://localhost:6379/0
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run migrations
alembic upgrade head

# Start server
uvicorn app:app --reload
```

### Docker Deployment

```bash
# Build and run all services
docker-compose -f infra/docker-compose.yml up -d

# Check status
docker-compose ps
```

---

## API Reference

### Authentication

All API endpoints require authentication via JWT or API key:

```bash
# JWT Authentication
curl -H "Authorization: Bearer <token>" ...

# API Key Authentication
curl -H "X-API-Key: <key>" ...
```

### Rate Limits

| Tier       | Requests/Minute | Burst |
| ---------- | --------------- | ----- |
| Free       | 60              | 10    |
| Pro        | 600             | 50    |
| Enterprise | 6000            | 200   |

### Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [{ "field": "age", "message": "Must be positive integer" }]
  }
}
```

---

## Best Practices

### Model Training

1. Always version your training data
2. Log all hyperparameters and metrics
3. Use reproducible random seeds
4. Validate model before deployment

### Deployment

1. Use canary deployments for production
2. Monitor error rates closely after deployment
3. Have rollback procedures ready
4. Test in staging first

### Monitoring

1. Set appropriate alert thresholds
2. Review drift reports regularly
3. Automate retraining triggers
4. Keep audit logs for compliance

---

## Troubleshooting

### Common Issues

**Model not loading**

- Check model registry connection
- Verify model version exists
- Check file permissions

**High latency**

- Enable model caching
- Consider model optimization
- Scale inference replicas

**Drift alerts firing**

- Review data pipeline changes
- Check for data quality issues
- Consider scheduled retraining

---

## Support

For issues and questions:

- GitHub Issues: [Report a bug](https://github.com/your-org/ai-ml-workflow-automation/issues)
- Documentation: [Full docs](https://docs.example.com)
- Email: mlops-support@example.com
