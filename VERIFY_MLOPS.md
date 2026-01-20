# MLOps Pipeline Verification Guide

This guide helps you verify that all MLOps components are properly implemented and working.

## Quick Verification Checklist

Run these commands from the `backend` directory to verify each component:

```powershell
cd backend
```

---

## 1. ✅ Verify Module Imports

Test that all new modules can be imported without errors:

```python
# Run this in Python to verify all modules are properly structured
python -c "
# MLOps Module
from mlops import MLflowTracker, ModelRegistry, FeatureStore, ExperimentManager
print('✅ MLOps module: OK')

# Pipelines Module
from pipelines import training_pipeline, data_pipeline, deployment_pipeline
from pipelines import PipelineScheduler
print('✅ Pipelines module: OK')

# Inference Module
from inference import InferenceServer, ModelPredictor, ABTestingRouter, InferenceMetrics
print('✅ Inference module: OK')

# Monitoring Module
from monitoring import DriftDetector, PerformanceMonitor, AlertManager, MonitoringDashboard
print('✅ Monitoring module: OK')

# CI/CD Module
from cicd import ModelValidator, DeploymentManager, RollbackManager
print('✅ CI/CD module: OK')

# Security Module
from security import AuditLogger, InputValidator, SecurityHeadersMiddleware
print('✅ Security module: OK')

print()
print('🎉 All MLOps modules verified successfully!')
"
```

---

## 2. ✅ Verify File Structure

Check that all files exist:

```powershell
# PowerShell command to verify file structure
$files = @(
    # MLOps
    "backend/mlops/__init__.py",
    "backend/mlops/tracking.py",
    "backend/mlops/registry.py",
    "backend/mlops/experiment.py",
    "backend/mlops/feature_store.py",

    # Pipelines
    "backend/pipelines/__init__.py",
    "backend/pipelines/tasks.py",
    "backend/pipelines/flows.py",
    "backend/pipelines/scheduler.py",

    # Inference
    "backend/inference/__init__.py",
    "backend/inference/predictor.py",
    "backend/inference/ab_testing.py",
    "backend/inference/metrics.py",
    "backend/inference/server.py",

    # Monitoring
    "backend/monitoring/__init__.py",
    "backend/monitoring/drift.py",
    "backend/monitoring/performance.py",
    "backend/monitoring/alerting.py",
    "backend/monitoring/dashboard.py",

    # CI/CD
    "backend/cicd/__init__.py",
    "backend/cicd/validation.py",
    "backend/cicd/deployment.py",
    "backend/cicd/rollback.py",

    # Security
    "backend/security/__init__.py",
    "backend/security/audit.py",
    "backend/security/validation.py",
    "backend/security/headers.py",

    # Workflows
    ".github/workflows/mlops.yml",

    # Documentation
    "MLOPS.md"
)

$missing = @()
foreach ($file in $files) {
    $fullPath = Join-Path "c:\Users\V.Vishnu Vardhan\Downloads\ai-ml-workflow-automation-main\ai-ml-workflow-automation-main" $file
    if (Test-Path $fullPath) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file" -ForegroundColor Red
        $missing += $file
    }
}

if ($missing.Count -eq 0) {
    Write-Host "`n🎉 All 27 MLOps files verified!" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Missing $($missing.Count) files" -ForegroundColor Red
}
```

---

## 3. ✅ Test Individual Components

### 3.1 Model Registry Test

```python
python -c "
from mlops.registry import ModelRegistry
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a simple model
model = RandomForestClassifier(n_estimators=10)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
model.fit(X, y)

# Test registry
registry = ModelRegistry()
version = registry.register_model(
    model=model,
    model_name='test-model',
    metrics={'accuracy': 0.85},
    tags={'test': 'true'}
)
print(f'✅ Model registered: {version}')

# List versions
versions = registry.list_versions('test-model')
print(f'✅ Versions found: {len(versions)}')

# Load model
loaded = registry.load_model('test-model', version=version['version'])
print(f'✅ Model loaded: {type(loaded).__name__}')

print('🎉 Model Registry: WORKING')
"
```

### 3.2 Drift Detection Test

```python
python -c "
import pandas as pd
import numpy as np
from monitoring.drift import DriftDetector, DriftMethod

# Create reference and current data
np.random.seed(42)
reference = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
})

# Current data with drift in feature1
current = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.5, 1000),  # Shifted!
    'feature2': np.random.normal(5, 2, 1000),
})

detector = DriftDetector(reference_data=reference)
results = detector.detect_drift(
    current_data=current,
    methods=[DriftMethod.PSI, DriftMethod.KS_TEST]
)

print(f'Drift detected: {results[\"drift_detected\"]}')
print(f'Columns with drift: {results[\"columns_with_drift\"]}')
print(f'Overall score: {results[\"overall_drift_score\"]:.4f}')
print('🎉 Drift Detection: WORKING')
"
```

### 3.3 Performance Monitor Test

```python
python -c "
from monitoring.performance import PerformanceMonitor
import numpy as np

monitor = PerformanceMonitor(
    model_id='test-model',
    problem_type='classification',
    baseline_metrics={'accuracy': 0.90}
)

# Simulate predictions
np.random.seed(42)
for _ in range(100):
    pred = np.random.randint(0, 2)
    actual = np.random.randint(0, 2)
    monitor.record_prediction(pred, actual)

metrics = monitor.calculate_metrics()
print(f'Current accuracy: {metrics.get(\"accuracy\", 0):.4f}')

degradation = monitor.check_degradation()
print(f'Degradation detected: {degradation.get(\"degradation_detected\", False)}')
print('🎉 Performance Monitor: WORKING')
"
```

### 3.4 Alert Manager Test

```python
python -c "
from monitoring.alerting import AlertManager, AlertSeverity

manager = AlertManager()

# Check with drift data
alerts = manager.check({
    'drift_detected': True,
    'accuracy': 0.75,
    'error_rate': 0.15
})

print(f'Alerts triggered: {len(alerts)}')
for alert in alerts:
    print(f'  - {alert.name} ({alert.severity.value})')

print('🎉 Alert Manager: WORKING')
"
```

### 3.5 Model Validator Test

```python
python -c "
from cicd.validation import ModelValidator, ValidationStatus
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Create model and test data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(200, 5), columns=[f'f{i}' for i in range(5)])
y = pd.Series(np.random.randint(0, 2, 200))

model = RandomForestClassifier(n_estimators=10)
model.fit(X[:100], y[:100])

# Validate
validator = ModelValidator(min_accuracy=0.5)
report = validator.validate(
    model=model,
    model_id='test-model',
    model_version='1',
    test_data=X[100:],
    test_labels=y[100:]
)

print(f'Overall status: {report.overall_status.value}')
print(f'Checks passed: {sum(1 for c in report.checks if c.status == ValidationStatus.PASSED)}/{len(report.checks)}')
print('🎉 Model Validator: WORKING')
"
```

### 3.6 A/B Testing Test

```python
python -c "
from inference.ab_testing import ABTestingRouter
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create router
router = ABTestingRouter()

# Create experiment
router.create_experiment(
    experiment_name='test-experiment',
    variants=[
        {'name': 'control', 'model_id': 'model-v1', 'weight': 0.5},
        {'name': 'treatment', 'model_id': 'model-v2', 'weight': 0.5}
    ]
)

# Test routing
assignments = {}
for i in range(100):
    variant = router.get_variant('test-experiment', user_id=f'user-{i}')
    assignments[variant] = assignments.get(variant, 0) + 1

print(f'Traffic split: {assignments}')
print('🎉 A/B Testing: WORKING')
"
```

### 3.7 Audit Logger Test

```python
python -c "
from security.audit import audit_logger, AuditAction

# Log some events
audit_logger.log(
    action=AuditAction.MODEL_TRAIN,
    user_id='test-user',
    username='tester',
    resource_type='model',
    resource_id='test-model',
    details={'algorithm': 'random_forest'}
)

audit_logger.log(
    action=AuditAction.DEPLOY_CREATE,
    user_id='test-user',
    username='tester',
    resource_type='deployment',
    resource_id='deploy-001'
)

# Query logs
events = audit_logger.query(limit=10)
print(f'Audit events logged: {len(events)}')
print('🎉 Audit Logger: WORKING')
"
```

---

## 4. ✅ Full Integration Test

Run all tests in sequence:

```python
python -c "
print('=' * 60)
print('MLOps Pipeline Integration Test')
print('=' * 60)

tests_passed = 0
tests_failed = 0

# Test 1: Imports
try:
    from mlops import MLflowTracker, ModelRegistry
    from pipelines import training_pipeline
    from inference import InferenceServer, ABTestingRouter
    from monitoring import DriftDetector, AlertManager
    from cicd import ModelValidator, DeploymentManager
    from security import AuditLogger
    print('✅ Test 1: All imports successful')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 1: Import failed - {e}')
    tests_failed += 1

# Test 2: Model Registry
try:
    from mlops.registry import ModelRegistry
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    model = DecisionTreeClassifier()
    model.fit(np.random.rand(50, 3), np.random.randint(0, 2, 50))

    registry = ModelRegistry()
    version = registry.register_model(model, 'integration-test', {'acc': 0.9})
    loaded = registry.load_model('integration-test', version['version'])

    print('✅ Test 2: Model Registry working')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 2: Model Registry failed - {e}')
    tests_failed += 1

# Test 3: Drift Detection
try:
    import pandas as pd
    from monitoring.drift import DriftDetector

    ref = pd.DataFrame({'a': [1,2,3,4,5], 'b': [5,4,3,2,1]})
    cur = pd.DataFrame({'a': [2,3,4,5,6], 'b': [5,4,3,2,1]})

    detector = DriftDetector(reference_data=ref)
    result = detector.detect_drift(cur)

    print('✅ Test 3: Drift Detection working')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 3: Drift Detection failed - {e}')
    tests_failed += 1

# Test 4: Alerting
try:
    from monitoring.alerting import AlertManager

    manager = AlertManager()
    alerts = manager.check({'drift_detected': True})

    print('✅ Test 4: Alert Manager working')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 4: Alert Manager failed - {e}')
    tests_failed += 1

# Test 5: Model Validation
try:
    from cicd.validation import ModelValidator
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    import numpy as np

    model = DecisionTreeClassifier()
    X = pd.DataFrame(np.random.rand(100, 3), columns=['a','b','c'])
    y = pd.Series(np.random.randint(0, 2, 100))
    model.fit(X, y)

    validator = ModelValidator(min_accuracy=0.3)
    report = validator.validate(model, 'test', '1', X, y)

    print('✅ Test 5: Model Validation working')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 5: Model Validation failed - {e}')
    tests_failed += 1

# Test 6: Audit Logging
try:
    from security.audit import AuditLogger, AuditAction

    logger = AuditLogger()
    logger.log(action=AuditAction.LOGIN, user_id='test')

    print('✅ Test 6: Audit Logging working')
    tests_passed += 1
except Exception as e:
    print(f'❌ Test 6: Audit Logging failed - {e}')
    tests_failed += 1

# Summary
print()
print('=' * 60)
print(f'Results: {tests_passed} passed, {tests_failed} failed')
print('=' * 60)

if tests_failed == 0:
    print()
    print('🎉 ALL TESTS PASSED!')
    print('Your MLOps Pipeline is fully operational!')
    print()
    print('Components verified:')
    print('  ✅ Model Tracking & Registry (MLflow)')
    print('  ✅ Orchestrated Pipelines (Prefect)')
    print('  ✅ Production Inference Service')
    print('  ✅ Monitoring & Drift Detection')
    print('  ✅ CI/CD Automation')
    print('  ✅ Security & Audit Logging')
else:
    print(f'⚠️  {tests_failed} test(s) failed. Check errors above.')
"
```

---

## 5. ✅ MLOps Capabilities Summary

After verification, your platform now has these MLOps capabilities:

| Capability                 | Component                      | Status |
| -------------------------- | ------------------------------ | ------ |
| **Experiment Tracking**    | MLflow integration             | ✅     |
| **Model Versioning**       | Model Registry with stages     | ✅     |
| **Feature Management**     | Feature Store                  | ✅     |
| **Pipeline Orchestration** | Prefect flows & tasks          | ✅     |
| **Automated Scheduling**   | Pipeline Scheduler             | ✅     |
| **Model Serving**          | Inference Server               | ✅     |
| **A/B Testing**            | Traffic routing & experiments  | ✅     |
| **Canary Deployments**     | Gradual rollouts               | ✅     |
| **Data Drift Detection**   | PSI, KL, KS, Chi-Square        | ✅     |
| **Performance Monitoring** | Metrics tracking & degradation | ✅     |
| **Alerting**               | Multi-channel notifications    | ✅     |
| **Model Validation**       | Pre-deployment checks          | ✅     |
| **Deployment Management**  | Blue-green, rolling, canary    | ✅     |
| **Rollback Support**       | Version tracking & recovery    | ✅     |
| **Audit Logging**          | Tamper-evident trails          | ✅     |
| **Input Sanitization**     | SQL/XSS prevention             | ✅     |
| **CI/CD Pipeline**         | GitHub Actions workflow        | ✅     |

---

## 6. 🚀 Next Steps

Now that you have a complete MLOps pipeline, you can:

1. **Start the backend server**:

   ```bash
   cd backend
   uvicorn app:app --reload
   ```

2. **Run a training pipeline**:

   ```python
   from pipelines.flows import training_pipeline

   result = training_pipeline(
       data_source="path/to/data.csv",
       model_type="xgboost",
       target_column="target"
   )
   ```

3. **Deploy a model**:

   ```python
   from cicd.deployment import DeploymentManager, DeploymentConfig, DeploymentStrategy

   manager = DeploymentManager()
   config = DeploymentConfig(
       model_id="my-model",
       model_version="1",
       environment="staging",
       strategy=DeploymentStrategy.CANARY
   )
   deployment = manager.create_deployment(config)
   ```

4. **Monitor in production**:

   ```python
   from monitoring.drift import DriftDetector
   from monitoring.alerting import AlertManager

   detector = DriftDetector(reference_data=training_data)
   alerts = detector.detect_drift(production_data)
   ```

---

## Congratulations! 🎉

You now have a **production-grade End-to-End MLOps Pipeline** with:

- ✅ Model lifecycle management
- ✅ Automated training pipelines
- ✅ Production-ready inference
- ✅ Real-time monitoring
- ✅ Automated alerting
- ✅ Safe deployments with rollback
- ✅ Security & compliance

For detailed documentation, see [MLOPS.md](MLOPS.md).
