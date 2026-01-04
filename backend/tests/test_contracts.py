"""
Contract Tests for API Request/Response Schemas

These tests verify that the frontend ↔ inference payload shapes match expectations.
Run with: pytest tests/test_contracts.py -v
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

# Import schemas
from schemas.prediction import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
)
from schemas.auth import (
    LoginRequest,
    LoginResponse,
    TokenResponse,
    UserResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
)
from schemas.deployment import (
    DeployModelRequest,
    DeploymentResponse,
    DeploymentStatus,
    DeploymentConfig,
)
from schemas.monitoring import (
    DriftMonitoringRequest,
    DriftMonitoringResponse,
    DriftMetrics,
    DriftAlert,
)
from schemas.common import (
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)


# ============================================================================
# Prediction Schema Tests
# ============================================================================

class TestPredictRequest:
    """Tests for PredictRequest schema."""
    
    def test_valid_request(self):
        """Test valid prediction request."""
        data = {
            "model_id": "mdl-abc123",
            "data": {"age": 35, "income": 75000.0, "category": "A"}
        }
        request = PredictRequest(**data)
        assert request.model_id == "mdl-abc123"
        assert request.data["age"] == 35
    
    def test_request_without_model_id(self):
        """Test request without model_id (optional)."""
        data = {"data": {"feature1": "value1"}}
        request = PredictRequest(**data)
        assert request.model_id is None
    
    def test_request_with_various_types(self):
        """Test request with various data types."""
        data = {
            "data": {
                "string_val": "text",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "null_val": None,
            }
        }
        request = PredictRequest(**data)
        assert request.data["string_val"] == "text"
        assert request.data["int_val"] == 42
        assert request.data["float_val"] == 3.14
        assert request.data["bool_val"] is True
        assert request.data["null_val"] is None
    
    def test_missing_data_field(self):
        """Test that missing data field raises error."""
        with pytest.raises(ValidationError):
            PredictRequest(model_id="test")
    
    def test_serialization_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = PredictRequest(
            model_id="mdl-123",
            data={"feature": 1.0}
        )
        json_str = original.model_dump_json()
        restored = PredictRequest.model_validate_json(json_str)
        assert restored.model_id == original.model_id
        assert restored.data == original.data


class TestPredictResponse:
    """Tests for PredictResponse schema."""
    
    def test_classification_response(self):
        """Test classification prediction response."""
        response = PredictResponse(
            prediction=1,
            confidence=0.92,
            probabilities={"class_a": 0.92, "class_b": 0.08},
            label="class_a",
            model_id="mdl-123",
            model_name="RandomForest",
            latency_ms=15.3,
        )
        assert response.prediction == 1
        assert response.confidence == 0.92
        assert response.label == "class_a"
    
    def test_regression_response(self):
        """Test regression prediction response."""
        response = PredictResponse(
            prediction=42.5,
            label="42.5000",
            model_id="mdl-123",
        )
        assert response.prediction == 42.5
        assert response.confidence is None
        assert response.probabilities is None
    
    def test_minimal_response(self):
        """Test response with only required fields."""
        response = PredictResponse(prediction=1)
        assert response.prediction == 1
        assert response.timestamp is not None


class TestBatchPredictRequest:
    """Tests for BatchPredictRequest schema."""
    
    def test_valid_batch_request(self):
        """Test valid batch prediction request."""
        request = BatchPredictRequest(
            model_id="mdl-123",
            data=[
                {"age": 35, "income": 75000},
                {"age": 28, "income": 55000},
            ]
        )
        assert len(request.data) == 2
    
    def test_empty_batch_rejected(self):
        """Test that empty batch is rejected."""
        with pytest.raises(ValidationError):
            BatchPredictRequest(model_id="test", data=[])
    
    def test_max_batch_size(self):
        """Test maximum batch size limit."""
        # Should accept up to 1000 items
        large_batch = [{"feature": i} for i in range(1000)]
        request = BatchPredictRequest(data=large_batch)
        assert len(request.data) == 1000
        
        # Should reject more than 1000 items
        with pytest.raises(ValidationError):
            BatchPredictRequest(data=[{"f": i} for i in range(1001)])


class TestBatchPredictResponse:
    """Tests for BatchPredictResponse schema."""
    
    def test_valid_batch_response(self):
        """Test valid batch response."""
        response = BatchPredictResponse(
            predictions=[
                PredictionResult(prediction=1, confidence=0.9, label="a"),
                PredictionResult(prediction=0, confidence=0.8, label="b"),
            ],
            count=2,
            model_id="mdl-123",
        )
        assert response.count == 2
        assert len(response.predictions) == 2


# ============================================================================
# Authentication Schema Tests
# ============================================================================

class TestLoginRequest:
    """Tests for LoginRequest schema."""
    
    def test_valid_login(self):
        """Test valid login request."""
        request = LoginRequest(username="admin", password="password123")
        assert request.username == "admin"
    
    def test_short_password_rejected(self):
        """Test that short password is rejected."""
        with pytest.raises(ValidationError):
            LoginRequest(username="admin", password="short")
    
    def test_empty_username_rejected(self):
        """Test that empty username is rejected."""
        with pytest.raises(ValidationError):
            LoginRequest(username="", password="password123")


class TestTokenResponse:
    """Tests for TokenResponse schema."""
    
    def test_valid_token_response(self):
        """Test valid token response."""
        response = TokenResponse(
            access_token="eyJ...",
            token_type="bearer",
            expires_in=3600,
        )
        assert response.token_type == "bearer"
        assert response.expires_in == 3600


class TestUserResponse:
    """Tests for UserResponse schema."""
    
    def test_valid_user_response(self):
        """Test valid user response."""
        response = UserResponse(
            id="usr-001",
            username="admin",
            email="admin@example.com",
            role="admin",
            permissions=["dataset:read", "model:train"],
        )
        assert response.role == "admin"
        assert "dataset:read" in response.permissions


# ============================================================================
# Deployment Schema Tests
# ============================================================================

class TestDeploymentSchemas:
    """Tests for deployment schemas."""
    
    def test_deploy_request(self):
        """Test deployment request."""
        request = DeployModelRequest(
            model_id="mdl-123",
            platform="kubernetes",
            config=DeploymentConfig(replicas=2, auto_scale=True),
        )
        assert request.platform == "kubernetes"
        assert request.config.replicas == 2
    
    def test_deployment_response(self):
        """Test deployment response."""
        response = DeploymentResponse(
            deployment_id="dep-123",
            model_id="mdl-123",
            platform="kubernetes",
            status=DeploymentStatus.RUNNING,
            url="https://model.example.com",
        )
        assert response.status == DeploymentStatus.RUNNING
    
    def test_canary_percentage_validation(self):
        """Test canary percentage validation."""
        # Valid percentage
        request = DeployModelRequest(
            model_id="mdl-123",
            platform="kubernetes",
            canary_percentage=10,
        )
        assert request.canary_percentage == 10
        
        # Invalid percentage
        with pytest.raises(ValidationError):
            DeployModelRequest(
                model_id="mdl-123",
                platform="kubernetes",
                canary_percentage=150,  # Over 100
            )


# ============================================================================
# Monitoring Schema Tests
# ============================================================================

class TestMonitoringSchemas:
    """Tests for monitoring schemas."""
    
    def test_drift_monitoring_request(self):
        """Test drift monitoring request."""
        request = DriftMonitoringRequest(
            model_id="mdl-123",
            reference_dataset_id="ds-456",
            check_interval_seconds=3600,
            alert_threshold=0.05,
        )
        assert request.check_interval_seconds == 3600
    
    def test_drift_metrics(self):
        """Test drift metrics schema."""
        metrics = DriftMetrics(
            feature_name="age",
            drift_score=0.15,
            p_value=0.02,
            is_drifted=True,
            test_type="ks_test",
        )
        assert metrics.is_drifted is True
        assert metrics.drift_score == 0.15
    
    def test_drift_alert(self):
        """Test drift alert schema."""
        alert = DriftAlert(
            alert_id="alert-001",
            model_id="mdl-123",
            drift_type="data_drift",
            severity="high",
            drift_score=0.15,
            threshold=0.05,
            message="Drift detected",
            metrics=DriftMetrics(
                drift_score=0.15,
                is_drifted=True,
            ),
        )
        assert alert.severity == "high"


# ============================================================================
# Common Schema Tests
# ============================================================================

class TestCommonSchemas:
    """Tests for common schemas."""
    
    def test_health_response(self):
        """Test health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5,
        )
        assert response.status == "healthy"
    
    def test_ready_response(self):
        """Test ready response."""
        response = ReadyResponse(
            ready=True,
            status="ready",
            checks={"model_loaded": True, "db_connected": True},
        )
        assert response.ready is True
        assert response.checks["model_loaded"] is True
    
    def test_error_response(self):
        """Test error response."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            code="VAL_001",
            request_id="req-123",
        )
        assert response.error == "ValidationError"
        assert response.code == "VAL_001"


# ============================================================================
# Frontend Contract Tests
# ============================================================================

class TestFrontendContracts:
    """
    Tests to verify frontend expectations match backend responses.
    These simulate the exact payloads the frontend sends/receives.
    """
    
    def test_frontend_prediction_request(self):
        """Test that frontend prediction format is valid."""
        # This is what the frontend sends
        frontend_payload = {
            "model_id": "mdl-abc123",
            "data": {
                "age": 35,
                "income": 75000,
                "category": "premium",
                "is_active": True
            }
        }
        
        # Should be valid
        request = PredictRequest(**frontend_payload)
        assert request.model_id == "mdl-abc123"
    
    def test_frontend_expects_prediction_response_shape(self):
        """Test that backend response matches frontend expectations."""
        # Backend response
        response = PredictResponse(
            prediction=1,
            confidence=0.95,
            probabilities={"yes": 0.95, "no": 0.05},
            label="yes",
            model_id="mdl-123",
            model_name="XGBoost",
            latency_ms=12.5,
        )
        
        # Convert to dict (as JSON would be)
        data = response.model_dump()
        
        # Frontend expects these fields
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "label" in data
        assert "model_id" in data
        assert "timestamp" in data
    
    def test_frontend_batch_prediction_contract(self):
        """Test batch prediction contract."""
        # Frontend sends
        request_data = {
            "data": [
                {"feature_a": 1, "feature_b": "x"},
                {"feature_a": 2, "feature_b": "y"},
            ]
        }
        
        request = BatchPredictRequest(**request_data)
        assert len(request.data) == 2
        
        # Backend responds
        response = BatchPredictResponse(
            predictions=[
                PredictionResult(prediction=0, label="no"),
                PredictionResult(prediction=1, label="yes"),
            ],
            count=2,
        )
        
        data = response.model_dump()
        
        # Frontend expects
        assert "predictions" in data
        assert "count" in data
        assert isinstance(data["predictions"], list)
    
    def test_frontend_error_response_contract(self):
        """Test error response format frontend expects."""
        error = ErrorResponse(
            error="UnauthorizedError",
            message="Authentication required",
            code="AUTH_001",
        )
        
        data = error.model_dump()
        
        # Frontend expects standard error format
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data


# ============================================================================
# Inference Service Contract Tests
# ============================================================================

class TestInferenceServiceContracts:
    """
    Tests to verify inference service contracts.
    Ensures the standalone inference service matches the main backend.
    """
    
    def test_inference_health_endpoint(self):
        """Test health endpoint contract."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=1000.0,
        )
        
        data = response.model_dump()
        assert data["status"] == "healthy"
    
    def test_inference_ready_endpoint(self):
        """Test readiness endpoint contract."""
        response = ReadyResponse(
            ready=True,
            status="ready",
            checks={
                "models_loaded": True,
                "model_store_ready": True,
            }
        )
        
        data = response.model_dump()
        assert data["ready"] is True
        assert "checks" in data
    
    def test_inference_predict_matches_backend(self):
        """Test that inference /predict matches backend /api/predict."""
        # Same request should work for both
        request = PredictRequest(
            data={"age": 30, "income": 50000}
        )
        
        # Same response format
        response = PredictResponse(
            prediction=1,
            confidence=0.8,
            label="approved",
        )
        
        # Verify serialization
        request_json = request.model_dump_json()
        response_json = response.model_dump_json()
        
        # Should be valid JSON
        import json
        assert json.loads(request_json)
        assert json.loads(response_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
