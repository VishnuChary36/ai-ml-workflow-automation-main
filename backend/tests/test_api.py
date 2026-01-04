"""Basic tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_upload_missing_file():
    """Test upload endpoint without file."""
    response = client.post("/api/upload")
    assert response.status_code == 422  # Unprocessable Entity


def test_get_nonexistent_dataset():
    """Test getting a dataset that doesn't exist."""
    response = client.get("/api/datasets/nonexistent")
    assert response.status_code == 404


def test_get_nonexistent_task():
    """Test getting a task that doesn't exist."""
    response = client.get("/api/task/nonexistent/status")
    assert response.status_code == 404


def test_list_tasks():
    """Test listing tasks."""
    response = client.get("/api/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert isinstance(data["tasks"], list)


def test_list_models():
    """Test listing models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
