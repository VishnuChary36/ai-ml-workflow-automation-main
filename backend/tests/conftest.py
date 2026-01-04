"""Test configuration and fixtures."""
import pytest
import os

# Set test environment variables
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/mlworkflow_test'
os.environ['REDIS_URL'] = 'redis://localhost:6379/1'
os.environ['DEBUG'] = 'True'
