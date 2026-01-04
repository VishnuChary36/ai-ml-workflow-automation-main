"""Tests for suggestion engine."""
import pandas as pd
import pytest
from services.suggestor import SuggestionEngine


def test_suggest_pipeline():
    """Test pipeline suggestion."""
    df = pd.DataFrame({
        'age': [25, 30, None, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC'],
        'target': [0, 1, 0, 1, 0]
    })
    
    from services.profiler import DataProfiler
    profile = DataProfiler.profile_dataset(df)
    
    suggestions = SuggestionEngine.suggest_pipeline(df, profile, 'target')
    
    assert len(suggestions) > 0
    assert all('id' in s for s in suggestions)
    assert all('type' in s for s in suggestions)
    assert all('confidence' in s for s in suggestions)
    assert all('rationale' in s for s in suggestions)


def test_suggest_models_classification():
    """Test model suggestion for classification."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    
    from services.profiler import DataProfiler
    profile = DataProfiler.profile_dataset(df)
    
    suggestions = SuggestionEngine.suggest_models(df, profile, 'target', 'classification')
    
    assert len(suggestions) > 0
    assert all('model' in s for s in suggestions)
    assert all('params' in s for s in suggestions)
    assert all('confidence' in s for s in suggestions)
    assert 'XGBoostClassifier' in [s['model'] for s in suggestions]


def test_suggest_models_regression():
    """Test model suggestion for regression."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [100, 200, 300, 400, 500]
    })
    
    from services.profiler import DataProfiler
    profile = DataProfiler.profile_dataset(df)
    
    suggestions = SuggestionEngine.suggest_models(df, profile, 'target', 'regression')
    
    assert len(suggestions) > 0
    assert 'XGBoostRegressor' in [s['model'] for s in suggestions]
