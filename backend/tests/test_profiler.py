"""Tests for profiler service."""
import pandas as pd
import pytest
from services.profiler import DataProfiler


def test_profile_dataset():
    """Test dataset profiling."""
    # Create sample dataset
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, None],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC']
    })
    
    profile = DataProfiler.profile_dataset(df)
    
    assert profile['rows'] == 5
    assert profile['columns'] == 3
    assert profile['summary']['numeric_columns'] == 2
    assert profile['summary']['categorical_columns'] == 1
    assert len(profile['columns_info']) == 3


def test_profile_column_numeric():
    """Test profiling numeric column."""
    df = pd.DataFrame({'age': [25, 30, 35, 40, 45]})
    
    col_info = DataProfiler._profile_column(df, 'age')
    
    assert col_info['name'] == 'age'
    assert col_info['type'] == 'int64'
    assert col_info['missing_count'] == 0
    assert col_info['mean'] == 35.0


def test_profile_column_categorical():
    """Test profiling categorical column."""
    df = pd.DataFrame({'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC']})
    
    col_info = DataProfiler._profile_column(df, 'city')
    
    assert col_info['name'] == 'city'
    assert col_info['unique_count'] == 2
    assert 'top_values' in col_info


def test_detect_target_column():
    """Test target column detection."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'target': [0, 1, 0]
    })
    
    profile = DataProfiler.profile_dataset(df)
    target = DataProfiler.detect_target_column(df, profile)
    
    assert target == 'target'
