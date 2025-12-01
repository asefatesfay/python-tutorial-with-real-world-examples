"""
pytest Configuration File

Shared fixtures and configuration for all tests.
This file is automatically loaded by pytest.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Fixture providing sample features for testing."""
    return [0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [40000, 50000, 60000, 70000, 80000],
        'score': [0.7, 0.8, 0.75, 0.9, 0.85]
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Fixture providing DataFrame with missing values."""
    return pd.DataFrame({
        'age': [25, np.nan, 35, 40, np.nan],
        'income': [40000, 50000, np.nan, 70000, 80000],
        'score': [0.7, 0.8, 0.75, np.nan, 0.85]
    })


@pytest.fixture
def sample_training_data():
    """Fixture providing training data."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    y = pd.Series([0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def sample_test_data():
    """Fixture providing test data."""
    return pd.DataFrame({
        'feature1': [2.5, 4.5],
        'feature2': [0.25, 0.45]
    })


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Fixture providing a mock ML model."""
    class MockModel:
        def __init__(self):
            self.is_trained = False
        
        def fit(self, X, y):
            self.is_trained = True
            return self
        
        def predict(self, X):
            if not self.is_trained:
                raise ValueError("Model not trained")
            return np.array([1] * len(X))
        
        def predict_proba(self, X):
            if not self.is_trained:
                raise ValueError("Model not trained")
            n_samples = len(X)
            return np.array([[0.3, 0.7]] * n_samples)
    
    return MockModel()


@pytest.fixture
def mock_api_response():
    """Fixture providing mock API response."""
    return {
        'prediction': 0.85,
        'confidence': 0.92,
        'model_version': 'v1.0',
        'latency_ms': 45.2
    }


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_random_features(n_samples: int = 100, n_features: int = 10) -> np.ndarray:
    """Generate random feature matrix."""
    return np.random.randn(n_samples, n_features)


def generate_random_labels(n_samples: int = 100, n_classes: int = 2) -> np.ndarray:
    """Generate random labels."""
    return np.random.randint(0, n_classes, n_samples)


@pytest.fixture
def random_data():
    """Fixture providing random data."""
    X = generate_random_features(n_samples=100, n_features=10)
    y = generate_random_labels(n_samples=100, n_classes=2)
    return X, y


# ============================================================================
# Pytest Hooks
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    # If using other libraries:
    # torch.manual_seed(42)
    # tf.random.set_seed(42)


@pytest.fixture(autouse=True)
def print_test_name(request):
    """Print test name before running (helpful for debugging)."""
    print(f"\n{'=' * 70}")
    print(f"Running: {request.node.name}")
    print('=' * 70)
    yield
    print(f"{'=' * 70}")
    print(f"Finished: {request.node.name}")
    print('=' * 70)


# ============================================================================
# Custom Assertions
# ============================================================================

def assert_valid_predictions(predictions: np.ndarray, expected_shape: tuple):
    """Assert that predictions are valid."""
    assert isinstance(predictions, np.ndarray), "Predictions must be numpy array"
    assert predictions.shape == expected_shape, f"Wrong shape: {predictions.shape}"
    assert not np.isnan(predictions).any(), "Predictions contain NaN"
    assert not np.isinf(predictions).any(), "Predictions contain Inf"


def assert_valid_probabilities(probas: np.ndarray):
    """Assert that probabilities are valid."""
    assert isinstance(probas, np.ndarray), "Probabilities must be numpy array"
    assert len(probas.shape) == 2, "Probabilities must be 2D"
    assert np.all(probas >= 0), "Probabilities must be >= 0"
    assert np.all(probas <= 1), "Probabilities must be <= 1"
    assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities must sum to 1"


def assert_dataframe_valid(df: pd.DataFrame, required_columns: List[str]):
    """Assert that DataFrame is valid."""
    assert isinstance(df, pd.DataFrame), "Must be DataFrame"
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0, "DataFrame is empty"


# Make custom assertions available to all tests
@pytest.fixture
def assertions():
    """Fixture providing custom assertions."""
    return {
        'valid_predictions': assert_valid_predictions,
        'valid_probabilities': assert_valid_probabilities,
        'dataframe_valid': assert_dataframe_valid
    }


# ============================================================================
# Pytest Configuration Options
# ============================================================================

"""
Add to pyproject.toml:

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["unit_tests", "integration_tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
python_classes = ["Test*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--disable-warnings",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
]

Running tests:
```bash
# Run all tests
pytest

# Run only unit tests
pytest unit_tests/

# Run specific file
pytest unit_tests/01_testing_basics.py

# Run specific test
pytest unit_tests/01_testing_basics.py::test_function_name

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run verbose
pytest -v

# Run with print statements
pytest -s

# Run and stop at first failure
pytest -x

# Run and show local variables on failure
pytest -l
```
"""
