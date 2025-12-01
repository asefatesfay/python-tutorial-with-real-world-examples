# Module 11: Testing ML Systems

**Goal**: Write production-quality tests for machine learning systems.

**Why Critical**: Untested ML code = production disasters. ML systems need different testing strategies than traditional software.

## 📚 What You'll Learn

### Unit Testing ML Code
- Testing data preprocessing pipelines
- Testing model predictions
- Testing feature engineering
- Mocking expensive operations (LLM calls, DB queries)
- Testing non-deterministic models

### Integration Testing
- End-to-end pipeline tests
- API endpoint tests
- Database integration tests
- External service mocks (OpenAI, Pinecone)

### Data Validation
- Input data validation with Pydantic
- Schema validation for datasets
- Data drift detection in tests
- Outlier detection

### ML-Specific Testing
- Model performance regression tests
- Golden dataset testing
- Property-based testing
- Test data generation

## 🎯 Why ML Testing is Different

Traditional software:
```python
def add(a, b):
    return a + b

# Easy to test
assert add(2, 3) == 5  ✅
```

ML systems:
```python
def predict(features):
    return model.predict(features)

# What do we test?
# - Model accuracy? (needs labeled data)
# - Prediction format? (could be probabilistic)
# - Latency? (varies)
# - Consistency? (non-deterministic)
```

**Challenges**:
1. **Non-deterministic**: Same input → different outputs (GPT-4)
2. **Expensive**: Model inference costs money/time
3. **No "correct answer"**: Can't assert `prediction == expected`
4. **Data-dependent**: Tests need realistic data
5. **External dependencies**: APIs, databases, cloud storage

## 📂 Module Structure

```
11-testing-ml-systems/
├── README.md (you are here)
├── unit_tests/
│   ├── 01_testing_basics.py          # pytest fundamentals
│   ├── 02_testing_models.py          # Test ML models
│   ├── 03_testing_pipelines.py       # Test data pipelines
│   ├── 04_testing_apis.py            # Test FastAPI endpoints
│   └── 05_data_validation.py         # Validate inputs
├── integration_tests/
│   ├── 01_end_to_end.py              # Full pipeline tests
│   ├── 02_mock_external_apis.py      # Mock OpenAI, Pinecone
│   ├── 03_database_tests.py          # Test with DB
│   └── 04_performance_tests.py       # Latency, throughput
├── fixtures/
│   ├── sample_data.py                # Test data fixtures
│   ├── mock_models.py                # Mock ML models
│   └── mock_services.py              # Mock external services
└── examples/
    ├── conftest.py                   # pytest configuration
    └── pytest.ini                    # pytest settings
```

## 🔧 Setup

Install testing dependencies:
```bash
poetry add --group dev pytest pytest-asyncio pytest-cov pytest-mock responses
```

Run tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest unit_tests/01_testing_basics.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_model"
```

## 💡 Testing Strategies for ML

### 1. **Smoke Tests** (Fast, run always)
```python
def test_model_loads():
    """Ensure model file exists and loads."""
    model = load_model("model.pkl")
    assert model is not None

def test_prediction_shape():
    """Ensure prediction has correct shape."""
    model = load_model("model.pkl")
    prediction = model.predict([[1, 2, 3]])
    assert prediction.shape == (1,)
```

### 2. **Golden Dataset Tests** (Regression tests)
```python
def test_predictions_unchanged():
    """Ensure model predictions haven't changed."""
    model = load_model("model.pkl")
    test_data = load_golden_dataset()
    
    predictions = model.predict(test_data)
    expected = load_expected_predictions()
    
    # Allow small numerical differences
    assert np.allclose(predictions, expected, rtol=1e-5)
```

### 3. **Property-Based Tests** (Edge cases)
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100), min_size=3, max_size=3))
def test_model_predictions_in_range(features):
    """Model predictions should always be between 0 and 1."""
    model = load_model("model.pkl")
    prediction = model.predict([features])
    assert 0 <= prediction[0] <= 1
```

### 4. **Performance Tests**
```python
import time

def test_prediction_latency():
    """Ensure predictions are fast enough."""
    model = load_model("model.pkl")
    
    start = time.time()
    model.predict([[1, 2, 3]])
    latency = time.time() - start
    
    assert latency < 0.1  # 100ms max
```

### 5. **Mock External Services**
```python
from unittest.mock import patch, MagicMock

def test_llm_call_with_mock():
    """Test LLM integration without actual API call."""
    with patch('openai.ChatCompletion.create') as mock_openai:
        # Setup mock response
        mock_openai.return_value = {
            'choices': [{'message': {'content': 'Mock response'}}]
        }
        
        # Test your code
        response = call_openai("test prompt")
        
        # Verify
        assert response == 'Mock response'
        mock_openai.assert_called_once()
```

## 🎓 Testing Best Practices

### DO ✅
- Test at multiple levels (unit, integration, e2e)
- Use fixtures for common test data
- Mock expensive operations (API calls, model loading)
- Test edge cases and error conditions
- Keep tests fast (<1 second each)
- Use descriptive test names
- Test one thing per test
- Use parametrized tests for similar cases

### DON'T ❌
- Test framework code (scikit-learn, PyTorch)
- Make real API calls in tests (expensive!)
- Skip error case testing
- Write flaky tests (non-deterministic)
- Test implementation details
- Copy-paste test code (use fixtures)

## 📊 Test Coverage Goals

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Data preprocessing | 90%+ | High |
| Feature engineering | 90%+ | High |
| Model wrapper code | 80%+ | High |
| API endpoints | 95%+ | Critical |
| Utility functions | 90%+ | High |
| ML model internals | 50%+ | Low |

## 🚀 Quick Start

1. **Install pytest**:
```bash
poetry add --group dev pytest pytest-cov
```

2. **Create first test**:
```python
# test_example.py
def test_addition():
    assert 1 + 1 == 2
```

3. **Run tests**:
```bash
pytest
```

4. **Add coverage**:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## 🎯 Learning Path

1. **Start here**: `unit_tests/01_testing_basics.py`
2. **Test ML models**: `unit_tests/02_testing_models.py`
3. **Test pipelines**: `unit_tests/03_testing_pipelines.py`
4. **Test APIs**: `unit_tests/04_testing_apis.py`
5. **Integration tests**: `integration_tests/01_end_to_end.py`
6. **Mock services**: `integration_tests/02_mock_external_apis.py`

## 💰 Testing ROI

**Without Tests**:
- Production bug: 4 hours to debug
- Uncertain deployments: 2 hours testing manually
- Fear of refactoring: Technical debt grows
- Customer-facing issues: Lost revenue

**With Tests**:
- Production bug: 15 minutes (tests caught it)
- Confident deployments: 5 minutes (CI/CD)
- Easy refactoring: Tests verify correctness
- Catch issues pre-deployment: Happy customers

**Time investment**: 20% more development time
**Time saved**: 80% less debugging time
**ROI**: ~300%

## 📚 Resources

- pytest docs: https://docs.pytest.org
- Testing ML systems: https://madewithml.com/courses/mlops/testing/
- Property-based testing: https://hypothesis.readthedocs.io

---

**Ready to write bulletproof ML code?** Start with `unit_tests/01_testing_basics.py`! 🧪
