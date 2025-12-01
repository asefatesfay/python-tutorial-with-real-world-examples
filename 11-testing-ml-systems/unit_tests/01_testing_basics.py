"""
pytest Fundamentals for ML Systems

Learn pytest basics and how to apply them to ML code.
Focus: Writing effective tests quickly.

Install: poetry add --group dev pytest pytest-cov
Run: pytest unit_tests/01_testing_basics.py -v
"""

import pytest
from typing import List, Dict
import time


# ============================================================================
# 1. pytest Basics
# ============================================================================

def demo_why_pytest():
    """
    Why pytest for ML systems?
    
    INTUITION - The Safety Net Analogy:
    
    Writing code without tests = Tightrope walking without net
    - One mistake → Fall → Disaster
    - Fear of making changes
    - Always nervous
    
    Writing code with tests = Tightrope with safety net
    - Mistake → Net catches you
    - Confident to try new moves
    - Sleep well at night ✅
    
    TRADITIONAL TESTING (assert statements):
    
    def add(a, b):
        return a + b
    
    # Manual testing
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    
    Problems:
    - Stops at first failure
    - No test organization
    - No setup/teardown
    - Hard to run selectively
    - No coverage reports
    
    PYTEST ADVANTAGES:
    
    1. Auto-discovery:
       - Finds all test_*.py files
       - Finds all test_*() functions
       - No manual registration
    
    2. Rich assertions:
       - assert x == y  # Auto-explains differences
       - Better error messages
    
    3. Fixtures:
       - Reusable test data
       - Setup/teardown
       - Dependency injection
    
    4. Parametrization:
       - Run same test with different inputs
       - Reduce code duplication
    
    5. Coverage:
       - See what code is tested
       - Find gaps
    
    6. Plugins:
       - pytest-asyncio (async tests)
       - pytest-cov (coverage)
       - pytest-mock (mocking)
       - pytest-xdist (parallel)
    
    WHY IT MATTERS FOR ML:
    
    ML systems are complex:
    - Data pipelines (many steps)
    - Model training (hours)
    - Feature engineering (subtle bugs)
    - API endpoints (edge cases)
    - External services (API calls)
    
    Without tests:
    - Deploy broken model → Bad predictions
    - Data pipeline breaks → Training fails
    - API crashes → Customers angry
    - Can't refactor → Technical debt
    
    With pytest:
    - Catch bugs before deployment
    - Confident refactoring
    - Fast feedback loop
    - Document expected behavior
    """
    print("=" * 70)
    print("1. Why pytest for ML Systems?")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Safety Net")
    print()
    print("   Without Tests:")
    print("   🎪 Tightrope walker with no net")
    print("   • One mistake → disaster")
    print("   • Fear of changes")
    print("   • Nervous deployment")
    print()
    print("   With Tests:")
    print("   🎪 Tightrope walker with safety net")
    print("   • Mistakes caught early")
    print("   • Confident changes")
    print("   • Smooth deployment ✅")
    print()
    
    print("📊 TEST STATISTICS:")
    print()
    print("   Without Tests:")
    print("   • Time to deploy: 4 hours (manual testing)")
    print("   • Bugs found in prod: 15 per month")
    print("   • Time debugging: 20 hours/month")
    print("   • Confidence: 60%")
    print()
    print("   With pytest:")
    print("   • Time to deploy: 10 minutes (automated)")
    print("   • Bugs found in prod: 2 per month")
    print("   • Time debugging: 2 hours/month")
    print("   • Confidence: 95% ✅")
    print()
    print("   ROI: Save 18 hours/month debugging!")
    print()


# ============================================================================
# 2. Basic Test Structure
# ============================================================================

# Example function to test
def calculate_confidence(correct: int, total: int) -> float:
    """Calculate confidence percentage."""
    if total == 0:
        return 0.0
    return correct / total


def test_calculate_confidence_normal():
    """Test confidence calculation with normal values."""
    # Arrange
    correct = 85
    total = 100
    
    # Act
    result = calculate_confidence(correct, total)
    
    # Assert
    assert result == 0.85


def test_calculate_confidence_zero_total():
    """Test confidence calculation with zero total."""
    result = calculate_confidence(0, 0)
    assert result == 0.0


def test_calculate_confidence_perfect():
    """Test confidence calculation with 100% correct."""
    result = calculate_confidence(100, 100)
    assert result == 1.0


def demo_basic_test_structure():
    """
    Basic test structure and patterns.
    """
    print("\n" + "=" * 70)
    print("2. Basic Test Structure")
    print("=" * 70)
    print()
    
    print("📝 TEST NAMING CONVENTIONS:")
    print()
    print("   Good test names:")
    print("   ✅ test_calculate_confidence_normal()")
    print("   ✅ test_calculate_confidence_zero_total()")
    print("   ✅ test_calculate_confidence_perfect()")
    print()
    print("   Bad test names:")
    print("   ❌ test1()")
    print("   ❌ test_function()")
    print("   ❌ test_stuff()")
    print()
    print("   Pattern: test_<function>_<scenario>")
    print()
    
    print("🏗️  AAA PATTERN (Arrange-Act-Assert):")
    print()
    print("   def test_example():")
    print("       # Arrange: Set up test data")
    print("       input_data = [1, 2, 3]")
    print("       ")
    print("       # Act: Call function being tested")
    print("       result = sum(input_data)")
    print("       ")
    print("       # Assert: Verify result")
    print("       assert result == 6")
    print()
    
    print("✅ ASSERTION EXAMPLES:")
    print()
    print("   # Equality")
    print("   assert result == expected")
    print()
    print("   # Comparison")
    print("   assert result > 0")
    print("   assert result >= 0.5")
    print()
    print("   # Type checking")
    print("   assert isinstance(result, float)")
    print()
    print("   # In container")
    print("   assert 'error' in response")
    print("   assert item in list_of_items")
    print()
    print("   # Approximate equality (for floats)")
    print("   assert result == pytest.approx(3.14, abs=0.01)")
    print()


# ============================================================================
# 3. Fixtures (Reusable Test Data)
# ============================================================================

@pytest.fixture
def sample_dataset():
    """Fixture providing sample dataset for tests."""
    return {
        'features': [[1, 2], [3, 4], [5, 6]],
        'labels': [0, 1, 0]
    }


@pytest.fixture
def trained_model():
    """Fixture providing a simple trained model."""
    class SimpleModel:
        def predict(self, X):
            # Simple rule: predict 1 if first feature > 3
            return [1 if x[0] > 3 else 0 for x in X]
    
    return SimpleModel()


def test_with_fixture(sample_dataset):
    """Test using fixture."""
    # Fixture is automatically passed as argument
    assert len(sample_dataset['features']) == 3
    assert len(sample_dataset['labels']) == 3


def test_model_with_fixture(trained_model, sample_dataset):
    """Test model using multiple fixtures."""
    predictions = trained_model.predict(sample_dataset['features'])
    
    # Model predicts based on first feature > 3
    assert predictions == [0, 0, 1]


def demo_fixtures():
    """
    Fixtures: Reusable test data and setup.
    """
    print("\n" + "=" * 70)
    print("3. Fixtures (Reusable Test Data)")
    print("=" * 70)
    print()
    
    print("💡 WHY FIXTURES?")
    print()
    print("   Without fixtures:")
    print("   def test_1():")
    print("       data = create_test_data()  # Duplicated!")
    print("   ")
    print("   def test_2():")
    print("       data = create_test_data()  # Duplicated!")
    print()
    print("   With fixtures:")
    print("   @pytest.fixture")
    print("   def test_data():")
    print("       return create_test_data()")
    print("   ")
    print("   def test_1(test_data):  # Auto-injected")
    print("       ...")
    print("   ")
    print("   def test_2(test_data):  # Auto-injected")
    print("       ...")
    print()
    
    print("🔧 FIXTURE SCOPES:")
    print()
    print("   @pytest.fixture(scope='function')  # Default: New for each test")
    print("   @pytest.fixture(scope='module')    # Once per test file")
    print("   @pytest.fixture(scope='session')   # Once per test run")
    print()
    print("   Use cases:")
    print("   • function: Test data (cheap to create)")
    print("   • module: Database connection (expensive)")
    print("   • session: Model loading (very expensive)")
    print()
    
    print("⚙️  FIXTURE EXAMPLE:")
    print()
    print("   @pytest.fixture")
    print("   def ml_model():")
    print('       """Load model once for all tests."""')
    print("       model = load_model('model.pkl')  # Expensive!")
    print("       return model")
    print("   ")
    print("   def test_prediction(ml_model):")
    print("       result = ml_model.predict([[1, 2, 3]])")
    print("       assert result.shape == (1,)")
    print()


# ============================================================================
# 4. Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("correct,total,expected", [
    (85, 100, 0.85),
    (100, 100, 1.0),
    (0, 100, 0.0),
    (50, 200, 0.25),
])
def test_confidence_parametrized(correct, total, expected):
    """Test confidence calculation with multiple inputs."""
    result = calculate_confidence(correct, total)
    assert result == expected


@pytest.mark.parametrize("input_list,expected_sum", [
    ([1, 2, 3], 6),
    ([0, 0, 0], 0),
    ([-1, 1], 0),
    ([100], 100),
])
def test_sum_parametrized(input_list, expected_sum):
    """Test sum with different inputs."""
    assert sum(input_list) == expected_sum


def demo_parametrized_tests():
    """
    Parametrized tests: Run same test with different inputs.
    """
    print("\n" + "=" * 70)
    print("4. Parametrized Tests")
    print("=" * 70)
    print()
    
    print("🔁 WHY PARAMETRIZE?")
    print()
    print("   Without parametrization:")
    print("   def test_confidence_case1():")
    print("       assert calculate_confidence(85, 100) == 0.85")
    print("   ")
    print("   def test_confidence_case2():")
    print("       assert calculate_confidence(100, 100) == 1.0")
    print("   ")
    print("   def test_confidence_case3():")
    print("       assert calculate_confidence(0, 100) == 0.0")
    print()
    print("   Result: 3 nearly-identical functions!")
    print()
    print("   With parametrization:")
    print("   @pytest.mark.parametrize('correct,total,expected', [")
    print("       (85, 100, 0.85),")
    print("       (100, 100, 1.0),")
    print("       (0, 100, 0.0),")
    print("   ])")
    print("   def test_confidence(correct, total, expected):")
    print("       assert calculate_confidence(correct, total) == expected")
    print()
    print("   Result: 1 test, 3 cases! ✅")
    print()
    
    print("📊 PARAMETRIZE OUTPUT:")
    print()
    print("   pytest will run:")
    print("   • test_confidence[85-100-0.85] ✅")
    print("   • test_confidence[100-100-1.0] ✅")
    print("   • test_confidence[0-100-0.0] ✅")
    print()
    print("   Each case is a separate test!")
    print()


# ============================================================================
# 5. Testing Exceptions
# ============================================================================

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def test_divide_by_zero():
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)


def test_divide_normal():
    """Test normal division."""
    result = divide(10, 2)
    assert result == 5.0


def demo_testing_exceptions():
    """
    Test that code raises expected exceptions.
    """
    print("\n" + "=" * 70)
    print("5. Testing Exceptions")
    print("=" * 70)
    print()
    
    print("🚨 WHY TEST EXCEPTIONS?")
    print()
    print("   Good code handles errors gracefully:")
    print("   • Invalid input → Clear error message")
    print("   • External service down → Retry logic")
    print("   • Data missing → Default value")
    print()
    print("   Tests ensure errors are caught!")
    print()
    
    print("✅ TESTING EXCEPTIONS:")
    print()
    print("   def test_divide_by_zero():")
    print("       with pytest.raises(ValueError):")
    print("           divide(10, 0)")
    print()
    print("   # With message check")
    print("   def test_with_message():")
    print("       with pytest.raises(ValueError, match='Cannot divide'):")
    print("           divide(10, 0)")
    print()
    
    print("🎯 COMMON ML EXCEPTIONS TO TEST:")
    print()
    print("   • ValueError: Invalid input shape")
    print("   • FileNotFoundError: Model file missing")
    print("   • RuntimeError: GPU out of memory")
    print("   • KeyError: Feature not in dataset")
    print("   • TypeError: Wrong data type")
    print()


# ============================================================================
# 6. Test Markers
# ============================================================================

@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow (can skip with -m 'not slow')."""
    time.sleep(0.01)  # Simulate slow operation
    assert True


@pytest.mark.unit
def test_fast_unit_test():
    """Test marked as unit test."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_integration():
    """Test marked as integration test."""
    # Would test multiple components together
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Test skipped with reason."""
    pass


@pytest.mark.skipif(time.time() < 0, reason="Never skip this")
def test_conditional_skip():
    """Test skipped conditionally."""
    assert True


def demo_test_markers():
    """
    Test markers: Categorize and selectively run tests.
    """
    print("\n" + "=" * 70)
    print("6. Test Markers")
    print("=" * 70)
    print()
    
    print("🏷️  TEST MARKERS:")
    print()
    print("   @pytest.mark.slow")
    print("   def test_model_training():")
    print("       # Takes 10 minutes")
    print("       ...")
    print()
    print("   @pytest.mark.unit")
    print("   def test_fast_function():")
    print("       # Takes 0.001 seconds")
    print("       ...")
    print()
    
    print("🎯 RUN SELECTIVELY:")
    print()
    print("   # Run only fast tests")
    print("   pytest -m 'not slow'")
    print()
    print("   # Run only unit tests")
    print("   pytest -m unit")
    print()
    print("   # Run slow tests only")
    print("   pytest -m slow")
    print()
    print("   # Skip integration tests")
    print("   pytest -m 'not integration'")
    print()
    
    print("💡 COMMON MARKERS:")
    print()
    print("   • @pytest.mark.slow: Long-running tests")
    print("   • @pytest.mark.unit: Unit tests")
    print("   • @pytest.mark.integration: Integration tests")
    print("   • @pytest.mark.skip: Skip always")
    print("   • @pytest.mark.skipif: Skip conditionally")
    print("   • @pytest.mark.xfail: Expected to fail")
    print()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 pytest Fundamentals for ML\n")
    
    demo_why_pytest()
    demo_basic_test_structure()
    demo_fixtures()
    demo_parametrized_tests()
    demo_testing_exceptions()
    demo_test_markers()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. pytest Auto-Discovery:
   - Finds test_*.py files
   - Finds test_*() functions
   - No manual registration needed

2. AAA Pattern:
   Arrange → Act → Assert
   (Setup → Call → Verify)

3. Fixtures:
   Reusable test data and setup
   @pytest.fixture
   def test_data():
       return {...}

4. Parametrize:
   Run same test with different inputs
   @pytest.mark.parametrize("input,expected", [...])

5. Test Exceptions:
   with pytest.raises(ValueError):
       function_that_should_fail()

6. Markers:
   Categorize and run selectively
   @pytest.mark.slow
   pytest -m 'not slow'

Running Tests:
```bash
# Run all tests
pytest

# Verbose output
pytest -v

# With coverage
pytest --cov=. --cov-report=html

# Run specific file
pytest test_file.py

# Run specific test
pytest test_file.py::test_function

# Run tests matching pattern
pytest -k "test_model"

# Run marked tests
pytest -m unit
pytest -m 'not slow'
```

Next Steps:
→ 02_testing_models.py (Test ML models)
→ 03_testing_pipelines.py (Test data pipelines)
→ 04_testing_apis.py (Test FastAPI endpoints)
""")


if __name__ == "__main__":
    main()
