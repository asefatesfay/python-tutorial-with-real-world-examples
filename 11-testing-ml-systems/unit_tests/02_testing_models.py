"""
Testing ML Models with pytest

Learn how to test machine learning models effectively.
Focus: Model predictions, performance, edge cases.

Install: poetry add --group dev pytest scikit-learn numpy
Run: pytest unit_tests/02_testing_models.py -v
"""

import pytest
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


# ============================================================================
# 1. Why Testing ML Models is Different
# ============================================================================

def demo_why_ml_testing_is_different():
    """
    ML testing is different from traditional software testing.
    """
    print("=" * 70)
    print("1. Why ML Testing is Different")
    print("=" * 70)
    print()
    
    print("🤔 TRADITIONAL SOFTWARE:")
    print()
    print("   def add(a, b):")
    print("       return a + b")
    print("   ")
    print("   # Test:")
    print("   assert add(2, 3) == 5  # Always 5! Deterministic ✅")
    print()
    
    print("🧠 MACHINE LEARNING:")
    print()
    print("   model.predict([[2, 3]])")
    print("   # Output: 0.847  ← Not deterministic!")
    print("   # ")
    print("   # Questions:")
    print("   # • Is 0.847 correct? (No ground truth)")
    print("   # • Should it be exactly 0.847? (Random seeds)")
    print("   # • What if it's 0.846? (Close enough?)")
    print()
    
    print("🎯 KEY DIFFERENCES:")
    print()
    print("   Traditional Testing           ML Testing")
    print("   ─────────────────────────────────────────────────")
    print("   ✓ Deterministic               ✗ Probabilistic")
    print("   ✓ Exact answers               ✗ Approximate")
    print("   ✓ Fast execution              ✗ Can be slow")
    print("   ✓ Clear correctness           ✗ Subjective")
    print("   ✓ No randomness               ✗ Random seeds matter")
    print()
    
    print("💡 WHAT TO TEST IN ML:")
    print()
    print("   Instead of 'Is prediction correct?':")
    print()
    print("   ✅ 1. Output Shape:")
    print("      predict([[1, 2]]) → shape (1,)")
    print("   ")
    print("   ✅ 2. Output Type:")
    print("      predict(...) → numpy array, not list")
    print("   ")
    print("   ✅ 3. Output Range:")
    print("      predict(...) → values between 0 and 1")
    print("   ")
    print("   ✅ 4. Edge Cases:")
    print("      predict([]) → raises ValueError")
    print("      predict(None) → raises TypeError")
    print("   ")
    print("   ✅ 5. Invariants:")
    print("      predict(X) twice → same result (reproducibility)")
    print("      predict([X1, X2]) == [predict(X1), predict(X2)]")
    print("   ")
    print("   ✅ 6. Performance:")
    print("      predict(1000 samples) < 1 second")
    print()


# ============================================================================
# 2. Simple Model to Test
# ============================================================================

class SimpleClassifier:
    """
    Simple ML classifier for testing examples.
    
    Predicts class based on simple threshold rule.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.is_trained = False
        self.n_features = None
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train the model."""
        self.is_trained = True
        self.n_features = len(X[0]) if X else 0
        return self
    
    def predict(self, X: List[List[float]]) -> np.ndarray:
        """Predict classes."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not X:
            raise ValueError("Input X cannot be empty")
        
        # Check feature count
        for sample in X:
            if len(sample) != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {len(sample)}"
                )
        
        # Simple rule: predict 1 if average > threshold
        predictions = []
        for sample in X:
            avg = sum(sample) / len(sample)
            predictions.append(1 if avg > self.threshold else 0)
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        probas = []
        for sample in X:
            avg = sum(sample) / len(sample)
            # Convert to probability (simple sigmoid-like)
            prob_class_1 = avg / (1 + avg)
            probas.append([1 - prob_class_1, prob_class_1])
        
        return np.array(probas)


# ============================================================================
# 3. Testing Model Predictions
# ============================================================================

@pytest.fixture
def trained_model():
    """Fixture providing a trained model."""
    model = SimpleClassifier(threshold=0.5)
    X_train = [[0.1, 0.2], [0.8, 0.9], [0.3, 0.4]]
    y_train = [0, 1, 0]
    model.fit(X_train, y_train)
    return model


def test_model_is_trained(trained_model):
    """Test that model is marked as trained."""
    assert trained_model.is_trained is True


def test_model_predicts_correct_shape(trained_model):
    """Test prediction output shape."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    predictions = trained_model.predict(X_test)
    
    # Check shape
    assert predictions.shape == (2,)
    assert len(predictions) == 2


def test_model_predicts_correct_type(trained_model):
    """Test prediction output type."""
    X_test = [[0.6, 0.7]]
    predictions = trained_model.predict(X_test)
    
    # Check type
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype in [np.int32, np.int64]


def test_model_predicts_correct_range(trained_model):
    """Test prediction values are in valid range."""
    X_test = [[0.6, 0.7], [0.2, 0.3], [0.9, 0.8]]
    predictions = trained_model.predict(X_test)
    
    # Binary classification: only 0 or 1
    assert all(p in [0, 1] for p in predictions)


def test_model_predict_proba_shape(trained_model):
    """Test probability prediction shape."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    probas = trained_model.predict_proba(X_test)
    
    # Shape: (n_samples, n_classes)
    assert probas.shape == (2, 2)


def test_model_predict_proba_sum_to_one(trained_model):
    """Test probabilities sum to 1."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    probas = trained_model.predict_proba(X_test)
    
    # Each row sums to 1
    for proba_row in probas:
        assert pytest.approx(sum(proba_row), abs=0.01) == 1.0


def test_model_predict_proba_range(trained_model):
    """Test probabilities are between 0 and 1."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    probas = trained_model.predict_proba(X_test)
    
    # All probabilities between 0 and 1
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)


def demo_testing_predictions():
    """Demo testing model predictions."""
    print("\n" + "=" * 70)
    print("2. Testing Model Predictions")
    print("=" * 70)
    print()
    
    print("✅ WHAT TO TEST:")
    print()
    print("   1. Output Shape:")
    print("      predictions.shape == (n_samples,)")
    print("   ")
    print("   2. Output Type:")
    print("      isinstance(predictions, np.ndarray)")
    print("   ")
    print("   3. Output Range:")
    print("      all(p in [0, 1] for p in predictions)")
    print("   ")
    print("   4. Probabilities:")
    print("      sum(proba) ≈ 1.0  # Each row sums to 1")
    print("      0 ≤ proba ≤ 1     # Valid range")
    print()


# ============================================================================
# 4. Testing Edge Cases
# ============================================================================

def test_model_raises_error_when_not_trained():
    """Test that untrained model raises error."""
    model = SimpleClassifier()
    
    with pytest.raises(ValueError, match="Model not trained"):
        model.predict([[0.5, 0.6]])


def test_model_raises_error_on_empty_input(trained_model):
    """Test that empty input raises error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        trained_model.predict([])


def test_model_raises_error_on_wrong_feature_count(trained_model):
    """Test that wrong number of features raises error."""
    # Model trained with 2 features
    X_wrong = [[0.5, 0.6, 0.7]]  # 3 features!
    
    with pytest.raises(ValueError, match="Expected 2 features, got 3"):
        trained_model.predict(X_wrong)


def test_model_handles_single_sample(trained_model):
    """Test model with single sample."""
    X_single = [[0.6, 0.7]]
    predictions = trained_model.predict(X_single)
    
    assert predictions.shape == (1,)
    assert predictions[0] in [0, 1]


def test_model_handles_large_batch(trained_model):
    """Test model with large batch."""
    # Generate 1000 samples
    X_large = [[0.5, 0.6] for _ in range(1000)]
    predictions = trained_model.predict(X_large)
    
    assert predictions.shape == (1000,)


def demo_testing_edge_cases():
    """Demo testing edge cases."""
    print("\n" + "=" * 70)
    print("3. Testing Edge Cases")
    print("=" * 70)
    print()
    
    print("🚨 EDGE CASES TO TEST:")
    print()
    print("   1. Not Trained:")
    print("      model.predict(...) before fit()")
    print("      → ValueError ✅")
    print("   ")
    print("   2. Empty Input:")
    print("      model.predict([])")
    print("      → ValueError ✅")
    print("   ")
    print("   3. Wrong Features:")
    print("      Trained: 2 features, Input: 3 features")
    print("      → ValueError ✅")
    print("   ")
    print("   4. Single Sample:")
    print("      model.predict([[0.5, 0.6]])")
    print("      → Works correctly ✅")
    print("   ")
    print("   5. Large Batch:")
    print("      model.predict([[...]  * 1000)")
    print("      → Handles efficiently ✅")
    print()
    
    print("💡 WHY EDGE CASES MATTER:")
    print()
    print("   Production scenarios:")
    print("   • User uploads empty file → Empty input")
    print("   • Data schema changes → Wrong features")
    print("   • High traffic → Large batches")
    print("   • Service restart → Model not loaded")
    print()
    print("   Without tests:")
    print("   • App crashes in production")
    print("   • Poor user experience")
    print("   • Debugging takes hours")
    print()
    print("   With tests:")
    print("   • Catch before deployment ✅")
    print("   • Graceful error handling ✅")
    print("   • Fast debugging ✅")
    print()


# ============================================================================
# 5. Testing Model Invariants
# ============================================================================

def test_model_reproducibility(trained_model):
    """Test that predictions are reproducible."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    
    # Predict twice
    predictions_1 = trained_model.predict(X_test)
    predictions_2 = trained_model.predict(X_test)
    
    # Should be identical
    assert np.array_equal(predictions_1, predictions_2)


def test_model_prediction_consistency(trained_model):
    """Test that batch prediction matches individual predictions."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    
    # Batch prediction
    batch_predictions = trained_model.predict(X_test)
    
    # Individual predictions
    individual_predictions = [
        trained_model.predict([X_test[0]])[0],
        trained_model.predict([X_test[1]])[0],
    ]
    
    # Should match
    assert np.array_equal(batch_predictions, individual_predictions)


def test_model_prediction_order_independence(trained_model):
    """Test that prediction order doesn't matter."""
    X_test = [[0.6, 0.7], [0.2, 0.3]]
    X_test_reversed = [[0.2, 0.3], [0.6, 0.7]]
    
    predictions = trained_model.predict(X_test)
    predictions_reversed = trained_model.predict(X_test_reversed)
    
    # Reversed predictions should match reversed order
    assert predictions[0] == predictions_reversed[1]
    assert predictions[1] == predictions_reversed[0]


def demo_testing_invariants():
    """Demo testing model invariants."""
    print("\n" + "=" * 70)
    print("4. Testing Model Invariants")
    print("=" * 70)
    print()
    
    print("🔍 WHAT ARE INVARIANTS?")
    print()
    print("   Properties that should ALWAYS hold true.")
    print()
    print("   Examples:")
    print("   • Same input → Same output (reproducibility)")
    print("   • Batch prediction = Individual predictions (consistency)")
    print("   • Order doesn't matter (unless it should!)")
    print()
    
    print("✅ TESTING INVARIANTS:")
    print()
    print("   1. Reproducibility:")
    print("      predict(X) == predict(X)  # Always!")
    print("   ")
    print("   2. Consistency:")
    print("      predict([X1, X2]) == [predict(X1), predict(X2)]")
    print("   ")
    print("   3. Order Independence:")
    print("      predict([X1, X2]) ≠ predict([X2, X1])  # Order matters")
    print("      But: predict([X1, X2])[0] == predict([X2, X1])[1]")
    print()
    
    print("💡 WHY INVARIANTS MATTER:")
    print()
    print("   Bug example:")
    print("   • User gets prediction: 0.85")
    print("   • Refreshes page: 0.91  ← Different!")
    print("   • User confused: 'Which is correct?'")
    print()
    print("   With invariant tests:")
    print("   • Catch non-reproducibility early")
    print("   • Ensure consistent user experience")
    print("   • Build trust in ML system")
    print()


# ============================================================================
# 6. Testing Model Performance
# ============================================================================

import time

def test_model_prediction_speed(trained_model):
    """Test that predictions are fast enough."""
    # Generate 1000 samples
    X_test = [[0.5, 0.6] for _ in range(1000)]
    
    start_time = time.time()
    trained_model.predict(X_test)
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    # Should take less than 1 second
    assert elapsed < 1.0, f"Prediction too slow: {elapsed:.2f}s"


def test_model_memory_efficiency(trained_model):
    """Test that model doesn't consume excessive memory."""
    import sys
    
    X_test = [[0.5, 0.6] for _ in range(1000)]
    
    # Predict
    predictions = trained_model.predict(X_test)
    
    # Check memory size
    prediction_size = sys.getsizeof(predictions)
    
    # Should be reasonable (< 1MB for 1000 predictions)
    assert prediction_size < 1_000_000


def demo_testing_performance():
    """Demo testing model performance."""
    print("\n" + "=" * 70)
    print("5. Testing Model Performance")
    print("=" * 70)
    print()
    
    print("⚡ PERFORMANCE TESTS:")
    print()
    print("   1. Prediction Speed:")
    print("      • 1000 predictions < 1 second")
    print("      • API response < 100ms")
    print("   ")
    print("   2. Memory Usage:")
    print("      • Model size < 100MB")
    print("      • Prediction memory < 1GB")
    print("   ")
    print("   3. Throughput:")
    print("      • Handle 100 requests/second")
    print("   ")
    print("   4. Latency:")
    print("      • P95 latency < 200ms")
    print("      • P99 latency < 500ms")
    print()
    
    print("💡 WHY PERFORMANCE MATTERS:")
    print()
    print("   Real-world constraints:")
    print("   • API timeout: 30 seconds")
    print("   • User patience: 2 seconds")
    print("   • Server memory: 8GB")
    print("   • Cost: $0.10 per 1000 predictions")
    print()
    print("   Slow model:")
    print("   • Users leave (high bounce rate)")
    print("   • High cloud costs")
    print("   • Poor scalability")
    print()
    print("   Fast model:")
    print("   • Happy users ✅")
    print("   • Low costs ✅")
    print("   • Scales well ✅")
    print()


# ============================================================================
# 7. Parametrized Model Tests
# ============================================================================

@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_model_with_different_thresholds(threshold):
    """Test model with different threshold values."""
    model = SimpleClassifier(threshold=threshold)
    X_train = [[0.1, 0.2], [0.8, 0.9]]
    y_train = [0, 1]
    model.fit(X_train, y_train)
    
    # Predictions should work
    X_test = [[0.6, 0.7]]
    predictions = model.predict(X_test)
    
    assert predictions.shape == (1,)
    assert predictions[0] in [0, 1]


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_model_with_different_batch_sizes(trained_model, n_samples):
    """Test model with different batch sizes."""
    X_test = [[0.5, 0.6] for _ in range(n_samples)]
    predictions = trained_model.predict(X_test)
    
    assert predictions.shape == (n_samples,)


def demo_parametrized_model_tests():
    """Demo parametrized model tests."""
    print("\n" + "=" * 70)
    print("6. Parametrized Model Tests")
    print("=" * 70)
    print()
    
    print("🔁 PARAMETRIZE MODEL TESTS:")
    print()
    print("   @pytest.mark.parametrize('threshold', [0.3, 0.5, 0.7])")
    print("   def test_model_with_thresholds(threshold):")
    print("       model = SimpleClassifier(threshold=threshold)")
    print("       # Test with different thresholds")
    print("   ")
    print("   Result: Tests model with 3 different configurations!")
    print()
    
    print("💡 WHEN TO PARAMETRIZE:")
    print()
    print("   • Different hyperparameters")
    print("   • Different input sizes")
    print("   • Different data types")
    print("   • Different edge cases")
    print()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Testing ML Models\n")
    
    demo_why_ml_testing_is_different()
    demo_testing_predictions()
    demo_testing_edge_cases()
    demo_testing_invariants()
    demo_testing_performance()
    demo_parametrized_model_tests()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. ML Testing is Different:
   - No exact "correct" answers
   - Test shape, type, range instead
   - Test invariants and properties

2. What to Test:
   ✅ Output shape: predictions.shape == (n,)
   ✅ Output type: isinstance(predictions, np.ndarray)
   ✅ Output range: 0 ≤ predictions ≤ 1
   ✅ Reproducibility: predict(X) == predict(X)
   ✅ Edge cases: empty input, wrong features
   ✅ Performance: speed, memory

3. Edge Cases:
   - Model not trained
   - Empty input
   - Wrong feature count
   - Single sample
   - Large batch

4. Invariants:
   - Same input → Same output
   - Batch = Individual predictions
   - Order independence (usually)

5. Performance:
   - Prediction speed < 1s for 1000 samples
   - Memory usage reasonable
   - Handle production load

Testing Checklist:
```
Model Tests:
□ Output shape correct
□ Output type correct
□ Output range valid
□ Probabilities sum to 1
□ Handles not trained error
□ Handles empty input error
□ Handles wrong features error
□ Handles single sample
□ Handles large batch
□ Reproducible predictions
□ Consistent batch vs individual
□ Fast enough (<1s for 1000)
□ Memory efficient
```

Next Steps:
→ 03_testing_pipelines.py (Test data pipelines)
→ 04_testing_apis.py (Test ML APIs)
""")


if __name__ == "__main__":
    main()
