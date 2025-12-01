"""
End-to-End Integration Testing

Learn how to test complete ML pipelines from input to output.
Focus: Full workflow testing, realistic scenarios, data flow.

Install: poetry add --group dev pytest pandas scikit-learn
Run: pytest integration_tests/01_end_to_end.py -v
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict
import time


# ============================================================================
# 1. Why End-to-End Testing?
# ============================================================================

def demo_why_e2e_testing():
    """
    Why end-to-end testing is essential.
    """
    print("=" * 70)
    print("1. Why End-to-End Testing?")
    print("=" * 70)
    print()
    
    print("💭 UNIT TESTS vs E2E TESTS:")
    print()
    print("   Unit Tests:")
    print("   🔬 Test individual functions")
    print("   • Fast (milliseconds)")
    print("   • Isolated")
    print("   • Mock dependencies")
    print("   • Example: test_model_predict()")
    print()
    print("   E2E Tests:")
    print("   🌍 Test entire system")
    print("   • Slower (seconds)")
    print("   • Integrated")
    print("   • Real dependencies")
    print("   • Example: test_full_prediction_pipeline()")
    print()
    
    print("💥 WHY UNIT TESTS AREN'T ENOUGH:")
    print()
    print("   Scenario: Building a house")
    print()
    print("   Unit tests:")
    print("   ✅ Foundation strong")
    print("   ✅ Walls sturdy")
    print("   ✅ Roof waterproof")
    print()
    print("   But...")
    print("   ❌ Walls don't connect to foundation!")
    print("   ❌ Roof doesn't fit on walls!")
    print("   ❌ House collapses!")
    print()
    print("   E2E test:")
    print("   ✅ Complete house stands")
    print("   ✅ All parts work together")
    print("   ✅ Ready to live in!")
    print()
    
    print("🎯 REAL-WORLD E2E BUG:")
    print()
    print("   Unit tests: All passing ✅")
    print("   • test_data_preprocessing() ✅")
    print("   • test_feature_engineering() ✅")
    print("   • test_model_prediction() ✅")
    print()
    print("   Production bug:")
    print("   • Preprocessing outputs column 'age'")
    print("   • Feature engineering expects 'Age' (capital A!)")
    print("   • Model crashes: KeyError 'Age'")
    print()
    print("   Why unit tests missed it:")
    print("   • Each tested in isolation")
    print("   • Mocked data had correct names")
    print("   • Never tested full pipeline!")
    print()
    print("   E2E test would catch:")
    print("   def test_full_pipeline():")
    print("       raw_data → preprocess → features → predict")
    print("       → KeyError! ✅ Caught before deploy")
    print()


# ============================================================================
# 2. Complete ML Pipeline to Test
# ============================================================================

class DataPreprocessor:
    """Preprocess raw data."""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        df_clean = df.copy()
        
        # Fill missing values
        df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
        df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
        
        # Remove outliers (simple method)
        df_clean = df_clean[df_clean['age'] < 100]
        df_clean = df_clean[df_clean['income'] < 1_000_000]
        
        return df_clean


class FeatureEngineer:
    """Create features from preprocessed data."""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features."""
        df_features = df.copy()
        
        # Create new features
        df_features['age_group'] = pd.cut(
            df_features['age'],
            bins=[0, 30, 50, 100],
            labels=['young', 'middle', 'senior']
        )
        
        df_features['income_bracket'] = pd.cut(
            df_features['income'],
            bins=[0, 50000, 100000, 1000000],
            labels=['low', 'medium', 'high']
        )
        
        # Encode categorical
        df_features['age_group_encoded'] = df_features['age_group'].cat.codes
        df_features['income_bracket_encoded'] = df_features['income_bracket'].cat.codes
        
        return df_features


class MLModel:
    """Simple ML model."""
    
    def __init__(self):
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train model."""
        self.is_trained = True
        # In reality: model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Simple prediction logic
        predictions = []
        for _, row in X.iterrows():
            score = row['age'] * 0.01 + row['income'] * 0.00001
            prediction = 1 if score > 1.0 else 0
            predictions.append(prediction)
        
        return np.array(predictions)


class MLPipeline:
    """Complete ML pipeline."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = MLModel()
    
    def train(self, df: pd.DataFrame, y: pd.Series):
        """Train pipeline."""
        # Preprocess
        df_clean = self.preprocessor.process(df)
        
        # Engineer features
        df_features = self.feature_engineer.create_features(df_clean)
        
        # Select features for model
        X = df_features[['age', 'income', 'age_group_encoded', 'income_bracket_encoded']]
        
        # Train model
        self.model.train(X, y)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using full pipeline."""
        # Preprocess
        df_clean = self.preprocessor.process(df)
        
        # Engineer features
        df_features = self.feature_engineer.create_features(df_clean)
        
        # Select features
        X = df_features[['age', 'income', 'age_group_encoded', 'income_bracket_encoded']]
        
        # Predict
        return self.model.predict(X)


# ============================================================================
# 3. End-to-End Tests
# ============================================================================

@pytest.fixture
def training_data():
    """Fixture providing training data."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [40000, 50000, 60000, 70000, 80000, 90000],
    })


@pytest.fixture
def training_labels():
    """Fixture providing training labels."""
    return pd.Series([0, 0, 1, 1, 1, 1])


@pytest.fixture
def test_data():
    """Fixture providing test data."""
    return pd.DataFrame({
        'age': [28, 42],
        'income': [45000, 75000],
    })


def test_e2e_pipeline_training(training_data, training_labels):
    """Test that pipeline can be trained end-to-end."""
    pipeline = MLPipeline()
    
    # Should not raise any errors
    pipeline.train(training_data, training_labels)
    
    # Model should be trained
    assert pipeline.model.is_trained is True


def test_e2e_pipeline_prediction(training_data, training_labels, test_data):
    """Test that pipeline can make predictions end-to-end."""
    pipeline = MLPipeline()
    
    # Train
    pipeline.train(training_data, training_labels)
    
    # Predict
    predictions = pipeline.predict(test_data)
    
    # Check output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert all(p in [0, 1] for p in predictions)


def test_e2e_pipeline_with_missing_values(training_labels):
    """Test pipeline with missing values."""
    # Data with missing values
    data_with_missing = pd.DataFrame({
        'age': [25, np.nan, 35],
        'income': [40000, 50000, np.nan],
    })
    
    labels = pd.Series([0, 0, 1])
    
    pipeline = MLPipeline()
    
    # Train with missing values
    pipeline.train(data_with_missing, labels)
    
    # Predict with missing values
    test_with_missing = pd.DataFrame({
        'age': [np.nan, 42],
        'income': [45000, np.nan],
    })
    
    predictions = pipeline.predict(test_with_missing)
    
    # Should handle missing values gracefully
    assert predictions.shape == (2,)
    assert not np.isnan(predictions).any()


def test_e2e_pipeline_with_outliers(training_labels):
    """Test pipeline with outliers."""
    # Data with outliers
    data_with_outliers = pd.DataFrame({
        'age': [25, 30, 35, 999],  # 999 is outlier
        'income': [40000, 50000, 60000, 70000],
    })
    
    labels = pd.Series([0, 0, 1, 1])
    
    pipeline = MLPipeline()
    
    # Train (should remove outliers)
    pipeline.train(data_with_outliers, labels)
    
    # Predict
    test_data = pd.DataFrame({
        'age': [28, 42],
        'income': [45000, 75000],
    })
    
    predictions = pipeline.predict(test_data)
    
    # Should work despite outliers in training
    assert predictions.shape == (2,)


def test_e2e_pipeline_reproducibility(training_data, training_labels, test_data):
    """Test that predictions are reproducible."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    # Predict twice
    predictions_1 = pipeline.predict(test_data)
    predictions_2 = pipeline.predict(test_data)
    
    # Should be identical
    np.testing.assert_array_equal(predictions_1, predictions_2)


def test_e2e_pipeline_batch_consistency(training_data, training_labels):
    """Test that batch prediction matches individual predictions."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    # Batch prediction
    batch_data = pd.DataFrame({
        'age': [28, 42],
        'income': [45000, 75000],
    })
    batch_predictions = pipeline.predict(batch_data)
    
    # Individual predictions
    individual_predictions = []
    for idx in range(len(batch_data)):
        single = batch_data.iloc[[idx]]
        pred = pipeline.predict(single)
        individual_predictions.append(pred[0])
    
    # Should match
    np.testing.assert_array_equal(batch_predictions, individual_predictions)


def demo_e2e_tests():
    """Demo end-to-end tests."""
    print("\n" + "=" * 70)
    print("2. End-to-End Tests")
    print("=" * 70)
    print()
    
    print("✅ E2E TEST SCENARIOS:")
    print()
    print("   1. Full Pipeline Training:")
    print("      Raw data → Preprocess → Features → Train")
    print("      Verify: Model trained successfully")
    print("   ")
    print("   2. Full Pipeline Prediction:")
    print("      Raw data → Preprocess → Features → Predict")
    print("      Verify: Predictions valid")
    print("   ")
    print("   3. Missing Values:")
    print("      Data with NaN → Pipeline handles → Predictions")
    print("      Verify: No NaN in output")
    print("   ")
    print("   4. Outliers:")
    print("      Data with outliers → Pipeline filters → Predictions")
    print("      Verify: Outliers handled")
    print("   ")
    print("   5. Reproducibility:")
    print("      predict(X) == predict(X)")
    print("      Verify: Same input → Same output")
    print("   ")
    print("   6. Batch Consistency:")
    print("      predict([X1, X2]) == [predict(X1), predict(X2)]")
    print("      Verify: Batch = Individual predictions")
    print()


# ============================================================================
# 4. Testing Data Flow
# ============================================================================

def test_e2e_data_shape_preservation(training_data, training_labels, test_data):
    """Test that data shape is preserved through pipeline."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    # Predict
    predictions = pipeline.predict(test_data)
    
    # Number of predictions should match input rows
    assert len(predictions) == len(test_data)


def test_e2e_no_data_leakage(training_data, training_labels):
    """Test that training and test data don't leak."""
    # Split data
    train_data = training_data.iloc[:4]
    train_labels = training_labels.iloc[:4]
    test_data = training_data.iloc[4:]
    
    pipeline = MLPipeline()
    
    # Train only on training data
    pipeline.train(train_data, train_labels)
    
    # Predict on test data
    predictions = pipeline.predict(test_data)
    
    # Should work without seeing test data during training
    assert predictions.shape == (2,)


def test_e2e_column_names_consistent(training_data, training_labels, test_data):
    """Test that column names are consistent throughout pipeline."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    # Test data with same columns
    predictions = pipeline.predict(test_data)
    assert predictions.shape == (len(test_data),)
    
    # Test data with different column order (should still work)
    test_reordered = test_data[['income', 'age']]
    predictions_reordered = pipeline.predict(test_reordered)
    
    # Should handle different column order
    # (In production, you'd want to enforce order)
    assert predictions_reordered.shape == (len(test_data),)


# ============================================================================
# 5. Performance Testing
# ============================================================================

def test_e2e_training_performance(training_data, training_labels):
    """Test that training completes in reasonable time."""
    pipeline = MLPipeline()
    
    start_time = time.time()
    pipeline.train(training_data, training_labels)
    elapsed = time.time() - start_time
    
    # Should train in less than 1 second
    assert elapsed < 1.0, f"Training too slow: {elapsed:.2f}s"


def test_e2e_prediction_performance(training_data, training_labels, test_data):
    """Test that prediction completes in reasonable time."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    start_time = time.time()
    pipeline.predict(test_data)
    elapsed = time.time() - start_time
    
    # Should predict in less than 100ms
    assert elapsed < 0.1, f"Prediction too slow: {elapsed:.2f}s"


def test_e2e_large_batch_prediction(training_data, training_labels):
    """Test prediction with large batch."""
    pipeline = MLPipeline()
    pipeline.train(training_data, training_labels)
    
    # Generate large test set
    large_test = pd.DataFrame({
        'age': [30] * 1000,
        'income': [50000] * 1000,
    })
    
    start_time = time.time()
    predictions = pipeline.predict(large_test)
    elapsed = time.time() - start_time
    
    # Should handle 1000 predictions in < 1 second
    assert elapsed < 1.0
    assert predictions.shape == (1000,)


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 End-to-End Integration Testing\n")
    
    demo_why_e2e_testing()
    demo_e2e_tests()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why E2E Testing:
   - Unit tests aren't enough
   - Components work individually but fail together
   - E2E tests verify complete system
   - Catch integration bugs

2. What to Test E2E:
   ✅ Full pipeline training
   ✅ Full pipeline prediction
   ✅ Missing value handling
   ✅ Outlier handling
   ✅ Reproducibility
   ✅ Batch consistency
   ✅ Data flow correctness
   ✅ Performance

3. E2E Test Structure:
   raw_data → preprocess → features → model → predictions
   Test each step works together

4. Common E2E Bugs Caught:
   - Column name mismatches
   - Data type incompatibilities
   - Feature shape mismatches
   - Data leakage
   - State management errors

E2E Testing Checklist:
```
Training:
□ Pipeline trains successfully
□ Training completes in time
□ Handles missing values
□ Handles outliers
□ No data leakage

Prediction:
□ Pipeline predicts successfully
□ Predictions in valid range
□ Prediction completes in time
□ Handles missing test data
□ Batch = Individual predictions

Data Flow:
□ Shape preserved
□ Column names consistent
□ Data types preserved
□ No NaN/Inf in output

Performance:
□ Training < 1s (small data)
□ Prediction < 100ms
□ Handles large batches (1000+)
```

Unit vs E2E Tests:
```
Unit Tests:
• Fast (ms)
• Isolated
• Mock dependencies
• Test individual functions
• Run frequently (every commit)

E2E Tests:
• Slower (seconds)
• Integrated
• Real dependencies
• Test complete system
• Run less frequently (before deploy)

Use both! 🎯
Unit tests catch bugs early
E2E tests catch integration bugs
```

Next Steps:
→ 02_mock_external_apis.py (Mock external services)
→ 03_database_tests.py (Test with databases)
→ 04_performance_tests.py (Load testing)
""")


if __name__ == "__main__":
    main()
