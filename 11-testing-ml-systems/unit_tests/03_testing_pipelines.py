"""
Testing Data Pipelines with pytest

Learn how to test data preprocessing, feature engineering, and ML pipelines.
Focus: Data transformations, pipeline steps, data validation.

Install: poetry add --group dev pytest pandas scikit-learn
Run: pytest unit_tests/03_testing_pipelines.py -v
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# 1. Why Test Data Pipelines?
# ============================================================================

def demo_why_test_pipelines():
    """
    Why testing data pipelines is critical.
    """
    print("=" * 70)
    print("1. Why Test Data Pipelines?")
    print("=" * 70)
    print()
    
    print("💥 REAL-WORLD HORROR STORIES:")
    print()
    print("   Story 1: The Missing Value Bug")
    print("   • Pipeline: fillna(0)")
    print("   • Bug: Some columns had -999 as missing")
    print("   • Result: Model learned -999 = valid value")
    print("   • Impact: 30% accuracy drop in production 💀")
    print("   • Cost: 2 weeks debugging + retrain model")
    print()
    print("   Story 2: The Feature Scaling Disaster")
    print("   • Pipeline: StandardScaler()")
    print("   • Bug: Forgot to save scaler parameters")
    print("   • Result: Different scaling in production")
    print("   • Impact: Predictions completely wrong 💀")
    print("   • Cost: Emergency rollback + model retrain")
    print()
    print("   Story 3: The Data Leakage Nightmare")
    print("   • Pipeline: Impute missing with mean")
    print("   • Bug: Calculated mean on entire dataset")
    print("   • Result: Test data leaked into training")
    print("   • Impact: 95% accuracy in dev, 60% in prod 💀")
    print("   • Cost: Model unusable, 3 weeks to fix")
    print()
    
    print("🎯 WHY PIPELINES ARE RISKY:")
    print()
    print("   1. Many Steps:")
    print("      Clean → Impute → Scale → Encode → Feature Engineer")
    print("      Any step can break!")
    print("   ")
    print("   2. Silent Failures:")
    print("      fillna(0) never crashes")
    print("      But might be wrong!")
    print("   ")
    print("   3. State Management:")
    print("      Fit on train, transform on test")
    print("      Easy to mix up!")
    print("   ")
    print("   4. Data Drift:")
    print("      Pipeline works on old data")
    print("      Fails on new data!")
    print()
    
    print("✅ WHAT TESTS PREVENT:")
    print()
    print("   • Missing value handling bugs")
    print("   • Feature scaling errors")
    print("   • Data leakage")
    print("   • Schema mismatches")
    print("   • Type conversion errors")
    print("   • NaN/Inf in output")
    print("   • Feature name mismatches")
    print()
    
    print("💰 ROI OF PIPELINE TESTS:")
    print()
    print("   Without tests:")
    print("   • Bug found in production: 2 weeks debugging")
    print("   • Emergency fixes: 40 hours")
    print("   • Model retrain: 1 week")
    print("   • Lost revenue: $50,000")
    print()
    print("   With tests:")
    print("   • Bug found in dev: 10 minutes")
    print("   • Fix: 30 minutes")
    print("   • Lost revenue: $0")
    print()
    print("   ROI: 100:1 (Save 100x debugging time!) ✅")
    print()


# ============================================================================
# 2. Simple Data Pipeline
# ============================================================================

class DataPipeline:
    """
    Simple data preprocessing pipeline.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.feature_means = {}
        self.feature_stds = {}
    
    def fit(self, df: pd.DataFrame) -> 'DataPipeline':
        """Fit pipeline on training data."""
        # Calculate statistics
        self.feature_means = df.select_dtypes(include=[np.number]).mean().to_dict()
        self.feature_stds = df.select_dtypes(include=[np.number]).std().to_dict()
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted statistics."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        df_copy = df.copy()
        
        # Fill missing values with mean
        for col in self.feature_means:
            if col in df_copy.columns:
                df_copy[col].fillna(self.feature_means[col], inplace=True)
        
        # Standardize (z-score)
        for col in self.feature_stds:
            if col in df_copy.columns and self.feature_stds[col] > 0:
                df_copy[col] = (
                    (df_copy[col] - self.feature_means[col]) / self.feature_stds[col]
                )
        
        return df_copy
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


# ============================================================================
# 3. Testing Pipeline State
# ============================================================================

@pytest.fixture
def sample_data():
    """Fixture providing sample dataframe."""
    return pd.DataFrame({
        'age': [25, 30, np.nan, 40],
        'income': [50000, 60000, 70000, np.nan],
        'score': [0.8, 0.9, 0.7, 0.95]
    })


def test_pipeline_not_fitted_raises_error():
    """Test that unfitted pipeline raises error."""
    pipeline = DataPipeline()
    df = pd.DataFrame({'age': [25, 30]})
    
    with pytest.raises(ValueError, match="Pipeline not fitted"):
        pipeline.transform(df)


def test_pipeline_fit_sets_fitted_flag(sample_data):
    """Test that fit() sets is_fitted flag."""
    pipeline = DataPipeline()
    assert pipeline.is_fitted is False
    
    pipeline.fit(sample_data)
    assert pipeline.is_fitted is True


def test_pipeline_fit_calculates_statistics(sample_data):
    """Test that fit() calculates correct statistics."""
    pipeline = DataPipeline()
    pipeline.fit(sample_data)
    
    # Check means
    assert 'age' in pipeline.feature_means
    assert 'income' in pipeline.feature_means
    assert 'score' in pipeline.feature_means
    
    # Check specific values (approximately)
    expected_age_mean = (25 + 30 + 40) / 3  # NaN excluded
    assert pytest.approx(pipeline.feature_means['age'], abs=0.1) == expected_age_mean


def demo_testing_pipeline_state():
    """Demo testing pipeline state."""
    print("\n" + "=" * 70)
    print("2. Testing Pipeline State")
    print("=" * 70)
    print()
    
    print("🔒 WHY STATE MATTERS:")
    print()
    print("   Stateful pipeline:")
    print("   1. fit(train_data)   → Learn statistics")
    print("   2. transform(test_data) → Apply statistics")
    print()
    print("   Bug example:")
    print("   pipeline.transform(test_data)  ← Not fitted!")
    print("   → Wrong results or crash")
    print()
    
    print("✅ WHAT TO TEST:")
    print()
    print("   1. Not Fitted Error:")
    print("      pipeline.transform(...) before fit()")
    print("      → ValueError ✅")
    print("   ")
    print("   2. Fitted Flag:")
    print("      After fit(), is_fitted == True")
    print("   ")
    print("   3. Statistics Calculated:")
    print("      After fit(), means/stds exist")
    print("   ")
    print("   4. Statistics Correct:")
    print("      Verify calculated values")
    print()


# ============================================================================
# 4. Testing Data Transformations
# ============================================================================

def test_pipeline_fills_missing_values(sample_data):
    """Test that missing values are filled."""
    pipeline = DataPipeline()
    pipeline.fit(sample_data)
    
    # Transform data with missing values
    transformed = pipeline.transform(sample_data)
    
    # No missing values in numeric columns
    assert transformed['age'].isna().sum() == 0
    assert transformed['income'].isna().sum() == 0


def test_pipeline_standardizes_features(sample_data):
    """Test that features are standardized."""
    pipeline = DataPipeline()
    transformed = pipeline.fit_transform(sample_data)
    
    # Standardized features have mean ≈ 0 and std ≈ 1
    for col in ['age', 'income', 'score']:
        mean = transformed[col].mean()
        std = transformed[col].std()
        
        # Check mean close to 0
        assert pytest.approx(mean, abs=0.1) == 0.0
        
        # Check std close to 1 (if not constant)
        if std > 0.01:
            assert pytest.approx(std, abs=0.2) == 1.0


def test_pipeline_preserves_data_shape(sample_data):
    """Test that pipeline preserves data shape."""
    pipeline = DataPipeline()
    transformed = pipeline.fit_transform(sample_data)
    
    # Same shape
    assert transformed.shape == sample_data.shape
    
    # Same columns
    assert list(transformed.columns) == list(sample_data.columns)


def test_pipeline_preserves_data_types(sample_data):
    """Test that pipeline preserves numeric data types."""
    pipeline = DataPipeline()
    transformed = pipeline.fit_transform(sample_data)
    
    # All columns still numeric
    assert transformed['age'].dtype in [np.float64, np.int64]
    assert transformed['income'].dtype in [np.float64, np.int64]
    assert transformed['score'].dtype in [np.float64, np.int64]


def test_pipeline_no_nans_or_infs(sample_data):
    """Test that output has no NaN or Inf values."""
    pipeline = DataPipeline()
    transformed = pipeline.fit_transform(sample_data)
    
    # Check for NaN
    assert not transformed.isna().any().any()
    
    # Check for Inf
    assert not np.isinf(transformed.select_dtypes(include=[np.number])).any().any()


def demo_testing_transformations():
    """Demo testing data transformations."""
    print("\n" + "=" * 70)
    print("3. Testing Data Transformations")
    print("=" * 70)
    print()
    
    print("✅ TRANSFORMATION TESTS:")
    print()
    print("   1. Missing Values:")
    print("      • Input: [1, 2, NaN, 4]")
    print("      • Output: [1, 2, 2.33, 4]  ← NaN filled")
    print("   ")
    print("   2. Standardization:")
    print("      • Mean ≈ 0")
    print("      • Std ≈ 1")
    print("   ")
    print("   3. Shape Preservation:")
    print("      • Input: (100, 5)")
    print("      • Output: (100, 5)  ← Same shape")
    print("   ")
    print("   4. No Invalid Values:")
    print("      • No NaN in output")
    print("      • No Inf in output")
    print("   ")
    print("   5. Data Types:")
    print("      • Numeric stays numeric")
    print("      • Categorical stays categorical")
    print()


# ============================================================================
# 5. Testing Pipeline Invariants
# ============================================================================

def test_pipeline_train_test_consistency(sample_data):
    """Test that train and test transforms are consistent."""
    # Split data
    train_data = sample_data.iloc[:2]
    test_data = sample_data.iloc[2:]
    
    # Fit on train
    pipeline = DataPipeline()
    pipeline.fit(train_data)
    
    # Transform both
    train_transformed = pipeline.transform(train_data)
    test_transformed = pipeline.transform(test_data)
    
    # Both should have same columns
    assert list(train_transformed.columns) == list(test_transformed.columns)
    
    # Both should have no NaNs
    assert not train_transformed.isna().any().any()
    assert not test_transformed.isna().any().any()


def test_pipeline_multiple_transforms_identical(sample_data):
    """Test that multiple transforms give same result."""
    pipeline = DataPipeline()
    pipeline.fit(sample_data)
    
    # Transform twice
    result1 = pipeline.transform(sample_data)
    result2 = pipeline.transform(sample_data)
    
    # Should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_pipeline_fit_idempotent(sample_data):
    """Test that fitting twice gives same result."""
    pipeline1 = DataPipeline()
    pipeline1.fit(sample_data)
    result1 = pipeline1.transform(sample_data)
    
    # Fit again
    pipeline2 = DataPipeline()
    pipeline2.fit(sample_data)
    pipeline2.fit(sample_data)  # Fit twice!
    result2 = pipeline2.transform(sample_data)
    
    # Should be same (within numerical precision)
    pd.testing.assert_frame_equal(result1, result2, atol=0.01)


def demo_testing_invariants():
    """Demo testing pipeline invariants."""
    print("\n" + "=" * 70)
    print("4. Testing Pipeline Invariants")
    print("=" * 70)
    print()
    
    print("🔍 PIPELINE INVARIANTS:")
    print()
    print("   1. Train-Test Consistency:")
    print("      • Fit on train")
    print("      • Transform train and test")
    print("      • Both use same statistics ✅")
    print("   ")
    print("   2. Reproducibility:")
    print("      • transform(X) == transform(X)")
    print("      • Same input → Same output")
    print("   ")
    print("   3. Idempotency:")
    print("      • fit(X) then fit(X) again")
    print("      • Second fit same as first")
    print()
    
    print("💥 COMMON BUGS CAUGHT:")
    print()
    print("   Bug 1: Refit on test data")
    print("   pipeline.fit(train)")
    print("   pipeline.fit(test)  ← WRONG! Overwrites statistics")
    print()
    print("   Bug 2: Different scaling")
    print("   train_scaled = scale(train)")
    print("   test_scaled = scale(test)  ← WRONG! Different scaling")
    print()
    print("   Bug 3: Stateful transforms")
    print("   result1 = pipeline.transform(X)")
    print("   result2 = pipeline.transform(X)  ← Different! (Bug)")
    print()


# ============================================================================
# 6. Testing Edge Cases
# ============================================================================

def test_pipeline_with_empty_dataframe():
    """Test pipeline with empty dataframe."""
    pipeline = DataPipeline()
    df_empty = pd.DataFrame()
    
    # Fit on empty should work (but useless)
    pipeline.fit(df_empty)
    assert pipeline.is_fitted is True


def test_pipeline_with_single_row():
    """Test pipeline with single row."""
    df_single = pd.DataFrame({'age': [25], 'income': [50000]})
    
    pipeline = DataPipeline()
    result = pipeline.fit_transform(df_single)
    
    # Should work
    assert result.shape == (1, 2)


def test_pipeline_with_constant_column():
    """Test pipeline with constant column."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'constant': [1, 1, 1]  # No variance
    })
    
    pipeline = DataPipeline()
    result = pipeline.fit_transform(df)
    
    # Constant column stays constant (std=0, don't divide)
    assert result['constant'].std() == 0


def test_pipeline_with_all_missing():
    """Test pipeline with all missing values in a column."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'missing': [np.nan, np.nan, np.nan]
    })
    
    pipeline = DataPipeline()
    result = pipeline.fit_transform(df)
    
    # All missing → filled with NaN mean (which is NaN)
    # This might be a design choice
    assert result.shape == df.shape


def demo_testing_edge_cases():
    """Demo testing edge cases."""
    print("\n" + "=" * 70)
    print("5. Testing Edge Cases")
    print("=" * 70)
    print()
    
    print("🚨 EDGE CASES:")
    print()
    print("   1. Empty DataFrame:")
    print("      • 0 rows, 0 columns")
    print("      • Should fit without error")
    print("   ")
    print("   2. Single Row:")
    print("      • Can't calculate std properly")
    print("      • Handle gracefully")
    print("   ")
    print("   3. Constant Column:")
    print("      • All values same")
    print("      • std = 0 → Don't divide")
    print("   ")
    print("   4. All Missing:")
    print("      • Column all NaN")
    print("      • Can't impute with mean")
    print("   ")
    print("   5. Mixed Types:")
    print("      • Numeric + Categorical")
    print("      • Only process numeric")
    print()


# ============================================================================
# 7. Testing with Parametrize
# ============================================================================

@pytest.mark.parametrize("missing_strategy", ["mean", "median", "zero"])
def test_pipeline_different_imputation_strategies(missing_strategy):
    """Test pipeline with different imputation strategies."""
    # This would test different configurations
    # For now, just ensure it runs
    pass


@pytest.mark.parametrize("n_rows,n_cols", [
    (10, 3),
    (100, 5),
    (1000, 10),
])
def test_pipeline_with_different_sizes(n_rows, n_cols):
    """Test pipeline with different data sizes."""
    # Generate random data
    df = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'col{i}' for i in range(n_cols)]
    )
    
    pipeline = DataPipeline()
    result = pipeline.fit_transform(df)
    
    # Check shape preserved
    assert result.shape == (n_rows, n_cols)
    
    # Check no NaNs or Infs
    assert not result.isna().any().any()
    assert not np.isinf(result).any().any()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Testing Data Pipelines\n")
    
    demo_why_test_pipelines()
    demo_testing_pipeline_state()
    demo_testing_transformations()
    demo_testing_invariants()
    demo_testing_edge_cases()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Test Pipelines:
   - Silent failures common
   - Many transformation steps
   - State management errors
   - Data leakage risks
   - ROI: 100:1 (catch bugs early)

2. Pipeline State Tests:
   ✅ Not fitted → Error
   ✅ After fit → is_fitted = True
   ✅ Statistics calculated correctly
   ✅ Fit is idempotent

3. Transformation Tests:
   ✅ Missing values filled
   ✅ Features standardized (mean≈0, std≈1)
   ✅ Shape preserved
   ✅ Data types preserved
   ✅ No NaN/Inf in output

4. Invariant Tests:
   ✅ Train-test consistency
   ✅ Reproducibility: transform(X) == transform(X)
   ✅ Fit idempotent

5. Edge Cases:
   - Empty DataFrame
   - Single row
   - Constant column (std=0)
   - All missing values
   - Mixed data types

Testing Checklist:
```
Pipeline Tests:
□ Not fitted raises error
□ Fitted flag set after fit()
□ Statistics calculated correctly
□ Missing values handled
□ Features standardized
□ Shape preserved
□ Data types preserved
□ No NaN/Inf in output
□ Train-test consistency
□ Reproducible transforms
□ Idempotent fit
□ Empty DataFrame handled
□ Single row handled
□ Constant column handled
□ All missing handled
```

Common Bugs Prevented:
• Data leakage (fit on test)
• Wrong imputation strategy
• Feature scaling errors
• NaN propagation
• Type conversion errors
• Schema mismatches

Next Steps:
→ 04_testing_apis.py (Test ML APIs)
→ 05_data_validation.py (Validate input data)
""")


if __name__ == "__main__":
    main()
