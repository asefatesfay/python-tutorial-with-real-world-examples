"""
Data Validation with Pydantic and pytest

Learn how to validate input data for ML systems.
Focus: Schema validation, data quality checks, error handling.

Install: poetry add --group dev pytest pydantic pandas
Run: pytest unit_tests/05_data_validation.py -v
"""

import pytest
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# 1. Why Data Validation is Critical
# ============================================================================

def demo_why_data_validation():
    """
    Why data validation is critical for ML systems.
    """
    print("=" * 70)
    print("1. Why Data Validation is Critical")
    print("=" * 70)
    print()
    
    print("💥 REAL-WORLD VALIDATION HORROR STORIES:")
    print()
    print("   Story 1: The Million Dollar Bug")
    print("   • Input: age = 999")
    print("   • Expected: 18-100")
    print("   • Model: Learned 999 = default value")
    print("   • Result: Wrong predictions for 20% of users")
    print("   • Impact: $1M in lost revenue 💀")
    print("   • Cause: No validation!")
    print()
    print("   Story 2: The Negative Price")
    print("   • Input: price = -50")
    print("   • Expected: price > 0")
    print("   • Model: Crashed with math error")
    print("   • Result: 100% API failures")
    print("   • Impact: 4-hour outage 💀")
    print("   • Cause: No validation!")
    print()
    print("   Story 3: The Type Mismatch")
    print("   • Input: income = 'fifty thousand'")
    print("   • Expected: income = 50000 (number)")
    print("   • Result: TypeError in production")
    print("   • Impact: Angry users + support tickets 💀")
    print("   • Cause: No validation!")
    print()
    
    print("🎯 WHY VALIDATION MATTERS:")
    print()
    print("   Real-world data is messy:")
    print("   • Users make typos")
    print("   • APIs return unexpected formats")
    print("   • Files have corrupt data")
    print("   • Sensors malfunction")
    print("   • Data drift over time")
    print()
    print("   Without validation:")
    print("   • Garbage in → Garbage out")
    print("   • Silent failures")
    print("   • Wrong predictions")
    print("   • Production crashes")
    print()
    print("   With validation:")
    print("   • Catch errors early")
    print("   • Clear error messages")
    print("   • Reliable predictions")
    print("   • Happy users ✅")
    print()
    
    print("💰 ROI OF VALIDATION:")
    print()
    print("   Without validation:")
    print("   • Bug in production: 4-hour outage")
    print("   • Emergency debugging: 8 hours")
    print("   • Data cleanup: 40 hours")
    print("   • Lost revenue: $100,000")
    print()
    print("   With validation:")
    print("   • Bug caught in dev: 1 minute")
    print("   • Clear error message")
    print("   • User fixes input")
    print("   • Lost revenue: $0")
    print()
    print("   ROI: ∞ (Prevent disasters!) ✅")
    print()


# ============================================================================
# 2. Basic Pydantic Validation
# ============================================================================

class UserProfile(BaseModel):
    """User profile with validation."""
    user_id: int = Field(..., gt=0)
    age: int = Field(..., ge=18, le=120)
    income: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    email: str
    name: Optional[str] = None


def test_valid_user_profile():
    """Test that valid user profile passes validation."""
    user = UserProfile(
        user_id=123,
        age=30,
        income=50000.0,
        credit_score=700,
        email="user@example.com"
    )
    
    assert user.user_id == 123
    assert user.age == 30
    assert user.income == 50000.0


def test_invalid_age_too_young():
    """Test that age < 18 is rejected."""
    with pytest.raises(ValueError):
        UserProfile(
            user_id=123,
            age=15,  # Too young!
            income=50000.0,
            credit_score=700,
            email="user@example.com"
        )


def test_invalid_age_too_old():
    """Test that age > 120 is rejected."""
    with pytest.raises(ValueError):
        UserProfile(
            user_id=123,
            age=150,  # Too old!
            income=50000.0,
            credit_score=700,
            email="user@example.com"
        )


def test_invalid_negative_income():
    """Test that negative income is rejected."""
    with pytest.raises(ValueError):
        UserProfile(
            user_id=123,
            age=30,
            income=-1000.0,  # Negative!
            credit_score=700,
            email="user@example.com"
        )


def test_missing_required_field():
    """Test that missing required field is rejected."""
    with pytest.raises(ValueError):
        UserProfile(
            user_id=123,
            age=30,
            # Missing income!
            credit_score=700,
            email="user@example.com"
        )


def test_optional_field():
    """Test that optional field can be omitted."""
    user = UserProfile(
        user_id=123,
        age=30,
        income=50000.0,
        credit_score=700,
        email="user@example.com"
        # name is optional
    )
    
    assert user.name is None


def demo_basic_validation():
    """Demo basic Pydantic validation."""
    print("\n" + "=" * 70)
    print("2. Basic Pydantic Validation")
    print("=" * 70)
    print()
    
    print("✅ PYDANTIC FIELD CONSTRAINTS:")
    print()
    print("   class UserProfile(BaseModel):")
    print("       age: int = Field(..., ge=18, le=120)")
    print("       income: float = Field(..., gt=0)")
    print("       name: Optional[str] = None")
    print()
    print("   Constraints:")
    print("   • gt: greater than")
    print("   • ge: greater than or equal")
    print("   • lt: less than")
    print("   • le: less than or equal")
    print("   • ...: required field")
    print("   • Optional[T]: can be None")
    print()
    
    print("🚨 VALIDATION ERRORS:")
    print()
    print("   age=15  → ValidationError (< 18)")
    print("   age=150 → ValidationError (> 120)")
    print("   income=-1000 → ValidationError (< 0)")
    print("   Missing income → ValidationError (required)")
    print()


# ============================================================================
# 3. Custom Validators
# ============================================================================

class Transaction(BaseModel):
    """Transaction with custom validation."""
    transaction_id: str
    amount: float = Field(..., gt=0)
    currency: str = Field(..., pattern="^[A-Z]{3}$")
    timestamp: datetime
    
    @field_validator('amount')
    @classmethod
    def amount_not_too_large(cls, v):
        """Validate that amount is not suspiciously large."""
        if v > 1_000_000:
            raise ValueError("Amount too large (> $1M)")
        return v
    
    @field_validator('currency')
    @classmethod
    def currency_supported(cls, v):
        """Validate that currency is supported."""
        supported = ['USD', 'EUR', 'GBP']
        if v not in supported:
            raise ValueError(f"Currency {v} not supported. Use: {supported}")
        return v


def test_valid_transaction():
    """Test that valid transaction passes."""
    txn = Transaction(
        transaction_id="TXN123",
        amount=100.50,
        currency="USD",
        timestamp=datetime.now()
    )
    
    assert txn.amount == 100.50
    assert txn.currency == "USD"


def test_transaction_amount_too_large():
    """Test that amount > 1M is rejected."""
    with pytest.raises(ValueError, match="Amount too large"):
        Transaction(
            transaction_id="TXN123",
            amount=2_000_000,  # Too large!
            currency="USD",
            timestamp=datetime.now()
        )


def test_transaction_unsupported_currency():
    """Test that unsupported currency is rejected."""
    with pytest.raises(ValueError, match="not supported"):
        Transaction(
            transaction_id="TXN123",
            amount=100.50,
            currency="JPY",  # Not supported!
            timestamp=datetime.now()
        )


def test_transaction_invalid_currency_format():
    """Test that invalid currency format is rejected."""
    with pytest.raises(ValueError):
        Transaction(
            transaction_id="TXN123",
            amount=100.50,
            currency="usd",  # Lowercase!
            timestamp=datetime.now()
        )


def demo_custom_validators():
    """Demo custom validators."""
    print("\n" + "=" * 70)
    print("3. Custom Validators")
    print("=" * 70)
    print()
    
    print("🔧 CUSTOM VALIDATION:")
    print()
    print("   class Transaction(BaseModel):")
    print("       amount: float")
    print("       ")
    print("       @field_validator('amount')")
    print("       @classmethod")
    print("       def amount_not_too_large(cls, v):")
    print("           if v > 1_000_000:")
    print("               raise ValueError('Too large')")
    print("           return v")
    print()
    
    print("💡 WHEN TO USE CUSTOM VALIDATORS:")
    print()
    print("   • Business logic constraints")
    print("   • Cross-field validation")
    print("   • Complex rules")
    print("   • Data sanitization")
    print()


# ============================================================================
# 4. Model Validators (Cross-Field)
# ============================================================================

class LoanApplication(BaseModel):
    """Loan application with cross-field validation."""
    applicant_age: int = Field(..., ge=18, le=80)
    loan_amount: float = Field(..., gt=0)
    annual_income: float = Field(..., gt=0)
    loan_term_years: int = Field(..., ge=1, le=30)
    
    @model_validator(mode='after')
    def validate_debt_to_income(self):
        """Validate debt-to-income ratio."""
        monthly_payment = self.loan_amount / (self.loan_term_years * 12)
        monthly_income = self.annual_income / 12
        
        debt_to_income = monthly_payment / monthly_income
        
        if debt_to_income > 0.43:
            raise ValueError(
                f"Debt-to-income ratio too high: {debt_to_income:.2f} (max 0.43)"
            )
        
        return self


def test_valid_loan_application():
    """Test that valid loan application passes."""
    loan = LoanApplication(
        applicant_age=30,
        loan_amount=100_000,
        annual_income=60_000,
        loan_term_years=15
    )
    
    assert loan.loan_amount == 100_000


def test_loan_high_debt_to_income():
    """Test that high debt-to-income ratio is rejected."""
    with pytest.raises(ValueError, match="Debt-to-income ratio too high"):
        LoanApplication(
            applicant_age=30,
            loan_amount=500_000,  # Too high!
            annual_income=50_000,
            loan_term_years=10
        )


def demo_model_validators():
    """Demo model validators."""
    print("\n" + "=" * 70)
    print("4. Model Validators (Cross-Field)")
    print("=" * 70)
    print()
    
    print("🔗 CROSS-FIELD VALIDATION:")
    print()
    print("   class LoanApplication(BaseModel):")
    print("       loan_amount: float")
    print("       annual_income: float")
    print("       ")
    print("       @model_validator(mode='after')")
    print("       def validate_debt_to_income(self):")
    print("           ratio = self.loan_amount / self.annual_income")
    print("           if ratio > 0.43:")
    print("               raise ValueError('Too high')")
    print("           return self")
    print()
    
    print("💡 USE CASES:")
    print()
    print("   • Date ranges (start < end)")
    print("   • Ratios (debt/income)")
    print("   • Conditionals (if A then B required)")
    print("   • Complex business rules")
    print()


# ============================================================================
# 5. Validating DataFrames
# ============================================================================

class DataFrameSchema(BaseModel):
    """Schema for DataFrame validation."""
    
    class Config:
        arbitrary_types_allowed = True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate DataFrame schema and data quality."""
        errors = []
        
        # Check required columns
        required_cols = ['age', 'income', 'score']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
        
        if errors:
            raise ValueError(f"Schema errors: {', '.join(errors)}")
        
        # Check data types
        if df['age'].dtype not in [np.int64, np.float64]:
            raise ValueError("Column 'age' must be numeric")
        
        # Check value ranges
        if (df['age'] < 0).any() or (df['age'] > 120).any():
            raise ValueError("Column 'age' has invalid values")
        
        if (df['income'] < 0).any():
            raise ValueError("Column 'income' has negative values")
        
        # Check for missing values
        if df[required_cols].isna().any().any():
            raise ValueError("DataFrame contains missing values")
        
        return True


def test_valid_dataframe():
    """Test that valid DataFrame passes validation."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'score': [0.8, 0.9, 0.7]
    })
    
    assert DataFrameSchema.validate_dataframe(df) is True


def test_dataframe_missing_column():
    """Test that missing column is rejected."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
        # Missing 'score'!
    })
    
    with pytest.raises(ValueError, match="Missing column: score"):
        DataFrameSchema.validate_dataframe(df)


def test_dataframe_invalid_age():
    """Test that invalid age values are rejected."""
    df = pd.DataFrame({
        'age': [25, 150, 35],  # 150 is invalid!
        'income': [50000, 60000, 70000],
        'score': [0.8, 0.9, 0.7]
    })
    
    with pytest.raises(ValueError, match="invalid values"):
        DataFrameSchema.validate_dataframe(df)


def test_dataframe_missing_values():
    """Test that missing values are rejected."""
    df = pd.DataFrame({
        'age': [25, np.nan, 35],  # Missing!
        'income': [50000, 60000, 70000],
        'score': [0.8, 0.9, 0.7]
    })
    
    with pytest.raises(ValueError, match="missing values"):
        DataFrameSchema.validate_dataframe(df)


def demo_dataframe_validation():
    """Demo DataFrame validation."""
    print("\n" + "=" * 70)
    print("5. Validating DataFrames")
    print("=" * 70)
    print()
    
    print("📊 DATAFRAME VALIDATION:")
    print()
    print("   @staticmethod")
    print("   def validate_dataframe(df: pd.DataFrame):")
    print("       # Check required columns")
    print("       required = ['age', 'income']")
    print("       for col in required:")
    print("           if col not in df.columns:")
    print("               raise ValueError(f'Missing: {col}')")
    print("       ")
    print("       # Check data types")
    print("       if df['age'].dtype not in [int, float]:")
    print("           raise ValueError('age must be numeric')")
    print("       ")
    print("       # Check value ranges")
    print("       if (df['age'] < 0).any():")
    print("           raise ValueError('age cannot be negative')")
    print()
    
    print("✅ WHAT TO VALIDATE:")
    print()
    print("   1. Schema:")
    print("      • Required columns present")
    print("      • Column names correct")
    print("   ")
    print("   2. Data Types:")
    print("      • Numeric columns are numeric")
    print("      • Date columns are dates")
    print("   ")
    print("   3. Value Ranges:")
    print("      • No negative ages")
    print("      • Scores between 0 and 1")
    print("   ")
    print("   4. Data Quality:")
    print("      • No missing values")
    print("      • No duplicates")
    print("      • No outliers")
    print()


# ============================================================================
# 6. Validation Error Messages
# ============================================================================

def test_validation_error_messages():
    """Test that validation errors have clear messages."""
    try:
        UserProfile(
            user_id=123,
            age=15,  # Too young
            income=50000.0,
            credit_score=700,
            email="user@example.com"
        )
    except ValueError as e:
        error_msg = str(e)
        
        # Error message should mention field and constraint
        assert "age" in error_msg.lower()


def demo_error_messages():
    """Demo validation error messages."""
    print("\n" + "=" * 70)
    print("6. Validation Error Messages")
    print("=" * 70)
    print()
    
    print("💬 CLEAR ERROR MESSAGES:")
    print()
    print("   Bad error:")
    print("   ❌ 'Invalid input'")
    print()
    print("   Good error:")
    print("   ✅ 'Field age: value 15 is less than minimum 18'")
    print()
    
    print("📝 ERROR MESSAGE BEST PRACTICES:")
    print()
    print("   1. Specify field name")
    print("   2. Explain what's wrong")
    print("   3. Show expected value/range")
    print("   4. Suggest fix if possible")
    print()
    
    print("   Example:")
    print("   'Age must be between 18 and 120, got 15'")
    print()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Data Validation with Pydantic\n")
    
    demo_why_data_validation()
    demo_basic_validation()
    demo_custom_validators()
    demo_model_validators()
    demo_dataframe_validation()
    demo_error_messages()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Validation is Critical:
   - Real-world data is messy
   - Prevents production crashes
   - Catches errors early
   - ROI: ∞ (prevent disasters)

2. Pydantic Basics:
   - Field constraints (gt, ge, lt, le)
   - Required vs Optional
   - Type validation
   - Automatic error messages

3. Custom Validators:
   @field_validator('field_name')
   - Business logic validation
   - Complex constraints
   - Data sanitization

4. Model Validators:
   @model_validator(mode='after')
   - Cross-field validation
   - Complex business rules
   - Ratio constraints

5. DataFrame Validation:
   - Schema validation
   - Data type checks
   - Value range checks
   - Data quality checks

Validation Checklist:
```
Schema:
□ Required columns present
□ Column names correct
□ Data types correct

Value Constraints:
□ Ranges valid (age: 18-120)
□ No negative values (income > 0)
□ Formats correct (email, phone)

Data Quality:
□ No missing values
□ No duplicates
□ No outliers
□ No NaN/Inf

Business Rules:
□ Cross-field constraints
□ Ratios in valid range
□ Dates in order
```

Pydantic Examples:
```python
from pydantic import BaseModel, Field, field_validator

class User(BaseModel):
    age: int = Field(..., ge=18, le=120)
    income: float = Field(..., gt=0)
    email: str
    name: Optional[str] = None
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# Usage
try:
    user = User(age=30, income=50000, email='test@example.com')
except ValueError as e:
    print(f"Validation error: {e}")
```

Next Steps:
→ integration_tests/ (End-to-end testing)
→ Test complete ML pipelines
""")


if __name__ == "__main__":
    main()
