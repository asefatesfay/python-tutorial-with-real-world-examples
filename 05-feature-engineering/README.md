# Module 5: Feature Engineering for ML/AI

**Goal**: Transform raw data into features that make ML models work better.

**Key Insight**: "Garbage in, garbage out." Good features > Complex models.

## 📚 What You'll Learn

### Core Techniques
- Numerical features (scaling, binning, transformations)
- Categorical encoding (one-hot, label, target)
- Text features (TF-IDF, embeddings)
- Date/time features (extract temporal patterns)
- Feature interactions (combine features)
- Dimensionality reduction (PCA, feature selection)

### Advanced Techniques
- Handling missing data (imputation strategies)
- Outlier treatment (cap, remove, transform)
- Feature creation from domain knowledge
- Automated feature engineering (Featuretools)
- Feature importance analysis

## 🎯 Real-World Applications

- **E-commerce**: Customer lifetime value, purchase patterns
- **Finance**: Risk scoring, fraud detection
- **Healthcare**: Disease prediction, patient outcomes
- **NLP**: Text classification, sentiment analysis
- **Computer Vision**: Image preprocessing, augmentation
- **Time Series**: Forecasting, anomaly detection

## 📂 Module Structure

```
05-feature-engineering/
├── README.md (you are here)
├── examples/
│   ├── 01_numerical_features.py     # Scaling, binning, transforms
│   ├── 02_categorical_encoding.py   # Encoding techniques
│   ├── 03_text_features.py          # TF-IDF, embeddings
│   ├── 04_datetime_features.py      # Temporal patterns
│   ├── 05_feature_interactions.py   # Combining features
│   ├── 06_dimensionality_reduction.py  # PCA, feature selection
│   └── 07_complete_pipeline.py      # End-to-end example
└── mini_project/
    └── customer_churn_prediction.py # Real ML project
```

## 💡 Feature Engineering Principles

**1. Domain Knowledge Beats Algorithms**
- Understanding your data > fancy techniques
- Business logic → better features

**2. Start Simple, Add Complexity**
- Basic features first
- Test impact before adding more

**3. Avoid Data Leakage**
- Don't use future information
- Fit on train, transform on test

**4. Feature Quality > Quantity**
- 10 good features > 100 mediocre ones
- Remove redundant/correlated features

## 🔧 Common Feature Types

**Numerical**:
- Continuous: age, price, distance
- Discrete: count, rating, rank

**Categorical**:
- Nominal: color, category, country
- Ordinal: rating, education level

**Text**:
- Short: product names, tags
- Long: reviews, descriptions, documents

**DateTime**:
- Timestamps: order_date, login_time
- Durations: session_length, days_since

**Derived**:
- Ratios: price_per_sqft
- Aggregates: avg_purchase_last_30_days
- Interactions: age * income

## 🎯 Feature Engineering Workflow

```
Raw Data
   ↓
1. Understand Data (EDA)
   ↓
2. Handle Missing Values
   ↓
3. Encode Categoricals
   ↓
4. Scale Numericals
   ↓
5. Create New Features
   ↓
6. Select Best Features
   ↓
ML-Ready Data
```

## 📊 Impact on Model Performance

Example: House Price Prediction

| Features | R² Score |
|----------|----------|
| Raw features (5) | 0.65 |
| + Scaling | 0.70 |
| + Polynomial features | 0.78 |
| + Domain features | 0.85 |
| + Feature selection | 0.87 |

**5% improvement** from good feature engineering!

## 🚀 Quick Start

```bash
# Install dependencies
poetry add numpy pandas scikit-learn

# Start with numerical features
poetry run python 05-feature-engineering/examples/01_numerical_features.py

# Try the complete pipeline
poetry run python 05-feature-engineering/examples/07_complete_pipeline.py
```

---

**Remember**: Better features make simple models outperform complex ones! 🎯
