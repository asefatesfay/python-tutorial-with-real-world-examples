# Module 6: Machine Learning Fundamentals

**Goal**: Build, train, and evaluate ML models from scratch and with scikit-learn.

**Approach**: Implement algorithms yourself first, then use libraries efficiently.

## 📚 What You'll Learn

### Supervised Learning
- **Linear Regression**: Predict continuous values
- **Logistic Regression**: Binary classification
- **Decision Trees**: Interpretable models
- **Random Forests**: Ensemble power
- **Gradient Boosting**: XGBoost, LightGBM
- **Support Vector Machines**: Margin-based learning

### Unsupervised Learning
- **K-Means Clustering**: Group similar data
- **Hierarchical Clustering**: Dendrograms
- **PCA**: Dimensionality reduction
- **Anomaly Detection**: Outlier identification

### Model Evaluation
- Train/validation/test split
- Cross-validation
- Metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix
- Learning curves
- Bias-variance tradeoff

### Model Optimization
- Hyperparameter tuning (Grid/Random search)
- Feature selection
- Regularization (L1, L2)
- Ensemble methods

## 🎯 Real-World Projects

Each algorithm applied to real problems:
- **Linear Regression**: House price prediction
- **Logistic Regression**: Email spam detection
- **Decision Trees**: Customer churn prediction
- **Random Forest**: Credit risk scoring
- **K-Means**: Customer segmentation
- **Anomaly Detection**: Fraud detection

## 📂 Module Structure

```
06-machine-learning-fundamentals/
├── README.md (you are here)
├── from_scratch/
│   ├── 01_linear_regression.py      # Implement from scratch
│   ├── 02_logistic_regression.py
│   ├── 03_decision_tree.py
│   ├── 04_k_means.py
│   └── 05_neural_network.py         # Simple feedforward NN
├── with_sklearn/
│   ├── 01_regression_models.py      # Using scikit-learn
│   ├── 02_classification_models.py
│   ├── 03_ensemble_methods.py
│   ├── 04_model_evaluation.py
│   ├── 05_hyperparameter_tuning.py
│   └── 06_pipeline_complete.py      # End-to-end ML pipeline
└── projects/
    ├── house_price_prediction/
    ├── spam_classification/
    └── customer_segmentation/
```

## 💡 ML Workflow

```
1. Problem Definition
   ↓
2. Data Collection
   ↓
3. Exploratory Data Analysis (EDA)
   ↓
4. Feature Engineering
   ↓
5. Model Selection
   ↓
6. Training
   ↓
7. Evaluation
   ↓
8. Hyperparameter Tuning
   ↓
9. Deployment
```

## 🎓 Learning Approach

**Phase 1: Understanding**
- Implement algorithms from scratch
- Understand the math behind them
- See how gradient descent works

**Phase 2: Practice**
- Use scikit-learn efficiently
- Build complete pipelines
- Handle real-world data

**Phase 3: Projects**
- End-to-end ML projects
- Kaggle-style competitions
- Portfolio pieces

## 📊 Model Selection Guide

| Problem Type | Start With | Also Try |
|--------------|-----------|----------|
| Regression | Linear Regression | Random Forest, XGBoost |
| Binary Classification | Logistic Regression | Random Forest, XGBoost |
| Multi-class | Logistic Regression | Random Forest, Neural Net |
| Clustering | K-Means | DBSCAN, Hierarchical |
| Anomaly Detection | Isolation Forest | One-Class SVM |

## 🔧 Key Metrics

**Regression**:
- MSE, RMSE, MAE
- R² score
- Mean Absolute Percentage Error (MAPE)

**Classification**:
- Accuracy (use cautiously!)
- Precision, Recall, F1
- AUC-ROC, AUC-PR
- Confusion Matrix

**Clustering**:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

## ⚠️ Common Pitfalls

1. **Not splitting data properly** → Overfitting
2. **Using accuracy for imbalanced data** → Misleading
3. **Data leakage** → Unrealistically good results
4. **Not scaling features** → Poor performance
5. **Overfitting to validation set** → Use test set!

## 🚀 Quick Start

```bash
# Install dependencies
poetry add numpy pandas scikit-learn matplotlib seaborn

# Build from scratch
poetry run python 06-machine-learning-fundamentals/from_scratch/01_linear_regression.py

# Use scikit-learn
poetry run python 06-machine-learning-fundamentals/with_sklearn/01_regression_models.py

# Complete project
poetry run python 06-machine-learning-fundamentals/projects/house_price_prediction/main.py
```

## 🎯 Expected Outcomes

After this module:
- ✅ Understand how ML algorithms work internally
- ✅ Implement gradient descent from scratch
- ✅ Use scikit-learn effectively
- ✅ Evaluate models properly
- ✅ Build complete ML pipelines
- ✅ Handle real-world datasets
- ✅ Have portfolio projects

---

**Remember**: Understanding > Memorizing. Build intuition! 🧠
