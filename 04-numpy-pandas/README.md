# Module 4: NumPy & Pandas for ML/AI

**Goal**: Master efficient data manipulation and numerical computing for ML/AI.

**Why This Matters**: Raw Python is slow. NumPy/Pandas are 10-100x faster and handle the heavy lifting in ML pipelines.

## 📚 What You'll Learn

### NumPy - Numerical Computing
- N-dimensional arrays (the foundation of ML)
- Vectorized operations (no loops needed!)
- Broadcasting (smart array operations)
- Array manipulation (reshape, slice, stack)
- Mathematical operations (matrix ops, statistics)
- Random number generation (sampling, initialization)

### Pandas - Data Manipulation
- DataFrames (tables with superpowers)
- Reading data (CSV, JSON, SQL, APIs)
- Data cleaning (missing values, duplicates)
- Data transformation (filter, group, aggregate)
- Feature engineering (create new features)
- Time series operations

## 🎯 Real-World Applications

**NumPy**:
- Image processing (arrays of pixels)
- Neural network operations (matrix multiplication)
- Statistical computations (mean, std, correlation)
- Signal processing (audio, video)

**Pandas**:
- Loading datasets for ML
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing pipelines
- Handling real-world messy data

## 📂 Module Structure

```
04-numpy-pandas/
├── README.md (you are here)
├── examples/
│   ├── 01_numpy_basics.py           # Arrays, indexing, operations
│   ├── 02_numpy_advanced.py         # Broadcasting, vectorization, performance
│   ├── 03_pandas_basics.py          # DataFrames, reading, basic ops
│   ├── 04_pandas_data_cleaning.py   # Missing data, duplicates, outliers
│   ├── 05_pandas_feature_engineering.py  # Creating features for ML
│   └── 06_numpy_pandas_for_ml.py    # Real ML preprocessing pipeline
└── data/
    └── sample_datasets.csv          # Practice data
```

## 🔧 Prerequisites

**Python Skills**: Modules 0-3 completed
**Math Background**: Module 3 (understand vectors, matrices)

**Install**:
```bash
poetry add numpy pandas matplotlib seaborn scikit-learn
```

## 💡 Why NumPy & Pandas?

| Task | Pure Python | NumPy/Pandas | Speedup |
|------|-------------|--------------|---------|
| Sum 1M numbers | 100ms | 1ms | 100x |
| Matrix multiply | 10s | 100ms | 100x |
| Filter DataFrame | 1s | 10ms | 100x |
| Load CSV | 30s | 1s | 30x |

**Bottom line**: NumPy/Pandas make ML feasible at scale.

## 🚀 Quick Start

```bash
# Install dependencies
poetry add numpy pandas matplotlib

# Start with NumPy basics
poetry run python 04-numpy-pandas/examples/01_numpy_basics.py

# Then move to Pandas
poetry run python 04-numpy-pandas/examples/03_pandas_basics.py
```

## 📊 Key Concepts

**NumPy Arrays**:
- Like Python lists but much faster
- Fixed type (all elements same type)
- Vectorized operations (no explicit loops)
- Broadcasting (operations on different shapes)

**Pandas DataFrames**:
- Like Excel spreadsheet in Python
- Labeled rows and columns
- Built on NumPy (fast!)
- Rich data manipulation API

## 🎓 Learning Path

1. **NumPy Basics** → Understand arrays, indexing
2. **NumPy Advanced** → Vectorization, broadcasting
3. **Pandas Basics** → DataFrames, reading data
4. **Data Cleaning** → Handle real-world messiness
5. **Feature Engineering** → Create ML-ready features
6. **ML Pipeline** → Combine everything

---

**Ready?** Let's build fast, efficient data pipelines! 🚀
