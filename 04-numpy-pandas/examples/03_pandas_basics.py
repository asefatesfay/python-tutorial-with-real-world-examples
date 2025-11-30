"""
Pandas Basics for ML/AI Engineering

Real-world examples of data manipulation with Pandas.
Focus: Working with tabular data like CSVs, databases, APIs.

Install: poetry add pandas numpy
Run: poetry run python 04-numpy-pandas/examples/03_pandas_basics.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# 1. Why Pandas? Excel/SQL for Python
# ============================================================================

def demo_why_pandas():
    """
    INTUITION - Your Data is a Spreadsheet:
    
    Pandas DataFrame = Excel spreadsheet in Python
    - Rows: Records (users, transactions, events)
    - Columns: Features (name, age, amount, timestamp)
    - Operations: Filter, sort, group, aggregate
    
    Why better than Excel?
    - Handle millions of rows
    - Reproducible (code, not manual clicks)
    - Integrate with ML models
    - Automate repetitive tasks
    """
    print("=" * 70)
    print("1. Why Pandas? DataFrames vs Excel")
    print("=" * 70)
    print()
    print("💡 INTUITION: Excel Spreadsheet in Python")
    print("   DataFrame = Spreadsheet with superpowers")
    print("   - Handle millions of rows (Excel: 1M limit)")
    print("   - Reproducible analysis (no manual clicks)")
    print("   - Connect directly to ML models")
    print()
    
    # Create sample customer data
    data = {
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 32, 45, 28, 35],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'total_purchases': [1200, 450, 2300, 890, 1500],
        'account_created': ['2023-01-15', '2023-03-20', '2022-11-10', '2023-05-01', '2022-12-15']
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Customer DataFrame:")
    print(df)
    print()
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print()
    
    print("Quick Stats:")
    print(f"   Average age: {df['age'].mean():.1f} years")
    print(f"   Total revenue: ${df['total_purchases'].sum():,}")
    print(f"   Top city: {df['city'].value_counts().index[0]} ({df['city'].value_counts().values[0]} customers)")
    print()
    
    print("🚀 What you can do:")
    print("   - Filter: df[df['age'] > 30]")
    print("   - Sort: df.sort_values('total_purchases')")
    print("   - Group: df.groupby('city')['total_purchases'].sum()")
    print("   - Join: Combine multiple datasets")
    print("   - Export: Save to CSV, Excel, database")


# ============================================================================
# 2. Creating DataFrames - Loading Data
# ============================================================================

def demo_creating_dataframes():
    """
    INTUITION - Different Data Sources:
    
    Real projects get data from:
    - CSV files (exports, logs)
    - Dictionaries (API responses)
    - Databases (SQL queries)
    - Excel files (business reports)
    """
    print("\n" + "=" * 70)
    print("2. Creating DataFrames - Loading Data")
    print("=" * 70)
    print()
    print("💡 INTUITION: Data Sources in Real Projects")
    print("   - CSV: Log files, exports, datasets")
    print("   - Dict: API responses (JSON → DataFrame)")
    print("   - Database: SQL query results")
    print("   - Excel: Business reports, analytics")
    print()
    
    # Method 1: From dictionary (like API response)
    print("📡 From API Response (Dictionary):")
    api_response = {
        'product_id': [101, 102, 103],
        'name': ['Laptop', 'Mouse', 'Keyboard'],
        'price': [999.99, 29.99, 79.99],
        'stock': [15, 150, 87]
    }
    
    df_api = pd.DataFrame(api_response)
    print(df_api)
    print()
    
    # Method 2: From list of dictionaries (each row is a dict)
    print("📝 From List of Records:")
    records = [
        {'user': 'alice', 'score': 95, 'passed': True},
        {'user': 'bob', 'score': 67, 'passed': False},
        {'user': 'charlie', 'score': 88, 'passed': True}
    ]
    
    df_records = pd.DataFrame(records)
    print(df_records)
    print()
    
    # Method 3: From NumPy array
    print("🔢 From NumPy Array (ML Model Output):")
    # Simulate model predictions
    predictions = np.random.rand(5, 3)
    df_predictions = pd.DataFrame(
        predictions,
        columns=['cat_prob', 'dog_prob', 'bird_prob'],
        index=[f'image_{i}' for i in range(5)]
    )
    print(df_predictions.round(3))
    print()
    
    # Method 4: Read CSV (most common)
    print("📁 Reading CSV Files:")
    print("   df = pd.read_csv('data.csv')")
    print("   df = pd.read_csv('data.csv', parse_dates=['timestamp'])")
    print("   df = pd.read_csv('data.csv', usecols=['name', 'price'])")
    print()
    
    # Create sample CSV data
    csv_data = """date,product,quantity,revenue
2024-01-01,Laptop,2,1999.98
2024-01-01,Mouse,5,149.95
2024-01-02,Keyboard,3,239.97"""
    
    # In real code: df = pd.read_csv('sales.csv')
    from io import StringIO
    df_csv = pd.read_csv(StringIO(csv_data), parse_dates=['date'])
    print("Sample CSV data:")
    print(df_csv)


# ============================================================================
# 3. Selecting and Filtering - SQL for DataFrames
# ============================================================================

def demo_selecting_filtering():
    """
    INTUITION - Querying Your Data:
    
    Like SQL WHERE clause:
    - SELECT * FROM users WHERE age > 30
    - SELECT name, email FROM users WHERE city = 'NYC'
    
    Pandas equivalent:
    - df[df['age'] > 30]
    - df[df['city'] == 'NYC'][['name', 'email']]
    """
    print("\n" + "=" * 70)
    print("3. Selecting and Filtering")
    print("=" * 70)
    print()
    print("💡 INTUITION: SQL Queries in Python")
    print("   SQL: SELECT * FROM users WHERE age > 30")
    print("   Pandas: df[df['age'] > 30]")
    print()
    
    # Create sample e-commerce data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, 50),
        'customer_id': np.random.randint(1000, 1010, 50),
        'product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], 50),
        'quantity': np.random.randint(1, 5, 50),
        'price': np.random.choice([999.99, 29.99, 79.99, 299.99], 50),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 50)
    })
    
    df['revenue'] = df['quantity'] * df['price']
    
    print("📊 Sample E-commerce Data (first 5 rows):")
    print(df.head())
    print()
    
    # Select single column
    print("Select 'product' column:")
    print(df['product'].head())
    print()
    
    # Select multiple columns
    print("Select multiple columns:")
    print(df[['product', 'revenue']].head())
    print()
    
    # Filter rows (WHERE clause)
    print("Filter: High-value orders (revenue > $500):")
    high_value = df[df['revenue'] > 500]
    print(f"   Found {len(high_value)} orders")
    print(high_value[['customer_id', 'product', 'revenue']].head())
    print()
    
    # Multiple conditions (AND)
    print("Filter: Laptops in North region:")
    laptops_north = df[(df['product'] == 'Laptop') & (df['region'] == 'North')]
    print(f"   Found {len(laptops_north)} orders")
    print()
    
    # OR condition
    print("Filter: Laptops OR Monitors:")
    expensive = df[(df['product'] == 'Laptop') | (df['product'] == 'Monitor')]
    print(f"   Found {len(expensive)} orders")
    print()
    
    # isin (like SQL IN)
    print("Filter: Multiple products (IN clause):")
    selected_products = df[df['product'].isin(['Laptop', 'Monitor'])]
    print(f"   Found {len(selected_products)} orders")
    print()
    
    # String operations
    df_with_names = pd.DataFrame({
        'name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'David Smith'],
        'email': ['alice@example.com', 'bob@test.com', 'charlie@example.com', 'david@other.com']
    })
    
    print("String filtering (names containing 'Smith'):")
    smiths = df_with_names[df_with_names['name'].str.contains('Smith')]
    print(smiths)
    print()
    
    print("Email domain filter (@example.com):")
    example_users = df_with_names[df_with_names['email'].str.endswith('@example.com')]
    print(example_users)


# ============================================================================
# 4. GroupBy - Aggregations and Pivots
# ============================================================================

def demo_groupby():
    """
    INTUITION - Business Intelligence Queries:
    
    "Show me total sales BY product"
    "What's the average order value BY region"
    "Count orders BY customer BY month"
    
    This is the heart of data analysis!
    """
    print("\n" + "=" * 70)
    print("4. GroupBy - Aggregations")
    print("=" * 70)
    print()
    print("💡 INTUITION: Business Intelligence Queries")
    print("   'Show me total sales BY product'")
    print("   'Average order value BY region BY month'")
    print("   SQL: GROUP BY product, region")
    print()
    
    # Create realistic sales data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, 500),
        'product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'], 500),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
        'customer_type': np.random.choice(['New', 'Returning'], 500, p=[0.3, 0.7]),
        'quantity': np.random.randint(1, 5, 500),
        'price': np.random.choice([999, 29, 79, 299, 149], 500)
    })
    
    df['revenue'] = df['quantity'] * df['price']
    
    print("📊 Sales Data:")
    print(f"   {len(df)} transactions")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Simple groupby - total revenue by product
    print("💰 Total Revenue by Product:")
    by_product = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
    for product, revenue in by_product.items():
        print(f"   {product:10s}: ${revenue:,}")
    print()
    
    # Multiple aggregations
    print("📈 Product Statistics:")
    product_stats = df.groupby('product').agg({
        'revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    })
    print(product_stats.round(2))
    print()
    
    # Group by multiple columns
    print("🌍 Revenue by Region and Product (Top 10):")
    by_region_product = df.groupby(['region', 'product'])['revenue'].sum().sort_values(ascending=False)
    print(by_region_product.head(10))
    print()
    
    # Pivot table (Excel-style)
    print("📊 Pivot Table: Region × Product:")
    pivot = df.pivot_table(
        values='revenue',
        index='region',
        columns='product',
        aggfunc='sum',
        fill_value=0
    )
    print(pivot)
    print()
    
    # Time-based grouping
    df['month'] = df['date'].dt.to_period('M')
    print("📅 Monthly Revenue Trend:")
    monthly = df.groupby('month')['revenue'].sum()
    for month, revenue in monthly.head(6).items():
        print(f"   {month}: ${revenue:,}")
    print()
    
    # Customer segmentation
    print("👥 Customer Type Analysis:")
    customer_analysis = df.groupby('customer_type').agg({
        'revenue': ['sum', 'mean'],
        'quantity': 'sum',
        'customer_type': 'count'
    })
    customer_analysis.columns = ['total_revenue', 'avg_order', 'total_items', 'num_orders']
    print(customer_analysis.round(2))


# ============================================================================
# 5. Sorting and Ranking
# ============================================================================

def demo_sorting_ranking():
    """
    INTUITION - Leaderboards and Rankings:
    
    - Top customers by revenue
    - Best-selling products
    - Rank users by activity score
    - Find outliers (highest/lowest)
    """
    print("\n" + "=" * 70)
    print("5. Sorting and Ranking")
    print("=" * 70)
    print()
    print("💡 INTUITION: Leaderboards and Top N Analysis")
    print("   - Top 10 customers")
    print("   - Best-selling products")
    print("   - Rank by multiple criteria")
    print()
    
    # Create customer data
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': range(1000, 1050),
        'name': [f'Customer_{i}' for i in range(50)],
        'total_spent': np.random.randint(100, 5000, 50),
        'num_orders': np.random.randint(1, 50, 50),
        'account_age_days': np.random.randint(30, 365, 50),
        'last_order_days_ago': np.random.randint(0, 90, 50)
    })
    
    df['avg_order_value'] = df['total_spent'] / df['num_orders']
    
    print("📊 Customer Data Sample:")
    print(df.head())
    print()
    
    # Sort by single column
    print("💰 Top 10 Customers by Total Spent:")
    top_customers = df.sort_values('total_spent', ascending=False).head(10)
    print(top_customers[['name', 'total_spent', 'num_orders']])
    print()
    
    # Sort by multiple columns
    print("📈 Top Customers by Orders, then by Spend:")
    top_by_orders = df.sort_values(['num_orders', 'total_spent'], ascending=[False, False]).head(10)
    print(top_by_orders[['name', 'num_orders', 'total_spent']])
    print()
    
    # Ranking
    print("🏆 Customer Ranking (by total spent):")
    df['rank'] = df['total_spent'].rank(ascending=False, method='min')
    print(df.nsmallest(10, 'rank')[['name', 'total_spent', 'rank']])
    print()
    
    # Percentile ranking
    df['percentile'] = df['total_spent'].rank(pct=True)
    print("Top 10% Customers (percentile > 0.9):")
    top_10_pct = df[df['percentile'] > 0.9].sort_values('total_spent', ascending=False)
    print(top_10_pct[['name', 'total_spent', 'percentile']])
    print()
    
    # Multi-criteria scoring
    print("🎯 Composite Score (weighted ranking):")
    # Normalize and weight different metrics
    df['spend_score'] = (df['total_spent'] - df['total_spent'].min()) / (df['total_spent'].max() - df['total_spent'].min())
    df['order_score'] = (df['num_orders'] - df['num_orders'].min()) / (df['num_orders'].max() - df['num_orders'].min())
    df['recency_score'] = 1 - (df['last_order_days_ago'] / df['last_order_days_ago'].max())
    
    # Composite score: 50% spend, 30% orders, 20% recency
    df['composite_score'] = (
        0.5 * df['spend_score'] +
        0.3 * df['order_score'] +
        0.2 * df['recency_score']
    )
    
    top_composite = df.nlargest(10, 'composite_score')
    print(top_composite[['name', 'total_spent', 'num_orders', 'last_order_days_ago', 'composite_score']].round(3))


# ============================================================================
# 6. Handling Missing Data
# ============================================================================

def demo_missing_data():
    """
    INTUITION - Real-World Data is Messy:
    
    Missing data happens:
    - User didn't fill optional fields
    - Sensor failure (IoT data)
    - Integration errors (API timeout)
    - Historical data incomplete
    
    Strategies:
    - Drop rows/columns
    - Fill with mean/median/mode
    - Forward/backward fill (time series)
    - Predict missing values (advanced)
    """
    print("\n" + "=" * 70)
    print("6. Handling Missing Data")
    print("=" * 70)
    print()
    print("💡 INTUITION: Real Data is Always Messy")
    print("   Why missing?")
    print("   - User skipped optional fields")
    print("   - Sensor failure")
    print("   - API timeout")
    print("   Solutions: Drop, fill, or predict")
    print()
    
    # Create data with missing values
    df = pd.DataFrame({
        'user_id': range(1, 11),
        'age': [25, 30, np.nan, 45, 28, np.nan, 35, 42, 29, 31],
        'income': [50000, np.nan, 65000, np.nan, 48000, 70000, np.nan, 80000, 52000, 62000],
        'city': ['NYC', 'LA', None, 'Chicago', 'NYC', 'LA', None, 'Chicago', 'NYC', 'LA'],
        'purchases': [5, 3, 8, np.nan, 12, 6, 9, np.nan, 4, 7]
    })
    
    print("📊 Data with Missing Values:")
    print(df)
    print()
    
    # Check missing data
    print("Missing Value Summary:")
    missing = df.isnull().sum()
    print(missing)
    print()
    
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    print("Percentage Missing:")
    print(missing_pct)
    print()
    
    # Strategy 1: Drop rows with any missing values
    print("Strategy 1: Drop rows with ANY missing value:")
    df_dropped = df.dropna()
    print(f"   Original: {len(df)} rows")
    print(f"   After drop: {len(df_dropped)} rows")
    print(f"   Lost: {len(df) - len(df_dropped)} rows ({(len(df) - len(df_dropped))/len(df)*100:.0f}%)")
    print()
    
    # Strategy 2: Drop rows only if specific column is missing
    print("Strategy 2: Drop only if 'age' is missing:")
    df_dropped_age = df.dropna(subset=['age'])
    print(f"   Kept: {len(df_dropped_age)} rows")
    print()
    
    # Strategy 3: Fill with mean/median
    print("Strategy 3: Fill numerical with median:")
    df_filled = df.copy()
    df_filled['age'].fillna(df_filled['age'].median(), inplace=True)
    df_filled['income'].fillna(df_filled['income'].median(), inplace=True)
    df_filled['purchases'].fillna(df_filled['purchases'].median(), inplace=True)
    print(df_filled)
    print()
    
    # Strategy 4: Fill categorical with mode
    print("Strategy 4: Fill 'city' with most common value:")
    df_filled['city'].fillna(df_filled['city'].mode()[0], inplace=True)
    print(df_filled)
    print()
    
    # Strategy 5: Forward fill (time series)
    print("Strategy 5: Forward Fill (for time series):")
    time_series = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'temperature': [72, np.nan, np.nan, 75, 74, np.nan, 73, 72, np.nan, 71]
    })
    print("Before forward fill:")
    print(time_series)
    print()
    
    time_series['temperature'].fillna(method='ffill', inplace=True)
    print("After forward fill (use last known value):")
    print(time_series)


# ============================================================================
# 7. Real-World: Data Cleaning Pipeline
# ============================================================================

def demo_data_cleaning_pipeline():
    """
    INTUITION - End-to-End Data Cleaning:
    
    Raw data from API/CSV → Clean data for ML
    
    Steps:
    1. Load data
    2. Check data types
    3. Handle missing values
    4. Remove duplicates
    5. Fix inconsistencies
    6. Create features
    7. Export clean data
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Data Cleaning Pipeline")
    print("=" * 70)
    print()
    print("💡 INTUITION: Raw Data → Clean ML-Ready Data")
    print("   Real data is ALWAYS messy!")
    print("   Pipeline: Load → Clean → Transform → Export")
    print()
    
    # Simulate messy real-world data
    raw_data = {
        'customer_id': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9],  # Duplicate ID
        'name': ['Alice', 'Bob', None, 'Diana', 'Eve', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy'],
        'email': ['alice@example.com', 'bob@TEST.COM', 'charlie@example.com', None, 
                  'eve@example.com', 'eve@example.com', 'frank@Example.com', 
                  'grace@example.com', 'invalid-email', 'ivy@example.com'],
        'age': ['25', '32', '45', 'unknown', '28', '28', '35', '42', '29', '200'],  # Bad data
        'purchase_amount': [100.50, 250.75, None, 175.00, 89.99, 89.99, 
                           310.25, 125.50, -50.00, 199.99],  # Negative value
        'purchase_date': ['2024-01-15', '2024/01/20', '2024-01-25', '2024-02-01',
                         '2024-02-05', '2024-02-05', '2024-02-10', '2024-02-15',
                         'invalid-date', '2024-02-20']
    }
    
    df = pd.DataFrame(raw_data)
    
    print("📊 Raw Data (messy!):")
    print(df)
    print()
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print()
    
    # STEP 1: Remove duplicates
    print("STEP 1: Remove Duplicates")
    df_clean = df.drop_duplicates(subset=['customer_id'], keep='first')
    print(f"   Removed {len(df) - len(df_clean)} duplicate rows")
    print()
    
    # STEP 2: Fix data types
    print("STEP 2: Fix Data Types")
    
    # Clean age (convert to numeric, handle 'unknown')
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    print(f"   Converted age to numeric ({df_clean['age'].isnull().sum()} invalid values)")
    
    # Clean email (lowercase)
    df_clean['email'] = df_clean['email'].str.lower().str.strip()
    print(f"   Standardized email format")
    print()
    
    # STEP 3: Handle missing values
    print("STEP 3: Handle Missing Values")
    
    # Fill missing names
    df_clean['name'].fillna('Unknown', inplace=True)
    print(f"   Filled {df_clean['name'].isnull().sum()} missing names")
    
    # Fill missing purchase amounts with median
    median_purchase = df_clean['purchase_amount'].median()
    df_clean['purchase_amount'].fillna(median_purchase, inplace=True)
    print(f"   Filled missing purchase amounts with median: ${median_purchase:.2f}")
    
    # Fill missing age with median
    median_age = df_clean['age'].median()
    df_clean['age'].fillna(median_age, inplace=True)
    print(f"   Filled missing ages with median: {median_age:.0f}")
    print()
    
    # STEP 4: Fix inconsistencies
    print("STEP 4: Fix Inconsistencies")
    
    # Remove invalid ages (e.g., 200)
    df_clean = df_clean[(df_clean['age'] >= 18) & (df_clean['age'] <= 100)]
    print(f"   Removed invalid ages (outside 18-100 range)")
    
    # Remove negative purchase amounts
    df_clean = df_clean[df_clean['purchase_amount'] >= 0]
    print(f"   Removed negative purchase amounts")
    
    # Validate email format (simple check)
    df_clean = df_clean[df_clean['email'].str.contains('@', na=False)]
    print(f"   Removed invalid email formats")
    print()
    
    # STEP 5: Parse dates
    print("STEP 5: Parse Dates")
    df_clean['purchase_date'] = pd.to_datetime(df_clean['purchase_date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['purchase_date'])
    print(f"   Parsed dates, removed invalid formats")
    print()
    
    # STEP 6: Create features
    print("STEP 6: Create Features (Feature Engineering)")
    
    # Age group
    df_clean['age_group'] = pd.cut(
        df_clean['age'], 
        bins=[0, 25, 35, 50, 100], 
        labels=['18-25', '26-35', '36-50', '50+']
    )
    print(f"   Created age_group feature")
    
    # Days since purchase
    df_clean['days_since_purchase'] = (pd.Timestamp.now() - df_clean['purchase_date']).dt.days
    print(f"   Created days_since_purchase feature")
    
    # Email domain
    df_clean['email_domain'] = df_clean['email'].str.split('@').str[1]
    print(f"   Extracted email domain")
    print()
    
    # FINAL: Clean data
    print("✅ Clean Data Ready for ML:")
    print(df_clean)
    print()
    print(f"Final shape: {df_clean.shape}")
    print(f"Data types:\n{df_clean.dtypes}")
    print()
    
    print("📊 Quality Report:")
    print(f"   Original rows: {len(df)}")
    print(f"   Clean rows: {len(df_clean)}")
    print(f"   Data quality: {len(df_clean)/len(df)*100:.1f}%")
    print(f"   Missing values: {df_clean.isnull().sum().sum()}")
    print()
    
    print("💾 Export:")
    print("   df_clean.to_csv('clean_data.csv', index=False)")
    print("   df_clean.to_parquet('clean_data.parquet')  # Faster!")
    print("   df_clean.to_sql('customers', con=db_engine)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐼 Pandas Basics for ML/AI Engineering\n")
    print("Focus: Real-world data manipulation and cleaning!")
    print()
    
    demo_why_pandas()
    demo_creating_dataframes()
    demo_selecting_filtering()
    demo_groupby()
    demo_sorting_ranking()
    demo_missing_data()
    demo_data_cleaning_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. DataFrame = Excel spreadsheet in Python (but better!)
2. Load data: CSV, JSON, SQL, APIs, Excel
3. Filter: Boolean indexing, SQL-like WHERE clauses
4. Aggregate: GroupBy for business intelligence queries
5. Clean: Handle missing data, duplicates, inconsistencies
6. Transform: Create features, parse dates, normalize

Pandas vs Excel:
- Handle millions of rows (Excel: 1M limit)
- Reproducible (code, not clicks)
- Integrate with ML models
- Automate repetitive tasks
- Version control your analysis

Next Steps:
- 04_pandas_data_cleaning.py - Advanced cleaning techniques
- 05_pandas_feature_engineering.py - ML feature creation
- 06_numpy_pandas_for_ml.py - Complete ML preprocessing pipeline

Real Projects:
80% of ML work is data cleaning and preparation.
Master Pandas = Master ML engineering!
""")


if __name__ == "__main__":
    main()
