"""
Numerical Feature Engineering for ML/AI

Transform raw numbers into features that models love!
Focus: Real-world techniques you'll use daily in ML engineering.

Install: poetry add numpy pandas scikit-learn
Run: poetry run python 05-feature-engineering/examples/01_numerical_features.py
"""

import math
from typing import List, Tuple


# ============================================================================
# 1. Feature Scaling - Making Features Comparable
# ============================================================================

def demo_feature_scaling():
    """
    Feature Scaling: Make different features comparable by scaling to similar ranges.
    
    INTUITION - The Shouting Problem:
    
    You have two features:
    - Age: 25-65 (small numbers, "quiet voice")
    - Salary: $30,000-$200,000 (BIG numbers, "LOUD voice")
    
    Without scaling:
    Model: "Salary changed $1000? MASSIVE UPDATE!"
    Model: "Age changed 1 year? Barely notice it."
    
    But age might be MORE important for your prediction!
    
    Solution: Scale both to similar ranges so model "hears" them equally.
    
    WHY IT MATTERS:
    - Distance-based algorithms (KNN, SVM) are very sensitive to scale
    - Gradient descent converges 10-100x faster
    - Neural networks train much better
    - Some algorithms (trees) don't care, but scaling never hurts
    
    ML Use Cases:
    - Pre-processing for neural networks
    - K-Nearest Neighbors, SVM, K-Means
    - Any algorithm using distance or gradient descent
    """
    print("=" * 70)
    print("1. Feature Scaling - Making Features Comparable")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Meeting Room Problem")
    print("   Quiet person (age: 25-65) has great insights")
    print("   Loud person (salary: $30k-$200k) talks over them")
    print("   Solution: Give them equal voice → Scale features!")
    print()
    
    # Real-world scenario: Employee data
    employees = [
        {"age": 25, "salary": 50000, "years_exp": 2},
        {"age": 35, "salary": 80000, "years_exp": 10},
        {"age": 45, "salary": 120000, "years_exp": 20},
        {"age": 28, "salary": 60000, "years_exp": 4},
        {"age": 50, "salary": 150000, "years_exp": 25},
    ]
    
    # Extract features
    ages = [e["age"] for e in employees]
    salaries = [e["salary"] for e in employees]
    experience = [e["years_exp"] for e in employees]
    
    print("📊 Original Data:")
    print(f"   Ages:       {ages}")
    print(f"   Salaries:   {salaries}")
    print(f"   Experience: {experience}")
    print()
    print("   Problem: Salary dominates (100x larger scale)!")
    print()
    
    # Method 1: Standardization (Z-score normalization)
    def standardize(values: List[float]) -> List[float]:
        """Transform to mean=0, std=1."""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        
        if std == 0:  # All values are the same
            return [0.0] * len(values)
        
        return [(x - mean) / std for x in values]
    
    # Method 2: Min-Max Scaling
    def min_max_scale(values: List[float], new_min: float = 0, new_max: float = 1) -> List[float]:
        """Transform to range [new_min, new_max]."""
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:  # All values are the same
            return [(new_min + new_max) / 2] * len(values)
        
        range_val = max_val - min_val
        new_range = new_max - new_min
        
        return [((x - min_val) / range_val) * new_range + new_min for x in values]
    
    # Method 3: Robust Scaling (using median & IQR)
    def robust_scale(values: List[float]) -> List[float]:
        """Transform using median and IQR (robust to outliers)."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        # Median
        if n % 2 == 0:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            median = sorted_vals[n // 2]
        
        # Q1 and Q3
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        
        if iqr == 0:  # All values in IQR are the same
            return [0.0] * len(values)
        
        return [(x - median) / iqr for x in values]
    
    # Apply standardization
    print("🔧 Method 1: Standardization (Mean=0, Std=1)")
    ages_std = standardize(ages)
    salaries_std = standardize(salaries)
    exp_std = standardize(experience)
    
    print(f"   Ages:       {[f'{x:.2f}' for x in ages_std]}")
    print(f"   Salaries:   {[f'{x:.2f}' for x in salaries_std]}")
    print(f"   Experience: {[f'{x:.2f}' for x in exp_std]}")
    print()
    print("   ✅ All features now have similar scale!")
    print("   Use when: Neural networks, SVM, KNN")
    print()
    
    # Apply min-max scaling
    print("🔧 Method 2: Min-Max Scaling [0, 1]")
    ages_minmax = min_max_scale(ages)
    salaries_minmax = min_max_scale(salaries)
    exp_minmax = min_max_scale(experience)
    
    print(f"   Ages:       {[f'{x:.2f}' for x in ages_minmax]}")
    print(f"   Salaries:   {[f'{x:.2f}' for x in salaries_minmax]}")
    print(f"   Experience: {[f'{x:.2f}' for x in exp_minmax]}")
    print()
    print("   ✅ All features in [0, 1] range!")
    print("   Use when: Neural networks, image processing")
    print()
    
    # Apply robust scaling
    print("🔧 Method 3: Robust Scaling (Median & IQR)")
    ages_robust = robust_scale(ages)
    salaries_robust = robust_scale(salaries)
    exp_robust = robust_scale(experience)
    
    print(f"   Ages:       {[f'{x:.2f}' for x in ages_robust]}")
    print(f"   Salaries:   {[f'{x:.2f}' for x in salaries_robust]}")
    print(f"   Experience: {[f'{x:.2f}' for x in exp_robust]}")
    print()
    print("   ✅ Less affected by outliers!")
    print("   Use when: Data has extreme outliers")
    print()
    
    # Real impact demonstration
    print("🎯 REAL IMPACT: KNN Distance Calculation")
    print()
    print("   Scenario: Find similar employees")
    print("   Employee A: Age=30, Salary=$70k, Exp=5yr")
    print("   Employee B: Age=31, Salary=$170k, Exp=6yr")
    print()
    
    # Without scaling
    diff_age = 31 - 30  # 1
    diff_salary = 170000 - 70000  # 100,000
    diff_exp = 6 - 5  # 1
    
    distance_unscaled = math.sqrt(diff_age**2 + diff_salary**2 + diff_exp**2)
    
    print(f"   WITHOUT scaling:")
    print(f"   Distance = √(1² + 100000² + 1²) = {distance_unscaled:.0f}")
    print(f"   → Salary dominates! Age & experience ignored!")
    print()
    
    # With scaling (using standardization)
    def calc_scaled_distance(val1, val2, data):
        """Calculate distance after standardization."""
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance ** 0.5
        
        if std == 0:
            return 0.0
        
        return ((val2 - mean) / std) - ((val1 - mean) / std)
    
    # For this simple example, let's use approximate values
    # In reality, you'd standardize based on your training data
    diff_age_scaled = (31 - 36.6) / 10  # Using mean=36.6, std≈10
    diff_salary_scaled = (170000 - 92000) / 40000  # Using mean=92k, std≈40k
    diff_exp_scaled = (6 - 12.2) / 9  # Using mean=12.2, std≈9
    
    distance_scaled = math.sqrt(
        diff_age_scaled**2 + diff_salary_scaled**2 + diff_exp_scaled**2
    )
    
    print(f"   WITH scaling:")
    print(f"   Distance = {distance_scaled:.2f}")
    print(f"   → All features contribute fairly!")
    print()
    
    print("💡 DECISION GUIDE:")
    print()
    print("   ┌─────────────────────┬────────────────────────────┐")
    print("   │ Situation           │ Best Scaling Method        │")
    print("   ├─────────────────────┼────────────────────────────┤")
    print("   │ Neural Networks     │ Standardization or MinMax  │")
    print("   │ K-NN, SVM, K-Means  │ Standardization            │")
    print("   │ Data with outliers  │ Robust Scaling             │")
    print("   │ Image data (0-255)  │ MinMax to [0,1]            │")
    print("   │ Tree-based (RF, XGB)│ No scaling needed!         │")
    print("   └─────────────────────┴────────────────────────────┘")


# ============================================================================
# 2. Binning - Converting Continuous to Categorical
# ============================================================================

def demo_binning():
    """
    Binning: Group continuous values into discrete bins/categories.
    
    INTUITION - Age Groups in Marketing:
    
    Instead of: Age = 23.5 years (too specific, noisy)
    Use:       Age Group = "Young Adult (18-30)"
    
    Why? Patterns often work better with groups:
    - 24 year old and 26 year old behave similarly
    - But 18 year old and 65 year old are very different!
    
    Real Example: Credit scoring
    - Income $49k and $51k → Basically same credit risk
    - But $30k vs $150k → Very different risk profiles
    - Solution: Bin into "Low", "Medium", "High" income
    
    WHEN TO USE:
    ✅ Capture non-linear relationships
    ✅ Reduce noise from overly-precise measurements
    ✅ Make models interpretable ("High income customers churn less")
    ✅ Handle outliers (extreme values go to extreme bins)
    
    ⚠️  CAUTION:
    - Lose information (26 and 29 treated identically)
    - Need to choose bin boundaries carefully
    - Tree-based models already do this automatically!
    
    ML Use Cases:
    - Feature engineering for linear models
    - Creating interpretable segments
    - Reducing impact of outliers
    - Finding non-linear patterns
    """
    print("\n" + "=" * 70)
    print("2. Binning - Making Continuous Data Categorical")
    print("=" * 70)
    print()
    print("💭 INTUITION: Netflix Age Groups")
    print("   Instead of tracking age as 23, 24, 25...")
    print("   Group into: Kids (0-12), Teens (13-17), Adults (18-60), Seniors (60+)")
    print("   Why? 24 and 25 year olds watch similar content!")
    print("   Same strategy works for income, time, prices, etc.")
    print()
    
    # Real-world scenario: Customer purchase amounts
    purchases = [15, 25, 30, 45, 50, 55, 75, 90, 120, 150, 180, 250, 300, 500, 1200]
    
    print("📊 Scenario: E-commerce Purchase Amounts")
    print(f"   Data: ${', $'.join(str(x) for x in purchases)}")
    print()
    
    # Method 1: Equal-width binning
    def equal_width_bins(values: List[float], n_bins: int) -> List[Tuple[float, float]]:
        """Create bins of equal width."""
        min_val = min(values)
        max_val = max(values)
        width = (max_val - min_val) / n_bins
        
        bins = []
        for i in range(n_bins):
            start = min_val + i * width
            end = min_val + (i + 1) * width
            bins.append((start, end))
        
        return bins
    
    # Method 2: Equal-frequency binning (quantile-based)
    def equal_frequency_bins(values: List[float], n_bins: int) -> List[Tuple[float, float]]:
        """Create bins with equal number of samples."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        samples_per_bin = n // n_bins
        
        bins = []
        for i in range(n_bins):
            start_idx = i * samples_per_bin
            end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else n
            
            start = sorted_vals[start_idx]
            end = sorted_vals[end_idx - 1]
            
            bins.append((start, end))
        
        return bins
    
    # Method 3: Custom bins (domain knowledge)
    custom_bins = [
        (0, 50, "Low Spender"),
        (50, 150, "Medium Spender"),
        (150, 500, "High Spender"),
        (500, float('inf'), "VIP Customer")
    ]
    
    def assign_to_bin(value: float, bins: List[Tuple[float, float, str]]) -> str:
        """Assign value to custom bin."""
        for start, end, label in bins:
            if start <= value < end:
                return label
        return bins[-1][2]  # Last bin for values >= last threshold
    
    # Apply equal-width binning
    print("🔧 Method 1: Equal-Width Bins (4 bins)")
    ew_bins = equal_width_bins(purchases, 4)
    for i, (start, end) in enumerate(ew_bins):
        count = sum(1 for p in purchases if start <= p < end or (i == 3 and p == end))
        print(f"   Bin {i+1}: [${start:.0f}, ${end:.0f}] → {count} purchases")
    
    print()
    print("   ⚠️  Problem: Uneven distribution! Most purchases in first bin.")
    print()
    
    # Apply equal-frequency binning
    print("🔧 Method 2: Equal-Frequency Bins (4 bins)")
    ef_bins = equal_frequency_bins(purchases, 4)
    for i, (start, end) in enumerate(ef_bins):
        print(f"   Bin {i+1}: [${start:.0f}, ${end:.0f}]")
    
    print()
    print("   ✅ Better: Each bin has ~same number of samples!")
    print()
    
    # Apply custom binning
    print("🔧 Method 3: Custom Bins (Business Logic)")
    for start, end, label in custom_bins:
        count = sum(1 for p in purchases if start <= p < end)
        avg = sum(p for p in purchases if start <= p < end) / count if count > 0 else 0
        print(f"   {label:15} [${start:>3.0f}, ${end if end != float('inf') else '∞':>4}] → {count:2} purchases, avg ${avg:.0f}")
    
    print()
    print("   ✅ Most interpretable: Clear business segments!")
    print()
    
    # Real-world example: Age binning
    print("🎯 REAL EXAMPLE: Age Groups for Marketing")
    print()
    
    ages = [22, 25, 28, 32, 35, 38, 42, 45, 50, 55, 60, 65, 70]
    age_bins = [
        (18, 30, "Young Adult", "TikTok, Instagram"),
        (30, 45, "Middle Age", "Facebook, LinkedIn"),
        (45, 60, "Mature", "Facebook, Email"),
        (60, 100, "Senior", "TV, Email")
    ]
    
    for start, end, label, channels in age_bins:
        count = sum(1 for age in ages if start <= age < end)
        print(f"   {label:12} ({start}-{end}): {count:2} people → Target: {channels}")
    
    print()
    print("   💡 Marketing can now target by age group, not individual ages!")
    print()
    
    print("💡 DECISION GUIDE:")
    print()
    print("   Equal-Width:")
    print("   • Pro: Simple, interpretable")
    print("   • Con: Uneven distribution with skewed data")
    print("   • Use: When data is uniformly distributed")
    print()
    print("   Equal-Frequency:")
    print("   • Pro: Balanced bins, works with skewed data")
    print("   • Con: Bin widths vary")
    print("   • Use: When you want similar sample sizes per bin")
    print()
    print("   Custom Bins:")
    print("   • Pro: Incorporates domain knowledge, interpretable")
    print("   • Con: Requires expertise, manual tuning")
    print("   • Use: When you understand the business problem!")
    print()
    print("⚠️  REMEMBER:")
    print("   • Binning loses information (26 and 29 treated the same)")
    print("   • Tree-based models (Random Forest, XGBoost) do this automatically")
    print("   • Best for linear models that can't capture non-linearity")


# ============================================================================
# 3. Transformations - Handling Skewed Data
# ============================================================================

def demo_transformations():
    """
    Transformations: Apply mathematical functions to fix skewed distributions.
    
    INTUITION - Income Distribution Problem:
    
    Most people: $30k-$80k (clustered together)
    Few people: $1M-$10M (extreme outliers)
    
    Problem for ML:
    - Distance metrics get dominated by outliers
    - Linear models can't handle this well
    - Gradients become unstable
    
    Solution: Transform to make distribution more "normal"
    - log(income): Compresses large values, expands small values
    - √(income): Moderate compression
    - 1/income: Extreme compression (rarely used)
    
    WHY IT WORKS:
    log($30,000) = 4.48
    log($1,000,000) = 6.00
    
    Now difference is 1.52 instead of $970,000!
    Large values are "pulled in", making data more manageable.
    
    COMMON USE CASES:
    - Income, revenue, prices (right-skewed)
    - Website traffic, user engagement (power law)
    - Population, counts (exponential growth)
    - Time until event (survival analysis)
    
    WHEN TO USE:
    ✅ Data is right-skewed (long tail on right)
    ✅ Spans multiple orders of magnitude (1 to 1,000,000)
    ✅ Model assumptions require normality
    ✅ Want to reduce impact of outliers
    
    ⚠️  CAUTION:
    - Changes interpretation (1 unit = multiplicative change)
    - Can't log-transform zero or negative values
    - May need to transform back for predictions
    """
    print("\n" + "=" * 70)
    print("3. Transformations - Fixing Skewed Distributions")
    print("=" * 70)
    print()
    print("💭 INTUITION: Income Data Problem")
    print("   Most people: $30k-$80k (normal incomes)")
    print("   Few people: $500k-$5M (millionaires pulling the scale!)")
    print()
    print("   Problem: Outliers dominate distance calculations")
    print("   Solution: log(income) compresses large values")
    print()
    print("   Before: $50k vs $5M = $4.95M difference")
    print("   After:  log($50k)=4.7 vs log($5M)=6.7 = 2.0 difference")
    print("   Now the difference is manageable!")
    print()
    
    # Real-world scenario: Website daily visitors (heavily skewed)
    daily_visitors = [
        50, 75, 100, 150, 200, 250, 300, 400, 500,  # Small sites
        600, 800, 1000, 1500, 2000, 3000,  # Medium sites
        5000, 8000, 12000, 20000,  # Popular sites
        50000, 100000, 500000  # Viral sites (outliers!)
    ]
    
    print("📊 Scenario: Website Daily Visitors (Right-Skewed)")
    print(f"   Range: {min(daily_visitors):,} to {max(daily_visitors):,} visitors")
    print(f"   Mean: {sum(daily_visitors) / len(daily_visitors):,.0f}")
    
    sorted_visitors = sorted(daily_visitors)
    median_idx = len(sorted_visitors) // 2
    median = sorted_visitors[median_idx]
    print(f"   Median: {median:,}")
    print()
    print("   ⚠️  Mean >> Median = Data is heavily right-skewed!")
    print()
    
    # Method 1: Log transformation
    def log_transform(values: List[float]) -> List[float]:
        """Apply log transformation (base 10)."""
        return [math.log10(x) if x > 0 else 0 for x in values]
    
    # Method 2: Square root transformation
    def sqrt_transform(values: List[float]) -> List[float]:
        """Apply square root transformation."""
        return [math.sqrt(x) if x >= 0 else 0 for x in values]
    
    # Method 3: Box-Cox transformation (simplified version: log(x + 1))
    def log1p_transform(values: List[float]) -> List[float]:
        """Apply log(x + 1) transformation (handles zeros)."""
        return [math.log10(x + 1) for x in values]
    
    # Apply transformations
    visitors_log = log_transform(daily_visitors)
    visitors_sqrt = sqrt_transform(daily_visitors)
    visitors_log1p = log1p_transform(daily_visitors)
    
    print("🔧 Transformation Results:")
    print()
    
    # Calculate statistics for each
    def calc_stats(values: List[float]) -> Tuple[float, float, float, float]:
        """Calculate min, max, mean, std."""
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
        return min(values), max(values), mean, std
    
    print("   Original Data:")
    min_v, max_v, mean_v, std_v = calc_stats(daily_visitors)
    print(f"   Range: {min_v:,.0f} to {max_v:,.0f}")
    print(f"   Mean ± Std: {mean_v:,.0f} ± {std_v:,.0f}")
    print(f"   Skewness: Very high (long right tail)")
    print()
    
    print("   After log10(x):")
    min_v, max_v, mean_v, std_v = calc_stats(visitors_log)
    print(f"   Range: {min_v:.2f} to {max_v:.2f}")
    print(f"   Mean ± Std: {mean_v:.2f} ± {std_v:.2f}")
    print(f"   ✅ Much more compact! Orders of magnitude → simple numbers")
    print()
    
    print("   After √(x):")
    min_v, max_v, mean_v, std_v = calc_stats(visitors_sqrt)
    print(f"   Range: {min_v:.0f} to {max_v:.0f}")
    print(f"   Mean ± Std: {mean_v:.0f} ± {std_v:.0f}")
    print(f"   ✅ Moderate compression (less aggressive than log)")
    print()
    
    print("   After log10(x + 1):")
    min_v, max_v, mean_v, std_v = calc_stats(visitors_log1p)
    print(f"   Range: {min_v:.2f} to {max_v:.2f}")
    print(f"   Mean ± Std: {mean_v:.2f} ± {std_v:.2f}")
    print(f"   ✅ Like log, but handles zeros!")
    print()
    
    # Real-world interpretation
    print("🎯 REAL IMPACT: Model Training")
    print()
    print("   WITHOUT transformation:")
    print("   • Model focuses on high-traffic sites (500k visitors)")
    print("   • Small sites (50-500 visitors) get ignored")
    print("   • Gradient descent unstable (huge value differences)")
    print()
    print("   WITH log transformation:")
    print("   • All sites get fair representation")
    print("   • Patterns visible across all scales")
    print("   • Faster, more stable training")
    print()
    
    # Choosing the right transformation
    print("💡 WHICH TRANSFORMATION TO USE?")
    print()
    print("   Log Transform (log(x)):")
    print("   • Use: Strong right skew, multiple orders of magnitude")
    print("   • Examples: Income, prices, web traffic")
    print("   • ⚠️  Can't handle x=0 or negative values")
    print()
    print("   Log1p Transform (log(x+1)):")
    print("   • Use: Same as log, but data includes zeros")
    print("   • Examples: Count data (0 purchases, 0 clicks)")
    print("   • ✅ Safe for zeros!")
    print()
    print("   Square Root (√x):")
    print("   • Use: Moderate skew")
    print("   • Examples: Areas, counts with moderate range")
    print("   • ✅ Works on zeros, less aggressive than log")
    print()
    print("   Reciprocal (1/x):")
    print("   • Use: Left skew (rare)")
    print("   • Examples: Time until event (small times are important)")
    print("   • ⚠️  Can't handle x=0, flips order")
    print()
    
    # Practical example
    print("🔍 DECISION TREE:")
    print()
    print("   Is data right-skewed (long tail on right)? ")
    print("   ├─ Yes")
    print("   │  └─ Contains zeros?")
    print("   │     ├─ Yes → Use log1p(x)")
    print("   │     └─ No  → Use log(x)")
    print("   └─ No (approximately normal)")
    print("      └─ No transformation needed!")
    print()
    
    # Reversing transformations for predictions
    print("⚠️  IMPORTANT: Reverse Transform for Predictions")
    print()
    print("   Trained on log-transformed data?")
    print("   Model predicts: log(visitors) = 4.5")
    print("   Actual prediction: 10^4.5 = 31,623 visitors")
    print()
    print("   Always remember:")
    print("   • Transform input features before prediction")
    print("   • Reverse-transform output predictions if target was transformed")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all numerical feature engineering demonstrations."""
    print("\n🔢 Numerical Feature Engineering for ML/AI\n")
    print("Focus: Transform raw numbers into ML-ready features!")
    print()
    
    demo_feature_scaling()
    demo_binning()
    demo_transformations()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Feature Scaling: Make features comparable
   • Standardization: mean=0, std=1 (most common)
   • Min-Max: scale to [0,1] range
   • Robust: use median & IQR (outlier-resistant)
   → Use for: Neural nets, KNN, SVM, K-Means

2. Binning: Convert continuous to categorical
   • Equal-width: simple, but uneven distribution
   • Equal-frequency: balanced bins
   • Custom: domain knowledge (best when possible)
   → Use for: Non-linear patterns, interpretability

3. Transformations: Fix skewed distributions
   • Log: strong skew, multiple magnitudes
   • Log1p: like log, but handles zeros
   • Sqrt: moderate skew
   → Use for: Income, prices, traffic, counts

GOLDEN RULES:
✅ Always scale before distance-based algorithms
✅ Fit scaler on training data, transform train & test
✅ Tree-based models don't need scaling!
✅ Log-transform heavy-tailed distributions
✅ Document all transformations (reverse for predictions)

Next: 02_categorical_encoding.py - Handle categories in ML!
""")


if __name__ == "__main__":
    main()
